// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, HashSet};
use std::sync::{Arc, Mutex};

use kvbm_common::{BlockId, LogicalLayoutHandle, LogicalResourceId, SequenceHash};
use kvbm_logical::ImmutableBlock;
use kvbm_physical::transfer::TransferCompleteNotification;
use kvbm_protocols::connector::{
    ActionFailure, ActionId, ActionStatus, BundleOnboardPlan, FindBlocksHandle, LeaderEngine,
    LeaderEngineError, OnboardHandle, RequestId, ResourceDestination, ResourceOnboard,
};

use super::{BundleOnboard, OnboardTransition};
use crate::G2;
use crate::tiering::engine::driver::ActionRecord;
use crate::tiering::engine::inflight::InflightKey;
use crate::tiering::engine::local::LocalConnectorEngine;

type ResourcePins = Vec<ImmutableBlock<G2>>;
type SourceLeases = BTreeMap<LogicalResourceId, ResourcePins>;
type BundleOnboardTransaction = Arc<Mutex<BundleOnboard<ResourcePins>>>;

struct DispatchedResource {
    resource: LogicalResourceId,
    destination_block_ids: Vec<BlockId>,
    notification: TransferCompleteNotification,
}

struct DispatchFailure {
    resource: LogicalResourceId,
    destination_block_ids: Vec<BlockId>,
}

struct DispatchBatch {
    transfers: Vec<DispatchedResource>,
    failure: Option<DispatchFailure>,
}

struct CompletedResource {
    resource: LogicalResourceId,
    destination_block_ids: Vec<BlockId>,
    result: anyhow::Result<()>,
}

impl LocalConnectorEngine {
    pub(in crate::tiering::engine) fn start_bundle_onboard(
        self: Arc<Self>,
        req: &RequestId,
        plan: BundleOnboardPlan,
    ) -> Result<OnboardHandle, LeaderEngineError> {
        validate_resource_transfers(self.as_ref(), &plan.resources)?;
        let (source_leases, inflight_hashes) = self.acquire_bundle_sources(&plan)?;
        self.start_bundle_onboard_with_sources(req, plan, source_leases, inflight_hashes)
    }

    pub(in crate::tiering::engine) fn start_searched_bundle_onboard(
        self: Arc<Self>,
        handle: &FindBlocksHandle,
        destinations: Vec<ResourceDestination>,
        num_external_tokens: usize,
    ) -> Result<OnboardHandle, LeaderEngineError> {
        let search_id = handle
            .search_id()
            .ok_or(LeaderEngineError::SearchNotMatched)?;
        let state = self
            .bundle_searches
            .get(&search_id)
            .ok_or(LeaderEngineError::SearchNotMatched)?;
        if state.request_id != *handle.request_id() {
            return Err(LeaderEngineError::FindBlocksDesync);
        }
        if state.matched_tokens != num_external_tokens {
            return Err(LeaderEngineError::ExternalTokensMismatch {
                expected: state.matched_tokens,
                got: num_external_tokens,
            });
        }
        let (plan, source_leases, inflight_hashes) = searched_bundle_plan(&state, destinations)?;
        validate_resource_transfers(self.as_ref(), &plan.resources)?;
        let request_id = state.request_id.clone();
        drop(state);
        self.bundle_searches.remove(&search_id);
        self.start_bundle_onboard_with_sources(&request_id, plan, source_leases, inflight_hashes)
    }

    pub(in crate::tiering::engine) fn start_resource_onboard(
        self: Arc<Self>,
        req: &RequestId,
        resources: Vec<ResourceOnboard>,
    ) -> Result<OnboardHandle, LeaderEngineError> {
        if resources.is_empty() {
            return Err(LeaderEngineError::InvalidResourceOnboard {
                reason: "at least one resource is required".to_owned(),
            });
        }
        validate_resource_transfers(self.as_ref(), &resources)?;
        let action_id = ActionId::new();
        let cell = Arc::new(Mutex::new(ActionStatus::Pending));
        self.actions.insert(
            action_id,
            ActionRecord::new(req.clone(), Arc::downgrade(&cell)),
        );
        self.by_request
            .entry(req.clone())
            .or_default()
            .push(action_id);
        let destination_block_ids = resources
            .iter()
            .flat_map(|transfer| transfer.destination_block_ids.iter().copied())
            .collect::<Vec<_>>();
        let request_id = req.clone();
        let driver = Arc::clone(&self);
        let terminal_block_ids = destination_block_ids.clone();
        self.leader.runtime().spawn(async move {
            let (dispatch_failure, completed) = driver.execute_resource_onboards(resources).await;
            let outcome = fold_resource_onboard(dispatch_failure, completed);
            driver.finish_load_action(action_id, &request_id, outcome, terminal_block_ids);
        });

        let engine: Arc<dyn LeaderEngine> = self;
        Ok(OnboardHandle::new(
            action_id,
            Arc::downgrade(&engine),
            cell,
            destination_block_ids,
        ))
    }

    fn start_bundle_onboard_with_sources(
        self: Arc<Self>,
        req: &RequestId,
        plan: BundleOnboardPlan,
        source_leases: SourceLeases,
        inflight_hashes: Vec<SequenceHash>,
    ) -> Result<OnboardHandle, LeaderEngineError> {
        let mut transaction =
            BundleOnboard::new(plan.identity, plan.key, source_leases).map_err(|error| {
                LeaderEngineError::InvalidBundleTransfer {
                    reason: error.to_string(),
                }
            })?;
        transaction.start();
        let transaction = Arc::new(Mutex::new(transaction));

        let action_id = ActionId::new();
        let cell = Arc::new(Mutex::new(ActionStatus::Pending));
        self.actions.insert(
            action_id,
            ActionRecord::new(req.clone(), Arc::downgrade(&cell))
                .with_inflight(InflightKey::Action(action_id)),
        );
        self.inflight
            .lock()
            .expect("inflight-guard mutex poisoned")
            .record(InflightKey::Action(action_id), inflight_hashes);
        self.by_request
            .entry(req.clone())
            .or_default()
            .push(action_id);
        let handle_dest_ids = plan
            .resources
            .iter()
            .flat_map(|transfer| transfer.destination_block_ids.iter().copied())
            .collect::<Vec<_>>();
        let request_id = req.clone();
        let driver = Arc::clone(&self);
        let terminal_dest_ids = handle_dest_ids.clone();
        self.leader.runtime().spawn(async move {
            let outcome = driver
                .drive_bundle_onboard(plan.resources, transaction)
                .await;
            driver.finish_load_action(action_id, &request_id, outcome, terminal_dest_ids);
        });

        let engine: Arc<dyn LeaderEngine> = self;
        Ok(OnboardHandle::new(
            action_id,
            Arc::downgrade(&engine),
            cell,
            handle_dest_ids,
        ))
    }

    fn acquire_bundle_sources(
        &self,
        plan: &BundleOnboardPlan,
    ) -> Result<(SourceLeases, Vec<SequenceHash>), LeaderEngineError> {
        let lease = self
            .bundle_index
            .lock()
            .expect("bundle-index mutex poisoned")
            .lease_exact(&plan.identity, &plan.key)
            .ok_or_else(|| LeaderEngineError::InvalidBundleTransfer {
                reason: "bundle is not committed in the local index".to_owned(),
            })?;
        let mut leased_resources = lease.into_resources();
        let mut source_leases = BTreeMap::new();
        let mut inflight_hashes = Vec::new();
        for transfer in &plan.resources {
            let pins = leased_resources.remove(&transfer.resource).ok_or_else(|| {
                LeaderEngineError::InvalidBundleTransfer {
                    reason: format!(
                        "bundle lease is missing logical resource {:?}",
                        transfer.resource
                    ),
                }
            })?;
            let leased_ids = pins.iter().map(|pin| pin.block_id()).collect::<Vec<_>>();
            if leased_ids != transfer.source_block_ids {
                return Err(LeaderEngineError::InvalidBundleTransfer {
                    reason: format!(
                        "resource {:?} source ids do not match the committed bundle lease",
                        transfer.resource
                    ),
                });
            }
            inflight_hashes.extend(pins.iter().map(|pin| pin.sequence_hash()));
            source_leases.insert(transfer.resource, pins);
        }
        Ok((source_leases, inflight_hashes))
    }

    async fn drive_bundle_onboard(
        &self,
        resources: Vec<ResourceOnboard>,
        transaction: BundleOnboardTransaction,
    ) -> ActionStatus {
        let (dispatch_failure, completed) = self.execute_resource_onboards(resources).await;
        fold_bundle_onboard(transaction, dispatch_failure, completed)
    }

    async fn execute_resource_onboards(
        &self,
        resources: Vec<ResourceOnboard>,
    ) -> (Option<DispatchFailure>, Vec<CompletedResource>) {
        let batch = self.dispatch_bundle_onboard(resources);
        let completed =
            futures::future::join_all(batch.transfers.into_iter().map(|dispatched| async move {
                CompletedResource {
                    resource: dispatched.resource,
                    destination_block_ids: dispatched.destination_block_ids,
                    result: dispatched.notification.await,
                }
            }))
            .await;
        (batch.failure, completed)
    }

    fn dispatch_bundle_onboard(&self, resources: Vec<ResourceOnboard>) -> DispatchBatch {
        let mut dispatched = Vec::with_capacity(resources.len());
        for transfer in resources {
            match self.leader.execute_local_transfer_for_resource(
                transfer.resource,
                LogicalLayoutHandle::G2,
                LogicalLayoutHandle::G1,
                transfer.source_block_ids,
                transfer.destination_block_ids.clone(),
                kvbm_physical::TransferOptions::default(),
            ) {
                Ok(notification) => dispatched.push(DispatchedResource {
                    resource: transfer.resource,
                    destination_block_ids: transfer.destination_block_ids,
                    notification,
                }),
                Err(error) => {
                    tracing::error!(
                        error = %error,
                        resource = ?transfer.resource,
                        "resource onboard dispatch failed"
                    );
                    return DispatchBatch {
                        transfers: dispatched,
                        failure: Some(DispatchFailure {
                            resource: transfer.resource,
                            destination_block_ids: transfer.destination_block_ids,
                        }),
                    };
                }
            }
        }
        DispatchBatch {
            transfers: dispatched,
            failure: None,
        }
    }
}

fn fold_resource_onboard(
    dispatch_failure: Option<DispatchFailure>,
    completed: Vec<CompletedResource>,
) -> ActionStatus {
    if let Some(failure) = dispatch_failure {
        return ActionStatus::Failed(ActionFailure::Resource {
            resource: failure.resource,
            block_ids: Some(failure.destination_block_ids),
        });
    }
    for completed in completed {
        if let Err(error) = completed.result {
            tracing::error!(
                %error,
                resource = ?completed.resource,
                "same-request resource onboard transfer failed"
            );
            return ActionStatus::Failed(ActionFailure::Resource {
                resource: completed.resource,
                block_ids: Some(completed.destination_block_ids),
            });
        }
    }
    ActionStatus::Complete
}

fn searched_bundle_plan(
    state: &crate::tiering::engine::local::BundleSearchState,
    destinations: Vec<ResourceDestination>,
) -> Result<(BundleOnboardPlan, SourceLeases, Vec<SequenceHash>), LeaderEngineError> {
    let mut destination_map = BTreeMap::new();
    for destination in destinations {
        if destination_map
            .insert(destination.resource, destination.block_ids)
            .is_some()
        {
            return Err(invalid_bundle(format!(
                "duplicate destination for {:?}",
                destination.resource
            )));
        }
    }
    let mut destinations = destination_map;
    if destinations.len() != state.identity.resources().len() {
        return Err(invalid_bundle(
            "resource destinations do not match the manifest",
        ));
    }

    let lease = state
        .lease()
        .ok_or_else(|| invalid_bundle("bundle search has no committed lease"))?;
    let boundary = usize::try_from(lease.key().boundary_tokens())
        .map_err(|_| invalid_bundle("bundle boundary does not fit usize"))?;
    let computed = state.computed_tokens;
    let mut resources = Vec::with_capacity(state.identity.resources().len());
    let mut source_leases = BTreeMap::new();
    let mut inflight_hashes = Vec::new();
    for requirement in state.identity.resources() {
        let resource = requirement.resource();
        let native = requirement.native_block_tokens().get() as usize;
        let first_block = computed / native;
        let end_block = boundary.div_ceil(native);
        let destination = destinations
            .remove(&resource)
            .ok_or_else(|| invalid_bundle(format!("missing destination for {resource:?}")))?;
        let source = lease
            .resources()
            .get(&resource)
            .ok_or_else(|| invalid_bundle(format!("search lease is missing {resource:?}")))?;
        if first_block >= end_block || source.len() < end_block || destination.len() < end_block {
            return Err(invalid_bundle(format!(
                "resource {resource:?} cannot cover native block range {first_block}..{end_block}"
            )));
        }
        let pins = source[first_block..end_block].to_vec();
        let source_block_ids = pins.iter().map(ImmutableBlock::block_id).collect();
        inflight_hashes.extend(pins.iter().map(ImmutableBlock::sequence_hash));
        resources.push(ResourceOnboard {
            resource,
            source_block_ids,
            destination_block_ids: destination[first_block..end_block].to_vec(),
        });
        source_leases.insert(resource, pins);
    }
    Ok((
        BundleOnboardPlan {
            identity: state.identity.clone(),
            key: *lease.key(),
            resources,
        },
        source_leases,
        inflight_hashes,
    ))
}

fn invalid_bundle(reason: impl Into<String>) -> LeaderEngineError {
    LeaderEngineError::InvalidBundleTransfer {
        reason: reason.into(),
    }
}

fn validate_resource_transfers(
    engine: &LocalConnectorEngine,
    resources: &[ResourceOnboard],
) -> Result<(), LeaderEngineError> {
    if resources.is_empty() {
        return Err(LeaderEngineError::InvalidBundleTransfer {
            reason: "at least one resource is required".to_owned(),
        });
    }
    let mut seen = HashSet::new();
    for transfer in resources {
        if !seen.insert(transfer.resource) {
            return Err(LeaderEngineError::InvalidResourceOnboard {
                reason: format!("duplicate logical resource {:?}", transfer.resource),
            });
        }
        if transfer.source_block_ids.is_empty()
            || transfer.source_block_ids.len() != transfer.destination_block_ids.len()
        {
            return Err(LeaderEngineError::InvalidResourceOnboard {
                reason: format!(
                    "resource {:?} has {} G2 sources and {} G1 destinations",
                    transfer.resource,
                    transfer.source_block_ids.len(),
                    transfer.destination_block_ids.len()
                ),
            });
        }
        if engine.leader.g2_manager_for(transfer.resource).is_none() {
            return Err(LeaderEngineError::ResourceOnboardNotConfigured {
                resource: transfer.resource,
            });
        }
    }
    Ok(())
}

fn fold_bundle_onboard(
    transaction: BundleOnboardTransaction,
    dispatch_failure: Option<DispatchFailure>,
    completed: Vec<CompletedResource>,
) -> ActionStatus {
    let mut outcome = match dispatch_failure.as_ref() {
        Some(failure) => ActionStatus::Failed(ActionFailure::Resource {
            resource: failure.resource,
            block_ids: Some(failure.destination_block_ids.clone()),
        }),
        None => ActionStatus::Complete,
    };
    let mut coordinator = transaction.lock().expect("bundle-onboard mutex poisoned");
    if let Some(failure) = dispatch_failure {
        let _ = coordinator.fail(failure.resource, Some(failure.destination_block_ids));
    }
    for completed in completed {
        let transition = match completed.result {
            Ok(()) => coordinator.complete(completed.resource),
            Err(error) => {
                tracing::error!(
                    error = %error,
                    resource = ?completed.resource,
                    "resource onboard transfer failed"
                );
                coordinator.fail(
                    completed.resource,
                    Some(completed.destination_block_ids.clone()),
                )
            }
        };
        match transition {
            Ok(OnboardTransition::Visible(visible)) => {
                tracing::trace!(
                    computed_tokens = visible.computed_tokens(),
                    "bundle onboard crossed all-resource visibility barrier"
                );
                outcome = ActionStatus::Complete;
            }
            Ok(OnboardTransition::Aborted(failure)) => {
                outcome = ActionStatus::Failed(ActionFailure::Resource {
                    resource: failure.resource(),
                    block_ids: failure.failed_blocks().map(<[usize]>::to_vec),
                });
            }
            Ok(OnboardTransition::Pending | OnboardTransition::Settled(_)) => {}
            Err(error) => {
                tracing::error!(
                    %error,
                    resource = ?completed.resource,
                    "bundle onboard completion fold failed"
                );
                outcome = ActionStatus::Failed(ActionFailure::Resource {
                    resource: completed.resource,
                    block_ids: Some(completed.destination_block_ids),
                });
            }
        }
    }
    outcome
}
