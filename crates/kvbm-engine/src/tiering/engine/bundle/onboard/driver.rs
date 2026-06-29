// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, HashSet};
use std::sync::{Arc, Mutex};

use kvbm_common::{BlockId, LogicalLayoutHandle, LogicalResourceId, SequenceHash};
use kvbm_logical::ImmutableBlock;
use kvbm_physical::transfer::TransferCompleteNotification;
use kvbm_protocols::connector::{
    ActionFailure, ActionId, ActionStatus, BundleOnboardPlan, LeaderEngine, LeaderEngineError,
    OnboardHandle, RequestId, ResourceOnboard,
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
        fold_bundle_onboard(transaction, batch.failure, completed)
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
