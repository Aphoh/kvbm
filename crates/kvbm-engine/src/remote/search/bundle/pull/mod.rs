// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! All-resource remote session acquisition, pull, and local bundle commit.

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use futures::future::{BoxFuture, join_all};
use kvbm_common::{LogicalResourceId, SequenceHash};
use kvbm_protocols::cache_manifest::{BundleKey, CacheIdentity, ResourceRole};
use kvbm_protocols::control::client::LeaderControlClient;
use kvbm_protocols::control::modules::transfer::{
    CloseTransferSessionRequest, FindMode, OpenTransferSessionRequest, OpenTransferSessionResponse,
    PullFromSessionRequest, SearchMode, TierSelection, TransferSessionCapability,
};

use crate::leader::InstanceLeader;

use super::{BundleMissReason, BundlePullOutcome, RemoteBundleCandidate, unix_time_ms};

pub(crate) trait BundlePullTarget: Send + Sync {
    fn instance_leader(&self) -> Arc<InstanceLeader>;

    fn commit_pulled_bundle(
        &self,
        identity: CacheIdentity,
        key: BundleKey,
        generation: u64,
        lineages: BTreeMap<LogicalResourceId, Vec<SequenceHash>>,
    ) -> BoxFuture<'static, Result<()>>;
}

struct OpenedResource {
    capability: TransferSessionCapability,
    resource: LogicalResourceId,
}

trait BundleTransfer: Send + Sync {
    fn open(
        &self,
        resource: LogicalResourceId,
        hashes: Vec<SequenceHash>,
    ) -> BoxFuture<'_, Result<OpenTransferSessionResponse>>;

    fn pull(&self, resource: &OpenedResource) -> BoxFuture<'_, Result<()>>;

    fn close(&self, session_id: uuid::Uuid, reason: &str) -> BoxFuture<'_, ()>;
}

struct LeaderBundleTransfer {
    leader: Arc<InstanceLeader>,
    client: LeaderControlClient,
    owner: crate::InstanceId,
}

impl BundleTransfer for LeaderBundleTransfer {
    fn open(
        &self,
        resource: LogicalResourceId,
        hashes: Vec<SequenceHash>,
    ) -> BoxFuture<'_, Result<OpenTransferSessionResponse>> {
        Box::pin(async move {
            self.client
                .transfer()
                .open_session(OpenTransferSessionRequest {
                    sequence_hashes: hashes,
                    search_mode: SearchMode::Prefix,
                    find_mode: FindMode::Sync,
                    tiers: TierSelection::default(),
                    resource: Some(resource),
                    watchdog_ms: None,
                })
                .await
                .with_context(|| format!("open bundle resource {resource:?} on {}", self.owner))
        })
    }

    fn pull(&self, resource: &OpenedResource) -> BoxFuture<'_, Result<()>> {
        let leader = Arc::clone(&self.leader);
        let owner = self.owner;
        let session_id = resource.capability.session_id;
        let endpoint = resource.capability.endpoint.clone();
        let resource = resource.resource;
        Box::pin(async move {
            leader
                .pull_from_session(PullFromSessionRequest {
                    session_id,
                    source_instance_id: owner,
                    endpoint: Some(endpoint),
                    selector: None,
                    resource: Some(resource),
                })
                .await
                .map(|_| ())
                .map_err(anyhow::Error::from)
        })
    }

    fn close(&self, session_id: uuid::Uuid, reason: &str) -> BoxFuture<'_, ()> {
        let reason = reason.to_owned();
        Box::pin(async move {
            if let Err(error) = self
                .client
                .transfer()
                .close_session(CloseTransferSessionRequest {
                    session_id,
                    reason: Some(reason),
                })
                .await
            {
                tracing::debug!(
                    %error,
                    %session_id,
                    "bundle session close failed; holder watchdog will reclaim"
                );
            }
        })
    }
}

pub(crate) async fn pull_remote_bundle(
    target: Arc<dyn BundlePullTarget>,
    candidate: RemoteBundleCandidate,
    sequence_hashes: Arc<[SequenceHash]>,
    base_block_tokens: usize,
) -> Result<BundlePullOutcome> {
    let leader = target.instance_leader();
    let owner = candidate.advertisement().owner();
    let transfer = Arc::new(LeaderBundleTransfer {
        client: LeaderControlClient::new(leader.messenger().clone(), owner),
        leader,
        owner,
    });
    pull_remote_bundle_with_transfer(
        target,
        candidate,
        sequence_hashes,
        base_block_tokens,
        transfer,
    )
    .await
}

async fn pull_remote_bundle_with_transfer(
    target: Arc<dyn BundlePullTarget>,
    candidate: RemoteBundleCandidate,
    sequence_hashes: Arc<[SequenceHash]>,
    base_block_tokens: usize,
    transfer: Arc<dyn BundleTransfer>,
) -> Result<BundlePullOutcome> {
    if unix_time_ms() >= candidate.lease_expires_at_unix_ms() {
        return Ok(BundlePullOutcome::Miss(BundleMissReason::Expired));
    }
    let advertisement = candidate.advertisement();
    let identity = advertisement.identity().clone();
    let key = advertisement.key();
    let owner = advertisement.owner();
    let lineages = resource_lineages(&identity, key, &sequence_hashes, base_block_tokens)?;

    let mut opened = Vec::with_capacity(lineages.len());
    for (&resource, hashes) in &lineages {
        if unix_time_ms() >= candidate.lease_expires_at_unix_ms() {
            close_all(&transfer, &opened, "bundle directory lease expired").await;
            return Ok(BundlePullOutcome::Miss(BundleMissReason::Expired));
        }
        let response = transfer.open(resource, hashes.clone()).await;
        let response = match response {
            Ok(response) => response,
            Err(error) => {
                close_all(&transfer, &opened, "bundle acquisition failed").await;
                return Err(error);
            }
        };
        let (capability, committed) = match response {
            OpenTransferSessionResponse::Sync {
                capability,
                committed,
                ..
            } => (capability, committed),
            OpenTransferSessionResponse::NoBlocksFound => {
                close_all(&transfer, &opened, "bundle resource omitted").await;
                return Ok(BundlePullOutcome::Miss(BundleMissReason::Incomplete));
            }
            OpenTransferSessionResponse::Async { capability } => {
                opened.push(OpenedResource {
                    resource,
                    capability,
                });
                close_all(&transfer, &opened, "unexpected async bundle acquisition").await;
                return Ok(BundlePullOutcome::Miss(BundleMissReason::Incomplete));
            }
        };
        if capability.resource != resource || committed != *hashes {
            opened.push(OpenedResource {
                resource,
                capability,
            });
            close_all(&transfer, &opened, "incomplete bundle acquisition").await;
            return Ok(BundlePullOutcome::Miss(BundleMissReason::Incomplete));
        }
        opened.push(OpenedResource {
            capability,
            resource,
        });
    }

    if unix_time_ms() >= candidate.lease_expires_at_unix_ms() {
        close_all(&transfer, &opened, "bundle lease expired during pull").await;
        return Ok(BundlePullOutcome::Miss(BundleMissReason::Expired));
    }
    let pulls = opened.iter().map(|opened_resource| {
        let transfer = Arc::clone(&transfer);
        async move {
            transfer.pull(opened_resource).await.with_context(|| {
                format!(
                    "pull bundle resource {:?} from {owner}",
                    opened_resource.resource
                )
            })
        }
    });
    let error = join_all(pulls).await.into_iter().find_map(Result::err);
    if let Some(error) = error {
        close_all(&transfer, &opened, "bundle pull failed").await;
        return Err(error);
    }
    if unix_time_ms() >= candidate.lease_expires_at_unix_ms() {
        close_all(&transfer, &opened, "bundle lease expired after pull").await;
        return Ok(BundlePullOutcome::Miss(BundleMissReason::Expired));
    }
    close_all(&transfer, &opened, "bundle pull complete").await;

    target
        .commit_pulled_bundle(identity, key, advertisement.generation(), lineages)
        .await?;
    Ok(BundlePullOutcome::Pulled(key))
}

fn resource_lineages(
    identity: &CacheIdentity,
    key: BundleKey,
    sequence_hashes: &[SequenceHash],
    base_block_tokens: usize,
) -> Result<BTreeMap<LogicalResourceId, Vec<SequenceHash>>> {
    if base_block_tokens == 0 || !key.is_compatible_with(identity) {
        bail!("invalid bundle lineage inputs");
    }
    let boundary = usize::try_from(key.boundary_tokens())?;
    let mut lineages = BTreeMap::new();
    for requirement in identity.resources() {
        let native = requirement.native_block_tokens().get() as usize;
        if native < base_block_tokens || !native.is_multiple_of(base_block_tokens) {
            bail!(
                "resource {:?} native span is not base-block aligned",
                requirement.resource()
            );
        }
        let hashes = match requirement.role() {
            ResourceRole::PrefixHistory => (native..=boundary)
                .step_by(native)
                .map(|end| {
                    let index = end / base_block_tokens - 1;
                    sequence_hashes.get(index).copied().ok_or_else(|| {
                        anyhow!(
                            "sequence does not cover resource {:?} boundary",
                            requirement.resource()
                        )
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            ResourceRole::BoundaryCapsule => vec![key.boundary_hash()],
        };
        if hashes.is_empty() {
            bail!(
                "resource {:?} has no hashes to pull",
                requirement.resource()
            );
        }
        lineages.insert(requirement.resource(), hashes);
    }
    Ok(lineages)
}

async fn close_all(transfer: &Arc<dyn BundleTransfer>, opened: &[OpenedResource], reason: &str) {
    for resource in opened {
        transfer.close(resource.capability.session_id, reason).await;
    }
}

#[cfg(test)]
mod tests;
