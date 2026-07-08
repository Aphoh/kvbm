// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded all-resource remote acquisition and atomic local publication.

mod deadline;
mod metrics;
#[cfg(any(test, feature = "testing"))]
pub(crate) mod test_support;
mod transaction;
mod transfer;

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, anyhow, bail};
use futures::future::{BoxFuture, join_all};
use kvbm_common::{LogicalResourceId, SequenceHash};
use kvbm_protocols::cache_manifest::{BundleKey, CacheIdentity};
use kvbm_protocols::control::modules::transfer::OpenTransferSessionResponse;
use tokio_util::sync::CancellationToken;

use crate::leader::InstanceLeader;

use super::{BundleMissReason, BundlePullOutcome, RemoteBundleCandidate, unix_time_ms};
use deadline::{BundlePullLimits, PullDeadline, bounded};
use metrics::PullMetrics;
pub(crate) use transaction::StagedBundle;
#[cfg(test)]
use transfer::BundleTransferError;
use transfer::{
    BundleTransfer, LeaderBundleTransfer, OpenedResource, close_all, spawn_draining_pull,
};

/// One complete-bundle acquisition attempt.
///
/// The object owns the expected manifest, request cancellation, one bounded
/// deadline, and the transfer adapter. Its execution publishes no physical
/// destination until every manifest resource has staged successfully.
struct RemoteBundlePull {
    target: Arc<dyn BundlePullTarget>,
    candidate: RemoteBundleCandidate,
    expected_identity: CacheIdentity,
    cancel: CancellationToken,
    search_deadline: tokio::time::Instant,
    transfer: Arc<dyn BundleTransfer>,
    limits: BundlePullLimits,
}

impl RemoteBundlePull {
    async fn execute(self) -> Result<BundlePullOutcome> {
        let mut metrics = PullMetrics::new(
            self.target
                .instance_leader()
                .observability()
                .map(|observability| observability.bundle_metrics().clone()),
            self.expected_identity
                .resources()
                .iter()
                .map(|requirement| requirement.resource())
                .collect(),
        );
        let outcome = self.execute_inner(&mut metrics).await;
        metrics.record(&outcome);
        outcome
    }

    async fn execute_inner(&self, metrics: &mut PullMetrics) -> Result<BundlePullOutcome> {
        if self.cancel.is_cancelled() {
            return Ok(BundlePullOutcome::Miss(BundleMissReason::Canceled));
        }
        if unix_time_ms() >= self.candidate.lease_expires_at_unix_ms() {
            return Ok(BundlePullOutcome::Miss(BundleMissReason::Expired));
        }
        let advertisement = self.candidate.advertisement();
        let expected_owner = advertisement.owner();
        if advertisement.identity() != &self.expected_identity
            || !advertisement
                .key()
                .is_compatible_with(&self.expected_identity)
        {
            return Ok(BundlePullOutcome::Miss(BundleMissReason::Incompatible));
        }
        let lineages = match resource_lineages(&self.expected_identity, advertisement) {
            Ok(lineages) => lineages,
            Err(error) => {
                tracing::debug!(%error, "remote bundle lineage was incompatible");
                return Ok(BundlePullOutcome::Miss(BundleMissReason::Incompatible));
            }
        };
        let generation = match self.target.reserve_publication_generation() {
            Ok(generation) => generation,
            Err(error) => {
                tracing::warn!(%error, "remote bundle publication generation unavailable");
                return Ok(BundlePullOutcome::Miss(BundleMissReason::CommitFailed));
            }
        };
        let deadline = PullDeadline::new(
            self.candidate.lease_expires_at_unix_ms(),
            self.search_deadline,
        );
        let mut opened = Vec::with_capacity(lineages.len());
        for (&resource, hashes) in &lineages {
            let watchdog = deadline
                .remaining()
                .min(self.limits.holder_watchdog)
                .max(Duration::from_millis(1));
            let response = bounded(
                &self.cancel,
                deadline.at,
                self.transfer.open(resource, hashes.clone(), watchdog),
            )
            .await;
            let response = match response {
                Ok(Ok(response)) => response,
                Ok(Err(error)) => {
                    close_all(
                        &self.transfer,
                        &opened,
                        "bundle acquisition failed",
                        self.limits.cleanup_timeout,
                    )
                    .await;
                    return Ok(BundlePullOutcome::Miss(error.reason()));
                }
                Err(interruption) => {
                    close_all(
                        &self.transfer,
                        &opened,
                        "bundle acquisition interrupted",
                        self.limits.cleanup_timeout,
                    )
                    .await;
                    return Ok(BundlePullOutcome::Miss(deadline.reason(interruption)));
                }
            };
            let (capability, committed) = match response {
                OpenTransferSessionResponse::Sync {
                    capability,
                    committed,
                    ..
                } => {
                    if capability.instance_id != expected_owner {
                        opened.push(OpenedResource {
                            resource,
                            hashes: hashes.clone(),
                            capability,
                        });
                        close_all(
                            &self.transfer,
                            &opened,
                            "bundle holder identity changed",
                            self.limits.cleanup_timeout,
                        )
                        .await;
                        return Ok(BundlePullOutcome::Miss(BundleMissReason::OwnerLost));
                    }
                    (capability, committed)
                }
                OpenTransferSessionResponse::NoBlocksFound => {
                    close_all(
                        &self.transfer,
                        &opened,
                        "bundle resource omitted",
                        self.limits.cleanup_timeout,
                    )
                    .await;
                    return Ok(BundlePullOutcome::Miss(BundleMissReason::Incomplete));
                }
                OpenTransferSessionResponse::Async { capability } => {
                    let reason = if capability.instance_id == expected_owner {
                        BundleMissReason::Incomplete
                    } else {
                        BundleMissReason::OwnerLost
                    };
                    opened.push(OpenedResource {
                        resource,
                        hashes: hashes.clone(),
                        capability,
                    });
                    close_all(
                        &self.transfer,
                        &opened,
                        "unexpected async bundle acquisition",
                        self.limits.cleanup_timeout,
                    )
                    .await;
                    return Ok(BundlePullOutcome::Miss(reason));
                }
            };
            if capability.resource != resource || committed != *hashes {
                opened.push(OpenedResource {
                    resource,
                    hashes: hashes.clone(),
                    capability,
                });
                close_all(
                    &self.transfer,
                    &opened,
                    "incomplete bundle acquisition",
                    self.limits.cleanup_timeout,
                )
                .await;
                return Ok(BundlePullOutcome::Miss(BundleMissReason::Incomplete));
            }
            opened.push(OpenedResource {
                capability,
                resource,
                hashes: hashes.clone(),
            });
        }

        let pulls = opened.iter().cloned().map(|resource| {
            let transfer = Arc::clone(&self.transfer);
            let cancel = self.cancel.clone();
            let pull = spawn_draining_pull(transfer, resource, self.limits.cleanup_timeout);
            async move { bounded(&cancel, deadline.at, pull).await }
        });
        let pull_results = join_all(pulls).await;
        let mut staged = Vec::with_capacity(opened.len());
        let mut miss = None;
        for (opened_resource, result) in opened.iter().zip(pull_results) {
            match result {
                Ok(Ok(Ok(resource)))
                    if resource.resource() == opened_resource.resource
                        && resource.hashes() == opened_resource.hashes =>
                {
                    let transferred_bytes = self
                        .target
                        .resource_bytes(resource.resource(), resource.hashes().len())
                        .unwrap_or(0);
                    metrics.observe_transferred_bytes(resource.resource(), transferred_bytes);
                    staged.push(resource);
                }
                Ok(Ok(Ok(_))) => miss = merge_reason(miss, BundleMissReason::Incomplete),
                Ok(Ok(Err(error))) => {
                    tracing::debug!(
                        resource = ?opened_resource.resource,
                        %error,
                        "remote bundle resource pull failed"
                    );
                    miss = merge_reason(miss, error.reason());
                }
                Ok(Err(error)) => {
                    tracing::error!(
                        resource = ?opened_resource.resource,
                        %error,
                        "detached bundle pull task failed"
                    );
                    miss = merge_reason(miss, BundleMissReason::TransferFailed);
                }
                Err(interruption) => {
                    miss = merge_reason(miss, deadline.reason(interruption));
                }
            }
        }
        if let Some(reason) = miss {
            return Ok(BundlePullOutcome::Miss(reason));
        }
        if self.cancel.is_cancelled() {
            return Ok(BundlePullOutcome::Miss(BundleMissReason::Canceled));
        }
        if unix_time_ms() >= self.candidate.lease_expires_at_unix_ms() {
            return Ok(BundlePullOutcome::Miss(BundleMissReason::Expired));
        }
        let bundle = match StagedBundle::new(lineages, staged) {
            Ok(bundle) => bundle,
            Err(error) => {
                tracing::debug!(%error, "remote bundle staging was incomplete");
                return Ok(BundlePullOutcome::Miss(BundleMissReason::Incomplete));
            }
        };
        let key = advertisement.key();
        let identity = self.expected_identity.clone();
        match bounded(
            &self.cancel,
            deadline.at,
            self.target
                .commit_pulled_bundle(identity, key, generation, bundle),
        )
        .await
        {
            Ok(Ok(())) => Ok(BundlePullOutcome::Pulled(key)),
            Ok(Err(error)) => {
                tracing::debug!(%error, "remote bundle local commit failed");
                Ok(BundlePullOutcome::Miss(BundleMissReason::CommitFailed))
            }
            Err(interruption) => Ok(BundlePullOutcome::Miss(deadline.reason(interruption))),
        }
    }
}

pub(crate) trait BundlePullTarget: Send + Sync {
    fn instance_leader(&self) -> Arc<InstanceLeader>;

    /// Reserve the local owner's publication generation before any remote I/O.
    /// Late attempts can therefore never publish over newer completed pulls.
    fn reserve_publication_generation(&self) -> Result<u64>;

    /// Return the configured physical bytes represented by `logical_blocks`
    /// for resource-keyed transfer metrics. Targets without byte geometry may
    /// omit the observation without affecting transaction correctness.
    fn resource_bytes(&self, _resource: LogicalResourceId, _logical_blocks: usize) -> Option<u64> {
        None
    }

    fn commit_pulled_bundle(
        &self,
        identity: CacheIdentity,
        key: BundleKey,
        generation: u64,
        bundle: StagedBundle,
    ) -> BoxFuture<'static, Result<()>>;
}

pub(crate) async fn pull_remote_bundle(
    target: Arc<dyn BundlePullTarget>,
    candidate: RemoteBundleCandidate,
    expected_identity: CacheIdentity,
    cancel: CancellationToken,
    search_deadline: tokio::time::Instant,
) -> Result<BundlePullOutcome> {
    let leader = target.instance_leader();
    let advertisement = candidate.advertisement();
    // `execute_inner` validates the advertised manifest and every resource
    // lineage before opening; the holder independently proves that the
    // directory hit still names its current registration lifecycle.
    let transfer = Arc::new(LeaderBundleTransfer::new(
        leader,
        advertisement.owner(),
        advertisement.registration_epoch(),
    ));
    pull_remote_bundle_with_transfer(
        target,
        candidate,
        expected_identity,
        cancel,
        search_deadline,
        transfer,
        BundlePullLimits::PRODUCTION,
    )
    .await
}

async fn pull_remote_bundle_with_transfer(
    target: Arc<dyn BundlePullTarget>,
    candidate: RemoteBundleCandidate,
    expected_identity: CacheIdentity,
    cancel: CancellationToken,
    search_deadline: tokio::time::Instant,
    transfer: Arc<dyn BundleTransfer>,
    limits: BundlePullLimits,
) -> Result<BundlePullOutcome> {
    RemoteBundlePull {
        target,
        candidate,
        expected_identity,
        cancel,
        search_deadline,
        transfer,
        limits,
    }
    .execute()
    .await
}

fn resource_lineages(
    identity: &CacheIdentity,
    advertisement: &super::BundleAdvertisement,
) -> Result<BTreeMap<LogicalResourceId, Vec<SequenceHash>>> {
    if advertisement.identity() != identity || !advertisement.key().is_compatible_with(identity) {
        bail!("bundle advertisement is incompatible with the expected identity");
    }
    let mut lineages = BTreeMap::new();
    for requirement in identity.resources() {
        let resource = requirement.resource();
        let lineage = advertisement
            .lineage(resource)
            .ok_or_else(|| anyhow!("bundle advertisement is missing resource {resource:?}"))?;
        lineages.insert(resource, lineage.hashes().to_vec());
    }
    Ok(lineages)
}

fn merge_reason(
    current: Option<BundleMissReason>,
    incoming: BundleMissReason,
) -> Option<BundleMissReason> {
    let rank = |reason| match reason {
        BundleMissReason::Canceled => 7,
        BundleMissReason::OwnerLost => 6,
        BundleMissReason::Expired => 5,
        BundleMissReason::TimedOut => 4,
        BundleMissReason::TransferFailed => 3,
        BundleMissReason::ChecksumFailed => 4,
        BundleMissReason::Incomplete => 2,
        BundleMissReason::CommitFailed
        | BundleMissReason::Incompatible
        | BundleMissReason::NotFound => 1,
    };
    Some(match current {
        Some(current) if rank(current) >= rank(incoming) => current,
        _ => incoming,
    })
}

#[cfg(test)]
mod tests;
