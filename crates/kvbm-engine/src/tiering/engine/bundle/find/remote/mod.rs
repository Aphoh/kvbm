// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Remote candidate selection before conditional-prefill placement.

use std::sync::Arc;

use kvbm_logical::ImmutableBlock;
use kvbm_protocols::cache_manifest::{BundleKey, CacheIdentity};
use kvbm_protocols::connector::LocalPrefillEstimate;
use tokio_util::sync::CancellationToken;

use crate::G2;
use crate::leader::RemoteDiscoveryHandle;
use crate::remote::search::bundle::{
    BundleDiscoveryOutcome, BundleDiscoveryQuery, BundleMissReason, BundlePullOutcome,
    BundlePullTarget, pull_remote_bundle, unix_time_ms,
};
use crate::tiering::engine::bundle::BundlePrefillRequest;
use crate::tiering::engine::local::{LocalConnectorEngine, RemoteBundleResolution};

type LocalBundleLease = crate::tiering::engine::bundle::BundleLease<Vec<ImmutableBlock<G2>>>;

/// One bounded remote lookup, optional seed pull, and placement decision.
pub(super) struct RemoteBundleFind {
    engine: Arc<LocalConnectorEngine>,
    directory: RemoteDiscoveryHandle,
    request_id: String,
    identity: CacheIdentity,
    candidates: Vec<BundleKey>,
    target: BundleKey,
    fallback: Option<BundleKey>,
    num_computed_tokens: usize,
    total_tokens: usize,
    local_prefill_estimate: Option<LocalPrefillEstimate>,
    cancel: CancellationToken,
    deadline: tokio::time::Instant,
}

pub(super) struct RemoteBundleFindInputs {
    pub(super) engine: Arc<LocalConnectorEngine>,
    pub(super) directory: RemoteDiscoveryHandle,
    pub(super) request_id: String,
    pub(super) identity: CacheIdentity,
    pub(super) candidates: Vec<BundleKey>,
    pub(super) target: BundleKey,
    pub(super) fallback: Option<BundleKey>,
    pub(super) num_computed_tokens: usize,
    pub(super) total_tokens: usize,
    pub(super) local_prefill_estimate: Option<LocalPrefillEstimate>,
    pub(super) cancel: CancellationToken,
    pub(super) deadline: tokio::time::Instant,
}

impl RemoteBundleFind {
    pub(super) fn new(inputs: RemoteBundleFindInputs) -> Self {
        Self {
            engine: inputs.engine,
            directory: inputs.directory,
            request_id: inputs.request_id,
            identity: inputs.identity,
            candidates: inputs.candidates,
            target: inputs.target,
            fallback: inputs.fallback,
            num_computed_tokens: inputs.num_computed_tokens,
            total_tokens: inputs.total_tokens,
            local_prefill_estimate: inputs.local_prefill_estimate,
            cancel: inputs.cancel,
            deadline: inputs.deadline,
        }
    }

    pub(super) async fn execute(self) -> RemoteBundleResolution {
        let metrics = self
            .engine
            .leader
            .observability()
            .map(|observability| observability.bundle_metrics().clone());
        let query = BundleDiscoveryQuery::new(
            self.identity.clone(),
            self.candidates.clone(),
            unix_time_ms(),
        );
        let discovery = tokio::select! {
            biased;
            _ = self.cancel.cancelled() => return self.fallback_resolution(),
            result = tokio::time::timeout_at(self.deadline, self.directory.discover_bundle(query)) => result,
        };
        let mut seed_lease = None;
        let mut seed = match discovery {
            Err(_) => {
                if let Some(metrics) = metrics.as_ref() {
                    metrics.record_find("remote_miss", "timed_out");
                }
                return self.fallback_resolution();
            }
            Ok(Err(error)) => {
                if let Some(metrics) = metrics.as_ref() {
                    metrics.record_find("remote_miss", "directory_failed");
                }
                tracing::warn!(%error, "remote complete-bundle directory failed");
                return self.fallback_resolution();
            }
            Ok(Ok(BundleDiscoveryOutcome::Miss(BundleMissReason::NotFound))) => self.fallback,
            Ok(Ok(BundleDiscoveryOutcome::Miss(reason))) => {
                if let Some(metrics) = metrics.as_ref() {
                    metrics.record_find("remote_miss", reason.as_label());
                }
                return self.fallback_resolution();
            }
            Ok(Ok(BundleDiscoveryOutcome::Hit(candidate))) => {
                let candidate = *candidate;
                let key = candidate.advertisement().key();
                if !self.candidates.contains(&key) {
                    if let Some(metrics) = metrics.as_ref() {
                        metrics.record_find("remote_miss", "incompatible");
                    }
                    return self.fallback_resolution();
                }
                let target: Arc<dyn BundlePullTarget> =
                    Arc::clone(&self.engine) as Arc<dyn BundlePullTarget>;
                match pull_remote_bundle(
                    target,
                    candidate,
                    self.identity.clone(),
                    self.cancel.clone(),
                    self.deadline,
                )
                .await
                {
                    Ok(BundlePullOutcome::Pulled(key)) => {
                        let Some(lease) = self.lease_exact(key) else {
                            if let Some(metrics) = metrics.as_ref() {
                                metrics.record_find("remote_miss", "commit_failed");
                            }
                            return self.fallback_resolution();
                        };
                        if let Some(metrics) = metrics.as_ref() {
                            metrics.record_find("remote_hit", "complete");
                            metrics.observe_boundary(key.boundary_tokens());
                        }
                        if key == self.target {
                            return RemoteBundleResolution::Selected(lease);
                        }
                        seed_lease = Some(lease);
                        Some(key)
                    }
                    Ok(BundlePullOutcome::Miss(reason)) => {
                        if let Some(metrics) = metrics.as_ref() {
                            metrics.record_find("remote_miss", reason.as_label());
                        }
                        return self.fallback_resolution();
                    }
                    Err(error) => {
                        if let Some(metrics) = metrics.as_ref() {
                            metrics.record_find("remote_miss", "transfer_failed");
                        }
                        tracing::warn!(%error, "remote bundle pull failed; retaining best seed");
                        return self.fallback_resolution();
                    }
                }
            }
        };

        // Close the search/dispatch race: a concurrent local publication may
        // land while discovery is in flight. Re-read the canonical catalog
        // before placement and prefer its greatest candidate. In particular,
        // a newly-local target B2 suppresses dispatch entirely.
        if let Some((local_key, local_lease)) = self.best_local_candidate()
            && seed.is_none_or(|seed| local_key.boundary_tokens() > seed.boundary_tokens())
        {
            if local_key == self.target {
                return RemoteBundleResolution::Selected(local_lease);
            }
            seed = Some(local_key);
            seed_lease = Some(local_lease);
        }

        let result = self
            .engine
            .clone()
            .run_bundle_prefill(
                Arc::clone(&self.directory),
                BundlePrefillRequest {
                    request_id: self.request_id.clone(),
                    identity: self.identity.clone(),
                    target: self.target,
                    seed,
                    num_computed_tokens: self.num_computed_tokens,
                    total_tokens: self.total_tokens,
                    local_prefill_estimate: self.local_prefill_estimate,
                },
                self.cancel.clone(),
                self.deadline,
            )
            .await;
        let selected = match result {
            Ok(selected) => selected,
            Err(error) => {
                tracing::warn!(%error, "bundle prefill failed; retaining best seed");
                seed
            }
        };
        self.resolve_selected(selected, seed, seed_lease)
    }

    fn resolve_selected(
        &self,
        selected: Option<BundleKey>,
        seed: Option<BundleKey>,
        seed_lease: Option<LocalBundleLease>,
    ) -> RemoteBundleResolution {
        let Some(selected) = selected else {
            return self.fallback_resolution();
        };
        if self.fallback == Some(selected) {
            return RemoteBundleResolution::Fallback;
        }
        if seed == Some(selected)
            && let Some(lease) = seed_lease
        {
            return RemoteBundleResolution::Selected(lease);
        }
        self.lease_exact(selected).map_or_else(
            || self.fallback_resolution(),
            RemoteBundleResolution::Selected,
        )
    }

    fn fallback_resolution(&self) -> RemoteBundleResolution {
        if self.fallback.is_some() {
            RemoteBundleResolution::Fallback
        } else {
            RemoteBundleResolution::Miss
        }
    }

    fn lease_exact(&self, key: BundleKey) -> Option<LocalBundleLease> {
        self.engine
            .bundle_catalog
            .lock()
            .expect("bundle-catalog mutex poisoned")
            .lease_exact(&self.identity, &key)
    }

    fn best_local_candidate(&self) -> Option<(BundleKey, LocalBundleLease)> {
        let catalog = self
            .engine
            .bundle_catalog
            .lock()
            .expect("bundle-catalog mutex poisoned");
        self.candidates.iter().find_map(|key| {
            catalog
                .lease_exact(&self.identity, key)
                .map(|lease| (*key, lease))
        })
    }
}
