// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Local engine lifecycle for manifest-scoped bundle finds.

use std::num::NonZeroUsize;
use std::sync::Arc;

use kvbm_protocols::cache_manifest::CacheIdentity;
use kvbm_protocols::connector::{
    FindBlocksHandle, FindBlocksOutcome, FindBlocksRequest, LeaderEngine, LeaderEngineError,
    SearchId,
};

use super::BundleFindQuery;
use crate::remote::search::bundle::{
    BundleDiscoveryOutcome, BundleDiscoveryQuery, BundlePullOutcome, BundlePullTarget,
    pull_remote_bundle, unix_time_ms,
};
use crate::tiering::engine::bundle::BundlePrefillRequest;
use crate::tiering::engine::find::DerivedWindow;
use crate::tiering::engine::local::{BundleSearchSource, BundleSearchState, LocalConnectorEngine};

impl LocalConnectorEngine {
    pub(in crate::tiering::engine) fn start_bundle_find(
        self: &Arc<Self>,
        req: &FindBlocksRequest,
        identity: &CacheIdentity,
        derived: &DerivedWindow,
    ) -> Result<FindBlocksOutcome, LeaderEngineError> {
        let query = self.bundle_find_query(req, identity, derived);
        let found = query.find(
            &self
                .bundle_index
                .lock()
                .expect("bundle-index mutex poisoned"),
        );
        let Some(found) = found else {
            if self.search_remote {
                return self.start_remote_bundle_find(req, identity, query);
            }
            return Ok(FindBlocksOutcome::Resolved {
                matched_tokens: 0,
                minted: None,
                release_parked: false,
            });
        };
        let matched_tokens = found.matched_tokens();
        let search_id = SearchId::new();
        self.bundle_searches.insert(
            search_id,
            BundleSearchState {
                request_id: req.request_id.clone(),
                identity: identity.clone(),
                source: BundleSearchSource::Local(found.into_lease()),
                computed_tokens: req.num_computed_tokens,
                matched_tokens,
            },
        );
        let engine: Arc<dyn LeaderEngine> = Arc::clone(self) as Arc<dyn LeaderEngine>;
        Ok(FindBlocksOutcome::Resolved {
            matched_tokens,
            minted: Some(FindBlocksHandle::search(
                req.request_id.clone(),
                search_id,
                Arc::downgrade(&engine),
            )),
            release_parked: false,
        })
    }

    fn start_remote_bundle_find(
        self: &Arc<Self>,
        req: &FindBlocksRequest,
        identity: &CacheIdentity,
        query: BundleFindQuery<'_>,
    ) -> Result<FindBlocksOutcome, LeaderEngineError> {
        let Some(directory) = self.leader.remote_discovery() else {
            return Ok(FindBlocksOutcome::Resolved {
                matched_tokens: 0,
                minted: None,
                release_parked: false,
            });
        };
        let candidates = query.candidate_keys();
        if candidates.is_empty() {
            return Ok(FindBlocksOutcome::Resolved {
                matched_tokens: 0,
                minted: None,
                release_parked: false,
            });
        }
        let search_id = SearchId::new();
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.bundle_searches.insert(
            search_id,
            BundleSearchState {
                request_id: req.request_id.clone(),
                identity: identity.clone(),
                source: BundleSearchSource::Remote(rx),
                computed_tokens: req.num_computed_tokens,
                matched_tokens: 0,
            },
        );
        let target: Arc<dyn BundlePullTarget> = Arc::clone(self) as Arc<dyn BundlePullTarget>;
        let target_key = candidates[0];
        let sequence_hashes = Arc::clone(&req.sequence_hashes);
        let block_size = self.block_size;
        let identity = identity.clone();
        let engine = self.weak_self.clone();
        let request_id = req.request_id.clone();
        let num_computed_tokens = req.num_computed_tokens;
        let total_tokens = req.total_tokens;
        let metrics = self
            .leader
            .observability()
            .map(|observability| observability.bundle_metrics().clone());
        self.leader.runtime().spawn(async move {
            let mut remaining = candidates;
            let mut result = Ok(None);
            while !remaining.is_empty() {
                let query =
                    BundleDiscoveryQuery::new(identity.clone(), remaining.clone(), unix_time_ms());
                let candidate = match directory.discover_bundle(query).await {
                    Ok(BundleDiscoveryOutcome::Hit(candidate)) => *candidate,
                    Ok(BundleDiscoveryOutcome::Miss(reason)) => {
                        if let Some(metrics) = metrics.as_ref() {
                            metrics.record_find("remote_miss", reason.as_label());
                        }
                        tracing::debug!(
                            resource_reason = reason.as_label(),
                            "remote complete-bundle directory miss"
                        );
                        break;
                    }
                    Err(error) => {
                        if let Some(metrics) = metrics.as_ref() {
                            metrics.record_find("remote_miss", "directory_failed");
                        }
                        result = Err(error);
                        break;
                    }
                };
                let attempted = candidate.advertisement().key();
                match pull_remote_bundle(
                    Arc::clone(&target),
                    candidate,
                    Arc::clone(&sequence_hashes),
                    block_size,
                )
                .await
                {
                    Ok(BundlePullOutcome::Pulled(key)) => {
                        if let Some(metrics) = metrics.as_ref() {
                            metrics.record_find("remote_hit", "complete");
                            metrics.observe_boundary(key.boundary_tokens());
                        }
                        result = Ok(Some(key));
                        break;
                    }
                    Ok(BundlePullOutcome::Miss(reason)) => {
                        if let Some(metrics) = metrics.as_ref() {
                            metrics.record_find("remote_miss", reason.as_label());
                        }
                    }
                    Err(error) => {
                        if let Some(metrics) = metrics.as_ref() {
                            metrics.record_find("remote_miss", "transfer_failed");
                        }
                        tracing::debug!(
                            %error,
                            boundary_tokens = attempted.boundary_tokens(),
                            "remote bundle pull failed; retrying an earlier boundary"
                        );
                    }
                }
                remaining.retain(|key| key.boundary_tokens() < attempted.boundary_tokens());
            }
            if matches!(result, Ok(None))
                && let Some(engine) = engine.upgrade()
            {
                result = engine
                    .run_bundle_prefill(
                        Arc::clone(&directory),
                        BundlePrefillRequest::new(
                            request_id,
                            identity,
                            target_key,
                            sequence_hashes,
                            num_computed_tokens,
                            total_tokens,
                        ),
                    )
                    .await;
            }
            let _ = tx.send(result.map_err(|error| error.to_string()));
        });

        let engine: Arc<dyn LeaderEngine> = Arc::clone(self) as Arc<dyn LeaderEngine>;
        Ok(FindBlocksOutcome::Searching {
            minted: Some(FindBlocksHandle::search(
                req.request_id.clone(),
                search_id,
                Arc::downgrade(&engine),
            )),
        })
    }

    pub(in crate::tiering::engine) fn refresh_bundle_find(
        &self,
        req: &FindBlocksRequest,
        identity: &CacheIdentity,
        search_id: SearchId,
        derived: &DerivedWindow,
    ) -> Result<FindBlocksOutcome, LeaderEngineError> {
        let mut state = self
            .bundle_searches
            .get_mut(&search_id)
            .ok_or(LeaderEngineError::FindBlocksDesync)?;
        if state.request_id != req.request_id || state.identity != *identity {
            return Err(LeaderEngineError::FindBlocksDesync);
        }
        let query = self.bundle_find_query(req, identity, derived);
        let remote_result = match &mut state.source {
            BundleSearchSource::Local(_) => None,
            BundleSearchSource::Remote(remote) => Some(remote.try_recv()),
        };
        if let Some(remote_result) = remote_result {
            match remote_result {
                Ok(Ok(Some(key))) => {
                    let lease = self
                        .bundle_index
                        .lock()
                        .expect("bundle-index mutex poisoned")
                        .lease_exact(identity, &key)
                        .ok_or(LeaderEngineError::FindBlocksDesync)?;
                    state.source = BundleSearchSource::Local(lease);
                }
                Ok(Ok(None) | Err(_)) | Err(tokio::sync::oneshot::error::TryRecvError::Closed) => {
                    drop(state);
                    self.bundle_searches.remove(&search_id);
                    return Ok(FindBlocksOutcome::Resolved {
                        matched_tokens: 0,
                        minted: None,
                        release_parked: true,
                    });
                }
                Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                    return Ok(FindBlocksOutcome::Searching { minted: None });
                }
            }
        }
        let lease = state.lease().ok_or(LeaderEngineError::FindBlocksDesync)?;
        if let Some(matched_tokens) = query.matched_tokens_for(lease.key()) {
            state.computed_tokens = req.num_computed_tokens;
            state.matched_tokens = matched_tokens;
            return Ok(FindBlocksOutcome::Resolved {
                matched_tokens,
                minted: None,
                release_parked: false,
            });
        }
        let replacement = query.find(
            &self
                .bundle_index
                .lock()
                .expect("bundle-index mutex poisoned"),
        );
        if let Some(found) = replacement {
            let matched_tokens = found.matched_tokens();
            state.source = BundleSearchSource::Local(found.into_lease());
            state.computed_tokens = req.num_computed_tokens;
            state.matched_tokens = matched_tokens;
            return Ok(FindBlocksOutcome::Resolved {
                matched_tokens,
                minted: None,
                release_parked: false,
            });
        }
        drop(state);
        self.bundle_searches.remove(&search_id);
        Ok(FindBlocksOutcome::Resolved {
            matched_tokens: 0,
            minted: None,
            release_parked: true,
        })
    }

    fn bundle_find_query<'a>(
        &self,
        req: &'a FindBlocksRequest,
        identity: &'a CacheIdentity,
        derived: &DerivedWindow,
    ) -> BundleFindQuery<'a> {
        BundleFindQuery::new(
            identity,
            &req.sequence_hashes,
            req.num_computed_tokens,
            derived.range.end,
            NonZeroUsize::new(self.block_size).expect("engine block size must be nonzero"),
        )
    }
}
