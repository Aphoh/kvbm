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

use super::remote::{RemoteBundleFind, RemoteBundleFindInputs};
use super::{BundleFindMatch, BundleFindQuery};
use crate::tiering::engine::find::DerivedWindow;
use crate::tiering::engine::local::{
    BundleSearchSource, BundleSearchState, LocalConnectorEngine, RemoteBundleResolution,
    RemoteBundleSearch,
};

const REMOTE_BUNDLE_DISCOVERY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

impl LocalConnectorEngine {
    pub(in crate::tiering::engine) fn start_bundle_find(
        self: &Arc<Self>,
        req: &FindBlocksRequest,
        identity: &CacheIdentity,
        derived: &DerivedWindow,
    ) -> Result<FindBlocksOutcome, LeaderEngineError> {
        let query = self.bundle_find_query(req, identity, derived);
        let local = {
            let catalog = self
                .bundle_catalog
                .lock()
                .expect("bundle-catalog mutex poisoned");
            query.find(catalog.index())
        };
        let candidates = query.candidate_keys();
        let Some(target) = candidates.first().copied() else {
            return Ok(zero_find(false));
        };
        if local.as_ref().is_some_and(|found| *found.key() == target) {
            return Ok(self.install_local_find(req, identity, local.expect("checked local match")));
        }
        if !self
            .leader
            .remote_search_eligible(self.search_remote, derived.range.len())
        {
            return Ok(local.map_or_else(
                || zero_find(false),
                |found| self.install_local_find(req, identity, found),
            ));
        }
        let Some(directory) = self.leader.remote_discovery() else {
            return Ok(local.map_or_else(
                || zero_find(false),
                |found| self.install_local_find(req, identity, found),
            ));
        };
        let fallback_boundary = local
            .as_ref()
            .map(|found| found.key().boundary_tokens())
            .unwrap_or_default();
        let remote_candidates = candidates
            .into_iter()
            .filter(|key| key.boundary_tokens() > fallback_boundary)
            .collect::<Vec<_>>();
        if remote_candidates.is_empty() {
            return Ok(local.map_or_else(
                || zero_find(false),
                |found| self.install_local_find(req, identity, found),
            ));
        }
        self.start_remote_bundle_find(req, identity, target, remote_candidates, local, directory)
    }

    fn install_local_find(
        self: &Arc<Self>,
        req: &FindBlocksRequest,
        identity: &CacheIdentity,
        found: BundleFindMatch<Vec<kvbm_logical::ImmutableBlock<crate::G2>>>,
    ) -> FindBlocksOutcome {
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
        FindBlocksOutcome::Resolved {
            matched_tokens,
            minted: Some(FindBlocksHandle::search(
                req.request_id.clone(),
                search_id,
                Arc::downgrade(&engine),
            )),
            release_parked: false,
        }
    }

    fn start_remote_bundle_find(
        self: &Arc<Self>,
        req: &FindBlocksRequest,
        identity: &CacheIdentity,
        target: kvbm_protocols::cache_manifest::BundleKey,
        candidates: Vec<kvbm_protocols::cache_manifest::BundleKey>,
        fallback: Option<BundleFindMatch<Vec<kvbm_logical::ImmutableBlock<crate::G2>>>>,
        directory: crate::leader::RemoteDiscoveryHandle,
    ) -> Result<FindBlocksOutcome, LeaderEngineError> {
        let search_id = SearchId::new();
        let (tx, rx) = tokio::sync::oneshot::channel();
        let cancel = tokio_util::sync::CancellationToken::new();
        let deadline = tokio::time::Instant::now() + REMOTE_BUNDLE_DISCOVERY_TIMEOUT;
        let fallback_key = fallback.as_ref().map(|found| *found.key());
        let fallback_matched = fallback
            .as_ref()
            .map(BundleFindMatch::matched_tokens)
            .unwrap_or_default();
        let fallback_lease = fallback.map(BundleFindMatch::into_lease);
        self.bundle_searches.insert(
            search_id,
            BundleSearchState {
                request_id: req.request_id.clone(),
                identity: identity.clone(),
                source: BundleSearchSource::Remote(RemoteBundleSearch::new(
                    rx,
                    cancel.clone(),
                    fallback_lease,
                )),
                computed_tokens: req.num_computed_tokens,
                matched_tokens: fallback_matched,
            },
        );
        let task = RemoteBundleFind::new(RemoteBundleFindInputs {
            engine: Arc::clone(self),
            directory,
            request_id: req.request_id.clone(),
            identity: identity.clone(),
            candidates,
            target,
            fallback: fallback_key,
            num_computed_tokens: req.num_computed_tokens,
            total_tokens: req.total_tokens,
            local_prefill_estimate: req.local_prefill_estimate,
            cancel,
            deadline,
        });
        self.leader.runtime().spawn(async move {
            let _ = tx.send(Ok(task.execute().await));
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
        let terminal = match &mut state.source {
            BundleSearchSource::Local(_) => None,
            BundleSearchSource::Remote(remote) => match remote.try_recv() {
                Ok(result) => Some((result, remote.take_fallback())),
                Err(tokio::sync::oneshot::error::TryRecvError::Closed) => {
                    Some((Ok(RemoteBundleResolution::Fallback), remote.take_fallback()))
                }
                Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                    return Ok(FindBlocksOutcome::Searching { minted: None });
                }
            },
        };
        if let Some((resolution, fallback)) = terminal {
            let selected = match resolution {
                Ok(RemoteBundleResolution::Selected(lease)) => Some(lease),
                Ok(RemoteBundleResolution::Fallback) | Err(_) => fallback,
                Ok(RemoteBundleResolution::Miss) => None,
            };
            let Some(lease) = selected else {
                drop(state);
                self.bundle_searches.remove(&search_id);
                return Ok(zero_find(true));
            };
            state.source = BundleSearchSource::Local(lease);
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
        let replacement = {
            let catalog = self
                .bundle_catalog
                .lock()
                .expect("bundle-catalog mutex poisoned");
            query.find(catalog.index())
        };
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
        Ok(zero_find(true))
    }

    fn bundle_find_query<'a>(
        &self,
        req: &'a FindBlocksRequest,
        identity: &'a CacheIdentity,
        derived: &DerivedWindow,
    ) -> BundleFindQuery<'a> {
        let minimum_boundary = req
            .transfer_params
            .as_ref()
            .and_then(|params| params.remote_prefill.as_ref())
            .and_then(|params| params.bundle.as_ref())
            .and_then(|context| context.initial_bundle())
            .and_then(|key| usize::try_from(key.boundary_tokens()).ok())
            .unwrap_or_default();
        BundleFindQuery::new(
            identity,
            &req.sequence_hashes,
            req.num_computed_tokens,
            derived.range.end,
            NonZeroUsize::new(self.block_size).expect("engine block size must be nonzero"),
        )
        .with_minimum_boundary(minimum_boundary)
    }
}

fn zero_find(release_parked: bool) -> FindBlocksOutcome {
    FindBlocksOutcome::Resolved {
        matched_tokens: 0,
        minted: None,
        release_parked,
    }
}
