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
use crate::tiering::engine::find::DerivedWindow;
use crate::tiering::engine::local::{BundleSearchState, LocalConnectorEngine};

impl LocalConnectorEngine {
    pub(in crate::tiering::engine) fn start_bundle_find(
        self: &Arc<Self>,
        req: &FindBlocksRequest,
        identity: &CacheIdentity,
        derived: &DerivedWindow,
    ) -> Result<FindBlocksOutcome, LeaderEngineError> {
        let query = self.bundle_find_query(req, identity, derived);
        let Some(found) = query.find(
            &self
                .bundle_index
                .lock()
                .expect("bundle-index mutex poisoned"),
        ) else {
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
                lease: found.into_lease(),
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
        if let Some(matched_tokens) = query.matched_tokens_for(state.lease.key()) {
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
            state.lease = found.into_lease();
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
