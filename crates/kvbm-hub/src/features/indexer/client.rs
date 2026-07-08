// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Client-side velo lookup wrapper for the KV indexer feature.
//!
//! Mirrors [`ConditionalDisaggClient`](crate::features::disagg::client::ConditionalDisaggClient):
//! a thin wrapper over an [`Arc<Messenger>`] that knows the indexer
//! [`QUERY_HANDLER`] name and the hub's velo [`InstanceId`], exposing a single
//! per-request [`find_blocks`](IndexerLookupClient::find_blocks) call. Construct
//! it via [`HubClient::indexer_lookup_client`](crate::HubClient::indexer_lookup_client),
//! which gates on the indexer being enabled and supplies the hub's `InstanceId`.

use std::sync::Arc;

use anyhow::Result;
use kvbm_logical::SequenceHash;
use kvbm_protocols::cache_manifest::RegistrationEpoch;
use velo::Messenger;
use velo_ext::InstanceId;

use super::protocol::{
    BUNDLE_INVALIDATE_HANDLER, BUNDLE_PUBLISH_HANDLER, BUNDLE_QUERY_HANDLER,
    BundleAdvertisementRecord, BundleInvalidateRequest, BundleInvalidationRecord,
    BundlePublishRequest, BundleQueryOutcome, BundleQueryRequest, FindBlocksHit, QUERY_HANDLER,
    QueryRequest,
};
use crate::protocol::MutationCredential;

/// Velo-plane lookup client for the hub's KV block index.
pub struct IndexerLookupClient {
    messenger: Arc<Messenger>,
    /// Hub's velo `InstanceId` — the target of the lookup unary RPC.
    hub_velo_id: InstanceId,
    mutation_credential: MutationCredential,
    registration_epoch: RegistrationEpoch,
}

impl std::fmt::Debug for IndexerLookupClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IndexerLookupClient")
            .field("hub_velo_id", &self.hub_velo_id)
            .finish()
    }
}

impl IndexerLookupClient {
    /// Wrap a [`Messenger`] targeting the hub at `hub_velo_id`.
    pub(crate) fn new(
        messenger: Arc<Messenger>,
        hub_velo_id: InstanceId,
        mutation_credential: MutationCredential,
        registration_epoch: RegistrationEpoch,
    ) -> Arc<Self> {
        Arc::new(Self {
            messenger,
            hub_velo_id,
            mutation_credential,
            registration_epoch,
        })
    }

    /// The hub's velo `InstanceId` this client targets.
    pub fn hub_velo_id(&self) -> InstanceId {
        self.hub_velo_id
    }

    /// Resolve a candidate block sequence to the deepest indexed block and its
    /// holders, over velo.
    ///
    /// `hashes` are the block-sequence PLHs in position order (low → high). The
    /// hub walks them and returns the deepest one present — so `[x, y, z]` with
    /// `z` missing but `y` indexed yields `Some(hit)` where `hit.matched == y`
    /// and `hit.candidates` are the instances holding `y`. A full miss returns
    /// `Ok(None)`.
    pub async fn find_blocks(&self, hashes: Vec<SequenceHash>) -> Result<Option<FindBlocksHit>> {
        let req = QueryRequest { hashes };
        let hit = self
            .messenger
            .typed_unary::<Option<FindBlocksHit>>(QUERY_HANDLER)?
            .payload(&req)?
            .instance(self.hub_velo_id)
            .send()
            .await?;
        Ok(hit)
    }

    pub async fn publish_bundle(&self, advertisement: BundleAdvertisementRecord) -> Result<()> {
        validate_registration_epoch(advertisement.registration_epoch, self.registration_epoch)?;
        let request = BundlePublishRequest {
            credential: self.mutation_credential.clone(),
            advertisement,
        };
        self.messenger
            .typed_unary::<()>(BUNDLE_PUBLISH_HANDLER)?
            .payload(&request)?
            .instance(self.hub_velo_id)
            .send()
            .await?;
        Ok(())
    }

    pub async fn invalidate_bundle(&self, invalidation: BundleInvalidationRecord) -> Result<bool> {
        let request = BundleInvalidateRequest {
            credential: self.mutation_credential.clone(),
            key: invalidation.key,
            generation: invalidation.generation,
            owner: invalidation.owner,
            retain_until_unix_ms: invalidation.retain_until_unix_ms,
        };
        self.messenger
            .typed_unary::<bool>(BUNDLE_INVALIDATE_HANDLER)?
            .payload(&request)?
            .instance(self.hub_velo_id)
            .send()
            .await
    }

    pub async fn find_bundle(&self, request: BundleQueryRequest) -> Result<BundleQueryOutcome> {
        self.messenger
            .typed_unary::<BundleQueryOutcome>(BUNDLE_QUERY_HANDLER)?
            .payload(&request)?
            .instance(self.hub_velo_id)
            .send()
            .await
    }
}

fn validate_registration_epoch(
    advertised: Option<RegistrationEpoch>,
    registered: RegistrationEpoch,
) -> Result<()> {
    anyhow::ensure!(
        advertised == Some(registered),
        "bundle advertisement registration epoch does not match the indexer client registration"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundle_publication_rejects_missing_or_mismatched_registration_epoch() {
        let registered = RegistrationEpoch::new();

        assert!(validate_registration_epoch(None, registered).is_err());
        assert!(validate_registration_epoch(Some(RegistrationEpoch::new()), registered).is_err());
        assert!(validate_registration_epoch(Some(registered), registered).is_ok());
    }
}
