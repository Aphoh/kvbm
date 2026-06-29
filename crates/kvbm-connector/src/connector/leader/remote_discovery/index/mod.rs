//! Hub index protocol adapter for remote discovery.

use std::sync::Arc;

use anyhow::{Context, Result};
use futures::future::BoxFuture;
use kvbm_engine::remote::search::bundle::{
    BundleAdvertisement, BundleDiscoveryOutcome, BundleDiscoveryQuery, BundleInvalidation,
    BundleMissReason, RemoteBundleCandidate,
};
use kvbm_hub::{
    BundleAdvertisementRecord, BundleInvalidateRequest, BundlePublishRequest, BundleQueryHit,
    BundleQueryMissReason, BundleQueryOutcome, BundleQueryRequest, FindBlocksHit,
    IndexerLookupClient,
};
use kvbm_logical::SequenceHash;

pub(super) trait BlockIndex: Send + Sync {
    fn find_blocks(
        &self,
        hashes: Vec<SequenceHash>,
    ) -> BoxFuture<'static, Result<Option<FindBlocksHit>>>;

    fn find_bundle(
        &self,
        _query: BundleDiscoveryQuery,
    ) -> BoxFuture<'static, Result<BundleDiscoveryOutcome>> {
        Box::pin(async { Ok(BundleDiscoveryOutcome::Miss(BundleMissReason::NotFound)) })
    }

    fn publish_bundle(
        &self,
        _advertisement: BundleAdvertisement,
    ) -> BoxFuture<'static, Result<()>> {
        Box::pin(async { Ok(()) })
    }

    fn invalidate_bundle(
        &self,
        _invalidation: BundleInvalidation,
    ) -> BoxFuture<'static, Result<()>> {
        Box::pin(async { Ok(()) })
    }
}

pub(super) struct HubBlockIndex(pub(super) Arc<IndexerLookupClient>);

impl BlockIndex for HubBlockIndex {
    fn find_blocks(
        &self,
        hashes: Vec<SequenceHash>,
    ) -> BoxFuture<'static, Result<Option<FindBlocksHit>>> {
        let index = Arc::clone(&self.0);
        Box::pin(async move {
            index
                .find_blocks(hashes)
                .await
                .context("query KVBM hub block index")
        })
    }

    fn find_bundle(
        &self,
        query: BundleDiscoveryQuery,
    ) -> BoxFuture<'static, Result<BundleDiscoveryOutcome>> {
        let index = Arc::clone(&self.0);
        Box::pin(async move {
            let request = BundleQueryRequest {
                manifest: query.identity().manifest(),
                required_resources: query
                    .identity()
                    .resources()
                    .iter()
                    .map(|requirement| requirement.resource())
                    .collect(),
                candidates: query.candidates().to_vec(),
                now_unix_ms: query.now_unix_ms(),
            };
            match index
                .find_bundle(request)
                .await
                .context("query KVBM hub bundle index")?
            {
                BundleQueryOutcome::Hit(hit) => directory_hit(query, hit)
                    .map(Box::new)
                    .map(BundleDiscoveryOutcome::Hit),
                BundleQueryOutcome::Miss(reason) => {
                    Ok(BundleDiscoveryOutcome::Miss(translate_miss_reason(reason)))
                }
            }
        })
    }

    fn publish_bundle(&self, advertisement: BundleAdvertisement) -> BoxFuture<'static, Result<()>> {
        let index = Arc::clone(&self.0);
        Box::pin(async move {
            index
                .publish_bundle(BundlePublishRequest {
                    advertisement: BundleAdvertisementRecord {
                        key: advertisement.key(),
                        generation: advertisement.generation(),
                        owner: advertisement.owner(),
                        resources: advertisement.resources().collect(),
                        expires_at_unix_ms: advertisement.expires_at_unix_ms(),
                    },
                })
                .await
                .context("publish KVBM bundle advertisement")
        })
    }

    fn invalidate_bundle(
        &self,
        invalidation: BundleInvalidation,
    ) -> BoxFuture<'static, Result<()>> {
        let index = Arc::clone(&self.0);
        Box::pin(async move {
            index
                .invalidate_bundle(BundleInvalidateRequest {
                    key: invalidation.key,
                    generation: invalidation.generation,
                    owner: invalidation.owner,
                })
                .await
                .context("invalidate KVBM bundle advertisement")?;
            Ok(())
        })
    }
}

const fn translate_miss_reason(reason: BundleQueryMissReason) -> BundleMissReason {
    match reason {
        BundleQueryMissReason::NotFound => BundleMissReason::NotFound,
        BundleQueryMissReason::Incompatible => BundleMissReason::Incompatible,
        BundleQueryMissReason::Incomplete => BundleMissReason::Incomplete,
        BundleQueryMissReason::Expired => BundleMissReason::Expired,
    }
}

pub(super) fn directory_hit(
    query: BundleDiscoveryQuery,
    hit: BundleQueryHit,
) -> Result<RemoteBundleCandidate> {
    let record = hit.advertisement;
    let advertisement = BundleAdvertisement::new(
        query.identity().clone(),
        record.key,
        record.generation,
        record.owner,
        record.expires_at_unix_ms,
        record.resources,
    )?;
    anyhow::ensure!(
        advertisement.matches(&query),
        "hub returned an incompatible bundle advertisement"
    );
    RemoteBundleCandidate::new(advertisement, hit.lease_id, hit.lease_expires_at_unix_ms)
        .map_err(anyhow::Error::from)
}
