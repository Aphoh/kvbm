// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use kvbm_common::{LogicalResourceId, SequenceHash};
use kvbm_protocols::cache_manifest::{BundleKey, CacheManifestId};
use velo_ext::InstanceId;

use super::{BundleDirectory, BundleDirectoryError};
use crate::features::indexer::protocol::{
    BundleAdvertisementRecord, BundleInvalidateRequest, BundlePublishRequest,
    BundleQueryMissReason, BundleQueryOutcome, BundleQueryRequest,
};

const CSA: LogicalResourceId = LogicalResourceId(10);
const HCA: LogicalResourceId = LogicalResourceId(11);
const CAPSULE: LogicalResourceId = LogicalResourceId(12);

fn key(manifest: CacheManifestId, boundary: u64) -> BundleKey {
    BundleKey::from_parts(
        manifest,
        SequenceHash::new(boundary / 4, None, boundary),
        boundary,
    )
    .unwrap()
}

fn advertisement(
    owner: InstanceId,
    manifest: CacheManifestId,
    boundary: u64,
    generation: u64,
    expires_at_unix_ms: u64,
    resources: Vec<LogicalResourceId>,
) -> BundlePublishRequest {
    BundlePublishRequest {
        advertisement: BundleAdvertisementRecord {
            key: key(manifest, boundary),
            generation,
            owner,
            resources,
            expires_at_unix_ms,
        },
    }
}

fn query(
    manifest: CacheManifestId,
    candidates: Vec<BundleKey>,
    now_unix_ms: u64,
) -> BundleQueryRequest {
    BundleQueryRequest {
        manifest,
        required_resources: vec![CSA, HCA, CAPSULE],
        candidates,
        now_unix_ms,
    }
}

#[test]
fn unknown_or_removed_owner_is_never_visible() {
    let directory = BundleDirectory::new(500);
    let owner = InstanceId::new_v4();
    let manifest = CacheManifestId::from_bytes([1; 32]);
    let publish = advertisement(owner, manifest, 8, 1, 10_000, vec![CSA, HCA, CAPSULE]);

    assert!(matches!(
        directory.publish(publish.clone()),
        Err(BundleDirectoryError::UnknownOwner { .. })
    ));
    directory.register_owner(owner);
    directory.publish(publish).unwrap();
    assert!(matches!(
        directory.query(query(manifest, vec![key(manifest, 8)], 1_000)),
        BundleQueryOutcome::Hit(_)
    ));
    directory.remove_owner(owner);
    assert_eq!(
        directory.query(query(manifest, vec![key(manifest, 8)], 1_000)),
        BundleQueryOutcome::Miss(BundleQueryMissReason::NotFound)
    );
}

#[test]
fn incomplete_manifest_mismatch_and_expired_have_distinct_miss_reasons() {
    let directory = BundleDirectory::new(500);
    let owner = InstanceId::new_v4();
    let manifest = CacheManifestId::from_bytes([2; 32]);
    directory.register_owner(owner);
    directory
        .publish(advertisement(owner, manifest, 8, 1, 999, vec![CSA, HCA]))
        .unwrap();

    assert_eq!(
        directory.query(query(manifest, vec![key(manifest, 8)], 1_000)),
        BundleQueryOutcome::Miss(BundleQueryMissReason::Expired)
    );
    directory
        .publish(advertisement(
            owner,
            manifest,
            16,
            1,
            10_000,
            vec![CSA, HCA],
        ))
        .unwrap();
    assert_eq!(
        directory.query(query(manifest, vec![key(manifest, 16)], 1_000)),
        BundleQueryOutcome::Miss(BundleQueryMissReason::Incomplete)
    );
    let other = CacheManifestId::from_bytes([3; 32]);
    assert_eq!(
        directory.query(query(other, vec![key(manifest, 8)], 1)),
        BundleQueryOutcome::Miss(BundleQueryMissReason::Incompatible)
    );
}

#[test]
fn duplicate_resources_are_incomplete_not_complete() {
    let directory = BundleDirectory::new(500);
    let owner = InstanceId::new_v4();
    let manifest = CacheManifestId::from_bytes([6; 32]);
    directory.register_owner(owner);
    directory
        .publish(advertisement(
            owner,
            manifest,
            8,
            1,
            10_000,
            vec![CSA, HCA, CAPSULE, CAPSULE],
        ))
        .unwrap();

    assert_eq!(
        directory.query(query(manifest, vec![key(manifest, 8)], 1_000)),
        BundleQueryOutcome::Miss(BundleQueryMissReason::Incomplete)
    );
}

#[test]
fn stale_generation_is_rejected_and_query_can_retry_earlier_boundary() {
    let directory = BundleDirectory::new(500);
    let owner = InstanceId::new_v4();
    let manifest = CacheManifestId::from_bytes([4; 32]);
    directory.register_owner(owner);
    directory
        .publish(advertisement(
            owner,
            manifest,
            8,
            2,
            10_000,
            vec![CSA, HCA, CAPSULE],
        ))
        .unwrap();
    assert!(matches!(
        directory.publish(advertisement(
            owner,
            manifest,
            8,
            1,
            10_000,
            vec![CSA, HCA, CAPSULE],
        )),
        Err(BundleDirectoryError::StaleGeneration { .. })
    ));

    let BundleQueryOutcome::Hit(hit) = directory.query(query(
        manifest,
        vec![key(manifest, 16), key(manifest, 8)],
        1_000,
    )) else {
        panic!("expected earlier complete bundle");
    };
    assert_eq!(hit.advertisement.key, key(manifest, 8));
    assert!(hit.lease_expires_at_unix_ms <= 1_500);
    assert!(hit.lease_expires_at_unix_ms <= hit.advertisement.expires_at_unix_ms);
}

#[test]
fn invalidation_requires_the_exact_owner_generation() {
    let directory = BundleDirectory::new(500);
    let owner = InstanceId::new_v4();
    let manifest = CacheManifestId::from_bytes([5; 32]);
    let bundle_key = key(manifest, 8);
    directory.register_owner(owner);
    directory
        .publish(advertisement(
            owner,
            manifest,
            8,
            7,
            10_000,
            vec![CSA, HCA, CAPSULE],
        ))
        .unwrap();

    assert!(!directory.invalidate(BundleInvalidateRequest {
        key: bundle_key,
        generation: 6,
        owner,
    }));
    assert!(directory.invalidate(BundleInvalidateRequest {
        key: bundle_key,
        generation: 7,
        owner,
    }));
    assert_eq!(
        directory.query(query(manifest, vec![bundle_key], 1_000)),
        BundleQueryOutcome::Miss(BundleQueryMissReason::NotFound)
    );
}
