// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use crate::InstanceId;
use kvbm_common::{LogicalResourceId, SequenceHash};
use kvbm_protocols::cache_manifest::{
    BundleKey, BundleResourceLineage, CacheManifest, ModelIdentity, RegistrationEpoch,
    ResourceRequirement, ResourceRole,
};

use super::{BundleAdvertisement, BundleDirectoryError, BundleDiscoveryQuery};

fn identity() -> kvbm_protocols::cache_manifest::CacheIdentity {
    CacheManifest::new(
        ModelIdentity::new("remote-test", "v1", [4; 32]).unwrap(),
        "remote-bundle-v1",
        vec![
            ResourceRequirement::new(LogicalResourceId(10), ResourceRole::PrefixHistory, 4)
                .unwrap(),
            ResourceRequirement::new(LogicalResourceId(11), ResourceRole::BoundaryCapsule, 8)
                .unwrap(),
        ],
        BTreeMap::new(),
    )
    .unwrap()
    .identity()
}

fn key(identity: &kvbm_protocols::cache_manifest::CacheIdentity) -> BundleKey {
    BundleKey::new(identity, hash(1), 8).unwrap()
}

fn hash(position: u64) -> SequenceHash {
    (1..=position).fold(SequenceHash::root(1), |parent, block| {
        parent.extend(block + 1)
    })
}

fn lineage(resource: u16, hashes: Vec<SequenceHash>) -> BundleResourceLineage {
    BundleResourceLineage::new(LogicalResourceId(resource), hashes).unwrap()
}

fn valid_lineages() -> [BundleResourceLineage; 2] {
    [
        lineage(10, vec![hash(0), hash(1)]),
        lineage(11, vec![hash(1)]),
    ]
}

fn registration_epoch() -> RegistrationEpoch {
    RegistrationEpoch::new()
}

#[test]
fn advertisement_requires_every_manifest_resource() {
    let identity = identity();
    let result = BundleAdvertisement::new(
        identity.clone(),
        key(&identity),
        3,
        InstanceId::new_v4(),
        registration_epoch(),
        10_000,
        [lineage(10, vec![hash(0), hash(1)])],
    );

    assert!(matches!(
        result,
        Err(BundleDirectoryError::IncompleteResources { .. })
    ));
}

#[test]
fn duplicate_manifest_resources_are_rejected() {
    let identity = identity();
    let result = BundleAdvertisement::new(
        identity.clone(),
        key(&identity),
        3,
        InstanceId::new_v4(),
        registration_epoch(),
        10_000,
        [
            lineage(10, vec![hash(0), hash(1)]),
            lineage(11, vec![hash(1)]),
            lineage(11, vec![hash(1)]),
        ],
    );

    assert!(matches!(
        result,
        Err(BundleDirectoryError::DuplicateResource(LogicalResourceId(
            11
        )))
    ));
}

#[test]
fn advertisement_rejects_role_native_block_count_mismatch() {
    let identity = identity();
    let result = BundleAdvertisement::new(
        identity.clone(),
        key(&identity),
        3,
        InstanceId::new_v4(),
        registration_epoch(),
        10_000,
        [lineage(10, vec![hash(0)]), lineage(11, vec![hash(1)])],
    );

    assert!(matches!(
        result,
        Err(BundleDirectoryError::InvalidResourceBlockCount {
            resource: LogicalResourceId(10),
            expected: 2,
            actual: 1,
            ..
        })
    ));
}

#[test]
fn advertisement_rejects_history_that_does_not_reach_native_boundary() {
    let identity = identity();
    let result = BundleAdvertisement::new(
        identity.clone(),
        key(&identity),
        3,
        InstanceId::new_v4(),
        registration_epoch(),
        10_000,
        [
            lineage(10, vec![hash(1), hash(2)]),
            lineage(11, vec![hash(1)]),
        ],
    );

    assert!(matches!(
        result,
        Err(BundleDirectoryError::ResourceBoundaryMismatch {
            resource: LogicalResourceId(10),
            boundary_tokens: 8,
        })
    ));
}

#[test]
fn advertisement_rejects_capsule_from_another_boundary() {
    let identity = identity();
    let result = BundleAdvertisement::new(
        identity.clone(),
        key(&identity),
        3,
        InstanceId::new_v4(),
        registration_epoch(),
        10_000,
        [
            lineage(10, vec![hash(0), hash(1)]),
            lineage(11, vec![hash(0)]),
        ],
    );

    assert!(matches!(
        result,
        Err(BundleDirectoryError::CapsuleBoundaryMismatch {
            resource: LogicalResourceId(11),
        })
    ));
}

#[test]
fn advertisement_requires_one_history_at_the_canonical_boundary_hash() {
    let identity = identity();
    let alternate_root = SequenceHash::root(50);
    let result = BundleAdvertisement::new(
        identity.clone(),
        key(&identity),
        3,
        InstanceId::new_v4(),
        registration_epoch(),
        10_000,
        [
            lineage(10, vec![alternate_root, alternate_root.extend(51)]),
            lineage(11, vec![hash(1)]),
        ],
    );

    assert_eq!(
        result,
        Err(BundleDirectoryError::MissingCanonicalHistoryBoundary)
    );
}

#[test]
fn manifest_mismatch_never_matches_a_query() {
    let identity = identity();
    let advertisement = BundleAdvertisement::new(
        identity.clone(),
        key(&identity),
        3,
        InstanceId::new_v4(),
        registration_epoch(),
        10_000,
        valid_lineages(),
    )
    .unwrap();
    let other = CacheManifest::new(
        ModelIdentity::new("remote-test", "other", [9; 32]).unwrap(),
        "remote-bundle-v1",
        identity.resources().to_vec(),
        BTreeMap::new(),
    )
    .unwrap()
    .identity();
    let query = BundleDiscoveryQuery::new(other, vec![advertisement.key()], 1_000);

    assert!(!advertisement.matches(&query));
}

#[test]
fn expired_advertisement_never_matches() {
    let identity = identity();
    let advertisement = BundleAdvertisement::new(
        identity.clone(),
        key(&identity),
        3,
        InstanceId::new_v4(),
        registration_epoch(),
        999,
        valid_lineages(),
    )
    .unwrap();
    let query = BundleDiscoveryQuery::new(identity, vec![advertisement.key()], 1_000);

    assert!(!advertisement.matches(&query));
}
