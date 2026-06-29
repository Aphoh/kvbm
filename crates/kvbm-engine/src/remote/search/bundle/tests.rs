// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use crate::InstanceId;
use kvbm_common::{LogicalResourceId, SequenceHash};
use kvbm_protocols::cache_manifest::{
    BundleKey, CacheManifest, ModelIdentity, ResourceRequirement, ResourceRole,
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
    BundleKey::new(identity, SequenceHash::new(1, None, 7), 8).unwrap()
}

#[test]
fn advertisement_requires_every_manifest_resource() {
    let identity = identity();
    let result = BundleAdvertisement::new(
        identity.clone(),
        key(&identity),
        3,
        InstanceId::new_v4(),
        10_000,
        [LogicalResourceId(10)],
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
        10_000,
        [
            LogicalResourceId(10),
            LogicalResourceId(11),
            LogicalResourceId(11),
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
fn manifest_mismatch_never_matches_a_query() {
    let identity = identity();
    let advertisement = BundleAdvertisement::new(
        identity.clone(),
        key(&identity),
        3,
        InstanceId::new_v4(),
        10_000,
        [LogicalResourceId(10), LogicalResourceId(11)],
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
        999,
        [LogicalResourceId(10), LogicalResourceId(11)],
    )
    .unwrap();
    let query = BundleDiscoveryQuery::new(identity, vec![advertisement.key()], 1_000);

    assert!(!advertisement.matches(&query));
}
