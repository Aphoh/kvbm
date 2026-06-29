#![allow(clippy::disallowed_macros)]

use std::sync::Arc;

use kvbm_common::{LogicalResourceId, SequenceHash};
use kvbm_protocols::cache_manifest::{
    BundleKey, CacheManifest, ModelIdentity, ResourceRequirement, ResourceRole,
};

use super::{BundleIndex, BundleIndexError};

const CSA: LogicalResourceId = LogicalResourceId(10);
const HCA: LogicalResourceId = LogicalResourceId(11);
const CAPSULE: LogicalResourceId = LogicalResourceId(12);

fn requirement(resource: LogicalResourceId, role: ResourceRole) -> ResourceRequirement {
    ResourceRequirement::new(resource, role, 256).expect("valid requirement")
}

fn manifest(revision: &str) -> CacheManifest {
    CacheManifest::new(
        ModelIdentity::new("deepseek_v4", revision, [9; 32]).expect("valid model"),
        "dsv4-test-cache-v1",
        vec![
            requirement(CSA, ResourceRole::PrefixHistory),
            requirement(HCA, ResourceRole::PrefixHistory),
            requirement(CAPSULE, ResourceRole::BoundaryCapsule),
        ],
        Default::default(),
    )
    .expect("valid manifest")
}

fn key(manifest: &CacheManifest, token: u64) -> BundleKey {
    BundleKey::new(
        &manifest.identity(),
        SequenceHash::new(token, None, token),
        token,
    )
    .expect("aligned key")
}

fn complete_pins() -> Vec<(LogicalResourceId, Arc<&'static str>)> {
    vec![
        (CSA, Arc::new("csa")),
        (HCA, Arc::new("hca")),
        (CAPSULE, Arc::new("capsule")),
    ]
}

#[test]
fn incomplete_candidates_never_enter_the_committed_index() {
    let manifest = manifest("revision-a");
    let identity = manifest.identity();
    let mut index = BundleIndex::new();

    let error = index
        .commit(
            &identity,
            key(&manifest, 256),
            1,
            vec![(CSA, Arc::new("csa"))],
        )
        .expect_err("CSA alone is not a complete bundle");
    assert_eq!(
        error,
        BundleIndexError::MissingResources {
            resources: vec![HCA, CAPSULE]
        }
    );
    assert!(
        index
            .find_longest(&identity, &[(SequenceHash::new(256, None, 256), 256)])
            .is_none()
    );

    let error = index
        .commit(
            &identity,
            key(&manifest, 256),
            1,
            vec![(CSA, Arc::new("csa")), (HCA, Arc::new("hca"))],
        )
        .expect_err("histories without a capsule are incomplete");
    assert_eq!(
        error,
        BundleIndexError::MissingResources {
            resources: vec![CAPSULE]
        }
    );
}

#[test]
fn complete_bundle_returns_the_longest_matching_boundary() {
    let manifest = manifest("revision-a");
    let identity = manifest.identity();
    let mut index = BundleIndex::new();
    index
        .commit(&identity, key(&manifest, 256), 1, complete_pins())
        .expect("commit 256");
    index
        .commit(&identity, key(&manifest, 512), 2, complete_pins())
        .expect("commit 512");

    let lease = index
        .find_longest(
            &identity,
            &[
                (SequenceHash::new(512, None, 512), 512),
                (SequenceHash::new(256, None, 256), 256),
            ],
        )
        .expect("complete bundle match");

    assert_eq!(lease.key().boundary_tokens(), 512);
    assert_eq!(lease.generation(), 2);
    assert_eq!(lease.resources().len(), 3);
}

#[test]
fn manifest_and_boundary_hash_mismatches_are_misses() {
    let stored = manifest("revision-a");
    let incompatible = manifest("revision-b");
    let mut index = BundleIndex::new();
    index
        .commit(&stored.identity(), key(&stored, 256), 1, complete_pins())
        .expect("commit complete bundle");

    assert!(
        index
            .find_longest(
                &incompatible.identity(),
                &[(SequenceHash::new(256, None, 256), 256)],
            )
            .is_none()
    );
    assert!(
        index
            .find_longest(
                &stored.identity(),
                &[(SequenceHash::new(999, None, 256), 256)],
            )
            .is_none()
    );
}

#[test]
fn bundle_lease_clones_every_resource_pin() {
    let manifest = manifest("revision-a");
    let identity = manifest.identity();
    let csa = Arc::new("csa");
    let hca = Arc::new("hca");
    let capsule = Arc::new("capsule");
    let mut index = BundleIndex::new();
    index
        .commit(
            &identity,
            key(&manifest, 256),
            1,
            vec![
                (CSA, Arc::clone(&csa)),
                (HCA, Arc::clone(&hca)),
                (CAPSULE, Arc::clone(&capsule)),
            ],
        )
        .expect("commit complete bundle");

    assert_eq!(Arc::strong_count(&csa), 2);
    let lease = index
        .find_longest(&identity, &[(SequenceHash::new(256, None, 256), 256)])
        .expect("complete bundle match");
    assert_eq!(Arc::strong_count(&csa), 3);
    assert_eq!(Arc::strong_count(&hca), 3);
    assert_eq!(Arc::strong_count(&capsule), 3);
    drop(lease);
    assert_eq!(Arc::strong_count(&csa), 2);
}
