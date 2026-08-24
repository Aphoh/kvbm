#![allow(clippy::disallowed_macros)]

use std::sync::{Arc, Weak};

use kvbm_common::{LogicalResourceId, SequenceHash};
use kvbm_protocols::cache_manifest::{
    BundleKey, CacheManifest, ModelIdentity, ResourceRequirement, ResourceRole,
};

use super::{BundleIndex, BundleIndexError, BundleResourcePin, BundleResourceReference};
use crate::tiering::policy::{BundleDependencyIndex, ResourceLineage};

const CSA: LogicalResourceId = LogicalResourceId(10);
const HCA: LogicalResourceId = LogicalResourceId(11);
const CAPSULE: LogicalResourceId = LogicalResourceId(12);

struct EphemeralPin(Arc<str>);

struct EphemeralReference(Weak<str>);

impl BundleResourcePin for EphemeralPin {
    type Reference = EphemeralReference;

    fn make_reference(&self) -> Self::Reference {
        EphemeralReference(Arc::downgrade(&self.0))
    }
}

impl BundleResourceReference for EphemeralReference {
    type Pin = EphemeralPin;

    fn reacquire(&self) -> Option<Self::Pin> {
        self.0.upgrade().map(EphemeralPin)
    }
}

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

fn text(value: &'static str) -> Arc<str> {
    Arc::from(value)
}

fn complete_pins() -> Vec<(LogicalResourceId, Arc<str>)> {
    vec![
        (CSA, text("csa")),
        (HCA, text("hca")),
        (CAPSULE, text("capsule")),
    ]
}

#[test]
fn incomplete_candidates_never_enter_the_committed_index() {
    let manifest = manifest("revision-a");
    let identity = manifest.identity();
    let mut index = BundleIndex::new();

    let error = index
        .commit(&identity, key(&manifest, 256), 1, vec![(CSA, text("csa"))])
        .expect_err("CSA alone is not a complete bundle");
    assert_eq!(
        error,
        BundleIndexError::MissingResources {
            resources: vec![HCA, CAPSULE]
        }
    );
    assert!(
        index
            .find_longest(&identity, [(SequenceHash::new(256, None, 256), 256)])
            .is_none()
    );

    let error = index
        .commit(
            &identity,
            key(&manifest, 256),
            1,
            vec![(CSA, text("csa")), (HCA, text("hca"))],
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
            [
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
                [(SequenceHash::new(256, None, 256), 256)],
            )
            .is_none()
    );
    assert!(
        index
            .find_longest(
                &stored.identity(),
                [(SequenceHash::new(999, None, 256), 256)],
            )
            .is_none()
    );
}

#[test]
fn bundle_lease_clones_every_resource_pin() {
    let manifest = manifest("revision-a");
    let identity = manifest.identity();
    let csa = text("csa");
    let hca = text("hca");
    let capsule = text("capsule");
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
        .find_longest(&identity, [(SequenceHash::new(256, None, 256), 256)])
        .expect("complete bundle match");
    assert_eq!(Arc::strong_count(&csa), 3);
    assert_eq!(Arc::strong_count(&hca), 3);
    assert_eq!(Arc::strong_count(&capsule), 3);
    drop(lease);
    assert_eq!(Arc::strong_count(&csa), 2);
}

#[test]
fn missing_child_fails_closed_and_drops_every_partial_reacquisition() {
    let manifest = manifest("revision-a");
    let identity = manifest.identity();
    let csa = text("csa");
    let hca = text("hca");
    let capsule = text("capsule");
    let bundle_key = key(&manifest, 256);
    let mut index = BundleIndex::new();
    index
        .commit(
            &identity,
            bundle_key,
            1,
            vec![
                (CSA, EphemeralPin(Arc::clone(&csa))),
                (HCA, EphemeralPin(Arc::clone(&hca))),
                (CAPSULE, EphemeralPin(Arc::clone(&capsule))),
            ],
        )
        .unwrap();
    assert_eq!(Arc::strong_count(&csa), 1, "the index must not pin CSA");
    drop(hca);

    assert!(index.lease_exact(&identity, &bundle_key).is_none());
    assert_eq!(
        Arc::strong_count(&csa),
        1,
        "a failed all-resource lookup must drop earlier reacquisitions"
    );
    assert_eq!(Arc::strong_count(&capsule), 1);
}

#[test]
fn returned_lease_survives_index_invalidation_and_source_eviction() {
    let manifest = manifest("revision-a");
    let identity = manifest.identity();
    let csa = text("csa");
    let hca = text("hca");
    let capsule = text("capsule");
    let bundle_key = key(&manifest, 256);
    let mut index = BundleIndex::new();
    index
        .commit(
            &identity,
            bundle_key,
            1,
            vec![
                (CSA, EphemeralPin(Arc::clone(&csa))),
                (HCA, EphemeralPin(Arc::clone(&hca))),
                (CAPSULE, EphemeralPin(Arc::clone(&capsule))),
            ],
        )
        .unwrap();
    let lease = index.lease_exact(&identity, &bundle_key).unwrap();
    drop((csa, hca, capsule));
    assert!(index.invalidate(bundle_key));

    assert_eq!(lease.resources().len(), 3);
    assert!(
        lease
            .resources()
            .values()
            .all(|pin| Arc::strong_count(&pin.0) == 1)
    );
}

#[test]
fn older_generation_cannot_overwrite_a_newer_capsule_bundle() {
    let manifest = manifest("revision-a");
    let identity = manifest.identity();
    let bundle_key = key(&manifest, 256);
    let original = text("new-generation");
    let mut index = BundleIndex::new();
    index
        .commit(
            &identity,
            bundle_key,
            8,
            vec![
                (CSA, Arc::clone(&original)),
                (HCA, text("hca")),
                (CAPSULE, text("capsule")),
            ],
        )
        .unwrap();

    assert_eq!(
        index.commit(&identity, bundle_key, 7, complete_pins()),
        Err(BundleIndexError::StaleGeneration {
            current: 8,
            attempted: 7,
        })
    );
    index
        .commit(&identity, bundle_key, 8, complete_pins())
        .expect("same-generation retry is idempotent");
    let lease = index
        .find_longest(&identity, [(SequenceHash::new(256, None, 256), 256)])
        .unwrap();
    assert_eq!(lease.generation(), 8);
    assert_eq!(Arc::strong_count(&original), 3);
}

#[test]
fn resource_invalidation_removes_every_dependent_bundle_from_visibility() {
    let manifest = manifest("revision-a");
    let identity = manifest.identity();
    let shallow = key(&manifest, 256);
    let deep = key(&manifest, 512);
    let shallow_hash = shallow.boundary_hash();
    let mut bundles = BundleIndex::new();
    bundles
        .commit(&identity, shallow, 1, complete_pins())
        .unwrap();
    bundles.commit(&identity, deep, 2, complete_pins()).unwrap();

    let mut dependencies = BundleDependencyIndex::new();
    dependencies
        .track(
            shallow,
            [ResourceLineage::new(
                CSA,
                ResourceRole::PrefixHistory,
                vec![shallow_hash],
            )],
        )
        .unwrap();
    dependencies
        .track(
            deep,
            [ResourceLineage::new(
                CSA,
                ResourceRole::PrefixHistory,
                vec![shallow_hash, deep.boundary_hash()],
            )],
        )
        .unwrap();

    dependencies.invalidate(CSA, shallow_hash, |event| {
        bundles.invalidate(event.key());
    });

    assert!(bundles.lease_exact(&identity, &shallow).is_none());
    assert!(bundles.lease_exact(&identity, &deep).is_none());
}
