// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use kvbm_common::{LogicalResourceId, SequenceHash};

use super::{
    BundleLineageValidationError, BundleResourceLineage, BundleResourceLineageError,
    validate_bundle_lineages,
};
use crate::cache_manifest::{BundleKey, CacheManifestId, ResourceRequirement, ResourceRole};

const HISTORY: LogicalResourceId = LogicalResourceId(7);
const CAPSULE: LogicalResourceId = LogicalResourceId(8);
const SECONDARY_HISTORY: LogicalResourceId = LogicalResourceId(9);

fn hash(position: u64) -> SequenceHash {
    SequenceHash::new(position + 1, None, position)
}

fn chain(block_hashes: impl IntoIterator<Item = u64>) -> Vec<SequenceHash> {
    let mut block_hashes = block_hashes.into_iter();
    let root = SequenceHash::root(block_hashes.next().expect("chain must not be empty"));
    std::iter::once(root)
        .chain(block_hashes.scan(root, |parent, block_hash| {
            *parent = parent.extend(block_hash);
            Some(*parent)
        }))
        .collect()
}

fn requirements() -> Vec<ResourceRequirement> {
    vec![
        ResourceRequirement::new(HISTORY, ResourceRole::PrefixHistory, 4).unwrap(),
        ResourceRequirement::new(CAPSULE, ResourceRole::BoundaryCapsule, 4).unwrap(),
    ]
}

fn bundle_key(hashes: &[SequenceHash]) -> BundleKey {
    BundleKey::from_parts(CacheManifestId::from_bytes([1; 32]), hashes[1], 8).unwrap()
}

fn lineage(resource: LogicalResourceId, hashes: Vec<SequenceHash>) -> BundleResourceLineage {
    BundleResourceLineage::new(resource, hashes).unwrap()
}

#[test]
fn rejects_empty_and_duplicate_lineages() {
    let resource = LogicalResourceId(7);
    assert_eq!(
        BundleResourceLineage::new(resource, Vec::new()),
        Err(BundleResourceLineageError::Empty { resource })
    );
    assert!(matches!(
        BundleResourceLineage::new(resource, vec![hash(0), hash(0)]),
        Err(BundleResourceLineageError::DuplicateHash { .. })
    ));
}

#[test]
fn wire_deserialization_enforces_constructor_invariants() {
    let bytes = rmp_serde::to_vec_named(&serde_json::json!({
        "resource": LogicalResourceId(7),
        "hashes": [hash(0), hash(0)],
    }))
    .unwrap();

    assert!(rmp_serde::from_slice::<BundleResourceLineage>(&bytes).is_err());
}

#[test]
fn rejects_nonconsecutive_positions() {
    let resource = LogicalResourceId(7);
    let hashes = chain([11, 12, 13]);

    assert_eq!(
        BundleResourceLineage::new(resource, vec![hashes[0], hashes[2]]),
        Err(BundleResourceLineageError::NonConsecutivePositions {
            resource,
            parent_position: 0,
            child_position: 2,
        })
    );
}

#[test]
fn rejects_adjacent_hashes_with_a_different_parent() {
    let resource = LogicalResourceId(7);
    let parent_a = SequenceHash::root(11);
    let child_of_c = SequenceHash::root(33).extend(22);

    assert_eq!(
        BundleResourceLineage::new(resource, vec![parent_a, child_of_c]),
        Err(BundleResourceLineageError::ParentHashMismatch {
            resource,
            parent: parent_a,
            child: child_of_c,
        })
    );
}

#[test]
fn accepts_a_genuine_parent_child_chain() {
    let resource = LogicalResourceId(7);
    let hashes = chain([11, 12, 13]);

    let lineage = BundleResourceLineage::new(resource, hashes.clone()).unwrap();

    assert_eq!(lineage.hashes(), hashes);
}

#[test]
fn accepts_a_genuine_edge_across_a_position_encoding_boundary() {
    let resource = LogicalResourceId(7);
    let hashes = chain(1..=257);

    assert!(BundleResourceLineage::new(resource, hashes).is_ok());
}

#[test]
fn canonical_projection_rebases_a_genuine_chain_across_a_position_encoding_boundary() {
    let resource = LogicalResourceId(7);
    let canonical = chain(1..=520);

    let projected = BundleResourceLineage::project_from_canonical(resource, &canonical, 2).unwrap();

    assert_eq!(projected.hashes().len(), 260);
    for (position, hash) in projected.hashes().iter().enumerate() {
        assert_eq!(hash.position(), position as u64);
        assert_eq!(
            hash.current_sequence_hash(),
            canonical[position * 2 + 1].current_sequence_hash()
        );
    }
    let parent = projected.hashes()[255];
    let child = projected.hashes()[256];
    assert_eq!(
        child.parent_hash_fragment(),
        parent.parent_fragment_for_child_position(child.position())
    );
    assert!(BundleResourceLineage::new(resource, projected.hashes().to_vec()).is_ok());
}

#[test]
fn wire_deserialization_rejects_a_disconnected_chain() {
    let resource = LogicalResourceId(7);
    let parent_a = SequenceHash::root(11);
    let child_of_c = SequenceHash::root(33).extend(22);
    let bytes = rmp_serde::to_vec_named(&serde_json::json!({
        "resource": resource,
        "hashes": [parent_a, child_of_c],
    }))
    .unwrap();

    assert!(rmp_serde::from_slice::<BundleResourceLineage>(&bytes).is_err());
}

#[test]
fn wire_deserialization_rejects_an_invalid_hash_mode_without_panicking() {
    let mut invalid_hash = [0u8; 16];
    invalid_hash[0] = 0b1100_0000;
    let bytes = rmp_serde::to_vec_named(&serde_json::json!({
        "resource": HISTORY,
        "hashes": [invalid_hash, hash(1)],
    }))
    .unwrap();

    let decoded =
        std::panic::catch_unwind(|| rmp_serde::from_slice::<BundleResourceLineage>(&bytes));

    assert!(decoded.is_ok(), "malformed wire data must not panic");
    assert!(
        decoded.unwrap().is_err(),
        "invalid hash mode must be rejected"
    );
}

#[test]
fn bundle_geometry_accepts_complete_role_correct_lineages() {
    let hashes = chain([11, 12]);
    let lineages = [
        lineage(HISTORY, hashes.clone()),
        lineage(CAPSULE, vec![hashes[1]]),
    ];

    assert!(validate_bundle_lineages(bundle_key(&hashes), &requirements(), &lineages).is_ok());
}

#[test]
fn bundle_geometry_rejects_wrong_role_specific_counts() {
    let hashes = chain([11, 12]);
    let short_history = [
        lineage(HISTORY, vec![hashes[0]]),
        lineage(CAPSULE, vec![hashes[1]]),
    ];
    let multi_block_capsule = [
        lineage(HISTORY, hashes.clone()),
        lineage(CAPSULE, hashes.clone()),
    ];

    assert_eq!(
        validate_bundle_lineages(bundle_key(&hashes), &requirements(), &short_history),
        Err(BundleLineageValidationError::InvalidResourceBlockCount {
            resource: HISTORY,
            role: ResourceRole::PrefixHistory,
            boundary_tokens: 8,
            expected: 2,
            actual: 1,
        })
    );
    assert_eq!(
        validate_bundle_lineages(bundle_key(&hashes), &requirements(), &multi_block_capsule,),
        Err(BundleLineageValidationError::InvalidResourceBlockCount {
            resource: CAPSULE,
            role: ResourceRole::BoundaryCapsule,
            boundary_tokens: 8,
            expected: 1,
            actual: 2,
        })
    );
}

#[test]
fn bundle_geometry_rejects_a_missing_canonical_history_boundary() {
    let canonical = chain([11, 12]);
    let alternate = chain([21, 22]);
    let lineages = [
        lineage(HISTORY, alternate),
        lineage(CAPSULE, vec![canonical[1]]),
    ];

    assert_eq!(
        validate_bundle_lineages(bundle_key(&canonical), &requirements(), &lineages),
        Err(BundleLineageValidationError::MissingCanonicalHistoryBoundary)
    );
}

#[test]
fn bundle_geometry_rejects_a_disconnected_secondary_history() {
    let canonical = chain([11, 12]);
    let disconnected = chain([21, 22]);
    let requirements = [
        ResourceRequirement::new(HISTORY, ResourceRole::PrefixHistory, 4).unwrap(),
        ResourceRequirement::new(SECONDARY_HISTORY, ResourceRole::PrefixHistory, 4).unwrap(),
        ResourceRequirement::new(CAPSULE, ResourceRole::BoundaryCapsule, 4).unwrap(),
    ];
    let lineages = [
        lineage(HISTORY, canonical.clone()),
        lineage(SECONDARY_HISTORY, disconnected),
        lineage(CAPSULE, vec![canonical[1]]),
    ];

    assert_eq!(
        validate_bundle_lineages(bundle_key(&canonical), &requirements, &lineages),
        Err(BundleLineageValidationError::ResourceBoundaryMismatch {
            resource: SECONDARY_HISTORY,
            boundary_tokens: 8,
        })
    );
}

#[test]
fn bundle_geometry_rejects_a_capsule_from_another_boundary() {
    let hashes = chain([11, 12]);
    let lineages = [
        lineage(HISTORY, hashes.clone()),
        lineage(CAPSULE, vec![SequenceHash::root(99)]),
    ];

    assert_eq!(
        validate_bundle_lineages(bundle_key(&hashes), &requirements(), &lineages),
        Err(BundleLineageValidationError::CapsuleBoundaryMismatch { resource: CAPSULE })
    );
}

#[test]
fn bundle_geometry_rejects_duplicate_and_missing_resources() {
    let hashes = chain([11, 12]);
    let duplicate = [
        lineage(HISTORY, hashes.clone()),
        lineage(CAPSULE, vec![hashes[1]]),
        lineage(CAPSULE, vec![hashes[1]]),
    ];
    let missing = [lineage(HISTORY, hashes.clone())];

    assert_eq!(
        validate_bundle_lineages(bundle_key(&hashes), &requirements(), &duplicate),
        Err(BundleLineageValidationError::DuplicateResource(CAPSULE))
    );
    assert_eq!(
        validate_bundle_lineages(bundle_key(&hashes), &requirements(), &missing),
        Err(BundleLineageValidationError::IncompleteResources {
            expected: vec![HISTORY, CAPSULE],
            actual: vec![HISTORY],
        })
    );
}

#[test]
fn bundle_geometry_rejects_duplicate_requirements() {
    let hashes = chain([11, 12]);
    let mut duplicate_requirements = requirements();
    duplicate_requirements
        .push(ResourceRequirement::new(HISTORY, ResourceRole::PrefixHistory, 4).unwrap());
    let lineages = [
        lineage(HISTORY, hashes.clone()),
        lineage(CAPSULE, vec![hashes[1]]),
    ];

    assert_eq!(
        validate_bundle_lineages(bundle_key(&hashes), &duplicate_requirements, &lineages),
        Err(BundleLineageValidationError::DuplicateRequirement(HISTORY))
    );
}

#[test]
fn bundle_geometry_rejects_an_empty_contract() {
    let key = BundleKey::from_parts(
        CacheManifestId::from_bytes([1; 32]),
        SequenceHash::root(11),
        4,
    )
    .unwrap();

    assert_eq!(
        validate_bundle_lineages(key, &[], &[]),
        Err(BundleLineageValidationError::NoRequirements)
    );
}

#[test]
fn bundle_geometry_rejects_a_boundary_unaligned_to_a_native_block() {
    let hash = SequenceHash::root(11);
    let key = BundleKey::from_parts(CacheManifestId::from_bytes([1; 32]), hash, 6).unwrap();
    let lineages = [lineage(HISTORY, vec![hash]), lineage(CAPSULE, vec![hash])];

    assert_eq!(
        validate_bundle_lineages(key, &requirements(), &lineages),
        Err(BundleLineageValidationError::UnalignedResourceBoundary {
            resource: HISTORY,
            boundary_tokens: 6,
            native_block_tokens: 4,
        })
    );
}
