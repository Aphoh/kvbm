// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::disallowed_macros)]

use std::num::NonZeroUsize;

use kvbm_common::{LogicalResourceId, SequenceHash};
use kvbm_protocols::cache_manifest::{
    BundleKey, CacheManifest, ModelIdentity, ResourceRequirement, ResourceRole,
};
use proptest::prelude::*;

use super::BundleFindQuery;
use crate::tiering::engine::bundle::BundleIndex;

const HISTORY: LogicalResourceId = LogicalResourceId(80);
const CAPSULE: LogicalResourceId = LogicalResourceId(81);
const BASE: usize = 4;

fn hash(block: usize) -> SequenceHash {
    SequenceHash::new(block as u64, None, block as u64)
}

fn eligible_blocks(total_tokens: usize, chain_len: usize) -> usize {
    (total_tokens.saturating_sub(1) / BASE).min(chain_len)
}

fn manifest(resources: impl IntoIterator<Item = LogicalResourceId>) -> CacheManifest {
    CacheManifest::new(
        ModelIdentity::new("bundle-find-test", "v1", [5; 32]).unwrap(),
        "bundle-find-test-v1",
        resources
            .into_iter()
            .map(|resource| {
                let (role, tokens) = if resource == CAPSULE {
                    (ResourceRole::BoundaryCapsule, 8)
                } else {
                    (ResourceRole::PrefixHistory, 4)
                };
                ResourceRequirement::new(resource, role, tokens).unwrap()
            })
            .collect(),
        Default::default(),
    )
    .unwrap()
}

fn commit(index: &mut BundleIndex<u8>, manifest: &CacheManifest, boundary: usize) {
    let identity = manifest.identity();
    index
        .commit(
            &identity,
            BundleKey::new(&identity, hash(boundary / BASE), boundary as u64).unwrap(),
            boundary as u64,
            [(HISTORY, 1), (CAPSULE, 2)],
        )
        .unwrap();
}

#[test]
fn fixed_point_returns_greatest_complete_aligned_boundary() {
    let manifest = manifest([HISTORY, CAPSULE]);
    let identity = manifest.identity();
    let chain = (1..=8).map(hash).collect::<Vec<_>>();
    let mut index = BundleIndex::new();
    commit(&mut index, &manifest, 8);
    commit(&mut index, &manifest, 24);

    let found = BundleFindQuery::new(
        &identity,
        &chain,
        0,
        eligible_blocks(8 * BASE + 1, chain.len()),
        NonZeroUsize::new(BASE).unwrap(),
    )
    .find(&index)
    .unwrap();

    assert_eq!(found.lease().key().boundary_tokens(), 24);
    assert_eq!(found.matched_tokens(), 24);
}

#[test]
fn missing_later_capsule_forces_an_earlier_complete_bundle() {
    let manifest = manifest([HISTORY, CAPSULE]);
    let identity = manifest.identity();
    let chain = (1..=8).map(hash).collect::<Vec<_>>();
    let mut index = BundleIndex::new();
    commit(&mut index, &manifest, 8);
    // History may extend farther, but without an atomic capsule commit there
    // is deliberately no complete bundle at 16/24/32.

    let found = BundleFindQuery::new(
        &identity,
        &chain,
        0,
        eligible_blocks(8 * BASE + 1, chain.len()),
        NonZeroUsize::new(BASE).unwrap(),
    )
    .find(&index)
    .unwrap();

    assert_eq!(found.lease().key().boundary_tokens(), 8);
}

#[test]
fn final_block_is_excluded_when_total_tokens_is_block_aligned() {
    let manifest = manifest([HISTORY, CAPSULE]);
    let identity = manifest.identity();
    let chain = (1..=4).map(hash).collect::<Vec<_>>();
    let mut index = BundleIndex::new();
    commit(&mut index, &manifest, 8);
    commit(&mut index, &manifest, 16);

    let found = BundleFindQuery::new(
        &identity,
        &chain,
        0,
        eligible_blocks(4 * BASE, chain.len()),
        NonZeroUsize::new(BASE).unwrap(),
    )
    .find(&index)
    .unwrap();

    assert_eq!(found.lease().key().boundary_tokens(), 8);
}

#[test]
fn result_is_invariant_to_manifest_resource_input_order() {
    let left = manifest([HISTORY, CAPSULE]);
    let right = manifest([CAPSULE, HISTORY]);
    assert_eq!(left.identity(), right.identity());
    let chain = (1..=4).map(hash).collect::<Vec<_>>();
    let mut left_index = BundleIndex::new();
    let mut right_index = BundleIndex::new();
    commit(&mut left_index, &left, 8);
    commit(&mut right_index, &right, 8);

    let left_found = BundleFindQuery::new(
        &left.identity(),
        &chain,
        0,
        eligible_blocks(4 * BASE + 1, chain.len()),
        NonZeroUsize::new(BASE).unwrap(),
    )
    .find(&left_index)
    .unwrap();
    let right_found = BundleFindQuery::new(
        &right.identity(),
        &chain,
        0,
        eligible_blocks(4 * BASE + 1, chain.len()),
        NonZeroUsize::new(BASE).unwrap(),
    )
    .find(&right_index)
    .unwrap();

    assert_eq!(left_found.lease().key(), right_found.lease().key());
}

proptest! {
    #[test]
    fn optimized_find_matches_a_slow_exhaustive_oracle(
        committed in prop::collection::vec(any::<bool>(), 16),
        computed_blocks in 0usize..12,
        total_tokens in 1usize..=(16 * BASE + 1),
    ) {
        let manifest = manifest([HISTORY, CAPSULE]);
        let identity = manifest.identity();
        let chain = (1..=16).map(hash).collect::<Vec<_>>();
        let mut index = BundleIndex::new();
        for (offset, present) in committed.iter().copied().enumerate() {
            let boundary = (offset + 1) * BASE;
            if present && boundary.is_multiple_of(8) {
                commit(&mut index, &manifest, boundary);
            }
        }
        let computed_tokens = computed_blocks * BASE;
        let query = BundleFindQuery::new(
            &identity,
            &chain,
            computed_tokens,
            eligible_blocks(total_tokens, chain.len()),
            NonZeroUsize::new(BASE).unwrap(),
        );
        let actual = query.find(&index).map(|found| *found.lease().key());
        let eligible = total_tokens.saturating_sub(1) / BASE;
        let expected = (1..=eligible.min(chain.len()))
            .map(|block| block * BASE)
            .filter(|boundary| *boundary > computed_tokens)
            .filter(|boundary| boundary.is_multiple_of(8))
            .filter(|boundary| committed[boundary / BASE - 1])
            .max()
            .map(|boundary| {
                BundleKey::new(&identity, hash(boundary / BASE), boundary as u64).unwrap()
            });

        prop_assert_eq!(actual, expected);
    }
}
