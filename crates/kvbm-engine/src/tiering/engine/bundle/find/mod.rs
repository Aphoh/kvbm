// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bundle-wide candidate generation and all-resource lease acquisition.

mod driver;

use std::num::NonZeroUsize;

use kvbm_common::SequenceHash;
use kvbm_protocols::cache_manifest::{BundleKey, CacheIdentity};

use super::{BundleIndex, BundleLease};

/// One request's eligible manifest-aligned boundaries in native-token space.
pub(in crate::tiering::engine) struct BundleFindQuery<'a> {
    identity: &'a CacheIdentity,
    sequence_hashes: &'a [SequenceHash],
    computed_tokens: usize,
    eligible_blocks: usize,
    base_block_tokens: NonZeroUsize,
}

impl<'a> BundleFindQuery<'a> {
    pub(in crate::tiering::engine) const fn new(
        identity: &'a CacheIdentity,
        sequence_hashes: &'a [SequenceHash],
        computed_tokens: usize,
        eligible_blocks: usize,
        base_block_tokens: NonZeroUsize,
    ) -> Self {
        Self {
            identity,
            sequence_hashes,
            computed_tokens,
            eligible_blocks,
            base_block_tokens,
        }
    }

    /// Find the greatest complete boundary and clone its all-resource lease.
    pub(in crate::tiering::engine) fn find<P: Clone>(
        &self,
        index: &BundleIndex<P>,
    ) -> Option<BundleFindMatch<P>> {
        let lease = index.find_longest(self.identity, self.candidates())?;
        let matched_tokens = usize::try_from(lease.key().boundary_tokens())
            .ok()?
            .checked_sub(self.computed_tokens)?;
        Some(BundleFindMatch {
            lease,
            matched_tokens,
        })
    }

    /// Whether a previously pinned key remains eligible for this poll.
    pub(in crate::tiering::engine) fn contains(&self, key: &BundleKey) -> bool {
        let block_tokens = self.base_block_tokens.get();
        let Ok(boundary) = usize::try_from(key.boundary_tokens()) else {
            return false;
        };
        if boundary <= self.computed_tokens || !boundary.is_multiple_of(block_tokens) {
            return false;
        }
        let Some(index) = boundary
            .checked_div(block_tokens)
            .and_then(|block| block.checked_sub(1))
        else {
            return false;
        };
        key.is_compatible_with(self.identity)
            && index < self.eligible_blocks.min(self.sequence_hashes.len())
            && self.sequence_hashes[index] == key.boundary_hash()
    }

    pub(in crate::tiering::engine) fn matched_tokens_for(&self, key: &BundleKey) -> Option<usize> {
        if !self.contains(key) {
            return None;
        }
        usize::try_from(key.boundary_tokens())
            .ok()?
            .checked_sub(self.computed_tokens)
    }

    pub(in crate::tiering::engine) fn candidate_keys(&self) -> Vec<BundleKey> {
        self.candidates()
            .filter_map(|(hash, tokens)| BundleKey::new(self.identity, hash, tokens).ok())
            .collect()
    }

    fn candidates(&self) -> impl Iterator<Item = (SequenceHash, u64)> + '_ {
        let block_tokens = self.base_block_tokens.get();
        let eligible_blocks = self.eligible_blocks.min(self.sequence_hashes.len());
        let computed_tokens = u64::try_from(self.computed_tokens).ok();
        let alignment = self.identity.alignment_tokens().get();
        (0..eligible_blocks).rev().filter_map(move |index| {
            let computed_tokens = computed_tokens?;
            let boundary = (index + 1).checked_mul(block_tokens)?;
            let boundary = u64::try_from(boundary).ok()?;
            (boundary > computed_tokens && boundary.is_multiple_of(alignment))
                .then_some((self.sequence_hashes[index], boundary))
        })
    }
}

/// A common scheduler-visible token count backed by one complete pinned lease.
pub(in crate::tiering::engine) struct BundleFindMatch<P> {
    lease: BundleLease<P>,
    matched_tokens: usize,
}

impl<P> BundleFindMatch<P> {
    #[cfg(test)]
    pub(in crate::tiering::engine) const fn lease(&self) -> &BundleLease<P> {
        &self.lease
    }

    pub(in crate::tiering::engine) const fn matched_tokens(&self) -> usize {
        self.matched_tokens
    }

    pub(in crate::tiering::engine) fn into_lease(self) -> BundleLease<P> {
        self.lease
    }
}

#[cfg(test)]
mod route_tests;
#[cfg(test)]
mod tests;
