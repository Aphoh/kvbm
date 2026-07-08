// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::num::NonZeroU64;

use kvbm_common::SequenceHash;
use serde::{Deserialize, Deserializer, Serialize};

use super::{CacheIdentity, CacheManifestId};

/// Manifest-scoped identity for one complete reusable prefix boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct BundleKey {
    manifest: CacheManifestId,
    boundary_hash: SequenceHash,
    boundary_tokens: NonZeroU64,
}

impl BundleKey {
    /// Reconstruct a key from its wire components when the full manifest is
    /// not locally available. Compatibility with a concrete identity must
    /// still be checked by the consumer.
    pub fn from_parts(
        manifest: CacheManifestId,
        boundary_hash: SequenceHash,
        boundary_tokens: u64,
    ) -> Result<Self, BundleKeyError> {
        let boundary_tokens =
            NonZeroU64::new(boundary_tokens).ok_or(BundleKeyError::ZeroBoundary)?;
        validate_hash_encoding(boundary_hash)?;
        Ok(Self {
            manifest,
            boundary_hash,
            boundary_tokens,
        })
    }

    pub fn new(
        identity: &CacheIdentity,
        boundary_hash: SequenceHash,
        boundary_tokens: u64,
    ) -> Result<Self, BundleKeyError> {
        let boundary_tokens =
            NonZeroU64::new(boundary_tokens).ok_or(BundleKeyError::ZeroBoundary)?;
        validate_hash_encoding(boundary_hash)?;
        let alignment = identity.alignment_tokens().get();
        if !boundary_tokens.get().is_multiple_of(alignment) {
            return Err(BundleKeyError::UnalignedBoundary {
                boundary_tokens: boundary_tokens.get(),
                alignment_tokens: alignment,
            });
        }
        Ok(Self {
            manifest: identity.manifest(),
            boundary_hash,
            boundary_tokens,
        })
    }

    pub const fn manifest(&self) -> CacheManifestId {
        self.manifest
    }

    pub const fn boundary_hash(&self) -> SequenceHash {
        self.boundary_hash
    }

    pub const fn boundary_tokens(&self) -> u64 {
        self.boundary_tokens.get()
    }

    pub fn is_compatible_with(&self, identity: &CacheIdentity) -> bool {
        self.manifest == identity.manifest()
            && self
                .boundary_tokens()
                .is_multiple_of(identity.alignment_tokens().get())
    }
}

impl<'de> Deserialize<'de> for BundleKey {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = BundleKeyWire::deserialize(deserializer)?;
        Self::from_parts(wire.manifest, wire.boundary_hash, wire.boundary_tokens)
            .map_err(serde::de::Error::custom)
    }
}

#[derive(Deserialize)]
struct BundleKeyWire {
    manifest: CacheManifestId,
    boundary_hash: SequenceHash,
    boundary_tokens: u64,
}

fn validate_hash_encoding(boundary_hash: SequenceHash) -> Result<(), BundleKeyError> {
    let mode = boundary_hash.mode();
    if mode > 2 {
        return Err(BundleKeyError::InvalidHashEncoding { mode });
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum BundleKeyError {
    #[error("bundle boundary must be greater than zero")]
    ZeroBoundary,
    #[error("bundle boundary hash has invalid encoding mode {mode}")]
    InvalidHashEncoding { mode: u8 },
    #[error("bundle boundary {boundary_tokens} is not aligned to {alignment_tokens} native tokens")]
    UnalignedBoundary {
        boundary_tokens: u64,
        alignment_tokens: u64,
    },
}
