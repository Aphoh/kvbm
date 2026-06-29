// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::num::NonZeroU64;

use kvbm_common::SequenceHash;
use serde::{Deserialize, Serialize};

use super::{CacheIdentity, CacheManifestId};

/// Manifest-scoped identity for one complete reusable prefix boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BundleKey {
    manifest: CacheManifestId,
    boundary_hash: SequenceHash,
    boundary_tokens: NonZeroU64,
}

impl BundleKey {
    pub fn new(
        identity: &CacheIdentity,
        boundary_hash: SequenceHash,
        boundary_tokens: u64,
    ) -> Result<Self, BundleKeyError> {
        let boundary_tokens =
            NonZeroU64::new(boundary_tokens).ok_or(BundleKeyError::ZeroBoundary)?;
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum BundleKeyError {
    #[error("bundle boundary must be greater than zero")]
    ZeroBoundary,
    #[error("bundle boundary {boundary_tokens} is not aligned to {alignment_tokens} native tokens")]
    UnalignedBoundary {
        boundary_tokens: u64,
        alignment_tokens: u64,
    },
}
