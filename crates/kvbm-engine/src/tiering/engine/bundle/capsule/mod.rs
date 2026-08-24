// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Atomic capsule storage and copy descriptors.

use kvbm_protocols::cache_manifest::BundleKey;
use kvbm_protocols::connector::BlockId;

/// One capsule allocation, even when its state spans several backing pools.
pub(super) struct CapsuleDescriptor {
    storage: CapsuleStorage,
    pools: Vec<CapsulePoolCopy>,
}

impl CapsuleDescriptor {
    pub(super) fn new(
        storage: CapsuleStorage,
        pools: Vec<CapsulePoolCopy>,
    ) -> Result<Self, CapsuleError> {
        if pools.is_empty() {
            return Err(CapsuleError::NoPools);
        }
        Ok(Self { storage, pools })
    }

    pub(super) const fn storage(&self) -> &CapsuleStorage {
        &self.storage
    }

    pub(super) fn copy_plan(&self) -> CapsuleCopyPlan<'_> {
        CapsuleCopyPlan {
            storage: &self.storage,
            pools: &self.pools,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CapsuleStorage {
    RequestSlots { generation: u64 },
    SharedPrefix { key: BundleKey, generation: u64 },
}

impl CapsuleStorage {
    pub(super) const fn generation(self) -> u64 {
        match self {
            Self::RequestSlots { generation } | Self::SharedPrefix { generation, .. } => generation,
        }
    }

    pub(super) const fn bundle_key(&self) -> Option<&BundleKey> {
        match self {
            Self::RequestSlots { .. } => None,
            Self::SharedPrefix { key, .. } => Some(key),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct CapsulePoolCopy {
    source_slots: Vec<BlockId>,
    destination_slots: Vec<BlockId>,
}

impl CapsulePoolCopy {
    pub(super) fn new(
        source_slots: Vec<BlockId>,
        destination_slots: Vec<BlockId>,
    ) -> Result<Self, CapsuleError> {
        if source_slots.is_empty() {
            return Err(CapsuleError::EmptyPool);
        }
        if source_slots.len() != destination_slots.len() {
            return Err(CapsuleError::PoolShapeMismatch {
                sources: source_slots.len(),
                destinations: destination_slots.len(),
            });
        }
        Ok(Self {
            source_slots,
            destination_slots,
        })
    }
}

pub(super) struct CapsuleCopyPlan<'a> {
    storage: &'a CapsuleStorage,
    pools: &'a [CapsulePoolCopy],
}

impl CapsuleCopyPlan<'_> {
    pub(super) fn generation(&self) -> u64 {
        self.storage.generation()
    }

    pub(super) const fn pools(&self) -> &[CapsulePoolCopy] {
        self.pools
    }

    pub(super) fn total_slots(&self) -> usize {
        self.pools.iter().map(|pool| pool.source_slots.len()).sum()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub(super) enum CapsuleError {
    #[error("capsule must contain at least one backing pool")]
    NoPools,
    #[error("capsule backing pool must contain at least one slot")]
    EmptyPool,
    #[error("capsule backing pool has {sources} sources and {destinations} destinations")]
    PoolShapeMismatch { sources: usize, destinations: usize },
}

#[cfg(test)]
mod tests;
