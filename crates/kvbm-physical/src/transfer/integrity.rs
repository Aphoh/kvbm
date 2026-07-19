// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Content digests for host-tier transport integrity.

use anyhow::{Result, anyhow, ensure};
use blake3::Hasher;
use kvbm_memory::StorageKind;
use serde::{Deserialize, Serialize};

use super::PhysicalLayout;
use crate::BlockId;

/// A BLAKE3 digest of one physical cache block's actual payload bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PayloadDigest([u8; 32]);

impl PayloadDigest {
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Digest local host/pinned blocks without introducing a device-to-host copy.
///
/// Remote bundle pulls land in G2. A Device layout is therefore a topology
/// error for this verifier and fails closed instead of adding a synchronous
/// D2H copy to the transfer hot path.
pub fn compute_host_block_digests(
    layout: &PhysicalLayout,
    block_ids: &[BlockId],
) -> Result<Vec<PayloadDigest>> {
    ensure!(
        matches!(layout.location(), StorageKind::System | StorageKind::Pinned),
        "transport integrity requires a local host/pinned G2 layout, got {:?}",
        layout.location()
    );
    block_ids
        .iter()
        .copied()
        .map(|block_id| compute_host_block_digest(layout, block_id))
        .collect()
}

fn compute_host_block_digest(layout: &PhysicalLayout, block_id: BlockId) -> Result<PayloadDigest> {
    let config = layout.layout().config();
    ensure!(
        block_id < config.num_blocks,
        "block ID {block_id} is outside layout capacity {}",
        config.num_blocks
    );

    let mut hasher = Hasher::new();
    hasher.update(b"kvbm-host-block-payload-v1\0");
    for layer_id in 0..config.num_layers {
        for outer_id in 0..config.outer_dim {
            let region = layout
                .memory_region(block_id, layer_id, outer_id)
                .map_err(|error| anyhow!("resolve block {block_id} payload region: {error:#}"))?;
            let bytes = unsafe {
                // The location check above limits this dereference to local
                // CPU-addressable System/Pinned storage.
                std::slice::from_raw_parts(region.addr() as *const u8, region.size())
            };
            hasher.update(&(layer_id as u64).to_le_bytes());
            hasher.update(&(outer_id as u64).to_le_bytes());
            hasher.update(&(bytes.len() as u64).to_le_bytes());
            hasher.update(bytes);
        }
    }
    Ok(PayloadDigest::from_bytes(*hasher.finalize().as_bytes()))
}

#[cfg(all(test, feature = "testing-kvbm"))]
mod tests {
    use super::super::tests::builder;
    use super::*;
    use crate::transfer::{FillPattern, fill_blocks};

    fn system_layout() -> PhysicalLayout {
        builder(2)
            .fully_contiguous()
            .allocate_system()
            .build()
            .unwrap()
    }

    #[test]
    fn host_payload_digest_changes_when_bytes_change() {
        let layout = system_layout();
        fill_blocks(&layout, &[0], FillPattern::Constant(7)).unwrap();
        let before = compute_host_block_digests(&layout, &[0]).unwrap();
        fill_blocks(&layout, &[0], FillPattern::Constant(9)).unwrap();
        let after = compute_host_block_digests(&layout, &[0]).unwrap();
        assert_ne!(before, after);
    }
}
