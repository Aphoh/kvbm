// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Physical distribution, collective, and logical-capacity resolution.

#[cfg(feature = "nccl")]
use anyhow::Context;
use anyhow::{Result, bail};
use kvbm_common::placement::StripedBlockPlacement;
use kvbm_engine::worker::CollectiveBootstrap;
use kvbm_physical::layout::LayoutConfig;

/// Resolve physical cache distribution from the registered tensor schema.
pub(in crate::connector::leader::construct) fn resolve_parallelism(
    configured: kvbm_config::ParallelismMode,
    layout: &LayoutConfig,
) -> kvbm_config::ParallelismMode {
    if layout.num_heads.is_none() {
        kvbm_config::ParallelismMode::ReplicatedData
    } else {
        configured
    }
}

fn collective_required(parallelism: kvbm_config::ParallelismMode, worker_count: usize) -> bool {
    parallelism == kvbm_config::ParallelismMode::ReplicatedData && worker_count > 1
}

pub(in crate::connector::leader::construct) fn build_collective_bootstrap(
    parallelism: kvbm_config::ParallelismMode,
    worker_count: usize,
) -> Result<Option<CollectiveBootstrap>> {
    if !collective_required(parallelism, worker_count) {
        return Ok(None);
    }

    #[cfg(feature = "nccl")]
    {
        let bootstrap = kvbm_engine::collectives::NcclBootstrap::generate(worker_count)
            .context("generating KVBM NCCL bootstrap for replicated cache workers")?;
        Ok(Some(CollectiveBootstrap::Nccl {
            serialized: bootstrap.serialize(),
        }))
    }

    #[cfg(not(feature = "nccl"))]
    bail!(
        "replicated cache data with {worker_count} workers requires the kvbm-connector `nccl` feature"
    )
}

/// Logical tier capacity represented by equal per-worker physical capacities.
///
/// Tensor-parallel blocks are sharded, so one block consumes one slot on every
/// rank and the logical capacity equals the per-rank capacity. Replicated MLA
/// blocks have one canonical lower-tier copy, so each rank contributes a
/// disjoint stripe and the logical capacity is the aggregate across ranks.
pub(in crate::connector::leader::construct) fn logical_tier_block_count(
    per_worker_blocks: usize,
    parallelism: kvbm_config::ParallelismMode,
    worker_count: usize,
) -> Result<usize> {
    if worker_count == 0 {
        bail!("cannot configure cache tiers without workers");
    }
    match parallelism {
        kvbm_config::ParallelismMode::TensorParallel => Ok(per_worker_blocks),
        kvbm_config::ParallelismMode::ReplicatedData => {
            StripedBlockPlacement::new(worker_count)?.global_capacity(per_worker_blocks)
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "nccl")]
    use super::build_collective_bootstrap;
    use super::{collective_required, logical_tier_block_count, resolve_parallelism};
    use kvbm_config::ParallelismMode;
    use kvbm_physical::layout::LayoutConfig;

    fn layout(num_heads: Option<usize>) -> LayoutConfig {
        LayoutConfig::builder()
            .num_blocks(128)
            .num_layers(2)
            .outer_dim(if num_heads.is_some() { 2 } else { 1 })
            .page_size(16)
            .inner_dim(512)
            .num_heads(num_heads)
            .build()
            .unwrap()
    }

    #[test]
    fn mla_layout_without_head_axis_selects_replicated_data() {
        assert_eq!(
            resolve_parallelism(ParallelismMode::TensorParallel, &layout(None)),
            ParallelismMode::ReplicatedData
        );
    }

    #[test]
    fn only_multi_worker_replicated_data_requires_a_collective() {
        assert!(!collective_required(ParallelismMode::ReplicatedData, 1));
        assert!(collective_required(ParallelismMode::ReplicatedData, 2));
        assert!(!collective_required(ParallelismMode::TensorParallel, 2));
    }

    #[cfg(feature = "nccl")]
    #[test]
    fn replicated_group_gets_one_decodable_nccl_bootstrap() {
        let collective = build_collective_bootstrap(ParallelismMode::ReplicatedData, 2)
            .unwrap()
            .expect("TP=2 replicated data requires a collective");
        let kvbm_engine::worker::CollectiveBootstrap::Nccl { serialized } = collective;
        let bootstrap = kvbm_engine::collectives::NcclBootstrap::deserialize(&serialized).unwrap();
        assert_eq!(bootstrap.world_size(), 2);
    }

    #[test]
    fn mha_layout_keeps_configured_parallelism() {
        assert_eq!(
            resolve_parallelism(ParallelismMode::TensorParallel, &layout(Some(4))),
            ParallelismMode::TensorParallel
        );
        assert_eq!(
            resolve_parallelism(ParallelismMode::ReplicatedData, &layout(Some(4))),
            ParallelismMode::ReplicatedData
        );
    }

    #[test]
    fn replicated_tiers_aggregate_worker_capacity() {
        assert_eq!(
            logical_tier_block_count(2_000, ParallelismMode::ReplicatedData, 2).unwrap(),
            4_000
        );
    }

    #[test]
    fn tensor_parallel_tiers_keep_per_worker_block_capacity() {
        assert_eq!(
            logical_tier_block_count(2_000, ParallelismMode::TensorParallel, 2).unwrap(),
            2_000
        );
    }

    #[test]
    fn tier_capacity_rejects_empty_groups_and_overflow() {
        assert!(logical_tier_block_count(1, ParallelismMode::ReplicatedData, 0).is_err());
        assert!(logical_tier_block_count(usize::MAX, ParallelismMode::ReplicatedData, 2).is_err());
    }
}
