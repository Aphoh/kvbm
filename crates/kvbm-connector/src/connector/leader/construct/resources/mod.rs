// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Validation and capacity planning for manifest-scoped worker resources.

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::{Result, anyhow, bail};
use kvbm_common::LogicalResourceId;
use kvbm_engine::worker::{ResourceTierConfig, WorkerCacheConfig};
use kvbm_physical::layout::LayoutConfig;
use kvbm_protocols::cache_manifest::CacheManifestId;

use crate::KvbmRuntime;

/// Validated rank-invariant resource topology and lower-tier capacities.
pub(super) struct ResourcePlan {
    pub(super) config: WorkerCacheConfig,
    pub(super) primary: LogicalResourceId,
    pub(super) primary_layout: LayoutConfig,
    pub(super) tiers: BTreeMap<LogicalResourceId, ResourceTierConfig>,
    pub(super) parallelism: BTreeMap<LogicalResourceId, kvbm_config::ParallelismMode>,
}

impl ResourcePlan {
    pub(super) fn build(
        runtime: &Arc<KvbmRuntime>,
        workers: &[WorkerCacheConfig],
        expected_manifest: Option<CacheManifestId>,
    ) -> Result<Self> {
        let config = workers
            .first()
            .ok_or_else(|| anyhow!("cannot plan resources without workers"))?
            .clone();
        config.validate()?;
        if config.manifest != expected_manifest {
            bail!(
                "worker cache manifest {:?} does not match leader manifest {:?}",
                config.manifest,
                expected_manifest
            );
        }
        for (rank, worker) in workers.iter().enumerate().skip(1) {
            worker.validate()?;
            if worker != &config {
                bail!("resource layout config mismatch between worker {rank} and worker 0");
            }
        }

        let primary = config.primary;
        let primary_layout = config
            .resources
            .get(&primary)
            .expect("validated primary resource")
            .clone();
        let tiers = resource_tiers(runtime, &config)?;
        let parallelism = config
            .resources
            .iter()
            .map(|(&resource, layout)| {
                (
                    resource,
                    resolve_parallelism(runtime.config().cache.parallelism, layout),
                )
            })
            .collect();
        Ok(Self {
            config,
            primary,
            primary_layout,
            tiers,
            parallelism,
        })
    }
}

/// Resolve physical cache distribution from the registered tensor schema.
pub(super) fn resolve_parallelism(
    configured: kvbm_config::ParallelismMode,
    layout: &LayoutConfig,
) -> kvbm_config::ParallelismMode {
    if layout.num_heads.is_none() {
        kvbm_config::ParallelismMode::ReplicatedData
    } else {
        configured
    }
}

fn resource_tiers(
    runtime: &Arc<KvbmRuntime>,
    resources: &WorkerCacheConfig,
) -> Result<BTreeMap<LogicalResourceId, ResourceTierConfig>> {
    let resource_count = resources.resources.len();
    resources
        .resources
        .iter()
        .map(|(&resource, layout)| {
            let bytes_per_block = layout
                .required_bytes()
                .checked_div(layout.num_blocks)
                .ok_or_else(|| anyhow!("resource {resource:?} has zero device blocks"))?;
            let host_block_count = runtime
                .config()
                .cache
                .host
                .compute_num_blocks(bytes_per_block)
                .unwrap_or(0)
                / resource_count;
            let disk_block_count = runtime
                .config()
                .cache
                .disk
                .as_ref()
                .and_then(|disk| disk.compute_num_blocks(bytes_per_block))
                .map(|count| count / resource_count);
            if host_block_count == 0 && disk_block_count.is_none_or(|count| count == 0) {
                bail!("no lower-tier capacity for logical resource {resource:?}");
            }
            Ok((
                resource,
                ResourceTierConfig {
                    host_block_count,
                    disk_block_count,
                },
            ))
        })
        .collect()
}
