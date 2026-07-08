// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end content integrity for host-tier peer pulls.

use anyhow::{Context, Result, ensure};
use blake3::Hasher;
use futures::future::join_all;
use kvbm_common::{BlockId, LogicalResourceId, SequenceHash, placement::StripedBlockPlacement};
use kvbm_config::ParallelismMode;
use kvbm_physical::transfer::PayloadDigest;
use serde::{Deserialize, Serialize};

use crate::leader::InstanceLeader;

/// A resource-, identity-, order-, topology-, and byte-bound payload checksum.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PayloadChecksum([u8; 32]);

/// One logical block whose physical G2 payload must be checksummed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PayloadBlock {
    pub(crate) hash: SequenceHash,
    pub(crate) block_id: BlockId,
    pub(crate) ordinal: u32,
}

impl PayloadChecksum {
    #[cfg(test)]
    pub(crate) const fn from_test_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    fn bind(
        resource: LogicalResourceId,
        block: PayloadBlock,
        placement: ChecksumPlacement,
        rank_digests: &[(usize, PayloadDigest)],
    ) -> Result<Self> {
        ensure!(
            !rank_digests.is_empty(),
            "payload checksum has no physical digests"
        );
        let mut hasher = Hasher::new();
        hasher.update(b"kvbm-transfer-payload-v1\0");
        hasher.update(&resource.0.to_le_bytes());
        hasher.update(&block.hash.as_u128().to_le_bytes());
        hasher.update(&block.ordinal.to_le_bytes());
        match placement {
            ChecksumPlacement::ReplicatedOwner => {
                ensure!(
                    rank_digests.len() == 1,
                    "replicated payload checksum requires exactly one owner digest"
                );
                hasher.update(b"replicated-owner\0");
                // The owner rank is intentionally omitted: striped leaders with
                // different world sizes can assign the same complete payload to
                // different ranks without changing its content identity.
                hasher.update(rank_digests[0].1.as_bytes());
            }
            ChecksumPlacement::TensorSharded { world_size } => {
                ensure!(
                    rank_digests.len() == world_size,
                    "tensor payload checksum expected {world_size} rank digests, got {}",
                    rank_digests.len()
                );
                hasher.update(b"tensor-sharded\0");
                hasher.update(&(world_size as u64).to_le_bytes());
                for (rank, digest) in rank_digests {
                    hasher.update(&(*rank as u64).to_le_bytes());
                    hasher.update(digest.as_bytes());
                }
            }
        }
        Ok(Self(*hasher.finalize().as_bytes()))
    }
}

#[derive(Clone, Copy)]
enum ChecksumPlacement {
    ReplicatedOwner,
    TensorSharded { world_size: usize },
}

impl InstanceLeader {
    /// Compute checksums from local G2 bytes in logical block order.
    pub(crate) async fn payload_checksums(
        &self,
        resource: LogicalResourceId,
        blocks: &[PayloadBlock],
    ) -> Result<Vec<PayloadChecksum>> {
        if blocks.is_empty() {
            return Ok(Vec::new());
        }
        let parallel = self
            .parallel_worker()
            .context("payload integrity requires a configured parallel worker")?;
        let workers = parallel.workers();
        ensure!(
            !workers.is_empty(),
            "payload integrity requires at least one worker"
        );
        let mode = self
            .parallelism_template_for_resource(resource)
            .map(|template| template.parallelism_mode)
            .unwrap_or(ParallelismMode::TensorParallel);

        let mut requests = vec![Vec::<(usize, BlockId)>::new(); workers.len()];
        match mode {
            ParallelismMode::ReplicatedData => {
                let placement = StripedBlockPlacement::new(workers.len())?;
                for (index, block) in blocks.iter().enumerate() {
                    let (rank, local_id) = placement.resolve(block.block_id);
                    requests[rank].push((index, local_id));
                }
            }
            ParallelismMode::TensorParallel => {
                for request in &mut requests {
                    request.extend(
                        blocks
                            .iter()
                            .enumerate()
                            .map(|(index, block)| (index, block.block_id)),
                    );
                }
            }
        }

        let results = join_all(
            workers
                .iter()
                .zip(requests.iter())
                .map(|(worker, request)| {
                    let block_ids = request.iter().map(|(_, id)| *id).collect::<Vec<_>>();
                    worker.compute_host_payload_digests(resource, block_ids)
                }),
        )
        .await;
        let mut per_block = vec![Vec::<(usize, PayloadDigest)>::new(); blocks.len()];
        for (rank, (request, result)) in requests.into_iter().zip(results).enumerate() {
            let digests = result.with_context(|| {
                format!("compute G2 payload digests for resource {resource:?} rank {rank}")
            })?;
            ensure!(
                digests.len() == request.len(),
                "payload digest worker {rank} returned {} digests for {} blocks",
                digests.len(),
                request.len()
            );
            for ((index, _), digest) in request.into_iter().zip(digests) {
                per_block[index].push((rank, digest));
            }
        }

        let placement = match mode {
            ParallelismMode::ReplicatedData => ChecksumPlacement::ReplicatedOwner,
            ParallelismMode::TensorParallel => ChecksumPlacement::TensorSharded {
                world_size: workers.len(),
            },
        };
        blocks
            .iter()
            .copied()
            .zip(per_block)
            .map(|(block, digests)| {
                Self::bind_payload_checksum(resource, block, placement, digests)
            })
            .collect()
    }

    fn bind_payload_checksum(
        resource: LogicalResourceId,
        block: PayloadBlock,
        placement: ChecksumPlacement,
        mut digests: Vec<(usize, PayloadDigest)>,
    ) -> Result<PayloadChecksum> {
        digests.sort_by_key(|(rank, _)| *rank);
        PayloadChecksum::bind(resource, block, placement, &digests)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(hash: u64, ordinal: u32) -> PayloadBlock {
        PayloadBlock {
            hash: SequenceHash::new(hash, None, u64::from(ordinal)),
            block_id: 9,
            ordinal,
        }
    }

    #[test]
    fn checksum_binds_resource_identity_order_and_rank_order() {
        let a = PayloadDigest::from_bytes([1; 32]);
        let b = PayloadDigest::from_bytes([2; 32]);
        let bind = |resource, block, digests: &[(usize, PayloadDigest)]| {
            PayloadChecksum::bind(
                resource,
                block,
                ChecksumPlacement::TensorSharded { world_size: 2 },
                digests,
            )
            .unwrap()
        };
        let baseline = bind(LogicalResourceId(3), block(7, 0), &[(0, a), (1, b)]);
        assert_ne!(
            baseline,
            bind(LogicalResourceId(4), block(7, 0), &[(0, a), (1, b)])
        );
        assert_ne!(
            baseline,
            bind(LogicalResourceId(3), block(8, 0), &[(0, a), (1, b)])
        );
        assert_ne!(
            baseline,
            bind(LogicalResourceId(3), block(7, 1), &[(0, a), (1, b)])
        );
        assert_ne!(
            baseline,
            bind(LogicalResourceId(3), block(7, 0), &[(0, b), (1, a)])
        );
    }

    #[test]
    fn replicated_checksum_is_independent_of_owner_rank() {
        let digest = PayloadDigest::from_bytes([5; 32]);
        let block = block(7, 2);
        let left = PayloadChecksum::bind(
            LogicalResourceId(3),
            block,
            ChecksumPlacement::ReplicatedOwner,
            &[(0, digest)],
        )
        .unwrap();
        let right = PayloadChecksum::bind(
            LogicalResourceId(3),
            block,
            ChecksumPlacement::ReplicatedOwner,
            &[(7, digest)],
        )
        .unwrap();
        assert_eq!(left, right);
    }
}
