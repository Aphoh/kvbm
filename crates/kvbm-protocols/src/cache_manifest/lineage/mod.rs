// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exact logical-resource lineage carried by complete-bundle advertisements.

use std::collections::HashSet;

use kvbm_common::{LogicalResourceId, SequenceHash};
use serde::{Deserialize, Deserializer, Serialize};

/// Exact ordered hashes owned for one resource in a complete bundle.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct BundleResourceLineage {
    resource: LogicalResourceId,
    hashes: Vec<SequenceHash>,
}

impl BundleResourceLineage {
    pub fn new(
        resource: LogicalResourceId,
        hashes: Vec<SequenceHash>,
    ) -> Result<Self, BundleResourceLineageError> {
        if hashes.is_empty() {
            return Err(BundleResourceLineageError::Empty { resource });
        }
        if let Some(mode) = hashes.iter().map(SequenceHash::mode).find(|mode| *mode > 2) {
            return Err(BundleResourceLineageError::InvalidHashEncoding { resource, mode });
        }
        let mut unique = HashSet::with_capacity(hashes.len());
        if let Some(hash) = hashes.iter().copied().find(|hash| !unique.insert(*hash)) {
            return Err(BundleResourceLineageError::DuplicateHash { resource, hash });
        }
        for edge in hashes.windows(2) {
            let parent = edge[0];
            let child = edge[1];
            if child.position() != parent.position() + 1 {
                return Err(BundleResourceLineageError::NonConsecutivePositions {
                    resource,
                    parent_position: parent.position(),
                    child_position: child.position(),
                });
            }
            if child.parent_hash_fragment()
                != parent.parent_fragment_for_child_position(child.position())
            {
                return Err(BundleResourceLineageError::ParentHashMismatch {
                    resource,
                    parent,
                    child,
                });
            }
        }
        Ok(Self { resource, hashes })
    }

    /// Project one canonical fine-grained PLH chain into a resource's coarser
    /// native-block coordinate.
    ///
    /// Every projected hash preserves the canonical endpoint's full sequence
    /// hash. Positions and parent fragments are rebased so the result remains a
    /// genuine, independently traversable PLH chain in resource-block order.
    pub fn project_from_canonical(
        resource: LogicalResourceId,
        canonical: &[SequenceHash],
        finer_blocks_per_resource_block: usize,
    ) -> Result<Self, BundleResourceLineageError> {
        let canonical = Self::new(resource, canonical.to_vec())?;
        if finer_blocks_per_resource_block == 0 {
            return Err(BundleResourceLineageError::ZeroProjectionFactor { resource });
        }
        if let Some((expected, hash)) = canonical
            .hashes
            .iter()
            .enumerate()
            .find(|(expected, hash)| hash.position() != *expected as u64)
        {
            return Err(BundleResourceLineageError::CanonicalPositionMismatch {
                resource,
                expected: expected as u64,
                actual: hash.position(),
            });
        }
        if !canonical
            .hashes
            .len()
            .is_multiple_of(finer_blocks_per_resource_block)
        {
            return Err(BundleResourceLineageError::UnalignedProjection {
                resource,
                canonical_blocks: canonical.hashes.len(),
                finer_blocks_per_resource_block,
            });
        }

        let mut parent = None;
        let projected = canonical
            .hashes
            .iter()
            .copied()
            .skip(finer_blocks_per_resource_block - 1)
            .step_by(finer_blocks_per_resource_block)
            .enumerate()
            .map(|(position, endpoint)| {
                let current = endpoint.current_sequence_hash();
                let hash = SequenceHash::new(current, parent, position as u64);
                parent = Some(current);
                hash
            })
            .collect();
        Self::new(resource, projected)
    }

    pub const fn resource(&self) -> LogicalResourceId {
        self.resource
    }

    pub fn hashes(&self) -> &[SequenceHash] {
        &self.hashes
    }
}

impl<'de> Deserialize<'de> for BundleResourceLineage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = BundleResourceLineageWire::deserialize(deserializer)?;
        Self::new(wire.resource, wire.hashes).map_err(serde::de::Error::custom)
    }
}

#[derive(Deserialize)]
struct BundleResourceLineageWire {
    resource: LogicalResourceId,
    hashes: Vec<SequenceHash>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum BundleResourceLineageError {
    #[error("bundle resource {resource:?} has an empty lineage")]
    Empty { resource: LogicalResourceId },
    #[error("bundle resource {resource:?} has invalid lineage-hash encoding mode {mode}")]
    InvalidHashEncoding {
        resource: LogicalResourceId,
        mode: u8,
    },
    #[error("bundle resource {resource:?} repeats lineage hash {hash}")]
    DuplicateHash {
        resource: LogicalResourceId,
        hash: SequenceHash,
    },
    #[error(
        "bundle resource {resource:?} lineage jumps from position {parent_position} to {child_position}"
    )]
    NonConsecutivePositions {
        resource: LogicalResourceId,
        parent_position: u64,
        child_position: u64,
    },
    #[error("bundle resource {resource:?} lineage child {child} does not descend from {parent}")]
    ParentHashMismatch {
        resource: LogicalResourceId,
        parent: SequenceHash,
        child: SequenceHash,
    },
    #[error("bundle resource {resource:?} lineage projection factor must be nonzero")]
    ZeroProjectionFactor { resource: LogicalResourceId },
    #[error(
        "bundle resource {resource:?} canonical lineage expected position {expected}, got {actual}"
    )]
    CanonicalPositionMismatch {
        resource: LogicalResourceId,
        expected: u64,
        actual: u64,
    },
    #[error(
        "bundle resource {resource:?} cannot project {canonical_blocks} canonical blocks in groups of {finer_blocks_per_resource_block}"
    )]
    UnalignedProjection {
        resource: LogicalResourceId,
        canonical_blocks: usize,
        finer_blocks_per_resource_block: usize,
    },
}

mod geometry;

pub use geometry::{BundleLineageValidationError, validate_bundle_lineages};

#[cfg(test)]
mod tests;
