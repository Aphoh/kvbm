// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reverse resource-lineage dependencies for committed bundle keys.

use std::collections::{HashMap, HashSet};

use kvbm_common::{LogicalResourceId, SequenceHash};
use kvbm_protocols::cache_manifest::{BundleKey, ResourceRole};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ResourceBlock {
    resource: LogicalResourceId,
    hash: SequenceHash,
}

#[derive(Debug, Default)]
pub(in crate::tiering) struct BundleDependencyIndex {
    by_block: HashMap<ResourceBlock, HashSet<BundleKey>>,
    by_bundle: HashMap<BundleKey, Vec<ResourceBlock>>,
}

impl BundleDependencyIndex {
    pub(in crate::tiering) fn new() -> Self {
        Self::default()
    }

    pub(in crate::tiering) fn track(
        &mut self,
        key: BundleKey,
        resources: impl IntoIterator<Item = ResourceLineage>,
    ) -> Result<(), DependencyError> {
        let resources = resources.into_iter().collect::<Vec<_>>();
        if resources.is_empty() {
            return Err(DependencyError::NoResources);
        }
        let mut blocks = Vec::new();
        let mut seen_resources = HashSet::new();
        for lineage in resources {
            if !seen_resources.insert(lineage.resource) {
                return Err(DependencyError::DuplicateResource {
                    resource: lineage.resource,
                });
            }
            lineage.validate(key)?;
            blocks.extend(lineage.hashes.into_iter().map(|hash| ResourceBlock {
                resource: lineage.resource,
                hash,
            }));
        }
        let mut unique_blocks = HashSet::new();
        blocks.retain(|block| unique_blocks.insert(*block));

        self.untrack(key);
        for block in &blocks {
            self.by_block.entry(*block).or_default().insert(key);
        }
        self.by_bundle.insert(key, blocks);
        Ok(())
    }

    pub(in crate::tiering) fn invalidate(
        &mut self,
        resource: LogicalResourceId,
        hash: SequenceHash,
        mut callback: impl FnMut(InvalidationEvent),
    ) {
        let block = ResourceBlock { resource, hash };
        let mut keys = self
            .by_block
            .get(&block)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect::<Vec<_>>();
        keys.sort_by_key(|key| {
            (
                *key.manifest().as_bytes(),
                key.boundary_tokens(),
                key.boundary_hash().as_u128(),
            )
        });
        for key in keys {
            self.untrack(key);
            callback(InvalidationEvent {
                key,
                resource,
                hash,
            });
        }
    }

    #[cfg(test)]
    pub(in crate::tiering) fn dependents(
        &self,
        resource: LogicalResourceId,
        hash: SequenceHash,
    ) -> Vec<BundleKey> {
        let mut keys = self
            .by_block
            .get(&ResourceBlock { resource, hash })
            .into_iter()
            .flat_map(|keys| keys.iter().copied())
            .collect::<Vec<_>>();
        keys.sort_by_key(|key| {
            (
                *key.manifest().as_bytes(),
                key.boundary_tokens(),
                key.boundary_hash().as_u128(),
            )
        });
        keys
    }

    fn untrack(&mut self, key: BundleKey) {
        let Some(blocks) = self.by_bundle.remove(&key) else {
            return;
        };
        for block in blocks {
            if let Some(keys) = self.by_block.get_mut(&block) {
                keys.remove(&key);
                if keys.is_empty() {
                    self.by_block.remove(&block);
                }
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::tiering) struct ResourceLineage {
    resource: LogicalResourceId,
    role: ResourceRole,
    hashes: Vec<SequenceHash>,
}

impl ResourceLineage {
    pub(in crate::tiering) fn new(
        resource: LogicalResourceId,
        role: ResourceRole,
        hashes: Vec<SequenceHash>,
    ) -> Self {
        Self {
            resource,
            role,
            hashes,
        }
    }

    fn validate(&self, key: BundleKey) -> Result<(), DependencyError> {
        if self.hashes.is_empty() {
            return Err(DependencyError::EmptyLineage {
                resource: self.resource,
            });
        }
        if self.hashes.last().copied() != Some(key.boundary_hash()) {
            return Err(DependencyError::BoundaryMismatch {
                resource: self.resource,
            });
        }
        if self.role == ResourceRole::BoundaryCapsule && self.hashes.len() != 1 {
            return Err(DependencyError::CapsuleNotAtomic {
                resource: self.resource,
            });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::tiering) struct InvalidationEvent {
    key: BundleKey,
    resource: LogicalResourceId,
    hash: SequenceHash,
}

impl InvalidationEvent {
    pub(in crate::tiering) const fn key(self) -> BundleKey {
        self.key
    }

    pub(in crate::tiering) const fn resource(self) -> LogicalResourceId {
        self.resource
    }

    pub(in crate::tiering) const fn hash(self) -> SequenceHash {
        self.hash
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub(in crate::tiering) enum DependencyError {
    #[error("a bundle dependency set requires at least one resource")]
    NoResources,
    #[error("resource {resource:?} occurs more than once in a dependency set")]
    DuplicateResource { resource: LogicalResourceId },
    #[error("resource {resource:?} has an empty block lineage")]
    EmptyLineage { resource: LogicalResourceId },
    #[error("resource {resource:?} does not end at the bundle boundary")]
    BoundaryMismatch { resource: LogicalResourceId },
    #[error("capsule resource {resource:?} must be one atomic logical object")]
    CapsuleNotAtomic { resource: LogicalResourceId },
}

#[cfg(test)]
mod tests {
    use kvbm_common::{LogicalResourceId, SequenceHash};
    use kvbm_protocols::cache_manifest::{
        BundleKey, CacheManifest, ModelIdentity, ResourceRequirement, ResourceRole,
    };

    use super::{BundleDependencyIndex, ResourceLineage};

    const HISTORY: LogicalResourceId = LogicalResourceId(70);
    const CAPSULE: LogicalResourceId = LogicalResourceId(71);

    fn hash(value: u8) -> SequenceHash {
        SequenceHash::new(u64::from(value), None, u64::from(value))
    }

    fn keys() -> (BundleKey, BundleKey) {
        let manifest = CacheManifest::new(
            ModelIdentity::new("policy-test", "v1", [9; 32]).unwrap(),
            "policy-test-v1",
            vec![
                ResourceRequirement::new(HISTORY, ResourceRole::PrefixHistory, 16).unwrap(),
                ResourceRequirement::new(CAPSULE, ResourceRole::BoundaryCapsule, 16).unwrap(),
            ],
            Default::default(),
        )
        .unwrap();
        let identity = manifest.identity();
        (
            BundleKey::new(&identity, hash(1), 16).unwrap(),
            BundleKey::new(&identity, hash(2), 32).unwrap(),
        )
    }

    #[test]
    fn history_eviction_invalidates_every_deeper_dependent_bundle() {
        let (shallow, deep) = keys();
        let mut index = BundleDependencyIndex::new();
        index
            .track(
                shallow,
                [ResourceLineage::new(
                    HISTORY,
                    ResourceRole::PrefixHistory,
                    vec![hash(1)],
                )],
            )
            .unwrap();
        index
            .track(
                deep,
                [ResourceLineage::new(
                    HISTORY,
                    ResourceRole::PrefixHistory,
                    vec![hash(1), hash(2)],
                )],
            )
            .unwrap();

        let mut invalidated = Vec::new();
        index.invalidate(HISTORY, hash(1), |event| invalidated.push(event.key()));
        invalidated.sort_by_key(BundleKey::boundary_tokens);

        assert_eq!(invalidated, vec![shallow, deep]);
        assert!(index.dependents(HISTORY, hash(1)).is_empty());
    }

    #[test]
    fn capsule_eviction_invalidates_only_its_exact_boundary() {
        let (shallow, deep) = keys();
        let mut index = BundleDependencyIndex::new();
        index
            .track(
                shallow,
                [ResourceLineage::new(
                    CAPSULE,
                    ResourceRole::BoundaryCapsule,
                    vec![hash(1)],
                )],
            )
            .unwrap();
        index
            .track(
                deep,
                [ResourceLineage::new(
                    CAPSULE,
                    ResourceRole::BoundaryCapsule,
                    vec![hash(2)],
                )],
            )
            .unwrap();

        let mut invalidated = Vec::new();
        index.invalidate(CAPSULE, hash(1), |event| invalidated.push(event.key()));

        assert_eq!(invalidated, vec![shallow]);
        assert_eq!(index.dependents(CAPSULE, hash(2)), vec![deep]);
    }
}
