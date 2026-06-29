// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Atomic multi-resource cache-bundle ownership.

mod barrier;
#[allow(
    dead_code,
    reason = "phase K1 defines capsule copy descriptors before the vLLM integration wires them in K4"
)]
mod capsule;
mod find;
mod offload;
mod onboard;
mod remote;

#[cfg(test)]
mod integration_tests;
#[cfg(test)]
mod test_support;

pub(super) use offload::{BundleCommitMetadata, BundleOffload, OffloadTransition};

use std::borrow::Borrow;
use std::collections::{BTreeMap, BTreeSet, HashMap};

use kvbm_common::{LogicalResourceId, SequenceHash};
use kvbm_protocols::cache_manifest::{BundleKey, CacheIdentity, CacheManifestId};

/// Complete committed bundle registry. P is the RAII pin type owned by one
/// logical resource; cloning it creates a lease pin.
pub(super) struct BundleIndex<P> {
    committed: HashMap<BundleKey, CommittedBundle<P>>,
}

struct CommittedBundle<P> {
    generation: u64,
    resources: BTreeMap<LogicalResourceId, P>,
}

/// All-resource lease returned by one bundle lookup.
pub(super) struct BundleLease<P> {
    key: BundleKey,
    #[allow(dead_code, reason = "phase K4 consumes the matched bundle generation")]
    generation: u64,
    resources: BTreeMap<LogicalResourceId, P>,
}

impl<P> BundleIndex<P> {
    pub(super) fn new() -> Self {
        Self {
            committed: HashMap::new(),
        }
    }

    pub(super) fn commit(
        &mut self,
        identity: &CacheIdentity,
        key: BundleKey,
        generation: u64,
        resources: impl IntoIterator<Item = (LogicalResourceId, P)>,
    ) -> Result<(), BundleIndexError> {
        if !key.is_compatible_with(identity) {
            return Err(BundleIndexError::ManifestMismatch {
                expected: identity.manifest(),
                actual: key.manifest(),
            });
        }

        let expected = identity
            .resources()
            .iter()
            .map(|requirement| requirement.resource())
            .collect::<BTreeSet<_>>();
        let mut owned = BTreeMap::new();
        for (resource, pin) in resources {
            if !expected.contains(&resource) {
                return Err(BundleIndexError::UnexpectedResource { resource });
            }
            if owned.insert(resource, pin).is_some() {
                return Err(BundleIndexError::DuplicateResource { resource });
            }
        }
        let missing = expected
            .into_iter()
            .filter(|resource| !owned.contains_key(resource))
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(BundleIndexError::MissingResources { resources: missing });
        }

        if let Some(current) = self.committed.get(&key) {
            if current.generation > generation {
                return Err(BundleIndexError::StaleGeneration {
                    current: current.generation,
                    attempted: generation,
                });
            }
            if current.generation == generation {
                return Ok(());
            }
        }

        self.committed.insert(
            key,
            CommittedBundle {
                generation,
                resources: owned,
            },
        );
        Ok(())
    }

    /// Remove one committed bundle and release all resource pins atomically.
    pub(super) fn invalidate(&mut self, key: BundleKey) -> bool {
        self.remove(key).is_some()
    }

    /// Remove a committed bundle and return the exact generation advertised
    /// for owner-scoped remote invalidation.
    pub(super) fn remove(&mut self, key: BundleKey) -> Option<u64> {
        self.committed.remove(&key).map(|bundle| bundle.generation)
    }
}

impl<P: Clone> BundleIndex<P> {
    pub(super) fn lease_exact(
        &self,
        identity: &CacheIdentity,
        key: &BundleKey,
    ) -> Option<BundleLease<P>> {
        if !key.is_compatible_with(identity) {
            return None;
        }
        self.committed.get(key).map(|bundle| BundleLease {
            key: *key,
            generation: bundle.generation,
            resources: bundle.resources.clone(),
        })
    }

    /// Return the greatest complete candidate boundary and clone every
    /// resource pin atomically into the resulting lease.
    pub(super) fn find_longest<I>(
        &self,
        identity: &CacheIdentity,
        candidates: I,
    ) -> Option<BundleLease<P>>
    where
        I: IntoIterator,
        I::Item: Borrow<(SequenceHash, u64)>,
    {
        candidates
            .into_iter()
            .filter_map(|candidate| {
                let &(hash, tokens) = candidate.borrow();
                BundleKey::new(identity, hash, tokens).ok()
            })
            .filter_map(|key| self.committed.get(&key).map(|bundle| (key, bundle)))
            .max_by_key(|(key, _)| key.boundary_tokens())
            .map(|(key, bundle)| BundleLease {
                key,
                generation: bundle.generation,
                resources: bundle.resources.clone(),
            })
    }
}

impl<P> BundleLease<P> {
    pub(super) const fn key(&self) -> &BundleKey {
        &self.key
    }

    #[cfg(test)]
    pub(super) const fn generation(&self) -> u64 {
        self.generation
    }

    pub(super) const fn resources(&self) -> &BTreeMap<LogicalResourceId, P> {
        &self.resources
    }

    pub(super) fn into_resources(self) -> BTreeMap<LogicalResourceId, P> {
        self.resources
    }
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub(super) enum BundleIndexError {
    #[error("bundle manifest mismatch: expected {expected}, got {actual}")]
    ManifestMismatch {
        expected: CacheManifestId,
        actual: CacheManifestId,
    },
    #[error("bundle contains unexpected resource {resource:?}")]
    UnexpectedResource { resource: LogicalResourceId },
    #[error("bundle contains duplicate resource {resource:?}")]
    DuplicateResource { resource: LogicalResourceId },
    #[error("bundle is missing required resources {resources:?}")]
    MissingResources { resources: Vec<LogicalResourceId> },
    #[error("bundle generation {attempted} is older than committed generation {current}")]
    StaleGeneration { current: u64, attempted: u64 },
}

#[cfg(test)]
mod tests;
