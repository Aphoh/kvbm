// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Atomic multi-resource cache-bundle ownership.

mod admission;
mod barrier;
#[allow(
    dead_code,
    reason = "phase K1 defines capsule copy descriptors before the vLLM integration wires them in K4"
)]
mod capsule;
mod catalog;
mod disagg;
mod find;
mod offload;
mod onboard;
mod remote;

#[cfg(test)]
mod integration_tests;
#[cfg(test)]
mod test_support;

pub(super) use admission::BundleAdmissionConfig;
pub(super) use catalog::{BUNDLE_DIRECTORY_TTL_MS, BundleCatalog, BundleCatalogError};
pub(super) use disagg::BundlePrefillRequest;
pub(super) use offload::{BundleOffload, OffloadTransition};
pub(super) use remote::{BundleDirectoryOrder, BundlePublicationRuntime};

use std::borrow::Borrow;
use std::collections::{BTreeMap, BTreeSet, HashMap};

use kvbm_common::{LogicalResourceId, SequenceHash};
use kvbm_logical::{BlockMetadata, ImmutableBlock, WeakBlock};
use kvbm_protocols::cache_manifest::{BundleKey, CacheIdentity, CacheManifestId};

/// Complete committed bundle registry. The index stores non-owning resource
/// references; a lookup reacquires every child before returning a strong lease.
pub(super) struct BundleIndex<P: BundleResourcePin> {
    committed: HashMap<BundleKey, CommittedBundle<P::Reference>>,
}

struct CommittedBundle<R> {
    generation: u64,
    resources: BTreeMap<LogicalResourceId, R>,
}

/// All-resource lease returned by one bundle lookup.
pub(super) struct BundleLease<P> {
    key: BundleKey,
    #[allow(dead_code, reason = "phase K4 consumes the matched bundle generation")]
    generation: u64,
    resources: BTreeMap<LogicalResourceId, P>,
}

impl<P: BundleResourcePin> BundleIndex<P> {
    pub(super) fn new() -> Self {
        Self {
            committed: HashMap::new(),
        }
    }

    #[cfg(test)]
    pub(super) fn commit(
        &mut self,
        identity: &CacheIdentity,
        key: BundleKey,
        generation: u64,
        resources: impl IntoIterator<Item = (LogicalResourceId, P)>,
    ) -> Result<(), BundleIndexError> {
        let mut owned = BTreeMap::new();
        for (resource, pin) in resources {
            if owned.insert(resource, pin).is_some() {
                return Err(BundleIndexError::DuplicateResource { resource });
            }
        }
        if let Some(prepared) = self.prepare_commit(identity, key, generation, &owned)? {
            self.install(key, prepared);
        }
        Ok(())
    }

    fn prepare_commit(
        &self,
        identity: &CacheIdentity,
        key: BundleKey,
        generation: u64,
        resources: &BTreeMap<LogicalResourceId, P>,
    ) -> Result<Option<CommittedBundle<P::Reference>>, BundleIndexError> {
        let resource_ids = resources.keys().copied().collect::<BTreeSet<_>>();
        if self.validate_commit(identity, key, generation, &resource_ids)?
            == CommitDisposition::Idempotent
        {
            return Ok(None);
        }
        Ok(Some(Self::prepare_validated(generation, resources)))
    }

    fn validate_commit(
        &self,
        identity: &CacheIdentity,
        key: BundleKey,
        generation: u64,
        resources: &BTreeSet<LogicalResourceId>,
    ) -> Result<CommitDisposition, BundleIndexError> {
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
        if let Some(&resource) = resources
            .iter()
            .find(|resource| !expected.contains(resource))
        {
            return Err(BundleIndexError::UnexpectedResource { resource });
        }
        let missing = expected
            .into_iter()
            .filter(|resource| !resources.contains(resource))
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
            // Same-generation publication is idempotent. Do not probe the weak
            // entries here: a probe may resurrect inactive blocks and perturb
            // their eviction policy. Manager eviction observers synchronously
            // remove reclaimed bundles; lookups fail closed during that race.
            if current.generation == generation {
                return Ok(CommitDisposition::Idempotent);
            }
        }

        Ok(CommitDisposition::Install)
    }

    fn prepare_validated(
        generation: u64,
        resources: &BTreeMap<LogicalResourceId, P>,
    ) -> CommittedBundle<P::Reference> {
        CommittedBundle {
            generation,
            resources: resources
                .iter()
                .map(|(&resource, pin)| (resource, pin.make_reference()))
                .collect(),
        }
    }

    fn install(&mut self, key: BundleKey, bundle: CommittedBundle<P::Reference>) {
        self.committed.insert(key, bundle);
    }

    fn generation(&self, key: &BundleKey) -> Option<u64> {
        self.committed.get(key).map(|bundle| bundle.generation)
    }

    /// Remove one committed bundle's weak ownership metadata atomically.
    #[cfg(test)]
    pub(super) fn invalidate(&mut self, key: BundleKey) -> bool {
        self.remove(key).is_some()
    }

    /// Remove a committed bundle and return the exact generation advertised
    /// for owner-scoped remote invalidation.
    pub(super) fn remove(&mut self, key: BundleKey) -> Option<u64> {
        self.committed.remove(&key).map(|bundle| bundle.generation)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CommitDisposition {
    Install,
    Idempotent,
}

impl<P: BundleResourcePin> BundleIndex<P> {
    pub(super) fn lease_exact(
        &self,
        identity: &CacheIdentity,
        key: &BundleKey,
    ) -> Option<BundleLease<P>> {
        if !key.is_compatible_with(identity) {
            return None;
        }
        let bundle = self.committed.get(key)?;
        Some(BundleLease {
            key: *key,
            generation: bundle.generation,
            resources: reacquire_resources(&bundle.resources)?,
        })
    }

    /// Return the greatest complete candidate boundary and reacquire every
    /// resource pin as one all-or-none lease.
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
            .filter_map(|key| {
                let bundle = self.committed.get(&key)?;
                Some(BundleLease {
                    key,
                    generation: bundle.generation,
                    resources: reacquire_resources(&bundle.resources)?,
                })
            })
            .max_by_key(|lease| lease.key.boundary_tokens())
    }
}

fn reacquire_resources<R: BundleResourceReference>(
    resources: &BTreeMap<LogicalResourceId, R>,
) -> Option<BTreeMap<LogicalResourceId, R::Pin>> {
    resources
        .iter()
        .map(|(&resource, reference)| Some((resource, reference.reacquire()?)))
        .collect()
}

/// Converts one strong logical-resource pin into the non-owning reference kept
/// by the index. This remains private so storage policy cannot leak into the
/// connector API.
pub(super) trait BundleResourcePin: Sized {
    type Reference: BundleResourceReference<Pin = Self>;

    fn make_reference(&self) -> Self::Reference;
}

/// Reacquires one complete logical-resource child for an all-resource lease.
pub(super) trait BundleResourceReference {
    type Pin;

    fn reacquire(&self) -> Option<Self::Pin>;
}

impl<T: BlockMetadata + Sync> BundleResourcePin for Vec<ImmutableBlock<T>> {
    type Reference = Vec<WeakBlock<T>>;

    fn make_reference(&self) -> Self::Reference {
        self.iter().map(ImmutableBlock::downgrade).collect()
    }
}

impl<T: BlockMetadata + Sync> BundleResourceReference for Vec<WeakBlock<T>> {
    type Pin = Vec<ImmutableBlock<T>>;

    fn reacquire(&self) -> Option<Self::Pin> {
        self.iter().map(WeakBlock::upgrade).collect()
    }
}

#[cfg(test)]
impl BundleResourcePin for u8 {
    type Reference = u8;

    fn make_reference(&self) -> Self::Reference {
        *self
    }
}

#[cfg(test)]
impl BundleResourceReference for u8 {
    type Pin = u8;

    fn reacquire(&self) -> Option<Self::Pin> {
        Some(*self)
    }
}

#[cfg(test)]
impl<T: ?Sized> BundleResourcePin for std::sync::Arc<T> {
    type Reference = std::sync::Arc<T>;

    fn make_reference(&self) -> Self::Reference {
        std::sync::Arc::clone(self)
    }
}

#[cfg(test)]
impl<T: ?Sized> BundleResourceReference for std::sync::Arc<T> {
    type Pin = std::sync::Arc<T>;

    fn reacquire(&self) -> Option<Self::Pin> {
        Some(std::sync::Arc::clone(self))
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
    #[cfg(test)]
    DuplicateResource { resource: LogicalResourceId },
    #[error("bundle is missing required resources {resources:?}")]
    MissingResources { resources: Vec<LogicalResourceId> },
    #[error("bundle generation {attempted} is older than committed generation {current}")]
    StaleGeneration { current: u64, attempted: u64 },
}

#[cfg(test)]
mod tests;
