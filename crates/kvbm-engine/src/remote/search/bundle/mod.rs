// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Complete-bundle remote directory contracts and pull orchestration.

use std::collections::BTreeSet;

use kvbm_common::LogicalResourceId;
use kvbm_protocols::cache_manifest::{BundleKey, CacheIdentity};

use crate::InstanceId;

mod pull;

pub(crate) use pull::{BundlePullTarget, pull_remote_bundle};

/// One owner's complete, manifest-scoped bundle advertisement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BundleAdvertisement {
    identity: CacheIdentity,
    key: BundleKey,
    generation: u64,
    owner: InstanceId,
    expires_at_unix_ms: u64,
    resources: BTreeSet<LogicalResourceId>,
}

impl BundleAdvertisement {
    pub fn new(
        identity: CacheIdentity,
        key: BundleKey,
        generation: u64,
        owner: InstanceId,
        expires_at_unix_ms: u64,
        resource_ids: impl IntoIterator<Item = LogicalResourceId>,
    ) -> Result<Self, BundleDirectoryError> {
        if !key.is_compatible_with(&identity) {
            return Err(BundleDirectoryError::ManifestMismatch);
        }
        let expected = identity
            .resources()
            .iter()
            .map(|requirement| requirement.resource())
            .collect::<BTreeSet<_>>();
        let mut resources = BTreeSet::new();
        for resource in resource_ids {
            if !resources.insert(resource) {
                return Err(BundleDirectoryError::DuplicateResource(resource));
            }
        }
        if resources != expected {
            return Err(BundleDirectoryError::IncompleteResources {
                expected: expected.into_iter().collect(),
                actual: resources.into_iter().collect(),
            });
        }
        Ok(Self {
            identity,
            key,
            generation,
            owner,
            expires_at_unix_ms,
            resources,
        })
    }

    pub const fn identity(&self) -> &CacheIdentity {
        &self.identity
    }

    pub const fn key(&self) -> BundleKey {
        self.key
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub const fn owner(&self) -> InstanceId {
        self.owner
    }

    pub const fn expires_at_unix_ms(&self) -> u64 {
        self.expires_at_unix_ms
    }

    pub fn resources(&self) -> impl Iterator<Item = LogicalResourceId> + '_ {
        self.resources.iter().copied()
    }

    pub fn matches(&self, query: &BundleDiscoveryQuery) -> bool {
        self.identity == query.identity
            && self.expires_at_unix_ms > query.now_unix_ms
            && query.candidates.contains(&self.key)
    }
}

/// Ordered bundle keys eligible for one remote lookup.
#[derive(Clone, Debug)]
pub struct BundleDiscoveryQuery {
    identity: CacheIdentity,
    candidates: Vec<BundleKey>,
    now_unix_ms: u64,
}

impl BundleDiscoveryQuery {
    pub fn new(identity: CacheIdentity, candidates: Vec<BundleKey>, now_unix_ms: u64) -> Self {
        Self {
            identity,
            candidates,
            now_unix_ms,
        }
    }

    pub const fn identity(&self) -> &CacheIdentity {
        &self.identity
    }

    pub fn candidates(&self) -> &[BundleKey] {
        &self.candidates
    }

    pub const fn now_unix_ms(&self) -> u64 {
        self.now_unix_ms
    }
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum BundleDirectoryError {
    #[error("bundle key does not match its cache identity")]
    ManifestMismatch,
    #[error("bundle resources are incomplete: expected {expected:?}, got {actual:?}")]
    IncompleteResources {
        expected: Vec<LogicalResourceId>,
        actual: Vec<LogicalResourceId>,
    },
    #[error("bundle resource {0:?} is duplicated")]
    DuplicateResource(LogicalResourceId),
    #[error("remote bundle lease outlives its advertisement")]
    LeaseOutlivesAdvertisement,
}

/// Stable classification for a complete-bundle directory miss.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BundleMissReason {
    NotFound,
    Incompatible,
    Incomplete,
    Expired,
}

impl BundleMissReason {
    pub const fn as_label(self) -> &'static str {
        match self {
            Self::NotFound => "not_found",
            Self::Incompatible => "incompatible",
            Self::Incomplete => "incomplete",
            Self::Expired => "expired",
        }
    }
}

/// Directory lookup outcome that keeps misses distinguishable for metrics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BundleDiscoveryOutcome {
    Hit(Box<RemoteBundleCandidate>),
    Miss(BundleMissReason),
}

/// Terminal result of attempting one directory-issued bundle lease.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BundlePullOutcome {
    Pulled(BundleKey),
    Miss(BundleMissReason),
}

/// Directory-issued lease for one remote complete-bundle owner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RemoteBundleCandidate {
    advertisement: BundleAdvertisement,
    lease_id: uuid::Uuid,
    lease_expires_at_unix_ms: u64,
}

impl RemoteBundleCandidate {
    pub fn new(
        advertisement: BundleAdvertisement,
        lease_id: uuid::Uuid,
        lease_expires_at_unix_ms: u64,
    ) -> Result<Self, BundleDirectoryError> {
        if lease_expires_at_unix_ms > advertisement.expires_at_unix_ms {
            return Err(BundleDirectoryError::LeaseOutlivesAdvertisement);
        }
        Ok(Self {
            advertisement,
            lease_id,
            lease_expires_at_unix_ms,
        })
    }

    pub const fn advertisement(&self) -> &BundleAdvertisement {
        &self.advertisement
    }

    pub const fn lease_id(&self) -> uuid::Uuid {
        self.lease_id
    }

    pub const fn lease_expires_at_unix_ms(&self) -> u64 {
        self.lease_expires_at_unix_ms
    }
}

/// Exact owner-generation invalidation emitted with local bundle eviction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BundleInvalidation {
    pub key: BundleKey,
    pub generation: u64,
    pub owner: InstanceId,
}

pub(crate) fn unix_time_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests;
