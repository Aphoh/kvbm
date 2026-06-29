// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Manifest-scoped complete-bundle directory.

use std::collections::{BTreeSet, HashSet};
use std::sync::RwLock;

use dashmap::DashMap;
use dashmap::mapref::entry::Entry;
use kvbm_protocols::cache_manifest::BundleKey;
use velo_ext::InstanceId;

use super::protocol::{
    BundleAdvertisementRecord, BundleInvalidateRequest, BundlePublishRequest, BundleQueryHit,
    BundleQueryMissReason, BundleQueryOutcome, BundleQueryRequest,
};

/// Live-owner directory for complete manifest-scoped bundles.
pub struct BundleDirectory {
    advertisements: DashMap<BundleKey, BundleAdvertisementRecord>,
    owners: RwLock<HashSet<InstanceId>>,
    lease_ttl_ms: u64,
}

impl BundleDirectory {
    pub fn new(lease_ttl_ms: u64) -> Self {
        Self {
            advertisements: DashMap::new(),
            owners: RwLock::new(HashSet::new()),
            lease_ttl_ms,
        }
    }

    pub fn register_owner(&self, owner: InstanceId) {
        if let Ok(mut owners) = self.owners.write() {
            owners.insert(owner);
        }
    }

    pub fn remove_owner(&self, owner: InstanceId) {
        if let Ok(mut owners) = self.owners.write() {
            owners.remove(&owner);
        }
        self.advertisements.retain(|_, entry| entry.owner != owner);
    }

    pub fn publish(&self, request: BundlePublishRequest) -> Result<(), BundleDirectoryError> {
        let advertisement = request.advertisement;
        if !self.owner_is_live(advertisement.owner) {
            return Err(BundleDirectoryError::UnknownOwner {
                owner: advertisement.owner,
            });
        }
        match self.advertisements.entry(advertisement.key) {
            Entry::Occupied(mut current) => {
                if current.get().generation > advertisement.generation {
                    return Err(BundleDirectoryError::StaleGeneration {
                        current: current.get().generation,
                        attempted: advertisement.generation,
                    });
                }
                current.insert(advertisement);
            }
            Entry::Vacant(entry) => {
                entry.insert(advertisement);
            }
        }
        Ok(())
    }

    pub fn query(&self, request: BundleQueryRequest) -> BundleQueryOutcome {
        let required = request
            .required_resources
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        if required.len() != request.required_resources.len() {
            return BundleQueryOutcome::Miss(BundleQueryMissReason::Incomplete);
        }
        let mut reason = BundleQueryMissReason::NotFound;
        for key in &request.candidates {
            if key.manifest() != request.manifest {
                if reason == BundleQueryMissReason::NotFound {
                    reason = BundleQueryMissReason::Incompatible;
                }
                continue;
            }
            let Some(entry) = self.advertisements.get(key) else {
                continue;
            };
            if !self.owner_is_live(entry.owner) {
                continue;
            }
            if entry.expires_at_unix_ms <= request.now_unix_ms {
                if reason == BundleQueryMissReason::NotFound {
                    reason = BundleQueryMissReason::Expired;
                }
                continue;
            }
            let resources = entry.resources.iter().copied().collect::<BTreeSet<_>>();
            if resources.len() != entry.resources.len() || resources != required {
                if reason == BundleQueryMissReason::NotFound {
                    reason = BundleQueryMissReason::Incomplete;
                }
                continue;
            }
            return BundleQueryOutcome::Hit(BundleQueryHit {
                advertisement: entry.clone(),
                lease_id: uuid::Uuid::new_v4(),
                lease_expires_at_unix_ms: entry
                    .expires_at_unix_ms
                    .min(request.now_unix_ms.saturating_add(self.lease_ttl_ms)),
            });
        }
        BundleQueryOutcome::Miss(reason)
    }

    pub fn invalidate(&self, request: BundleInvalidateRequest) -> bool {
        self.advertisements
            .remove_if(&request.key, |_, advertisement| {
                advertisement.owner == request.owner
                    && advertisement.generation == request.generation
            })
            .is_some()
    }

    fn owner_is_live(&self, owner: InstanceId) -> bool {
        self.owners
            .read()
            .map(|owners| owners.contains(&owner))
            .unwrap_or(false)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum BundleDirectoryError {
    #[error("bundle owner {owner} is not registered")]
    UnknownOwner { owner: InstanceId },
    #[error("bundle generation {attempted} is older than current generation {current}")]
    StaleGeneration { current: u64, attempted: u64 },
}

#[cfg(test)]
mod tests;
