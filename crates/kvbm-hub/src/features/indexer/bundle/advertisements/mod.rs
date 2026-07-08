// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Keyed storage and deterministic expiration for bundle advertisements.

use std::collections::{BTreeMap, HashMap, HashSet};

use kvbm_protocols::cache_manifest::BundleKey;
use velo_ext::InstanceId;

use super::super::protocol::BundleAdvertisementRecord;

/// Advertisement storage indexed for exact bundle-key lookup and expiration.
pub(super) struct AdvertisementIndex {
    by_key: HashMap<BundleKey, HashMap<InstanceId, BundleAdvertisementRecord>>,
    expirations: BTreeMap<u64, HashSet<(BundleKey, InstanceId)>>,
    owner_counts: HashMap<InstanceId, usize>,
    capacity_per_owner: usize,
    global_capacity: usize,
    len: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AdvertisementCapacity {
    Owner { owner: InstanceId, capacity: usize },
    Global { capacity: usize },
}

impl AdvertisementIndex {
    pub(super) fn new(capacity_per_owner: usize, global_capacity: usize) -> Self {
        Self {
            by_key: HashMap::new(),
            expirations: BTreeMap::new(),
            owner_counts: HashMap::new(),
            capacity_per_owner,
            global_capacity,
            len: 0,
        }
    }

    pub(super) fn get(
        &self,
        key: BundleKey,
        owner: InstanceId,
    ) -> Option<&BundleAdvertisementRecord> {
        self.by_key.get(&key)?.get(&owner)
    }

    pub(super) fn for_key(
        &self,
        key: &BundleKey,
    ) -> impl Iterator<Item = &BundleAdvertisementRecord> {
        self.by_key
            .get(key)
            .into_iter()
            .flat_map(|owners| owners.values())
    }

    pub(super) fn insert(
        &mut self,
        advertisement: BundleAdvertisementRecord,
    ) -> Result<(), AdvertisementCapacity> {
        let identity = (advertisement.key, advertisement.owner);
        let is_new = self.get(identity.0, identity.1).is_none();
        if is_new {
            if self.len >= self.global_capacity {
                return Err(AdvertisementCapacity::Global {
                    capacity: self.global_capacity,
                });
            }
            if self.owner_counts.get(&identity.1).copied().unwrap_or(0) >= self.capacity_per_owner {
                return Err(AdvertisementCapacity::Owner {
                    owner: identity.1,
                    capacity: self.capacity_per_owner,
                });
            }
        }
        let expiration = advertisement.expires_at_unix_ms;
        let previous = self
            .by_key
            .entry(identity.0)
            .or_default()
            .insert(identity.1, advertisement);
        if let Some(previous) = &previous {
            self.unschedule(identity, previous.expires_at_unix_ms);
        } else {
            self.len += 1;
            *self.owner_counts.entry(identity.1).or_default() += 1;
        }
        self.expirations
            .entry(expiration)
            .or_default()
            .insert(identity);
        Ok(())
    }

    pub(super) fn remove(
        &mut self,
        key: BundleKey,
        owner: InstanceId,
    ) -> Option<BundleAdvertisementRecord> {
        let (removed, key_is_empty) = {
            let owners = self.by_key.get_mut(&key)?;
            let removed = owners.remove(&owner)?;
            (removed, owners.is_empty())
        };
        if key_is_empty {
            self.by_key.remove(&key);
        }
        self.unschedule((key, owner), removed.expires_at_unix_ms);
        self.len -= 1;
        let remove_owner_count = self.owner_counts.get_mut(&owner).is_some_and(|count| {
            *count -= 1;
            *count == 0
        });
        if remove_owner_count {
            self.owner_counts.remove(&owner);
        }
        Some(removed)
    }

    pub(super) fn records_for_owner(&self, owner: InstanceId) -> Vec<BundleAdvertisementRecord> {
        self.by_key
            .values()
            .filter_map(|owners| owners.get(&owner).cloned())
            .collect()
    }

    pub(super) fn remove_owner(&mut self, owner: InstanceId) -> Vec<BundleAdvertisementRecord> {
        self.records_for_owner(owner)
            .into_iter()
            .filter_map(|record| self.remove(record.key, owner))
            .collect()
    }

    pub(super) fn prune_expired(&mut self, observed_unix_ms: u64) -> HashSet<BundleKey> {
        let mut expired = HashSet::new();
        loop {
            let ready = self
                .expirations
                .first_key_value()
                .is_some_and(|(expiration, _)| *expiration <= observed_unix_ms);
            if !ready {
                break;
            }
            let Some((_, identities)) = self.expirations.pop_first() else {
                break;
            };
            for (key, owner) in identities {
                let is_expired = self
                    .get(key, owner)
                    .is_some_and(|record| record.expires_at_unix_ms <= observed_unix_ms);
                if is_expired && self.remove(key, owner).is_some() {
                    expired.insert(key);
                }
            }
        }
        expired
    }

    pub(super) const fn len(&self) -> usize {
        self.len
    }

    pub(super) fn owner_len(&self, owner: InstanceId) -> usize {
        self.owner_counts.get(&owner).copied().unwrap_or(0)
    }

    #[cfg(test)]
    pub(super) fn scheduled_expiration_count(&self) -> usize {
        self.expirations.values().map(HashSet::len).sum()
    }

    fn unschedule(&mut self, identity: (BundleKey, InstanceId), expiration: u64) {
        let remove_bucket = self
            .expirations
            .get_mut(&expiration)
            .is_some_and(|identities| {
                identities.remove(&identity);
                identities.is_empty()
            });
        if remove_bucket {
            self.expirations.remove(&expiration);
        }
    }
}
