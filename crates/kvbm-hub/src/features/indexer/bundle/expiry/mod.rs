// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded recent-expiry classification for exact bundle keys.

use std::collections::{BTreeMap, HashMap};

use kvbm_protocols::cache_manifest::BundleKey;

/// Maximum number of recently expired exact keys retained for miss classification.
const MAX_RETAINED_KEYS: usize = 4_096;

/// A bounded, time-limited history used to distinguish a recent expiry from a
/// key that was never advertised.
pub(super) struct ExpiredAdvertisementHistory {
    entries: HashMap<BundleKey, u64>,
    eviction_order: BTreeMap<ExpiryOrder, BundleKey>,
    retention_ms: u64,
    capacity: usize,
}

type ExpiryOrder = (u64, [u8; 32], u64, u128);

impl ExpiredAdvertisementHistory {
    pub(super) fn new(retention_ms: u64) -> Self {
        Self::with_capacity(retention_ms, MAX_RETAINED_KEYS)
    }

    pub(super) fn record(
        &mut self,
        keys: impl IntoIterator<Item = BundleKey>,
        observed_unix_ms: u64,
    ) {
        self.prune(observed_unix_ms);
        let retain_until_unix_ms = observed_unix_ms.saturating_add(self.retention_ms);
        for key in keys {
            self.forget(key);
            if self.capacity == 0 || retain_until_unix_ms <= observed_unix_ms {
                continue;
            }
            self.entries.insert(key, retain_until_unix_ms);
            self.eviction_order
                .insert(expiry_order(key, retain_until_unix_ms), key);
        }
        self.enforce_capacity();
    }

    pub(super) fn contains(&self, key: BundleKey) -> bool {
        self.entries.contains_key(&key)
    }

    pub(super) fn forget(&mut self, key: BundleKey) {
        let Some(retain_until_unix_ms) = self.entries.remove(&key) else {
            return;
        };
        self.eviction_order
            .remove(&expiry_order(key, retain_until_unix_ms));
    }

    pub(super) fn prune(&mut self, observed_unix_ms: u64) {
        loop {
            let ready = self
                .eviction_order
                .first_key_value()
                .is_some_and(|(order, _)| order.0 <= observed_unix_ms);
            if !ready {
                break;
            }
            let Some((_, key)) = self.eviction_order.pop_first() else {
                break;
            };
            self.entries.remove(&key);
        }
    }

    fn with_capacity(retention_ms: u64, capacity: usize) -> Self {
        Self {
            entries: HashMap::new(),
            eviction_order: BTreeMap::new(),
            retention_ms,
            capacity,
        }
    }

    fn enforce_capacity(&mut self) {
        while self.entries.len() > self.capacity {
            let Some((_, key)) = self.eviction_order.pop_first() else {
                break;
            };
            self.entries.remove(&key);
        }
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }
}

fn expiry_order(key: BundleKey, retain_until_unix_ms: u64) -> ExpiryOrder {
    (
        retain_until_unix_ms,
        *key.manifest().as_bytes(),
        key.boundary_tokens(),
        key.boundary_hash().as_u128(),
    )
}

#[cfg(test)]
mod tests {
    use kvbm_common::SequenceHash;
    use kvbm_protocols::cache_manifest::CacheManifestId;

    use super::*;

    fn key(boundary_tokens: u64) -> BundleKey {
        BundleKey::from_parts(
            CacheManifestId::from_bytes([1; 32]),
            SequenceHash::root(boundary_tokens),
            boundary_tokens,
        )
        .unwrap()
    }

    #[test]
    fn retention_is_time_limited_and_exact_keyed() {
        let mut history = ExpiredAdvertisementHistory::with_capacity(10, 4);
        let expired = key(4);
        history.record([expired], 20);

        assert!(history.contains(expired));
        assert!(!history.contains(key(8)));
        history.prune(29);
        assert!(history.contains(expired));
        history.prune(30);
        assert!(!history.contains(expired));
    }

    #[test]
    fn same_deadline_capacity_eviction_is_deterministic() {
        let mut history = ExpiredAdvertisementHistory::with_capacity(10, 2);
        let mut reordered = ExpiredAdvertisementHistory::with_capacity(10, 2);
        let first = key(4);
        let second = key(8);
        let third = key(12);

        history.record([third, first, second], 20);
        reordered.record([first, second, third], 20);

        assert_eq!(history.len(), 2);
        assert!(!history.contains(first));
        assert!(history.contains(second));
        assert!(history.contains(third));
        assert!(!reordered.contains(first));
        assert!(reordered.contains(second));
        assert!(reordered.contains(third));
    }

    #[test]
    fn forgetting_one_key_removes_its_expiration_schedule() {
        let mut history = ExpiredAdvertisementHistory::with_capacity(10, 2);
        let forgotten = key(4);
        let retained = key(8);
        history.record([forgotten, retained], 20);

        history.forget(forgotten);
        assert_eq!(history.len(), 1);
        assert_eq!(history.eviction_order.len(), 1);
        assert!(!history.contains(forgotten));
        assert!(history.contains(retained));
        history.prune(30);

        assert_eq!(history.len(), 0);
    }
}
