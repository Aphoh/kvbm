// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded retired owner-local bundle generations.

use std::collections::HashMap;

use kvbm_protocols::cache_manifest::BundleKey;
use velo_ext::InstanceId;

type RetirementKey = (BundleKey, InstanceId);

/// High-water marks that prevent delayed publications from reviving an
/// invalidated generation. Absent-key entries consume a per-owner budget;
/// entries created by removing a live advertisement bypass that narrower
/// budget but remain subject to total owner and directory limits.
pub(super) struct RetiredGenerations {
    entries: HashMap<RetirementKey, RetiredGeneration>,
    owner_counts: HashMap<InstanceId, usize>,
    absent_counts: HashMap<InstanceId, usize>,
    absent_capacity_per_owner: usize,
    absent_global_capacity: usize,
    absent_len: usize,
    total_capacity_per_owner: usize,
    total_global_capacity: usize,
    max_absent_retention_ms: u64,
}

#[derive(Clone, Copy)]
struct RetiredGeneration {
    generation: u64,
    retain_until_unix_ms: u64,
    source: RetirementSource,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RetirementSource {
    Absent,
    LiveAdvertisement,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RetiredGenerationCapacity {
    AbsentOwner { owner: InstanceId, capacity: usize },
    AbsentGlobal { capacity: usize },
    TotalOwner { owner: InstanceId, capacity: usize },
    TotalGlobal { capacity: usize },
}

impl RetiredGenerations {
    pub(super) fn new(
        max_absent_retention_ms: u64,
        absent_capacity_per_owner: usize,
        absent_global_capacity: usize,
        total_capacity_per_owner: usize,
        total_global_capacity: usize,
    ) -> Self {
        Self {
            entries: HashMap::new(),
            owner_counts: HashMap::new(),
            absent_counts: HashMap::new(),
            absent_capacity_per_owner,
            absent_global_capacity,
            absent_len: 0,
            total_capacity_per_owner,
            total_global_capacity,
            max_absent_retention_ms,
        }
    }

    pub(super) fn rejected_generation(
        &self,
        key: BundleKey,
        owner: InstanceId,
        attempted: u64,
        observed_unix_ms: u64,
    ) -> Option<u64> {
        self.entries
            .get(&(key, owner))
            .filter(|retired| {
                retired.generation >= attempted && retired.retain_until_unix_ms > observed_unix_ms
            })
            .map(|retired| retired.generation)
    }

    pub(super) fn retire_absent(
        &mut self,
        key: BundleKey,
        owner: InstanceId,
        generation: u64,
        requested_retain_until_unix_ms: u64,
        observed_unix_ms: u64,
    ) -> Result<(), RetiredGenerationCapacity> {
        let retain_until_unix_ms =
            self.bounded_absent_deadline(requested_retain_until_unix_ms, observed_unix_ms);
        if retain_until_unix_ms <= observed_unix_ms {
            return Ok(());
        }
        let identity = (key, owner);
        if let Some(retired) = self.entries.get_mut(&identity) {
            merge_retirement(retired, generation, retain_until_unix_ms);
            return Ok(());
        }
        self.ensure_capacity(owner, true)?;
        let count = self.absent_counts.get(&owner).copied().unwrap_or(0);
        self.entries.insert(
            identity,
            RetiredGeneration {
                generation,
                retain_until_unix_ms,
                source: RetirementSource::Absent,
            },
        );
        *self.owner_counts.entry(owner).or_default() += 1;
        self.absent_counts.insert(owner, count + 1);
        self.absent_len += 1;
        Ok(())
    }

    pub(super) fn retire_live(
        &mut self,
        key: BundleKey,
        owner: InstanceId,
        generation: u64,
        advertisement_expires_at_unix_ms: u64,
        requested_retain_until_unix_ms: u64,
        observed_unix_ms: u64,
    ) -> Result<(), RetiredGenerationCapacity> {
        let retain_until_unix_ms = advertisement_expires_at_unix_ms
            .max(self.bounded_absent_deadline(requested_retain_until_unix_ms, observed_unix_ms));
        if retain_until_unix_ms <= observed_unix_ms {
            return Ok(());
        }
        let identity = (key, owner);
        let upgraded_absent = if let Some(retired) = self.entries.get_mut(&identity) {
            let upgraded = retired.source == RetirementSource::Absent;
            merge_retirement(retired, generation, retain_until_unix_ms);
            retired.source = RetirementSource::LiveAdvertisement;
            upgraded
        } else {
            self.ensure_capacity(owner, false)?;
            self.entries.insert(
                identity,
                RetiredGeneration {
                    generation,
                    retain_until_unix_ms,
                    source: RetirementSource::LiveAdvertisement,
                },
            );
            *self.owner_counts.entry(owner).or_default() += 1;
            false
        };
        if upgraded_absent {
            self.decrement_absent(owner);
        }
        Ok(())
    }

    pub(super) fn prune(&mut self, observed_unix_ms: u64) {
        let expired = self
            .entries
            .iter()
            .filter_map(|(identity, retired)| {
                (retired.retain_until_unix_ms <= observed_unix_ms)
                    .then_some((*identity, retired.source))
            })
            .collect::<Vec<_>>();
        for (identity, source) in expired {
            self.entries.remove(&identity);
            decrement_count(&mut self.owner_counts, identity.1);
            if source == RetirementSource::Absent {
                self.decrement_absent(identity.1);
            }
        }
    }

    pub(super) fn remove_owner(&mut self, owner: InstanceId) -> Vec<BundleKey> {
        let removed = self
            .entries
            .iter()
            .filter_map(|((key, entry_owner), retired)| {
                (*entry_owner == owner).then_some((*key, retired.source))
            })
            .collect::<Vec<_>>();
        for (key, source) in &removed {
            self.entries.remove(&(*key, owner));
            decrement_count(&mut self.owner_counts, owner);
            if *source == RetirementSource::Absent {
                self.decrement_absent(owner);
            }
        }
        removed.into_iter().map(|(key, _)| key).collect()
    }

    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(super) fn owner_len(&self, owner: InstanceId) -> usize {
        self.owner_counts.get(&owner).copied().unwrap_or(0)
    }

    pub(super) fn contains(&self, key: BundleKey, owner: InstanceId) -> bool {
        self.entries.contains_key(&(key, owner))
    }

    fn bounded_absent_deadline(
        &self,
        requested_retain_until_unix_ms: u64,
        observed_unix_ms: u64,
    ) -> u64 {
        requested_retain_until_unix_ms
            .min(observed_unix_ms.saturating_add(self.max_absent_retention_ms))
    }

    fn decrement_absent(&mut self, owner: InstanceId) {
        self.absent_len -= 1;
        decrement_count(&mut self.absent_counts, owner);
    }

    fn ensure_capacity(
        &self,
        owner: InstanceId,
        absent: bool,
    ) -> Result<(), RetiredGenerationCapacity> {
        if self.entries.len() >= self.total_global_capacity {
            return Err(RetiredGenerationCapacity::TotalGlobal {
                capacity: self.total_global_capacity,
            });
        }
        if self.owner_len(owner) >= self.total_capacity_per_owner {
            return Err(RetiredGenerationCapacity::TotalOwner {
                owner,
                capacity: self.total_capacity_per_owner,
            });
        }
        if absent && self.absent_len >= self.absent_global_capacity {
            return Err(RetiredGenerationCapacity::AbsentGlobal {
                capacity: self.absent_global_capacity,
            });
        }
        if absent
            && self.absent_counts.get(&owner).copied().unwrap_or(0)
                >= self.absent_capacity_per_owner
        {
            return Err(RetiredGenerationCapacity::AbsentOwner {
                owner,
                capacity: self.absent_capacity_per_owner,
            });
        }
        Ok(())
    }
}

fn decrement_count(counts: &mut HashMap<InstanceId, usize>, owner: InstanceId) {
    let remove = counts.get_mut(&owner).is_some_and(|count| {
        *count -= 1;
        *count == 0
    });
    if remove {
        counts.remove(&owner);
    }
}

fn merge_retirement(retired: &mut RetiredGeneration, generation: u64, retain_until_unix_ms: u64) {
    retired.generation = retired.generation.max(generation);
    retired.retain_until_unix_ms = retired.retain_until_unix_ms.max(retain_until_unix_ms);
}
