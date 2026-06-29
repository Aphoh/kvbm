// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Manifest-bundle scheduler state and atomic save planning.

use std::sync::Arc;

use kvbm_common::{BlockId, LogicalResourceId};
use kvbm_protocols::cache_manifest::{BundleKey, CacheIdentity, ResourceRole};
use kvbm_protocols::connector::{BundleOffloadPlan, OffloadMode, ResourceOffload};

use super::super::slot::RequestSlot;
use super::LeaderState;

impl LeaderState {
    pub(super) fn offload_bundle_step(
        &mut self,
        request_id: &str,
        desired_tokens: usize,
        identity: CacheIdentity,
    ) -> bool {
        let (plan, boundary) = {
            let Some(slot) = self.slots.get_mut(request_id) else {
                return false;
            };
            let alignment = match usize::try_from(identity.alignment_tokens().get()) {
                Ok(alignment) => alignment,
                Err(_) => return false,
            };
            let sequence_tokens = slot.sequence.blocks().len() * self.block_size;
            let allocation_tokens = identity
                .resources()
                .iter()
                .map(|requirement| {
                    slot.resource_block_ids
                        .get(&requirement.resource())
                        .and_then(|ids| {
                            ids.len()
                                .checked_mul(requirement.native_block_tokens().get() as usize)
                        })
                        .unwrap_or(0)
                })
                .min()
                .unwrap_or(0);
            let boundary =
                desired_tokens.min(sequence_tokens).min(allocation_tokens) / alignment * alignment;
            if boundary == 0 || boundary <= slot.evaluated_tokens {
                return false;
            }
            let hash_index = boundary / self.block_size - 1;
            let boundary_hash = slot.sequence_hash(hash_index);
            let key = match BundleKey::new(&identity, boundary_hash, boundary as u64) {
                Ok(key) => key,
                Err(error) => {
                    tracing::warn!(request_id, %error, "invalid bundle boundary");
                    return false;
                }
            };
            let mut resources = Vec::with_capacity(identity.resources().len());
            for requirement in identity.resources() {
                let native = requirement.native_block_tokens().get() as usize;
                if native < self.block_size || !native.is_multiple_of(self.block_size) {
                    tracing::warn!(
                        request_id,
                        resource = ?requirement.resource(),
                        native,
                        "resource native block is not aligned to the connector block size"
                    );
                    return false;
                }
                let ids = slot
                    .resource_block_ids
                    .get(&requirement.resource())
                    .expect("allocation boundary checked above");
                let blocks = match requirement.role() {
                    ResourceRole::PrefixHistory => (0..boundary / native)
                        .map(|index| {
                            let end_token = (index + 1) * native;
                            let hash = slot.sequence_hash(end_token / self.block_size - 1);
                            (hash, ids[index])
                        })
                        .collect(),
                    ResourceRole::BoundaryCapsule => {
                        vec![(boundary_hash, ids[boundary / native - 1])]
                    }
                };
                resources.push(ResourceOffload {
                    resource: requirement.resource(),
                    blocks,
                });
            }
            (
                BundleOffloadPlan {
                    identity,
                    key,
                    generation: boundary as u64,
                    mode: OffloadMode::Mirror,
                    resources,
                },
                boundary,
            )
        };

        match Arc::clone(&self.engine).offload_bundle(&request_id.to_owned(), plan) {
            Ok(handle) => {
                if let Some(slot) = self.slots.get_mut(request_id) {
                    slot.evaluated_tokens = boundary;
                    slot.offloads.push(handle);
                }
                true
            }
            Err(error) => {
                tracing::warn!(request_id, %error, "engine rejected bundle offload");
                false
            }
        }
    }
}

pub(super) fn sync_full_resource_lists(
    slot: &mut RequestSlot,
    resources: &[LogicalResourceId],
    groups: &[Vec<BlockId>],
) {
    if resources.len() != groups.len() {
        tracing::warn!(
            request_id = %slot.request_id,
            resources = resources.len(),
            groups = groups.len(),
            "scheduler resource mapping does not match block-id group count"
        );
        return;
    }
    for (&resource, ids) in resources.iter().zip(groups) {
        let current = slot.resource_block_ids.entry(resource).or_default();
        if ids.len() > current.len() {
            if ids[..current.len()] != current[..] {
                tracing::warn!(
                    request_id = %slot.request_id,
                    ?resource,
                    "resource block id prefix mismatch while syncing scheduler allocation"
                );
                continue;
            }
            current.extend_from_slice(&ids[current.len()..]);
        }
    }
}

pub(super) fn extend_resource_lists(
    slot: &mut RequestSlot,
    resources: &[LogicalResourceId],
    groups: &[Vec<BlockId>],
) {
    if resources.len() != groups.len() {
        tracing::warn!(
            request_id = %slot.request_id,
            resources = resources.len(),
            groups = groups.len(),
            "scheduler resource mapping does not match delta block-id group count"
        );
        return;
    }
    for (&resource, ids) in resources.iter().zip(groups) {
        slot.resource_block_ids
            .entry(resource)
            .or_default()
            .extend_from_slice(ids);
    }
}
