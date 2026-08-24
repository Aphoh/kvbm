// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use kvbm_common::{LogicalResourceId, SequenceHash};
use kvbm_protocols::cache_manifest::{
    BundleKey, CacheIdentity, CacheManifest, ModelIdentity, ResourceRequirement, ResourceRole,
};

pub(super) const CSA: LogicalResourceId = LogicalResourceId(10);
pub(super) const HCA: LogicalResourceId = LogicalResourceId(11);
pub(super) const CAPSULE: LogicalResourceId = LogicalResourceId(12);

pub(super) fn identity() -> CacheIdentity {
    CacheManifest::new(
        ModelIdentity::new("hybrid-cache", "revision-a", [9; 32]).expect("valid model"),
        "hybrid-cache-test-v1",
        vec![
            requirement(CSA, ResourceRole::PrefixHistory),
            requirement(HCA, ResourceRole::PrefixHistory),
            requirement(CAPSULE, ResourceRole::BoundaryCapsule),
        ],
        Default::default(),
    )
    .expect("valid manifest")
    .identity()
}

pub(super) fn bundle_key(tokens: u64) -> BundleKey {
    BundleKey::new(&identity(), SequenceHash::new(tokens, None, tokens), tokens)
        .expect("aligned key")
}

fn requirement(resource: LogicalResourceId, role: ResourceRole) -> ResourceRequirement {
    ResourceRequirement::new(resource, role, 256).expect("valid requirement")
}
