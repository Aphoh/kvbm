// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::num::{NonZeroU32, NonZeroU64};

use kvbm_common::LogicalResourceId;
use serde::{Deserialize, Serialize};

use super::ManifestError;

/// Semantic lifetime of one required cache resource.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceRole {
    PrefixHistory,
    BoundaryCapsule,
}

impl ResourceRole {
    pub(super) const fn canonical_tag(self) -> u8 {
        match self {
            Self::PrefixHistory => 0,
            Self::BoundaryCapsule => 1,
        }
    }
}

/// One logical resource required before a prefix can be reused.
///
/// Every listed resource is mandatory. Optional/advisory resources belong in
/// policy configuration, not in the correctness contract.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ResourceRequirement {
    resource: LogicalResourceId,
    role: ResourceRole,
    native_block_tokens: NonZeroU32,
}

impl ResourceRequirement {
    pub fn new(
        resource: LogicalResourceId,
        role: ResourceRole,
        native_block_tokens: u32,
    ) -> Result<Self, ManifestError> {
        let native_block_tokens = NonZeroU32::new(native_block_tokens)
            .ok_or(ManifestError::ZeroNativeBlockTokens { resource })?;
        Ok(Self {
            resource,
            role,
            native_block_tokens,
        })
    }

    pub const fn resource(&self) -> LogicalResourceId {
        self.resource
    }

    pub const fn role(&self) -> ResourceRole {
        self.role
    }

    pub const fn native_block_tokens(&self) -> NonZeroU32 {
        self.native_block_tokens
    }
}

pub(super) fn resource_alignment(
    resources: &[ResourceRequirement],
) -> Result<NonZeroU64, ManifestError> {
    let alignment = resources.iter().try_fold(1u64, |alignment, requirement| {
        checked_lcm(
            alignment,
            u64::from(requirement.native_block_tokens().get()),
        )
        .ok_or(ManifestError::AlignmentOverflow)
    })?;
    NonZeroU64::new(alignment).ok_or(ManifestError::AlignmentOverflow)
}

fn checked_lcm(left: u64, right: u64) -> Option<u64> {
    left.checked_div(gcd(left, right))?.checked_mul(right)
}

fn gcd(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        (left, right) = (right, left % right);
    }
    left
}
