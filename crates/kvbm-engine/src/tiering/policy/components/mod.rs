// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Checked per-resource physical component widths used by admission scoring.

use std::collections::BTreeMap;
use std::num::NonZeroU64;

use kvbm_common::LogicalResourceId;

/// Positive byte widths for every independently atomic component of one
/// logical block, keyed by logical resource.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ResourceComponentBytes(BTreeMap<LogicalResourceId, Box<[NonZeroU64]>>);

impl ResourceComponentBytes {
    pub const fn new() -> Self {
        Self(BTreeMap::new())
    }

    /// Insert one resource exactly once. Empty component sets are rejected.
    pub fn insert(
        &mut self,
        resource: LogicalResourceId,
        components: impl IntoIterator<Item = NonZeroU64>,
    ) -> Result<(), ResourceComponentBytesError> {
        if self.0.contains_key(&resource) {
            return Err(ResourceComponentBytesError::DuplicateResource { resource });
        }
        let components = components.into_iter().collect::<Box<[_]>>();
        if components.is_empty() {
            return Err(ResourceComponentBytesError::NoComponents { resource });
        }
        self.0.insert(resource, components);
        Ok(())
    }

    pub fn get(&self, resource: LogicalResourceId) -> Option<&[NonZeroU64]> {
        self.0.get(&resource).map(AsRef::as_ref)
    }

    pub fn iter(&self) -> impl Iterator<Item = (LogicalResourceId, &[NonZeroU64])> + '_ {
        self.0
            .iter()
            .map(|(&resource, components)| (resource, components.as_ref()))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ResourceComponentBytesError {
    #[error("logical resource {resource:?} has no physical components")]
    NoComponents { resource: LogicalResourceId },
    #[error("logical resource {resource:?} has more than one component-byte config")]
    DuplicateResource { resource: LogicalResourceId },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_and_duplicate_resource_entries() {
        let resource = LogicalResourceId(4);
        let mut bytes = ResourceComponentBytes::new();
        assert_eq!(
            bytes.insert(resource, []),
            Err(ResourceComponentBytesError::NoComponents { resource })
        );
        bytes
            .insert(resource, [NonZeroU64::new(64).unwrap()])
            .unwrap();
        assert_eq!(
            bytes.insert(resource, [NonZeroU64::new(32).unwrap()]),
            Err(ResourceComponentBytesError::DuplicateResource { resource })
        );
    }
}
