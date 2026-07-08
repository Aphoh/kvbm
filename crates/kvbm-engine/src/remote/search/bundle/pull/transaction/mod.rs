// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! All-resource ownership for a remote bundle before local publication.

use std::collections::{BTreeMap, BTreeSet};

use kvbm_common::{LogicalResourceId, SequenceHash};
use kvbm_logical::ImmutableBlock;

use crate::G2;
use crate::p2p::StagedPull;

/// A complete set of unregistered physical pulls and their logical lineages.
///
/// Dropping this value rolls every staged destination back. Publication is a
/// consuming, infallible transition used only after catalog preflight.
pub(crate) struct StagedBundle {
    lineages: BTreeMap<LogicalResourceId, Vec<SequenceHash>>,
    resources: BTreeMap<LogicalResourceId, StagedPull>,
}

impl StagedBundle {
    pub(crate) fn new(
        lineages: BTreeMap<LogicalResourceId, Vec<SequenceHash>>,
        resources: impl IntoIterator<Item = StagedPull>,
    ) -> Result<Self, StagedBundleError> {
        let mut staged = BTreeMap::new();
        for resource in resources {
            let resource_id = resource.resource();
            if staged.insert(resource_id, resource).is_some() {
                return Err(StagedBundleError::DuplicateResource(resource_id));
            }
        }
        let expected = lineages.keys().copied().collect::<BTreeSet<_>>();
        let actual = staged.keys().copied().collect::<BTreeSet<_>>();
        if expected != actual {
            return Err(StagedBundleError::ResourceSetMismatch {
                expected: expected.into_iter().collect(),
                actual: actual.into_iter().collect(),
            });
        }
        for (&resource, hashes) in &lineages {
            if staged[&resource].hashes() != hashes {
                return Err(StagedBundleError::LineageMismatch(resource));
            }
        }
        Ok(Self {
            lineages,
            resources: staged,
        })
    }

    pub(crate) fn lineages(&self) -> &BTreeMap<LogicalResourceId, Vec<SequenceHash>> {
        &self.lineages
    }

    pub(crate) fn resource_ids(&self) -> impl Iterator<Item = LogicalResourceId> + '_ {
        self.resources.keys().copied()
    }

    pub(crate) fn publish(self) -> BTreeMap<LogicalResourceId, Vec<ImmutableBlock<G2>>> {
        self.resources
            .into_iter()
            .map(|(resource, staged)| (resource, staged.publish()))
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum StagedBundleError {
    #[error("staged resource {0:?} occurs more than once")]
    DuplicateResource(LogicalResourceId),
    #[error("staged resources differ from lineages: expected {expected:?}, got {actual:?}")]
    ResourceSetMismatch {
        expected: Vec<LogicalResourceId>,
        actual: Vec<LogicalResourceId>,
    },
    #[error("staged resource {0:?} does not match its requested lineage")]
    LineageMismatch(LogicalResourceId),
}
