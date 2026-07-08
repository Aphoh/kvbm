// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;

use kvbm_common::LogicalResourceId;

use super::BundleResourceLineage;
use crate::cache_manifest::{BundleKey, ResourceRequirement, ResourceRole};

/// Validate complete-bundle resource coverage and role-specific geometry.
pub fn validate_bundle_lineages(
    key: BundleKey,
    requirements: &[ResourceRequirement],
    lineages: &[BundleResourceLineage],
) -> Result<(), BundleLineageValidationError> {
    if requirements.is_empty() {
        return Err(BundleLineageValidationError::NoRequirements);
    }
    let mut expected = BTreeSet::new();
    for requirement in requirements {
        let resource = requirement.resource();
        if !expected.insert(resource) {
            return Err(BundleLineageValidationError::DuplicateRequirement(resource));
        }
    }
    let mut actual = BTreeSet::new();
    for lineage in lineages {
        let resource = lineage.resource();
        if !actual.insert(resource) {
            return Err(BundleLineageValidationError::DuplicateResource(resource));
        }
    }
    if actual != expected {
        return Err(BundleLineageValidationError::IncompleteResources {
            expected: expected.into_iter().collect(),
            actual: actual.into_iter().collect(),
        });
    }

    let mut ordered_requirements = requirements.iter().collect::<Vec<_>>();
    ordered_requirements.sort_by_key(|requirement| requirement.resource());
    let mut ordered_lineages = lineages.iter().collect::<Vec<_>>();
    ordered_lineages.sort_by_key(|lineage| lineage.resource());

    let mut canonical_history: Option<(&ResourceRequirement, &BundleResourceLineage)> = None;
    for (requirement, lineage) in ordered_requirements.iter().zip(&ordered_lineages) {
        let resource = requirement.resource();
        let native_block_tokens = u64::from(requirement.native_block_tokens().get());
        if !key.boundary_tokens().is_multiple_of(native_block_tokens) {
            return Err(BundleLineageValidationError::UnalignedResourceBoundary {
                resource,
                boundary_tokens: key.boundary_tokens(),
                native_block_tokens: requirement.native_block_tokens().get(),
            });
        }
        let expected_blocks = match requirement.role() {
            ResourceRole::PrefixHistory => {
                usize::try_from(key.boundary_tokens() / native_block_tokens).unwrap_or(usize::MAX)
            }
            ResourceRole::BoundaryCapsule => 1,
        };
        if lineage.hashes().len() != expected_blocks {
            return Err(BundleLineageValidationError::InvalidResourceBlockCount {
                resource,
                role: requirement.role(),
                boundary_tokens: key.boundary_tokens(),
                expected: expected_blocks,
                actual: lineage.hashes().len(),
            });
        }
        match requirement.role() {
            ResourceRole::PrefixHistory => {
                if !lineage
                    .hashes()
                    .iter()
                    .enumerate()
                    .all(|(position, hash)| hash.position() == position as u64)
                {
                    return Err(BundleLineageValidationError::ResourceBoundaryMismatch {
                        resource,
                        boundary_tokens: key.boundary_tokens(),
                    });
                }
                if lineage.hashes().last().copied() == Some(key.boundary_hash())
                    && canonical_history.as_ref().is_none_or(|(canonical, _)| {
                        requirement.native_block_tokens() < canonical.native_block_tokens()
                    })
                {
                    canonical_history = Some((requirement, lineage));
                }
            }
            ResourceRole::BoundaryCapsule
                if lineage.hashes().first().copied() != Some(key.boundary_hash()) =>
            {
                return Err(BundleLineageValidationError::CapsuleBoundaryMismatch { resource });
            }
            ResourceRole::BoundaryCapsule => {}
        }
    }

    let Some((canonical_requirement, canonical_lineage)) = canonical_history else {
        return Err(BundleLineageValidationError::MissingCanonicalHistoryBoundary);
    };
    let canonical_native = canonical_requirement.native_block_tokens().get();
    for (requirement, actual) in ordered_requirements
        .iter()
        .zip(&ordered_lineages)
        .filter(|(requirement, _)| requirement.role() == ResourceRole::PrefixHistory)
    {
        let native = requirement.native_block_tokens().get();
        if !native.is_multiple_of(canonical_native) {
            return Err(BundleLineageValidationError::ResourceBoundaryMismatch {
                resource: requirement.resource(),
                boundary_tokens: key.boundary_tokens(),
            });
        }
        let expected = BundleResourceLineage::project_from_canonical(
            requirement.resource(),
            canonical_lineage.hashes(),
            (native / canonical_native) as usize,
        )
        .map_err(|_| BundleLineageValidationError::ResourceBoundaryMismatch {
            resource: requirement.resource(),
            boundary_tokens: key.boundary_tokens(),
        })?;
        if *actual != &expected {
            return Err(BundleLineageValidationError::ResourceBoundaryMismatch {
                resource: requirement.resource(),
                boundary_tokens: key.boundary_tokens(),
            });
        }
    }
    Ok(())
}

/// Manifest-neutral failure from complete-bundle lineage validation.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum BundleLineageValidationError {
    #[error("a bundle lineage contract requires at least one resource")]
    NoRequirements,
    #[error("bundle requirement resource {0:?} is duplicated")]
    DuplicateRequirement(LogicalResourceId),
    #[error("bundle resource {0:?} is duplicated")]
    DuplicateResource(LogicalResourceId),
    #[error("bundle resources are incomplete: expected {expected:?}, got {actual:?}")]
    IncompleteResources {
        expected: Vec<LogicalResourceId>,
        actual: Vec<LogicalResourceId>,
    },
    #[error(
        "bundle boundary {boundary_tokens} is not aligned to resource {resource:?} native block size {native_block_tokens}"
    )]
    UnalignedResourceBoundary {
        resource: LogicalResourceId,
        boundary_tokens: u64,
        native_block_tokens: u32,
    },
    #[error(
        "bundle resource {resource:?} has {actual} lineage blocks, expected {expected} for role {role:?} at boundary {boundary_tokens}"
    )]
    InvalidResourceBlockCount {
        resource: LogicalResourceId,
        role: ResourceRole,
        boundary_tokens: u64,
        expected: usize,
        actual: usize,
    },
    #[error("bundle history resource {resource:?} does not reach boundary {boundary_tokens}")]
    ResourceBoundaryMismatch {
        resource: LogicalResourceId,
        boundary_tokens: u64,
    },
    #[error("no bundle prefix history ends at the canonical boundary hash")]
    MissingCanonicalHistoryBoundary,
    #[error("bundle capsule resource {resource:?} does not match the bundle boundary hash")]
    CapsuleBoundaryMismatch { resource: LogicalResourceId },
}
