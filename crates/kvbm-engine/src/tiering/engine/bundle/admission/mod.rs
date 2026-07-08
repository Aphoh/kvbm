// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Atomic admission gate for complete multi-resource bundles.

use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU64;

use kvbm_common::LogicalResourceId;
use kvbm_protocols::cache_manifest::{ResourceRequirement, ResourceRole};
use kvbm_protocols::connector::{BundleOffloadPlan, LeaderEngineError, ResourceOffload};

use crate::tiering::engine::local::LocalConnectorEngine;
use crate::tiering::policy::{
    AdmissionScoreInputs, ResourceComponentBytes, ResourcePolicies, RetentionDecision,
    RetentionReason,
};

/// Immutable construction-time inputs for whole-bundle admission.
#[derive(Default)]
pub(in crate::tiering::engine) struct BundleAdmissionConfig {
    policies: ResourcePolicies,
    component_bytes: ResourceComponentBytes,
}

impl BundleAdmissionConfig {
    pub(in crate::tiering::engine) const fn new(
        policies: ResourcePolicies,
        component_bytes: ResourceComponentBytes,
    ) -> Self {
        Self {
            policies,
            component_bytes,
        }
    }

    pub(super) fn resource_bytes(
        &self,
        resource: LogicalResourceId,
        logical_blocks: usize,
    ) -> Option<u64> {
        let logical_blocks = u64::try_from(logical_blocks).ok()?;
        self.component_bytes
            .get(resource)?
            .iter()
            .try_fold(0_u64, |total, component| total.checked_add(component.get()))?
            .checked_mul(logical_blocks)
    }
}

impl LocalConnectorEngine {
    /// Evaluate every resource before the caller creates a transaction, action,
    /// drain, or buffered child. Any error therefore rejects the whole bundle
    /// without leaving partial orchestration state.
    pub(in crate::tiering::engine) fn admit_bundle(
        &self,
        plan: &BundleOffloadPlan,
    ) -> Result<BTreeMap<LogicalResourceId, u64>, LeaderEngineError> {
        let expected = plan
            .identity
            .resources()
            .iter()
            .map(ResourceRequirement::resource)
            .collect::<BTreeSet<_>>();
        let children = self.admission_children(plan, &expected)?;

        self.require_exact_policy_coverage(&expected)?;
        self.require_exact_component_coverage(&expected)?;

        let histories = plan
            .identity
            .resources()
            .iter()
            .filter(|requirement| requirement.role() == ResourceRole::PrefixHistory)
            .collect::<Vec<_>>();
        let mut planned_bytes = BTreeMap::new();
        for requirement in &histories {
            let resource = requirement.resource();
            planned_bytes.insert(
                resource,
                self.admit_resource(requirement, children[&resource], true)?,
            );
        }

        let history_dependency_present = !histories.is_empty();
        for requirement in plan
            .identity
            .resources()
            .iter()
            .filter(|requirement| requirement.role() == ResourceRole::BoundaryCapsule)
        {
            let resource = requirement.resource();
            planned_bytes.insert(
                resource,
                self.admit_resource(requirement, children[&resource], history_dependency_present)?,
            );
        }
        Ok(planned_bytes)
    }

    fn admission_children<'a>(
        &self,
        plan: &'a BundleOffloadPlan,
        expected: &BTreeSet<LogicalResourceId>,
    ) -> Result<BTreeMap<LogicalResourceId, &'a ResourceOffload>, LeaderEngineError> {
        let mut children = BTreeMap::new();
        for child in &plan.resources {
            if children.insert(child.resource, child).is_some() {
                return Err(self.config_rejection(
                    child.resource,
                    "duplicate_resource",
                    "resource occurs more than once in the bundle plan",
                ));
            }
        }
        let actual = children.keys().copied().collect::<BTreeSet<_>>();
        self.require_exact_set(expected, &actual, "bundle_plan")?;
        Ok(children)
    }

    fn require_exact_policy_coverage(
        &self,
        expected: &BTreeSet<LogicalResourceId>,
    ) -> Result<(), LeaderEngineError> {
        let actual = self
            .bundle_admission
            .policies
            .iter()
            .map(|(resource, _)| resource)
            .collect::<BTreeSet<_>>();
        self.require_exact_set(expected, &actual, "resource_policy")
    }

    fn require_exact_component_coverage(
        &self,
        expected: &BTreeSet<LogicalResourceId>,
    ) -> Result<(), LeaderEngineError> {
        let actual = self
            .bundle_admission
            .component_bytes
            .iter()
            .map(|(resource, _)| resource)
            .collect::<BTreeSet<_>>();
        self.require_exact_set(expected, &actual, "component_bytes")
    }

    fn require_exact_set(
        &self,
        expected: &BTreeSet<LogicalResourceId>,
        actual: &BTreeSet<LogicalResourceId>,
        source: &'static str,
    ) -> Result<(), LeaderEngineError> {
        if let Some(&resource) = expected.difference(actual).next() {
            return Err(self.config_rejection(
                resource,
                "missing_config",
                &format!("{source} is missing a required resource"),
            ));
        }
        if let Some(&resource) = actual.difference(expected).next() {
            return Err(self.config_rejection(
                resource,
                "unexpected_config",
                &format!("{source} configures a resource absent from the cache identity"),
            ));
        }
        Ok(())
    }

    fn admit_resource(
        &self,
        requirement: &ResourceRequirement,
        child: &ResourceOffload,
        history_dependency_present: bool,
    ) -> Result<u64, LeaderEngineError> {
        let resource = requirement.resource();
        let policy = self
            .bundle_admission
            .policies
            .get(resource)
            .expect("exact policy coverage was checked");
        if policy.role() != requirement.role() {
            return Err(self.config_rejection(
                resource,
                "role_mismatch",
                &format!(
                    "policy role {:?} disagrees with manifest role {:?}",
                    policy.role(),
                    requirement.role()
                ),
            ));
        }
        if child.blocks.is_empty() {
            return Err(self.config_rejection(
                resource,
                "empty_resource",
                "resource contains no logical blocks",
            ));
        }

        let manager = self.leader.g2_manager_for(resource).ok_or_else(|| {
            self.config_rejection(
                resource,
                "missing_manager",
                "resource has no G2 manager for frequency observation",
            )
        })?;
        let hits = child
            .blocks
            .iter()
            .map(|(hash, _)| u64::from(manager.block_registry().count(*hash)))
            .min()
            .expect("non-empty child was checked");
        let block_count = u64::try_from(child.blocks.len()).map_err(|_| {
            self.config_rejection(
                resource,
                "byte_overflow",
                "logical block count does not fit u64",
            )
        })?;
        let components = self
            .bundle_admission
            .component_bytes
            .get(resource)
            .expect("exact component coverage was checked")
            .iter()
            .map(|bytes: &NonZeroU64| {
                bytes
                    .get()
                    .checked_mul(block_count)
                    .and_then(NonZeroU64::new)
                    .ok_or_else(|| {
                        self.config_rejection(
                            resource,
                            "byte_overflow",
                            "component bytes overflow after scaling by logical block count",
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let inputs = if history_dependency_present {
            AdmissionScoreInputs::complete(hits, components)
        } else {
            AdmissionScoreInputs::orphan(hits, components)
        }
        .map_err(|error| self.config_rejection(resource, "invalid_input", &error.to_string()))?;
        let outcome = policy.evaluate_admission(&inputs);
        if let Some(observability) = self.leader.observability() {
            observability.bundle_metrics().record_admission(
                resource,
                decision_label(outcome.decision()),
                reason_label(outcome.reason()),
                outcome.score().bytes().get(),
            );
        }
        match outcome.decision() {
            RetentionDecision::Retain => Ok(outcome.score().bytes().get()),
            RetentionDecision::Drop => Err(LeaderEngineError::InvalidBundleTransfer {
                reason: format!(
                    "bundle admission rejected resource {resource:?}: {}",
                    reason_label(outcome.reason())
                ),
            }),
        }
    }

    fn config_rejection(
        &self,
        resource: LogicalResourceId,
        reason: &'static str,
        detail: &str,
    ) -> LeaderEngineError {
        if let Some(observability) = self.leader.observability() {
            observability
                .bundle_metrics()
                .record_admission(resource, "drop", reason, 0);
        }
        LeaderEngineError::InvalidBundleTransfer {
            reason: format!("bundle admission rejected resource {resource:?}: {detail}"),
        }
    }
}

const fn decision_label(decision: RetentionDecision) -> &'static str {
    match decision {
        RetentionDecision::Retain => "retain",
        RetentionDecision::Drop => "drop",
    }
}

const fn reason_label(reason: RetentionReason) -> &'static str {
    match reason {
        RetentionReason::ByteNormalizedValue => "byte_normalized_value",
        RetentionReason::BelowByteNormalizedThreshold => "below_byte_normalized_threshold",
        RetentionReason::IncompleteAtomicResource => "incomplete_atomic_resource",
        RetentionReason::MissingHistoryDependency => "missing_history_dependency",
    }
}
