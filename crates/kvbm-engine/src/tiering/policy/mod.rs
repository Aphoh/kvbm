// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Resource-keyed admission, retention, and inactive-backend policy.

mod components;
mod dependency;

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::num::{NonZeroU64, NonZeroUsize};

use kvbm_common::LogicalResourceId;
use kvbm_logical::manager::InactiveBackendConfig;
use kvbm_protocols::cache_manifest::ResourceRole;
use serde::{Deserialize, Serialize};

pub use components::{ResourceComponentBytes, ResourceComponentBytesError};
pub(in crate::tiering) use dependency::{BundleDependencyIndex, DependencyError, ResourceLineage};

/// All configurable policy for one logical cache resource.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourcePolicy {
    role: ResourceRole,
    g1_inactive: InactiveBackendConfig,
    g2_inactive: InactiveBackendConfig,
    atomic_components: NonZeroUsize,
    minimum_hits_per_mib: u64,
    #[serde(default)]
    minimum_admission_hits_per_mib: u64,
}

impl ResourcePolicy {
    pub fn new(
        role: ResourceRole,
        g1_inactive: InactiveBackendConfig,
        g2_inactive: InactiveBackendConfig,
    ) -> Self {
        Self {
            role,
            g1_inactive,
            g2_inactive,
            atomic_components: NonZeroUsize::MIN,
            minimum_hits_per_mib: 0,
            minimum_admission_hits_per_mib: 0,
        }
    }

    /// Require the declared physical components to admit and evict as one unit.
    pub fn with_atomic_components(mut self, components: NonZeroUsize) -> Self {
        self.atomic_components = components;
        self
    }

    /// Require at least this many expected reuse hits per MiB retained.
    pub fn with_minimum_hits_per_mib(mut self, minimum: u64) -> Self {
        self.minimum_hits_per_mib = minimum;
        self
    }

    /// Set the ingress density floor independently from pressure-time
    /// retention. It defaults to zero so a first-time bundle can enter before
    /// its blocks have accumulated reuse evidence in G2.
    pub fn with_minimum_admission_hits_per_mib(mut self, minimum: u64) -> Self {
        self.minimum_admission_hits_per_mib = minimum;
        self
    }

    pub const fn role(&self) -> ResourceRole {
        self.role
    }

    pub const fn g1_inactive(&self) -> &InactiveBackendConfig {
        &self.g1_inactive
    }

    pub const fn g2_inactive(&self) -> &InactiveBackendConfig {
        &self.g2_inactive
    }

    /// Whether pressure-time evaluation needs G1 reuse counts.
    ///
    /// This is deliberately independent of the inactive backend: an LRU can
    /// still use frequency as a retention signal. Admission reads G2's
    /// separately tracked registry and therefore does not affect this flag.
    pub const fn requires_g1_frequency_tracking(&self) -> bool {
        self.minimum_hits_per_mib > 0
    }

    /// Evaluate initial whole-bundle admission. Structural and dependency
    /// checks are shared with retention, but ingress has its own density floor.
    pub fn evaluate_admission(&self, input: &AdmissionScoreInputs) -> RetentionOutcome {
        self.evaluate_with_minimum(input, self.minimum_admission_hits_per_mib)
    }

    /// Evaluate whether an admitted resource remains valuable under pressure.
    pub fn evaluate(&self, input: &AdmissionScoreInputs) -> RetentionOutcome {
        self.evaluate_with_minimum(input, self.minimum_hits_per_mib)
    }

    fn evaluate_with_minimum(
        &self,
        input: &AdmissionScoreInputs,
        minimum_hits_per_mib: u64,
    ) -> RetentionOutcome {
        let score = input.score();
        if input.component_count() != self.atomic_components.get() {
            return RetentionOutcome::drop(score, RetentionReason::IncompleteAtomicResource);
        }
        if self.role == ResourceRole::BoundaryCapsule && !input.history_dependency_present {
            return RetentionOutcome::drop(score, RetentionReason::MissingHistoryDependency);
        }
        if !score.meets_hits_per_mib(minimum_hits_per_mib) {
            return RetentionOutcome::drop(score, RetentionReason::BelowByteNormalizedThreshold);
        }
        RetentionOutcome::retain(score, RetentionReason::ByteNormalizedValue)
    }
}

/// Resource-keyed policy configuration. No resource ID has built-in meaning.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ResourcePolicies(BTreeMap<LogicalResourceId, ResourcePolicy>);

impl ResourcePolicies {
    pub const fn new() -> Self {
        Self(BTreeMap::new())
    }

    pub fn insert(
        &mut self,
        resource: LogicalResourceId,
        policy: ResourcePolicy,
    ) -> Result<(), DuplicateResourcePolicy> {
        if self.0.contains_key(&resource) {
            return Err(DuplicateResourcePolicy { resource });
        }
        self.0.insert(resource, policy);
        Ok(())
    }

    pub fn get(&self, resource: LogicalResourceId) -> Option<&ResourcePolicy> {
        self.0.get(&resource)
    }

    pub fn iter(&self) -> impl Iterator<Item = (LogicalResourceId, &ResourcePolicy)> + '_ {
        self.0.iter().map(|(&resource, policy)| (resource, policy))
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("logical resource {resource:?} has more than one policy")]
pub struct DuplicateResourcePolicy {
    resource: LogicalResourceId,
}

/// Complete atomic-resource facts used for a byte-normalized admission decision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdmissionScoreInputs {
    hits: u64,
    component_bytes: Vec<NonZeroU64>,
    total_bytes: NonZeroU64,
    history_dependency_present: bool,
}

impl AdmissionScoreInputs {
    pub fn complete(
        hits: u64,
        component_bytes: impl IntoIterator<Item = NonZeroU64>,
    ) -> Result<Self, AdmissionInputError> {
        Self::new(hits, component_bytes, true)
    }

    pub fn orphan(
        hits: u64,
        component_bytes: impl IntoIterator<Item = NonZeroU64>,
    ) -> Result<Self, AdmissionInputError> {
        Self::new(hits, component_bytes, false)
    }

    fn new(
        hits: u64,
        component_bytes: impl IntoIterator<Item = NonZeroU64>,
        history_dependency_present: bool,
    ) -> Result<Self, AdmissionInputError> {
        let component_bytes = component_bytes.into_iter().collect::<Vec<_>>();
        if component_bytes.is_empty() {
            return Err(AdmissionInputError::NoComponents);
        }
        let total_bytes = component_bytes
            .iter()
            .try_fold(0u64, |total, bytes| total.checked_add(bytes.get()))
            .ok_or(AdmissionInputError::ByteCountOverflow)?;
        Ok(Self {
            hits,
            component_bytes,
            total_bytes: NonZeroU64::new(total_bytes).ok_or(AdmissionInputError::NoComponents)?,
            history_dependency_present,
        })
    }

    pub fn score(&self) -> AdmissionScore {
        AdmissionScore {
            hits: self.hits,
            bytes: self.total_bytes,
        }
    }

    const fn component_count(&self) -> usize {
        self.component_bytes.len()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum AdmissionInputError {
    #[error("admission input requires at least one physical component")]
    NoComponents,
    #[error("admission component byte count overflowed u64")]
    ByteCountOverflow,
}

/// Exact expected-hit density. Ordering compares ratios without floating point.
#[derive(Clone, Copy, Debug)]
pub struct AdmissionScore {
    hits: u64,
    bytes: NonZeroU64,
}

impl AdmissionScore {
    fn meets_hits_per_mib(self, minimum: u64) -> bool {
        u128::from(self.hits) * u128::from(1024u64 * 1024)
            >= u128::from(minimum) * u128::from(self.bytes.get())
    }

    pub const fn bytes(self) -> NonZeroU64 {
        self.bytes
    }
}

impl PartialEq for AdmissionScore {
    fn eq(&self, other: &Self) -> bool {
        u128::from(self.hits) * u128::from(other.bytes.get())
            == u128::from(other.hits) * u128::from(self.bytes.get())
    }
}

impl Eq for AdmissionScore {}

impl PartialOrd for AdmissionScore {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for AdmissionScore {
    fn cmp(&self, other: &Self) -> Ordering {
        (u128::from(self.hits) * u128::from(other.bytes.get()))
            .cmp(&(u128::from(other.hits) * u128::from(self.bytes.get())))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RetentionDecision {
    Retain,
    Drop,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RetentionReason {
    ByteNormalizedValue,
    BelowByteNormalizedThreshold,
    IncompleteAtomicResource,
    MissingHistoryDependency,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RetentionOutcome {
    decision: RetentionDecision,
    reason: RetentionReason,
    score: AdmissionScore,
}

impl RetentionOutcome {
    const fn retain(score: AdmissionScore, reason: RetentionReason) -> Self {
        Self {
            decision: RetentionDecision::Retain,
            reason,
            score,
        }
    }

    const fn drop(score: AdmissionScore, reason: RetentionReason) -> Self {
        Self {
            decision: RetentionDecision::Drop,
            reason,
            score,
        }
    }

    pub const fn decision(self) -> RetentionDecision {
        self.decision
    }

    pub const fn reason(self) -> RetentionReason {
        self.reason
    }

    pub const fn score(self) -> AdmissionScore {
        self.score
    }
}

#[cfg(test)]
mod tests;
