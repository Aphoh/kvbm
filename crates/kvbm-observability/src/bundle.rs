// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Manifest-scoped bundle lifecycle metrics.

use std::time::Duration;

use kvbm_common::LogicalResourceId;
use prometheus::{Histogram, HistogramOpts, HistogramVec, IntCounterVec, Opts, Registry};

/// Prometheus handles for complete-bundle cache and disaggregation decisions.
#[derive(Clone)]
pub struct BundleMetrics {
    find_total: IntCounterVec,
    boundary_tokens: Histogram,
    txn_total: IntCounterVec,
    transfer_bytes: IntCounterVec,
    transfer_seconds: HistogramVec,
    admission_total: IntCounterVec,
    admission_bytes_total: IntCounterVec,
    resource_outcome_total: IntCounterVec,
    resource_planned_bytes_total: IntCounterVec,
    resource_bytes_total: IntCounterVec,
    resource_duration_seconds: HistogramVec,
    dependency_invalidations_total: IntCounterVec,
    lease_seconds: HistogramVec,
    disagg_decision_total: IntCounterVec,
    disagg_estimated_seconds: Histogram,
    disagg_actual_seconds: Histogram,
    disagg_phase_estimated_seconds: HistogramVec,
    disagg_phase_actual_seconds: HistogramVec,
}

impl BundleMetrics {
    pub fn new() -> Self {
        Self {
            find_total: IntCounterVec::new(
                Opts::new(
                    "kvbm_bundle_find_total",
                    "Manifest-scoped bundle finds by outcome and resource-level reason",
                ),
                &["result", "resource_reason"],
            )
            .expect("valid metric"),
            boundary_tokens: Histogram::with_opts(HistogramOpts::new(
                "kvbm_bundle_boundary_tokens",
                "Token boundary selected for a complete bundle",
            ))
            .expect("valid metric"),
            txn_total: IntCounterVec::new(
                Opts::new(
                    "kvbm_bundle_txn_total",
                    "Complete-bundle transactions by operation and terminal outcome",
                ),
                &["operation", "outcome"],
            )
            .expect("valid metric"),
            transfer_bytes: IntCounterVec::new(
                Opts::new(
                    "kvbm_bundle_transfer_bytes",
                    "Bytes transferred for each logical resource and tier route",
                ),
                &["resource", "source_tier", "destination_tier"],
            )
            .expect("valid metric"),
            transfer_seconds: HistogramVec::new(
                HistogramOpts::new(
                    "kvbm_bundle_transfer_seconds",
                    "Bundle resource transfer duration by phase",
                ),
                &["resource", "phase"],
            )
            .expect("valid metric"),
            admission_total: IntCounterVec::new(
                Opts::new(
                    "kvbm_bundle_admission_total",
                    "Whole-bundle resource admission decisions by bounded reason",
                ),
                &["resource", "decision", "reason"],
            )
            .expect("valid metric"),
            admission_bytes_total: IntCounterVec::new(
                Opts::new(
                    "kvbm_bundle_admission_bytes_total",
                    "Bytes considered by whole-bundle resource admission decisions",
                ),
                &["resource", "decision"],
            )
            .expect("valid metric"),
            resource_outcome_total: IntCounterVec::new(
                Opts::new(
                    "kvbm_bundle_resource_outcome_total",
                    "Bounded outcomes for manifest resource operations",
                ),
                &["operation", "resource", "outcome", "reason"],
            )
            .expect("valid metric"),
            resource_planned_bytes_total: IntCounterVec::new(
                Opts::new(
                    "kvbm_bundle_resource_planned_bytes_total",
                    "Bytes planned for manifest resource operations",
                ),
                &["operation", "resource"],
            )
            .expect("valid metric"),
            resource_bytes_total: IntCounterVec::new(
                Opts::new(
                    "kvbm_bundle_resource_bytes_total",
                    "Actual-safe or scored bytes for manifest resource outcomes",
                ),
                &["operation", "resource", "outcome"],
            )
            .expect("valid metric"),
            resource_duration_seconds: HistogramVec::new(
                HistogramOpts::new(
                    "kvbm_bundle_resource_duration_seconds",
                    "Duration of manifest resource operations by bounded outcome",
                ),
                &["operation", "resource", "outcome"],
            )
            .expect("valid metric"),
            dependency_invalidations_total: IntCounterVec::new(
                Opts::new(
                    "kvbm_bundle_dependency_invalidations_total",
                    "Bundle invalidations caused by logical-resource eviction",
                ),
                &["resource"],
            )
            .expect("valid metric"),
            lease_seconds: HistogramVec::new(
                HistogramOpts::new(
                    "kvbm_bundle_lease_seconds",
                    "Complete-bundle lease duration by locality",
                ),
                &["locality"],
            )
            .expect("valid metric"),
            disagg_decision_total: IntCounterVec::new(
                Opts::new(
                    "kvbm_disagg_decision_total",
                    "Conditional-disaggregation decisions and reasons",
                ),
                &["decision", "reason"],
            )
            .expect("valid metric"),
            disagg_estimated_seconds: Histogram::with_opts(HistogramOpts::new(
                "kvbm_disagg_estimated_seconds",
                "Estimated duration used for a disaggregation decision",
            ))
            .expect("valid metric"),
            disagg_actual_seconds: Histogram::with_opts(HistogramOpts::new(
                "kvbm_disagg_actual_seconds",
                "Observed duration after a disaggregation decision",
            ))
            .expect("valid metric"),
            disagg_phase_estimated_seconds: HistogramVec::new(
                HistogramOpts::new(
                    "kvbm_disagg_phase_estimated_seconds",
                    "Estimated conditional-prefill cost by placement and phase",
                ),
                &["placement", "phase"],
            )
            .expect("valid metric"),
            disagg_phase_actual_seconds: HistogramVec::new(
                HistogramOpts::new(
                    "kvbm_disagg_phase_actual_seconds",
                    "Observed conditional-prefill cost by placement and phase",
                ),
                &["placement", "phase"],
            )
            .expect("valid metric"),
        }
    }

    pub fn register(&self, registry: &Registry) -> Result<(), prometheus::Error> {
        registry.register(Box::new(self.find_total.clone()))?;
        registry.register(Box::new(self.boundary_tokens.clone()))?;
        registry.register(Box::new(self.txn_total.clone()))?;
        registry.register(Box::new(self.transfer_bytes.clone()))?;
        registry.register(Box::new(self.transfer_seconds.clone()))?;
        registry.register(Box::new(self.admission_total.clone()))?;
        registry.register(Box::new(self.admission_bytes_total.clone()))?;
        registry.register(Box::new(self.resource_outcome_total.clone()))?;
        registry.register(Box::new(self.resource_planned_bytes_total.clone()))?;
        registry.register(Box::new(self.resource_bytes_total.clone()))?;
        registry.register(Box::new(self.resource_duration_seconds.clone()))?;
        registry.register(Box::new(self.dependency_invalidations_total.clone()))?;
        registry.register(Box::new(self.lease_seconds.clone()))?;
        registry.register(Box::new(self.disagg_decision_total.clone()))?;
        registry.register(Box::new(self.disagg_estimated_seconds.clone()))?;
        registry.register(Box::new(self.disagg_actual_seconds.clone()))?;
        registry.register(Box::new(self.disagg_phase_estimated_seconds.clone()))?;
        registry.register(Box::new(self.disagg_phase_actual_seconds.clone()))?;
        Ok(())
    }

    pub fn record_find(&self, result: &'static str, resource_reason: &'static str) {
        self.find_total
            .with_label_values(&[result, resource_reason])
            .inc();
    }

    pub fn observe_boundary(&self, tokens: u64) {
        self.boundary_tokens.observe(tokens as f64);
    }

    pub fn record_transaction(&self, operation: &'static str, outcome: &'static str) {
        self.txn_total
            .with_label_values(&[operation, outcome])
            .inc();
    }

    pub fn record_transfer_bytes(
        &self,
        resource: &str,
        source_tier: &'static str,
        destination_tier: &'static str,
        bytes: u64,
    ) {
        self.transfer_bytes
            .with_label_values(&[resource, source_tier, destination_tier])
            .inc_by(bytes);
    }

    pub fn observe_transfer(&self, resource: &str, phase: &'static str, duration: Duration) {
        self.transfer_seconds
            .with_label_values(&[resource, phase])
            .observe(duration.as_secs_f64());
    }

    /// Record one bounded admission decision. The resource label is the stable
    /// numeric model-local ID, never a user-controlled resource name.
    pub fn record_admission(
        &self,
        resource: LogicalResourceId,
        decision: &'static str,
        reason: &'static str,
        bytes: u64,
    ) {
        let resource = resource.0.to_string();
        self.admission_total
            .with_label_values(&[resource.as_str(), decision, reason])
            .inc();
        self.admission_bytes_total
            .with_label_values(&[resource.as_str(), decision])
            .inc_by(bytes);
    }

    /// Record a bounded outcome and its actual-safe (or policy-scored) bytes
    /// for one logical resource. Callers own the finite operation/outcome/reason
    /// vocabularies; the resource label is always the stable numeric ID.
    pub fn record_resource_outcome(
        &self,
        operation: &'static str,
        resource: LogicalResourceId,
        outcome: &'static str,
        reason: &'static str,
        bytes: u64,
    ) {
        let resource = resource.0.to_string();
        self.resource_outcome_total
            .with_label_values(&[operation, resource.as_str(), outcome, reason])
            .inc();
        self.resource_bytes_total
            .with_label_values(&[operation, resource.as_str(), outcome])
            .inc_by(bytes);
    }

    /// Record the byte budget planned before a logical-resource operation.
    pub fn record_resource_planned_bytes(
        &self,
        operation: &'static str,
        resource: LogicalResourceId,
        bytes: u64,
    ) {
        let resource = resource.0.to_string();
        self.resource_planned_bytes_total
            .with_label_values(&[operation, resource.as_str()])
            .inc_by(bytes);
    }

    /// Observe one logical-resource operation from submission to terminal.
    pub fn observe_resource_duration(
        &self,
        operation: &'static str,
        resource: LogicalResourceId,
        outcome: &'static str,
        duration: Duration,
    ) {
        let resource = resource.0.to_string();
        self.resource_duration_seconds
            .with_label_values(&[operation, resource.as_str(), outcome])
            .observe(duration.as_secs_f64());
    }

    pub fn record_dependency_invalidation(&self, resource: &str) {
        self.dependency_invalidations_total
            .with_label_values(&[resource])
            .inc();
    }

    pub fn observe_lease(&self, locality: &'static str, duration: Duration) {
        self.lease_seconds
            .with_label_values(&[locality])
            .observe(duration.as_secs_f64());
    }

    pub fn record_disagg_decision(&self, decision: &'static str, reason: &'static str) {
        self.disagg_decision_total
            .with_label_values(&[decision, reason])
            .inc();
    }

    pub fn observe_disagg_estimate(&self, duration: Duration) {
        self.disagg_estimated_seconds
            .observe(duration.as_secs_f64());
    }

    pub fn observe_disagg_actual(&self, duration: Duration) {
        self.disagg_actual_seconds.observe(duration.as_secs_f64());
    }

    pub fn observe_disagg_phase_estimate(
        &self,
        placement: &'static str,
        phase: &'static str,
        duration: Duration,
    ) {
        self.disagg_phase_estimated_seconds
            .with_label_values(&[placement, phase])
            .observe(duration.as_secs_f64());
    }

    pub fn observe_disagg_phase_actual(
        &self,
        placement: &'static str,
        phase: &'static str,
        duration: Duration,
    ) {
        self.disagg_phase_actual_seconds
            .with_label_values(&[placement, phase])
            .observe(duration.as_secs_f64());
    }
}

impl Default for BundleMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kvbm_common::LogicalResourceId;

    #[test]
    fn find_reasons_are_distinct_prometheus_series() {
        let metrics = BundleMetrics::new();
        let registry = Registry::new();
        metrics.register(&registry).unwrap();
        for reason in [
            "not_found",
            "incompatible",
            "incomplete",
            "expired",
            "timed_out",
            "owner_lost",
            "transfer_failed",
            "commit_failed",
        ] {
            metrics.record_find("remote_miss", reason);
        }

        let family = registry
            .gather()
            .into_iter()
            .find(|family| family.name() == "kvbm_bundle_find_total")
            .unwrap();
        assert_eq!(family.get_metric().len(), 8);
    }

    #[test]
    fn admission_metrics_use_numeric_resource_ids_and_bounded_outcomes() {
        let metrics = BundleMetrics::new();
        let registry = Registry::new();
        metrics.register(&registry).unwrap();

        metrics.record_admission(
            LogicalResourceId(12),
            "retain",
            "byte_normalized_value",
            4096,
        );
        metrics.record_admission(
            LogicalResourceId(12),
            "drop",
            "below_byte_normalized_threshold",
            1024,
        );

        let gathered = registry.gather();
        let outcomes = gathered
            .iter()
            .find(|family| family.name() == "kvbm_bundle_admission_total")
            .unwrap();
        assert_eq!(outcomes.get_metric().len(), 2);
        assert!(outcomes.get_metric().iter().all(|metric| {
            metric
                .get_label()
                .iter()
                .any(|label| label.name() == "resource" && label.value() == "12")
        }));

        let bytes = gathered
            .iter()
            .find(|family| family.name() == "kvbm_bundle_admission_bytes_total")
            .unwrap();
        assert_eq!(bytes.get_metric().len(), 2);
    }

    #[test]
    fn generic_resource_metrics_keep_ids_numeric_and_bytes_and_duration_bounded() {
        let metrics = BundleMetrics::new();
        let registry = Registry::new();
        metrics.register(&registry).unwrap();

        metrics.record_resource_planned_bytes("offload_transfer", LogicalResourceId(12), 4096);
        metrics.record_resource_outcome(
            "offload_transfer",
            LogicalResourceId(12),
            "complete",
            "complete",
            4096,
        );
        metrics.observe_resource_duration(
            "offload_transfer",
            LogicalResourceId(12),
            "complete",
            Duration::from_millis(5),
        );

        let gathered = registry.gather();
        for family_name in [
            "kvbm_bundle_resource_outcome_total",
            "kvbm_bundle_resource_planned_bytes_total",
            "kvbm_bundle_resource_bytes_total",
            "kvbm_bundle_resource_duration_seconds",
        ] {
            let family = gathered
                .iter()
                .find(|family| family.name() == family_name)
                .unwrap_or_else(|| panic!("missing metric family {family_name}"));
            assert_eq!(family.get_metric().len(), 1);
            assert!(
                family.get_metric()[0]
                    .get_label()
                    .iter()
                    .any(|label| { label.name() == "resource" && label.value() == "12" })
            );
        }
    }
}
