// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Manifest-scoped bundle lifecycle metrics.

use std::time::Duration;

use prometheus::{Histogram, HistogramOpts, HistogramVec, IntCounterVec, Opts, Registry};

/// Prometheus handles for complete-bundle cache and disaggregation decisions.
#[derive(Clone)]
pub struct BundleMetrics {
    find_total: IntCounterVec,
    boundary_tokens: Histogram,
    txn_total: IntCounterVec,
    transfer_bytes: IntCounterVec,
    transfer_seconds: HistogramVec,
    dependency_invalidations_total: IntCounterVec,
    lease_seconds: HistogramVec,
    disagg_decision_total: IntCounterVec,
    disagg_estimated_seconds: Histogram,
    disagg_actual_seconds: Histogram,
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
        }
    }

    pub fn register(&self, registry: &Registry) -> Result<(), prometheus::Error> {
        registry.register(Box::new(self.find_total.clone()))?;
        registry.register(Box::new(self.boundary_tokens.clone()))?;
        registry.register(Box::new(self.txn_total.clone()))?;
        registry.register(Box::new(self.transfer_bytes.clone()))?;
        registry.register(Box::new(self.transfer_seconds.clone()))?;
        registry.register(Box::new(self.dependency_invalidations_total.clone()))?;
        registry.register(Box::new(self.lease_seconds.clone()))?;
        registry.register(Box::new(self.disagg_decision_total.clone()))?;
        registry.register(Box::new(self.disagg_estimated_seconds.clone()))?;
        registry.register(Box::new(self.disagg_actual_seconds.clone()))?;
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
}

impl Default for BundleMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            "transfer_failed",
        ] {
            metrics.record_find("remote_miss", reason);
        }

        let family = registry
            .gather()
            .into_iter()
            .find(|family| family.name() == "kvbm_bundle_find_total")
            .unwrap();
        assert_eq!(family.get_metric().len(), 5);
    }
}
