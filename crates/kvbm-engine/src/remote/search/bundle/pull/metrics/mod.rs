// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded transaction and per-resource remote-pull observations.

use std::collections::BTreeMap;

use kvbm_common::LogicalResourceId;
use kvbm_observability::BundleMetrics;

use super::super::BundlePullOutcome;

pub(super) struct PullMetrics {
    metrics: Option<BundleMetrics>,
    transferred_bytes: BTreeMap<LogicalResourceId, u64>,
    started: std::time::Instant,
}

impl PullMetrics {
    pub(super) fn new(metrics: Option<BundleMetrics>, resources: Vec<LogicalResourceId>) -> Self {
        Self {
            metrics,
            transferred_bytes: resources
                .into_iter()
                .map(|resource| (resource, 0))
                .collect(),
            started: std::time::Instant::now(),
        }
    }

    pub(super) fn observe_transferred_bytes(&mut self, resource: LogicalResourceId, bytes: u64) {
        if let Some(transferred) = self.transferred_bytes.get_mut(&resource) {
            *transferred = bytes;
        }
    }

    pub(super) fn record(&self, result: &anyhow::Result<BundlePullOutcome>) {
        let Some(metrics) = self.metrics.as_ref() else {
            return;
        };
        let (transaction, resource_outcome, reason) = match result {
            Ok(BundlePullOutcome::Pulled(_)) => ("commit", "committed", "complete"),
            Ok(BundlePullOutcome::Miss(reason)) => ("abort", "aborted", reason.as_label()),
            Err(_) => ("abort", "aborted", "internal_error"),
        };
        metrics.record_transaction("remote_pull", transaction);
        let elapsed = self.started.elapsed();
        for (&resource, &bytes) in &self.transferred_bytes {
            metrics.record_resource_outcome(
                "remote_pull",
                resource,
                resource_outcome,
                reason,
                bytes,
            );
            metrics.observe_resource_duration("remote_pull", resource, resource_outcome, elapsed);
        }
    }
}
