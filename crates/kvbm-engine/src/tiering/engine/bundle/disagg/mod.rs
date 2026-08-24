// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conditional prefill composed over complete-bundle remote search.

mod decision;
mod reservation;

use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use kvbm_observability::BundleMetrics;
use kvbm_protocols::cache_manifest::{BundleKey, CacheIdentity};
use kvbm_protocols::connector::LocalPrefillEstimate;
use kvbm_protocols::disagg::BundlePrefillContext;
use tokio_util::sync::CancellationToken;

use self::decision::{PlacementDecision, PlacementInputs};
use self::reservation::BudgetReservation;
use crate::leader::RemoteDiscoveryHandle;
use crate::remote::cd::decode::{self, PlanInputs, PlanOutcome};
use crate::remote::cd::wire::PrefillDispatch;
use crate::remote::search::bundle::{
    BundleDiscoveryOutcome, BundleDiscoveryQuery, BundleMissReason, BundlePullOutcome,
    BundlePullTarget, pull_remote_bundle, unix_time_ms,
};
use crate::tiering::engine::local::LocalConnectorEngine;

/// Complete-bundle target and best seed retained across one placement attempt.
pub(in crate::tiering::engine) struct BundlePrefillRequest {
    pub(in crate::tiering::engine) request_id: String,
    pub(in crate::tiering::engine) identity: CacheIdentity,
    pub(in crate::tiering::engine) target: BundleKey,
    pub(in crate::tiering::engine) seed: Option<BundleKey>,
    pub(in crate::tiering::engine) num_computed_tokens: usize,
    pub(in crate::tiering::engine) total_tokens: usize,
    pub(in crate::tiering::engine) local_prefill_estimate: Option<LocalPrefillEstimate>,
}

impl LocalConnectorEngine {
    pub(in crate::tiering::engine) async fn run_bundle_prefill(
        self: Arc<Self>,
        directory: RemoteDiscoveryHandle,
        request: BundlePrefillRequest,
        cancel: CancellationToken,
        search_deadline: tokio::time::Instant,
    ) -> Result<Option<BundleKey>> {
        let fallback = request.seed;
        let Some(cd) = self.cd.as_ref() else {
            return Ok(fallback);
        };
        let target_tokens = usize::try_from(request.target.boundary_tokens()).unwrap_or(usize::MAX);
        let seed_tokens = request
            .seed
            .and_then(|key| usize::try_from(key.boundary_tokens()).ok())
            .unwrap_or_default();
        let estimated_bytes = u64::try_from(target_tokens)
            .unwrap_or(u64::MAX)
            .saturating_mul(cd.cfg.bundle_bytes_per_token);
        let placement = decision::decide(PlacementInputs {
            target_tokens,
            seed_tokens,
            computed_tokens: request.num_computed_tokens,
            bundle_bytes: estimated_bytes,
            local: request.local_prefill_estimate,
            remote: cd.cfg.cost,
            margin: cd.cfg.decision_margin,
        });
        let metrics = self
            .leader
            .observability()
            .map(|observability| observability.bundle_metrics().clone());
        let costs = match placement {
            PlacementDecision::Local { reason, costs } => {
                observe_estimates(metrics.as_ref(), costs);
                record_decision(metrics.as_ref(), "local", reason.label());
                return Ok(fallback);
            }
            PlacementDecision::Remote { costs } => costs,
        };
        observe_estimates(metrics.as_ref(), costs);

        let inputs = PlanInputs {
            total_tokens: target_tokens,
            num_computed_tokens: seed_tokens,
            matched_tokens: 0,
            block_size: self.block_size,
            bundle_bytes: estimated_bytes,
        };
        let reserved_tokens = match decode::plan(&cd.cfg, &cd.tier, &cd.budget, &inputs) {
            PlanOutcome::Remote {
                full_block_external_tokens,
            } => {
                record_decision(metrics.as_ref(), "remote", "remote_strictly_faster");
                full_block_external_tokens
            }
            PlanOutcome::Local { reason } => {
                record_decision(metrics.as_ref(), "local", local_reason(reason));
                return Ok(fallback);
            }
            PlanOutcome::Reject => {
                record_decision(metrics.as_ref(), "local", "budget_rejected");
                return Ok(fallback);
            }
        };
        let _reservation = BudgetReservation::new(Arc::clone(&cd.budget), reserved_tokens);
        let context = BundlePrefillContext::new(
            request.identity.manifest(),
            request
                .identity
                .resources()
                .iter()
                .map(|requirement| requirement.resource()),
            request.seed,
            request.target,
            estimated_bytes,
        )?;
        let window_end = target_tokens.saturating_add(1).min(request.total_tokens);
        let dispatch = PrefillDispatch {
            request_id: request.request_id,
            session_id: uuid::Uuid::new_v4(),
            decode_endpoint: None,
            num_provided_tokens: 0,
            num_window_tokens: window_end,
            bundle: Some(context),
        };
        let deadline =
            search_deadline.min(tokio::time::Instant::now() + cd.cfg.bundle_prefill_timeout);
        let pull_target: Arc<dyn BundlePullTarget> = Arc::clone(&self) as Arc<dyn BundlePullTarget>;
        let started = Instant::now();
        let result = tokio::select! {
            biased;
            _ = cancel.cancelled() => {
                record_decision(metrics.as_ref(), "local", "canceled");
                fallback
            },
            result = tokio::time::timeout_at(deadline, async {
                let dispatch_started = Instant::now();
                if let Err(error) = cd.plane.dispatch(dispatch).await {
                    record_decision(metrics.as_ref(), "local", "dispatch_failed");
                    tracing::warn!(%error, "bundle prefill dispatch failed; retaining best seed");
                    return fallback;
                }
                observe_actual(metrics.as_ref(), "remote", "dispatch", dispatch_started.elapsed());
                let publication_started = Instant::now();
                loop {
                    let query = BundleDiscoveryQuery::new(
                        request.identity.clone(),
                        vec![request.target],
                        unix_time_ms(),
                    );
                    match directory.discover_bundle(query).await {
                        Ok(BundleDiscoveryOutcome::Hit(candidate)) => {
                            if candidate.advertisement().identity() != &request.identity
                                || candidate.advertisement().key() != request.target
                            {
                                record_decision(metrics.as_ref(), "local", "target_incompatible");
                                return fallback;
                            }
                            observe_actual(
                                metrics.as_ref(),
                                "remote",
                                "publication_wait",
                                publication_started.elapsed(),
                            );
                            let pull_started = Instant::now();
                            let outcome = pull_remote_bundle(
                                Arc::clone(&pull_target),
                                *candidate,
                                request.identity.clone(),
                                cancel.clone(),
                                deadline,
                            )
                            .await;
                            observe_actual(metrics.as_ref(), "remote", "pull", pull_started.elapsed());
                            return match outcome {
                                Ok(BundlePullOutcome::Pulled(key)) => Some(key),
                                Ok(BundlePullOutcome::Miss(reason)) => {
                                    record_decision(metrics.as_ref(), "local", reason.as_label());
                                    fallback
                                }
                                Err(error) => {
                                    record_decision(metrics.as_ref(), "local", "transfer_failed");
                                    tracing::warn!(%error, "bundle target pull failed; retaining best seed");
                                    fallback
                                }
                            };
                        }
                        Ok(BundleDiscoveryOutcome::Miss(BundleMissReason::NotFound)) => {
                            tokio::time::sleep(cd.cfg.bundle_prefill_poll).await;
                        }
                        Ok(BundleDiscoveryOutcome::Miss(reason)) => {
                            record_decision(metrics.as_ref(), "local", reason.as_label());
                            return fallback;
                        }
                        Err(error) => {
                            record_decision(metrics.as_ref(), "local", "directory_failed");
                            tracing::warn!(%error, "bundle prefill directory failed; retaining best seed");
                            return fallback;
                        }
                    }
                }
            }) => match result {
                Ok(selected) => selected,
                Err(_) => {
                    record_decision(metrics.as_ref(), "local", "timed_out");
                    fallback
                }
            },
        };
        observe_actual(metrics.as_ref(), "remote", "total", started.elapsed());
        if let Some(metrics) = metrics.as_ref() {
            metrics.observe_disagg_actual(started.elapsed());
        }
        Ok(result)
    }
}

fn observe_estimates(metrics: Option<&BundleMetrics>, costs: decision::PlacementCosts) {
    let Some(metrics) = metrics else {
        return;
    };
    if let Some(local) = costs.local_estimate {
        metrics.observe_disagg_phase_estimate("local", "total", local);
    }
    metrics.observe_disagg_phase_estimate("remote", "total", costs.remote_estimate);
    metrics.observe_disagg_estimate(costs.remote_estimate);
}

fn observe_actual(
    metrics: Option<&BundleMetrics>,
    placement: &'static str,
    phase: &'static str,
    duration: std::time::Duration,
) {
    if let Some(metrics) = metrics {
        metrics.observe_disagg_phase_actual(placement, phase, duration);
    }
}

fn record_decision(metrics: Option<&BundleMetrics>, decision: &'static str, reason: &'static str) {
    if let Some(metrics) = metrics {
        metrics.record_disagg_decision(decision, reason);
    }
}

const fn local_reason(reason: decode::LocalReason) -> &'static str {
    match reason {
        decode::LocalReason::Policy => "policy",
        decode::LocalReason::BreakerHot => "breaker_hot",
        decode::LocalReason::ZeroBlock => "zero_block",
        decode::LocalReason::OverloadFallback => "overload",
        decode::LocalReason::CostGuard => "cost_guard",
    }
}

#[cfg(test)]
mod tests;
