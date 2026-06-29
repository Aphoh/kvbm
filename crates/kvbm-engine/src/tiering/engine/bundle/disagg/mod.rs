// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conditional prefill composed over complete-bundle remote search.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use kvbm_observability::BundleMetrics;
use kvbm_protocols::cache_manifest::{BundleKey, CacheIdentity};
use kvbm_protocols::connector::{FindBlocksRequest, LeaderEngineError};
use kvbm_protocols::disagg::BundlePrefillContext;

use crate::SequenceHash;
use crate::leader::RemoteDiscoveryHandle;
use crate::remote::cd::budget::InflightBudget;
use crate::remote::cd::decode::{self, PlanInputs, PlanOutcome};
use crate::remote::cd::wire::PrefillDispatch;
use crate::remote::search::bundle::{
    BundleDiscoveryOutcome, BundleDiscoveryQuery, BundlePullOutcome, BundlePullTarget,
    pull_remote_bundle, unix_time_ms,
};
use crate::tiering::engine::local::LocalConnectorEngine;

pub(in crate::tiering::engine) struct BundlePrefillRequest {
    request_id: String,
    identity: CacheIdentity,
    target: BundleKey,
    sequence_hashes: Arc<[SequenceHash]>,
    num_computed_tokens: usize,
    total_tokens: usize,
}

impl BundlePrefillRequest {
    pub(in crate::tiering::engine) fn new(
        request_id: String,
        identity: CacheIdentity,
        target: BundleKey,
        sequence_hashes: Arc<[SequenceHash]>,
        num_computed_tokens: usize,
        total_tokens: usize,
    ) -> Self {
        Self {
            request_id,
            identity,
            target,
            sequence_hashes,
            num_computed_tokens,
            total_tokens,
        }
    }
}

impl LocalConnectorEngine {
    pub(in crate::tiering::engine) async fn run_bundle_prefill(
        self: Arc<Self>,
        directory: RemoteDiscoveryHandle,
        request: BundlePrefillRequest,
    ) -> Result<Option<BundleKey>> {
        let Some(cd) = self.cd.as_ref() else {
            return Ok(None);
        };
        let BundlePrefillRequest {
            request_id,
            identity,
            target,
            sequence_hashes,
            num_computed_tokens,
            total_tokens,
        } = request;
        let estimated_bytes = target
            .boundary_tokens()
            .saturating_mul(cd.cfg.bundle_bytes_per_token);
        let inputs = PlanInputs {
            total_tokens,
            num_computed_tokens,
            matched_tokens: 0,
            block_size: self.block_size,
            bundle_bytes: estimated_bytes,
        };
        let metrics = self
            .leader
            .observability()
            .map(|observability| observability.bundle_metrics().clone());
        let reserved_tokens = match decode::plan(&cd.cfg, &cd.tier, &cd.budget, &inputs) {
            PlanOutcome::Remote {
                full_block_external_tokens,
            } => {
                record_decision(metrics.as_ref(), "remote", "bundle_cost_accepted");
                full_block_external_tokens
            }
            PlanOutcome::Local { reason } => {
                record_decision(metrics.as_ref(), "local", local_reason(reason));
                return Ok(None);
            }
            PlanOutcome::Reject => {
                record_decision(metrics.as_ref(), "local", "budget_rejected");
                return Ok(None);
            }
        };
        let reservation = BudgetReservation::new(Arc::clone(&cd.budget), reserved_tokens);
        let estimate = cd.cfg.cost.estimate(reserved_tokens, estimated_bytes);
        if let Some(metrics) = metrics.as_ref() {
            metrics.observe_disagg_estimate(estimate);
        }
        let started = Instant::now();
        let context = BundlePrefillContext::new(
            identity.manifest(),
            identity
                .resources()
                .iter()
                .map(|requirement| requirement.resource()),
            None,
            target,
            estimated_bytes,
        )?;
        let window_end = usize::try_from(target.boundary_tokens())
            .unwrap_or(usize::MAX)
            .saturating_add(1)
            .min(total_tokens);
        let dispatch = PrefillDispatch {
            request_id,
            session_id: uuid::Uuid::new_v4(),
            decode_endpoint: None,
            num_provided_tokens: 0,
            num_window_tokens: window_end,
            bundle: Some(context),
        };
        let timeout = cd.cfg.bundle_prefill_timeout;
        let poll = cd.cfg.bundle_prefill_poll;
        let plane = Arc::clone(&cd.plane);
        let block_size = self.block_size;
        let pull_target: Arc<dyn BundlePullTarget> = Arc::clone(&self) as Arc<dyn BundlePullTarget>;
        let result = tokio::time::timeout(timeout, async move {
            if let Err(error) = plane.dispatch(dispatch).await {
                tracing::warn!(%error, "bundle prefill dispatch failed; recomputing locally");
                return Ok(None);
            }
            loop {
                let query = BundleDiscoveryQuery::new(
                    identity.clone(),
                    vec![target],
                    unix_time_ms(),
                );
                match directory.discover_bundle(query).await {
                    Ok(BundleDiscoveryOutcome::Hit(candidate)) => {
                        if candidate.advertisement().identity() != &identity
                            || candidate.advertisement().key() != target
                        {
                            tracing::warn!(
                                expected_manifest = %identity.manifest(),
                                expected_boundary = target.boundary_tokens(),
                                "bundle prefill directory returned a non-target advertisement"
                            );
                            tokio::time::sleep(poll).await;
                            continue;
                        }
                        return match pull_remote_bundle(
                            Arc::clone(&pull_target),
                            *candidate,
                            Arc::clone(&sequence_hashes),
                            block_size,
                        )
                        .await
                        {
                            Ok(BundlePullOutcome::Pulled(key)) => Ok(Some(key)),
                            Ok(BundlePullOutcome::Miss(reason)) => {
                                tracing::warn!(
                                    reason = reason.as_label(),
                                    "bundle prefill target pull missed; recomputing locally"
                                );
                                Ok(None)
                            }
                            Err(error) => {
                                tracing::warn!(
                                    %error,
                                    "bundle prefill target pull failed; recomputing locally"
                                );
                                Ok(None)
                            }
                        };
                    }
                    Ok(BundleDiscoveryOutcome::Miss(_)) => tokio::time::sleep(poll).await,
                    Err(error) => {
                        tracing::warn!(%error, "bundle prefill directory failed; recomputing locally");
                        return Ok(None);
                    }
                }
            }
        })
        .await
        .unwrap_or_else(|_| {
            tracing::warn!("bundle prefill timed out; recomputing locally");
            Ok(None)
        });
        if let Some(metrics) = metrics.as_ref() {
            metrics.observe_disagg_actual(started.elapsed());
        }
        drop(reservation);
        result
    }
}

pub(in crate::tiering::engine) fn validate_prefill_context(
    context: &BundlePrefillContext,
    request: &FindBlocksRequest,
    block_size: usize,
) -> Result<(), LeaderEngineError> {
    let identity = request.cache.identity().ok_or_else(|| {
        invalid_prefill("bundle context requires a manifest-scoped cache request")
    })?;
    if context.manifest() != identity.manifest() {
        return Err(invalid_prefill(
            "bundle context manifest does not match the request cache",
        ));
    }
    let expected_resources = identity
        .resources()
        .iter()
        .map(|requirement| requirement.resource())
        .collect::<BTreeSet<_>>();
    if context.resources().collect::<BTreeSet<_>>() != expected_resources {
        return Err(invalid_prefill(
            "bundle context resources do not match the request cache",
        ));
    }
    if let Some(initial) = context.initial_bundle() {
        validate_request_key("initial", initial, identity, request, block_size)?;
    }
    validate_request_key(
        "target",
        context.target_bundle(),
        identity,
        request,
        block_size,
    )
}

fn validate_request_key(
    role: &str,
    key: BundleKey,
    identity: &CacheIdentity,
    request: &FindBlocksRequest,
    block_size: usize,
) -> Result<(), LeaderEngineError> {
    if block_size == 0 || !key.is_compatible_with(identity) {
        return Err(invalid_prefill(format!(
            "bundle {role} is incompatible with the request cache"
        )));
    }
    let boundary = usize::try_from(key.boundary_tokens())
        .map_err(|_| invalid_prefill(format!("bundle {role} boundary does not fit this worker")))?;
    let eligible_blocks = request.total_tokens.saturating_sub(1) / block_size;
    let Some(index) = boundary
        .checked_div(block_size)
        .filter(|_| boundary.is_multiple_of(block_size))
        .and_then(|blocks| blocks.checked_sub(1))
    else {
        return Err(invalid_prefill(format!(
            "bundle {role} boundary is not aligned to the worker block size"
        )));
    };
    if index >= eligible_blocks.min(request.sequence_hashes.len())
        || request.sequence_hashes[index] != key.boundary_hash()
    {
        return Err(invalid_prefill(format!(
            "bundle {role} is not a boundary in the request hash chain"
        )));
    }
    Ok(())
}

fn invalid_prefill(reason: impl Into<String>) -> LeaderEngineError {
    LeaderEngineError::InvalidPrefillRequest {
        reason: reason.into(),
    }
}

struct BudgetReservation {
    budget: Arc<InflightBudget>,
    tokens: usize,
}

impl BudgetReservation {
    fn new(budget: Arc<InflightBudget>, tokens: usize) -> Self {
        Self { budget, tokens }
    }
}

impl Drop for BudgetReservation {
    fn drop(&mut self) {
        self.budget.release(self.tokens);
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
