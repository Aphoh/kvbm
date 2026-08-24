// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cost comparison for complete-bundle conditional prefill.

use std::time::Duration;

use kvbm_protocols::connector::LocalPrefillEstimate;

use crate::remote::cd::cost::CostModel;

/// Placement outcome and the exact estimates that produced it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PlacementDecision {
    Local {
        reason: LocalPlacementReason,
        costs: PlacementCosts,
    },
    Remote {
        costs: PlacementCosts,
    },
}

/// Honest work and cost inputs retained for metrics and diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PlacementCosts {
    pub(super) local_work_tokens: usize,
    pub(super) remote_work_tokens: usize,
    pub(super) local_estimate: Option<Duration>,
    pub(super) remote_estimate: Duration,
}

/// Immutable inputs for one target `B2` and best complete seed `B`.
pub(super) struct PlacementInputs {
    pub(super) target_tokens: usize,
    pub(super) seed_tokens: usize,
    pub(super) computed_tokens: usize,
    pub(super) bundle_bytes: u64,
    pub(super) local: Option<LocalPrefillEstimate>,
    pub(super) remote: CostModel,
    pub(super) margin: Duration,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum LocalPlacementReason {
    MissingLocalEstimate,
    NoRemoteWork,
    CostComparison,
}

impl LocalPlacementReason {
    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::MissingLocalEstimate => "local_estimate_missing",
            Self::NoRemoteWork => "target_ready",
            Self::CostComparison => "local_not_slower",
        }
    }
}

pub(super) fn decide(inputs: PlacementInputs) -> PlacementDecision {
    let seed = inputs.seed_tokens.min(inputs.target_tokens);
    let local_ready = inputs.computed_tokens.max(seed).min(inputs.target_tokens);
    let local_work_tokens = inputs.target_tokens.saturating_sub(local_ready);
    let remote_work_tokens = inputs.target_tokens.saturating_sub(seed);
    let local_estimate = inputs.local.map(|model| model.estimate(local_work_tokens));
    let remote_estimate = inputs
        .remote
        .estimate(remote_work_tokens, inputs.bundle_bytes);
    let costs = PlacementCosts {
        local_work_tokens,
        remote_work_tokens,
        local_estimate,
        remote_estimate,
    };

    if remote_work_tokens == 0 {
        return PlacementDecision::Local {
            reason: LocalPlacementReason::NoRemoteWork,
            costs,
        };
    }
    let Some(local_estimate) = local_estimate else {
        return PlacementDecision::Local {
            reason: LocalPlacementReason::MissingLocalEstimate,
            costs,
        };
    };
    let remote_with_margin = remote_estimate.saturating_add(inputs.margin);
    if remote_with_margin >= local_estimate {
        return PlacementDecision::Local {
            reason: LocalPlacementReason::CostComparison,
            costs,
        };
    }
    PlacementDecision::Remote { costs }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs(remote: Duration, margin: Duration) -> PlacementInputs {
        PlacementInputs {
            target_tokens: 8,
            seed_tokens: 4,
            computed_tokens: 0,
            bundle_bytes: 0,
            local: Some(LocalPrefillEstimate::from_rate(
                Duration::from_millis(10),
                0,
            )),
            remote: CostModel::new(remote, 0, 0, None),
            margin,
        }
    }

    #[test]
    fn equality_stays_local() {
        assert!(matches!(
            decide(inputs(Duration::from_millis(10), Duration::ZERO)),
            PlacementDecision::Local {
                reason: LocalPlacementReason::CostComparison,
                ..
            }
        ));
    }

    #[test]
    fn configured_margin_must_also_be_won() {
        assert!(matches!(
            decide(inputs(Duration::from_millis(9), Duration::from_millis(2))),
            PlacementDecision::Local {
                reason: LocalPlacementReason::CostComparison,
                ..
            }
        ));
        assert!(matches!(
            decide(inputs(Duration::from_millis(9), Duration::ZERO)),
            PlacementDecision::Remote { .. }
        ));
    }

    #[test]
    fn work_is_measured_from_best_seed_and_local_readiness() {
        let mut inputs = inputs(Duration::ZERO, Duration::ZERO);
        inputs.target_tokens = 16;
        inputs.seed_tokens = 8;
        inputs.computed_tokens = 12;
        let PlacementDecision::Remote { costs } = decide(inputs) else {
            panic!("zero-cost remote should win")
        };
        assert_eq!(costs.local_work_tokens, 4);
        assert_eq!(costs.remote_work_tokens, 8);
    }
}
