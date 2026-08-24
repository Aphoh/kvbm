// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Static, deterministic cost guard for conditional prefill placement.

use std::time::Duration;

/// Immutable cost assumptions used for one request's placement decision.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CostModel {
    fixed: Duration,
    bytes_per_second: u64,
    tokens_per_second: u64,
    maximum: Option<Duration>,
}

impl CostModel {
    pub(crate) const fn new(
        fixed: Duration,
        bytes_per_second: u64,
        tokens_per_second: u64,
        maximum: Option<Duration>,
    ) -> Self {
        Self {
            fixed,
            bytes_per_second,
            tokens_per_second,
            maximum,
        }
    }

    pub(crate) fn estimate(&self, tokens: usize, bytes: u64) -> Duration {
        let fixed = self.fixed.as_nanos();
        let transfer = estimate_nanos(bytes, self.bytes_per_second);
        let tokens = u64::try_from(tokens).unwrap_or(u64::MAX);
        let compute = estimate_nanos(tokens, self.tokens_per_second);
        let nanos = fixed
            .saturating_add(transfer)
            .saturating_add(compute)
            .min(u128::from(u64::MAX));
        Duration::from_nanos(u64::try_from(nanos).unwrap_or(u64::MAX))
    }

    pub(crate) fn allows(&self, tokens: usize, bytes: u64) -> bool {
        self.maximum
            .is_none_or(|maximum| self.estimate(tokens, bytes) <= maximum)
    }
}

fn estimate_nanos(units: u64, units_per_second: u64) -> u128 {
    if units == 0 || units_per_second == 0 {
        return 0;
    }
    u128::from(units)
        .saturating_mul(1_000_000_000)
        .div_ceil(u128::from(units_per_second))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_combines_fixed_transfer_and_compute_without_floats() {
        let model = CostModel::new(Duration::from_millis(2), 1_000, 500, None);
        assert_eq!(model.estimate(500, 1_000), Duration::from_millis(2_002));
    }
}
