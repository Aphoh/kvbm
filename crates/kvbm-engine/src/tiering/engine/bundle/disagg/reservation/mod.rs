// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! RAII ownership for an already-admitted conditional-prefill budget.

use std::sync::Arc;

use crate::remote::cd::budget::InflightBudget;

/// Releases exactly one successful token reservation on every exit path.
pub(super) struct BudgetReservation {
    budget: Arc<InflightBudget>,
    tokens: usize,
}

impl BudgetReservation {
    pub(super) fn new(budget: Arc<InflightBudget>, tokens: usize) -> Self {
        Self { budget, tokens }
    }
}

impl Drop for BudgetReservation {
    fn drop(&mut self) {
        self.budget.release(self.tokens);
    }
}
