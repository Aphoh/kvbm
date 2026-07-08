// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One search-owned deadline shared by discovery and every candidate pull.

use std::future::Future;
use std::time::Duration;

use tokio_util::sync::CancellationToken;

use super::super::{BundleMissReason, unix_time_ms};

#[derive(Clone, Copy)]
pub(super) struct BundlePullLimits {
    pub(super) cleanup_timeout: Duration,
    pub(super) holder_watchdog: Duration,
}

impl BundlePullLimits {
    pub(super) const PRODUCTION: Self = Self {
        cleanup_timeout: Duration::from_secs(1),
        holder_watchdog: Duration::from_secs(30),
    };

    #[cfg(test)]
    pub(super) fn for_test(operation_timeout: Duration) -> Self {
        Self {
            cleanup_timeout: operation_timeout,
            holder_watchdog: operation_timeout,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct PullDeadline {
    pub(super) at: tokio::time::Instant,
    lease_limited: bool,
}

impl PullDeadline {
    pub(super) fn new(
        lease_expires_at_unix_ms: u64,
        search_deadline: tokio::time::Instant,
    ) -> Self {
        let lease_remaining =
            Duration::from_millis(lease_expires_at_unix_ms.saturating_sub(unix_time_ms()));
        let lease_deadline = tokio::time::Instant::now() + lease_remaining;
        Self {
            at: lease_deadline.min(search_deadline),
            lease_limited: lease_deadline <= search_deadline,
        }
    }

    pub(super) fn remaining(self) -> Duration {
        self.at
            .checked_duration_since(tokio::time::Instant::now())
            .unwrap_or_default()
    }

    pub(super) const fn reason(self, interruption: PullInterruption) -> BundleMissReason {
        match interruption {
            PullInterruption::Canceled => BundleMissReason::Canceled,
            PullInterruption::Deadline if self.lease_limited => BundleMissReason::Expired,
            PullInterruption::Deadline => BundleMissReason::TimedOut,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum PullInterruption {
    Canceled,
    Deadline,
}

pub(super) async fn bounded<F>(
    cancel: &CancellationToken,
    deadline: tokio::time::Instant,
    future: F,
) -> Result<F::Output, PullInterruption>
where
    F: Future,
{
    tokio::select! {
        biased;
        _ = cancel.cancelled() => Err(PullInterruption::Canceled),
        result = tokio::time::timeout_at(deadline, future) => {
            result.map_err(|_| PullInterruption::Deadline)
        }
    }
}
