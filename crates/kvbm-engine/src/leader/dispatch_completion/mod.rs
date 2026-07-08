// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Draining aggregation for leader-to-worker pull-plan fan-outs.

use std::sync::Arc;

use anyhow::{Context, Result};
use kvbm_physical::transfer::TransferCompleteNotification;
use velo::EventManager;

use crate::p2p::dispatch::WorkerPullPlan;

#[derive(Clone, Copy)]
pub(super) enum PullDispatchKind {
    Strict,
    Replicated,
}

impl PullDispatchKind {
    const fn label(self) -> &'static str {
        match self {
            Self::Strict => "strict",
            Self::Replicated => "replicated",
        }
    }
}

/// Dispatch every plan, retaining synchronous failures alongside successful
/// notifications so the latter always drain before the aggregate fails.
pub(super) fn dispatch_and_aggregate_pull_plans<F>(
    kind: PullDispatchKind,
    worker_count: usize,
    plans: Vec<(usize, WorkerPullPlan)>,
    events: &Arc<EventManager>,
    runtime: &tokio::runtime::Handle,
    mut dispatch: F,
) -> Result<TransferCompleteNotification>
where
    F: FnMut(usize, WorkerPullPlan) -> Result<TransferCompleteNotification>,
{
    let results = plans
        .into_iter()
        .map(|(local_rank, plan)| {
            if local_rank >= worker_count {
                anyhow::bail!(
                    "{} pull selected local rank {local_rank} but only {worker_count} workers are registered",
                    kind.label()
                );
            }
            dispatch(local_rank, plan).with_context(|| {
                format!("{} pull dispatch failed on local rank {local_rank}", kind.label())
            })
        })
        .collect();
    TransferCompleteNotification::aggregate_results(results, events, runtime)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use anyhow::{Result, anyhow};
    use kvbm_common::{LogicalLayoutHandle, LogicalResourceId};

    use super::*;
    use crate::p2p::dispatch::{WirePullOptions, WorkerPullPlan};

    #[tokio::test]
    async fn strict_dispatch_error_waits_for_prior_rank() -> Result<()> {
        assert_dispatch_drains(PullDispatchKind::Strict, "strict").await
    }

    #[tokio::test]
    async fn replicated_dispatch_error_waits_for_prior_rank() -> Result<()> {
        assert_dispatch_drains(PullDispatchKind::Replicated, "replicated").await
    }

    async fn assert_dispatch_drains(kind: PullDispatchKind, label: &str) -> Result<()> {
        let events = Arc::new(EventManager::local());
        let delayed_event = events.new_event()?;
        let mut delayed = Some(TransferCompleteNotification::from_awaiter(
            events.awaiter(delayed_event.handle())?,
        ));
        let plans = vec![(0, plan()), (1, plan())];
        let aggregate = dispatch_and_aggregate_pull_plans(
            kind,
            2,
            plans,
            &events,
            &tokio::runtime::Handle::current(),
            |rank, _| match rank {
                0 => Ok(delayed.take().unwrap()),
                _ => Err(anyhow!("injected synchronous dispatch failure")),
            },
        )?;
        let mut completion = tokio::spawn(aggregate.into_future());
        assert!(
            tokio::time::timeout(Duration::from_millis(20), &mut completion)
                .await
                .is_err()
        );
        delayed_event.trigger()?;
        let failure = tokio::time::timeout(Duration::from_secs(1), completion)
            .await??
            .expect_err("dispatch failure must poison after the prior rank drains");
        let message = failure.to_string();
        assert!(message.contains(label));
        assert!(message.contains("local rank 1"));
        assert!(message.contains("injected synchronous dispatch failure"));
        Ok(())
    }

    fn plan() -> WorkerPullPlan {
        WorkerPullPlan {
            remote_instance: uuid::Uuid::new_v4().into(),
            source_resource: LogicalResourceId::default(),
            source_layout: LogicalLayoutHandle::G2,
            dst_resource: LogicalResourceId::default(),
            dst_layout: LogicalLayoutHandle::G2,
            src_block_ids: vec![],
            dst_block_ids: vec![],
            shards: vec![],
            options: WirePullOptions::default(),
        }
    }
}
