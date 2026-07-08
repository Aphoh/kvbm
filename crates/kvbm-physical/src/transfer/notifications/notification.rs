// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transfer completion notification handle.

use anyhow::Result;
use futures::future::{Either, Ready, ready};
use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};
use velo::{Event, EventAwaiter, EventManager};

pub enum TransferAwaiter {
    Local(EventAwaiter),
    // Sync(SyncResult),
}

impl std::future::Future for TransferAwaiter {
    type Output = Result<()>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.get_mut() {
            Self::Local(waiter) => Pin::new(waiter).poll(cx),
            // Self::Sync(sync) => Pin::new(sync).poll(cx),
        }
    }
}

/// Notification handle for an in-progress transfer.
///
/// This object can be awaited to block until the transfer completes.
/// The transfer is tracked by a background handler that polls for completion
/// or processes notification events.
///
/// Uses `futures::Either` to avoid event system overhead for synchronous completions.
/// Pending transfers use `LocalEventWaiter` which avoids heap allocation and repeated
/// DashMap lookups when awaiting.
pub struct TransferCompleteNotification {
    awaiter: Either<Ready<Result<()>>, TransferAwaiter>,
}

impl TransferCompleteNotification {
    /// Create a notification that is already completed (for synchronous transfers).
    ///
    /// This is useful for transfers that complete immediately without needing
    /// background polling, such as memcpy operations.
    ///
    /// This is extremely efficient - no allocations, locks, or event system overhead.
    pub fn completed() -> Self {
        Self {
            awaiter: Either::Left(ready(Ok(()))),
        }
    }

    /// Create a notification from a `LocalEventWaiter`.
    ///
    /// This is the primary way to construct a notification when you already
    /// have an event waiter from the event system.
    pub fn from_awaiter(awaiter: EventAwaiter) -> Self {
        Self {
            awaiter: Either::Right(TransferAwaiter::Local(awaiter)),
        }
    }

    // /// Create a notification from a synchronous active message result.
    // pub fn from_sync_result(sync: SyncResult) -> Self {
    //     Self {
    //         awaiter: Either::Right(TransferAwaiter::Sync(sync)),
    //     }
    // }

    /// Check if the notification can yield the current task.
    ///
    /// The internal ::Left arm is guaranteed to be ready, while the ::Right arm is not.
    pub fn could_yield(&self) -> bool {
        matches!(self.awaiter, Either::Right(_))
    }

    /// Aggregate multiple notifications into one that completes when all are done.
    ///
    /// This is useful when a transfer is split across multiple workers and you want
    /// to wait for all of them to complete.
    ///
    /// # Arguments
    /// * `notifications` - The notifications to aggregate
    /// * `events` - The event system to create the aggregate event
    /// * `runtime` - The tokio runtime handle to spawn the aggregation task
    ///
    /// # Behavior
    /// - If the list is empty, returns an already-completed notification
    /// - If there's only one, returns it directly
    /// - Otherwise, creates a new event and spawns a task to await all notifications
    pub fn aggregate(
        notifications: Vec<Self>,
        events: &Arc<EventManager>,
        runtime: &tokio::runtime::Handle,
    ) -> Result<Self> {
        Self::aggregate_results(notifications.into_iter().map(Ok).collect(), events, runtime)
    }

    /// Aggregate dispatch results without abandoning transfers that launched
    /// before a later synchronous dispatch error.
    ///
    /// Every successful notification is drained. Synchronous dispatch errors
    /// and asynchronous completion errors are combined into one terminal
    /// failure, delivered only after all launched work has settled.
    pub fn aggregate_results(
        results: Vec<Result<Self>>,
        events: &Arc<EventManager>,
        runtime: &tokio::runtime::Handle,
    ) -> Result<Self> {
        let mut notifications = Vec::with_capacity(results.len());
        let mut dispatch_errors = Vec::new();
        for result in results {
            match result {
                Ok(notification) => notifications.push(notification),
                Err(error) => dispatch_errors.push(error),
            }
        }
        if notifications.is_empty() {
            return errors_or_completed(dispatch_errors);
        }
        if notifications.len() == 1 && dispatch_errors.is_empty() {
            return Ok(notifications.into_iter().next().unwrap());
        }

        // Check if all notifications are already complete (no yielding needed)
        if notifications.iter().all(|n| !n.could_yield()) {
            return errors_or_completed(dispatch_errors);
        }

        // Create a new event for the aggregate completion
        let event = events.new_event()?;
        let awaiter = events.awaiter(event.handle())?;

        // Spawn task that awaits all notifications and triggers/poisons the event
        runtime.spawn(await_all_notifications(
            notifications,
            dispatch_errors,
            event,
        ));

        Ok(Self::from_awaiter(awaiter))
    }
}

/// Awaits all transfer notifications and signals completion via the event.
///
/// This function awaits ALL notifications regardless of individual failures,
/// then triggers the event on success or poisons it with error details on failure.
async fn await_all_notifications(
    notifications: Vec<TransferCompleteNotification>,
    mut errors: Vec<anyhow::Error>,
    local_event: Event,
) {
    // Await all notifications, collecting results
    let results: Vec<Result<()>> =
        futures::future::join_all(notifications.into_iter().map(|n| n.into_future())).await;

    // Check for any failures
    errors.extend(results.into_iter().filter_map(|result| result.err()));

    if errors.is_empty() {
        // Ignore trigger error - if event system is shutdown, nothing to do
        let _ = local_event.trigger();
    } else {
        let error_msg = combined_error_message(&errors);
        // Ignore poison error - if event system is shutdown, nothing to do
        let _ = local_event.poison(error_msg);
    }
}

fn errors_or_completed(errors: Vec<anyhow::Error>) -> Result<TransferCompleteNotification> {
    if errors.is_empty() {
        Ok(TransferCompleteNotification::completed())
    } else {
        Err(anyhow::anyhow!(combined_error_message(&errors)))
    }
}

fn combined_error_message(errors: &[anyhow::Error]) -> String {
    errors
        .iter()
        .map(|error| format!("{error:#}"))
        .collect::<Vec<_>>()
        .join("; ")
}

impl std::future::IntoFuture for TransferCompleteNotification {
    type Output = Result<()>;
    type IntoFuture = Either<Ready<Result<()>>, TransferAwaiter>;

    fn into_future(self) -> Self::IntoFuture {
        self.awaiter
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use anyhow::{Result, anyhow};
    use velo::EventManager;

    use super::TransferCompleteNotification;

    #[tokio::test]
    async fn later_dispatch_error_waits_for_earlier_notification_to_drain() -> Result<()> {
        let events = Arc::new(EventManager::local());
        let delayed_event = events.new_event()?;
        let delayed =
            TransferCompleteNotification::from_awaiter(events.awaiter(delayed_event.handle())?);

        let aggregate = TransferCompleteNotification::aggregate_results(
            vec![
                Ok(delayed),
                Err(anyhow!("later synchronous dispatch failed")),
            ],
            &events,
            &tokio::runtime::Handle::current(),
        )?;
        let mut completion = tokio::spawn(aggregate.into_future());

        assert!(
            tokio::time::timeout(Duration::from_millis(20), &mut completion)
                .await
                .is_err(),
            "a synchronous error must not abandon an already-launched transfer"
        );
        delayed_event.trigger()?;
        let failure = tokio::time::timeout(Duration::from_secs(1), completion)
            .await??
            .expect_err("the deferred synchronous dispatch failure must poison completion");
        assert!(
            failure
                .to_string()
                .contains("later synchronous dispatch failed")
        );
        Ok(())
    }

    #[tokio::test]
    async fn dispatch_and_completion_errors_are_combined_after_drain() -> Result<()> {
        let events = Arc::new(EventManager::local());
        let delayed_event = events.new_event()?;
        let delayed =
            TransferCompleteNotification::from_awaiter(events.awaiter(delayed_event.handle())?);

        let aggregate = TransferCompleteNotification::aggregate_results(
            vec![Ok(delayed), Err(anyhow!("synchronous dispatch failure"))],
            &events,
            &tokio::runtime::Handle::current(),
        )?;
        delayed_event.poison("asynchronous completion failure")?;
        let failure = aggregate
            .await
            .expect_err("both terminal failures must poison aggregate completion");
        let message = failure.to_string();
        assert!(message.contains("synchronous dispatch failure"));
        assert!(message.contains("asynchronous completion failure"));
        Ok(())
    }
}
