// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Glue between the CD prefill queue and the per-worker
//! [`PrefillExecutionBackend`]s, mediated by the load-aware
//! [`Selector`].
//!
//! `PrefillRouter` implements [`PrefillRequestDispatcher`]: it picks a
//! worker for each dequeued request, spawns the backend `execute()` on
//! a tokio task (holding the per-worker permit and charge guard for the
//! lifetime of execution), and immediately returns
//! [`DispatchOutcome::Accepted`] to the queue pump so the pump can move
//! on. Completion telemetry is logged inside the spawned task.

use std::sync::Arc;

use anyhow::Result;
use futures::future::BoxFuture;

use super::dispatcher::{DispatchOutcome, PrefillRequestDispatcher};
use super::selection::{PickedSlot, Selector};
use crate::protocol::PrefillRequest;

/// CD queue → fleet glue. Owns the [`Selector`] and is itself a
/// [`PrefillRequestDispatcher`] the disagg manager hands queue items to.
pub struct PrefillRouter {
    selector: Arc<Selector>,
}

impl std::fmt::Debug for PrefillRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrefillRouter")
            .field("selector", &self.selector)
            .finish()
    }
}

impl PrefillRouter {
    /// Build a router on top of `selector`.
    pub fn new(selector: Arc<Selector>) -> Arc<Self> {
        Arc::new(Self { selector })
    }

    /// Underlying selector (for the manager's HTTP introspection
    /// endpoints).
    pub fn selector(&self) -> &Arc<Selector> {
        &self.selector
    }
}

impl PrefillRequestDispatcher for PrefillRouter {
    fn dispatch(&self, request: PrefillRequest) -> BoxFuture<'_, Result<DispatchOutcome>> {
        let selector = Arc::clone(&self.selector);
        Box::pin(async move {
            let net_new = selector.config().net_new(&request);

            // Block until some worker has capacity. The await is the
            // backpressure signal back to the CD queue pump (it polls
            // the queue with batch_size=1, so it can't move on until we
            // return).
            let PickedSlot {
                slot,
                permit,
                guard,
            } = selector.pick(net_new).await;

            tracing::info!(
                instance_id = %slot.instance_id,
                backend = slot.backend.label(),
                request_id = %request.request_id,
                net_new,
                "PrefillRouter: dispatching"
            );

            let backend = Arc::clone(&slot.backend);
            tokio::spawn(async move {
                // Permit + guard are dropped at the end of this task —
                // permit releases the per-worker slot, guard
                // decrements counters and wakes the next picker.
                let _permit = permit;
                let _guard = guard;
                let request_id = request.request_id.clone();
                let label = backend.label();
                match backend.execute(request).await {
                    Ok(DispatchOutcome::Accepted) => {
                        tracing::info!(
                            backend = label,
                            request_id = %request_id,
                            "PrefillRouter: completed"
                        );
                    }
                    Ok(DispatchOutcome::Rejected { reason }) => {
                        tracing::warn!(
                            backend = label,
                            request_id = %request_id,
                            reason = %reason,
                            "PrefillRouter: rejected"
                        );
                    }
                    Err(err) => {
                        tracing::error!(
                            backend = label,
                            request_id = %request_id,
                            error = %err,
                            "PrefillRouter: dispatcher error"
                        );
                    }
                }
            });

            Ok(DispatchOutcome::Accepted)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::prefill_router::execution::PrefillExecutionBackend;
    use crate::features::prefill_router::selection::SelectorConfig;
    use anyhow::Result;
    use async_trait::async_trait;
    use kvbm_protocols::disagg::DISAGG_PROTOCOL_VERSION;
    use parking_lot::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;
    use tokio::sync::Notify;
    use velo::Handler;
    use velo::transports::tcp::TcpTransportBuilder;
    use velo_ext::InstanceId;

    use crate::{PREFILL_DISPATCH_HANDLER, PrefillDispatchRequest, PrefillDispatchResponse};

    /// Test backend: records hit counts per `InstanceId`, optional
    /// per-call latency.
    struct MockBackend {
        id: InstanceId,
        latency: Duration,
        hits: Arc<AtomicUsize>,
        per_call_seen: Arc<Mutex<Vec<String>>>,
    }

    #[async_trait]
    impl PrefillExecutionBackend for MockBackend {
        fn instance_id(&self) -> InstanceId {
            self.id
        }
        fn label(&self) -> &'static str {
            "mock"
        }
        async fn execute(&self, req: PrefillRequest) -> Result<DispatchOutcome> {
            self.per_call_seen.lock().push(req.request_id.clone());
            if !self.latency.is_zero() {
                tokio::time::sleep(self.latency).await;
            }
            self.hits.fetch_add(1, Ordering::SeqCst);
            Ok(DispatchOutcome::Accepted)
        }
    }

    fn make_request(id: &str, n_tokens: usize, n_hashes: usize) -> PrefillRequest {
        use kvbm_protocols::disagg::KvHashingRequestEnvelope;
        PrefillRequest {
            protocol_version: DISAGG_PROTOCOL_VERSION,
            request_id: id.to_string(),
            session_id: uuid::Uuid::new_v4(),
            initiator_instance_id: InstanceId::new_v4(),
            decode_endpoint: None,
            token_ids: vec![0u32; n_tokens],
            num_provided_tokens: n_hashes * 16,
            request: KvHashingRequestEnvelope::default(),
            expected_hash_digest: None,
            bundle: None,
        }
    }

    async fn poll_until<F: FnMut() -> bool>(mut p: F, timeout: Duration) -> bool {
        let deadline = std::time::Instant::now() + timeout;
        while std::time::Instant::now() < deadline {
            if p() {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        p()
    }

    fn add_mock(
        sel: &Arc<Selector>,
        latency: Duration,
    ) -> (InstanceId, Arc<AtomicUsize>, Arc<Mutex<Vec<String>>>) {
        let id = InstanceId::new_v4();
        let hits = Arc::new(AtomicUsize::new(0));
        let seen = Arc::new(Mutex::new(Vec::new()));
        let backend = Arc::new(MockBackend {
            id,
            latency,
            hits: Arc::clone(&hits),
            per_call_seen: Arc::clone(&seen),
        });
        sel.add_worker(id, backend);
        (id, hits, seen)
    }

    #[tokio::test]
    async fn dispatch_records_and_clears_inflight() {
        let sel = Selector::new(SelectorConfig {
            per_worker_concurrency: 2,
            block_size: 16,
        });
        let (id_a, hits_a, _) = add_mock(&sel, Duration::ZERO);
        let (id_b, hits_b, _) = add_mock(&sel, Duration::ZERO);
        let router = PrefillRouter::new(Arc::clone(&sel));

        for i in 0..20 {
            let req = make_request(&format!("r{i}"), 64, 1);
            let out = router.dispatch(req).await.unwrap();
            assert_eq!(out, DispatchOutcome::Accepted);
        }

        let total = || hits_a.load(Ordering::SeqCst) + hits_b.load(Ordering::SeqCst);
        assert!(
            poll_until(|| total() == 20, Duration::from_secs(5)).await,
            "expected all 20 requests to reach the mock backends; got {}",
            total()
        );

        assert!(
            poll_until(
                || {
                    sel.snapshot().iter().all(|slot| {
                        let c = slot.counters();
                        c.inflight == 0 && c.load_net_new == 0
                    })
                },
                Duration::from_secs(2),
            )
            .await,
            "expected all per-worker counters to drain to zero"
        );

        // Both workers should have received traffic (selection isn't
        // strictly fair on a single sample, but neither should be
        // starved when both are equally available).
        assert!(hits_a.load(Ordering::SeqCst) > 0);
        assert!(hits_b.load(Ordering::SeqCst) > 0);
        // Silence unused-warnings
        let _ = (id_a, id_b);
    }

    #[tokio::test]
    async fn dispatch_blocks_when_fleet_is_full() {
        let sel = Selector::new(SelectorConfig {
            per_worker_concurrency: 1,
            block_size: 16,
        });
        let (_id, hits, _seen) = add_mock(&sel, Duration::from_millis(300));
        let router = PrefillRouter::new(Arc::clone(&sel));

        // First dispatch occupies the only permit immediately.
        let t0 = std::time::Instant::now();
        router.dispatch(make_request("r1", 32, 0)).await.unwrap();
        assert!(
            t0.elapsed() < Duration::from_millis(100),
            "first dispatch should return immediately (just spawned)"
        );

        // Second dispatch must wait for the first POST to complete
        // (drop the permit + guard).
        let t1 = std::time::Instant::now();
        router.dispatch(make_request("r2", 32, 0)).await.unwrap();
        let waited = t1.elapsed();
        assert!(
            waited >= Duration::from_millis(200),
            "second dispatch should have waited for the first; waited only {waited:?}"
        );

        assert!(
            poll_until(|| hits.load(Ordering::SeqCst) == 2, Duration::from_secs(2)).await,
            "expected both calls to land"
        );
    }

    #[tokio::test]
    async fn dispatch_blocks_until_worker_added() {
        let sel = Selector::new(SelectorConfig {
            per_worker_concurrency: 1,
            block_size: 16,
        });
        let router = PrefillRouter::new(Arc::clone(&sel));

        let r = router.dispatch(make_request("r1", 16, 0));
        tokio::pin!(r);
        tokio::select! {
            biased;
            _ = &mut r => panic!("dispatch should not complete without workers"),
            _ = tokio::time::sleep(Duration::from_millis(50)) => {}
        }

        let (_id, hits, _seen) = add_mock(&sel, Duration::ZERO);
        tokio::time::timeout(Duration::from_secs(1), r)
            .await
            .expect("dispatch should complete after worker added")
            .unwrap();
        assert!(poll_until(|| hits.load(Ordering::SeqCst) == 1, Duration::from_secs(1)).await);
    }

    #[tokio::test]
    async fn capacity_stays_charged_past_legacy_velo_timeout_until_worker_terminal() {
        let transport = |listener: std::net::TcpListener| {
            Arc::new(
                TcpTransportBuilder::new()
                    .from_listener(listener)
                    .unwrap()
                    .build()
                    .unwrap(),
            )
        };
        let hub_listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let worker_listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let hub = velo::Velo::builder()
            .add_transport(transport(hub_listener))
            .build()
            .await
            .unwrap();
        let worker = velo::Velo::builder()
            .add_transport(transport(worker_listener))
            .build()
            .await
            .unwrap();
        hub.register_peer(worker.peer_info()).unwrap();
        worker.register_peer(hub.peer_info()).unwrap();

        let started = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let started_for_handler = Arc::clone(&started);
        let release_for_handler = Arc::clone(&release);
        let handler =
            Handler::typed_unary_async::<PrefillDispatchRequest, PrefillDispatchResponse, _, _>(
                PREFILL_DISPATCH_HANDLER,
                move |_ctx| {
                    let started = Arc::clone(&started_for_handler);
                    let release = Arc::clone(&release_for_handler);
                    async move {
                        started.notify_one();
                        release.notified().await;
                        Ok(PrefillDispatchResponse {
                            ok: true,
                            error: None,
                        })
                    }
                },
            )
            .build();
        worker.messenger().register_handler(handler).unwrap();

        let selector = Selector::new(SelectorConfig {
            per_worker_concurrency: 1,
            block_size: 16,
        });
        selector.add_worker(
            worker.instance_id(),
            super::super::execution::VeloExecutionBackend::new(
                worker.instance_id(),
                hub.messenger().clone(),
            ),
        );
        let router = PrefillRouter::new(Arc::clone(&selector));

        router
            .dispatch(make_request("long-running", 64, 0))
            .await
            .unwrap();
        started.notified().await;
        assert_eq!(selector.available_permits(), 0);
        assert_eq!(selector.snapshot()[0].counters().inflight, 1);

        // Freeze only after delivery, then cross the removed 30-second
        // deadline without spending wall-clock time.
        tokio::time::pause();
        tokio::time::advance(Duration::from_secs(31)).await;
        tokio::task::yield_now().await;

        assert_eq!(selector.available_permits(), 0);
        assert_eq!(selector.snapshot()[0].counters().inflight, 1);

        release.notify_one();
        assert!(poll_until(|| selector.available_permits() == 1, Duration::from_secs(1)).await);
        assert_eq!(selector.snapshot()[0].counters().inflight, 0);
    }
}
