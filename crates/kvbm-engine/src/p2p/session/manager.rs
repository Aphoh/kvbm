// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! [`SessionManager`] — keeps opened [`Session`]s alive and evicts them
//! when their lifecycle ends.
//!
//! `VeloSessionFactory` returns an `Arc<dyn Session>` and immediately forgets
//! it — the caller owns the only handle. A handler that opens a session and
//! returns must therefore park the session somewhere, or it tears down before
//! the peer can attach. `SessionManager` is that home: it holds the
//! `Arc<dyn Session>` in a map keyed by [`SessionId`] and spawns a per-session
//! watcher that removes the entry when the session detaches, fails, or a
//! watchdog timeout elapses.
//!
//! This mirrors the connector coordinator's `spawn_lifecycle_watcher` pattern
//! (`kvbm-connector/.../disagg/lifecycle.rs`) but stands alone; unifying the
//! two is a flagged follow-up.

use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use futures::StreamExt;
use tokio::runtime::Handle;

use super::{LifecycleEvent, Session, SessionId};

/// Default watchdog: evict a session that has neither detached nor failed
/// within this window. Guards against callers that open a session but whose
/// peer never attaches.
pub const DEFAULT_SESSION_WATCHDOG: Duration = Duration::from_secs(30);

/// Tracks live [`Session`]s by [`SessionId`] and auto-evicts them on
/// lifecycle termination.
pub struct SessionManager {
    sessions: DashMap<SessionId, Arc<dyn Session>>,
    runtime: Handle,
    watchdog: Duration,
}

impl SessionManager {
    /// Create a manager that spawns watcher tasks on `runtime` and evicts
    /// un-terminated sessions after `watchdog`.
    pub fn new(runtime: Handle, watchdog: Duration) -> Arc<Self> {
        Arc::new(Self {
            sessions: DashMap::new(),
            runtime,
            watchdog,
        })
    }

    /// Convenience constructor using [`DEFAULT_SESSION_WATCHDOG`].
    pub fn with_default_watchdog(runtime: Handle) -> Arc<Self> {
        Self::new(runtime, DEFAULT_SESSION_WATCHDOG)
    }

    /// Park a session: insert it into the map and spawn a watcher that
    /// evicts it on `Detached` / `Failed` / watchdog timeout.
    pub fn register(self: &Arc<Self>, session: Arc<dyn Session>) {
        self.register_with_watchdog(session, self.watchdog);
    }

    /// Park a session with a caller-requested watchdog, capped by the
    /// manager's configured maximum. A zero-duration request is raised to one
    /// millisecond so every parked session still receives a runnable cleanup
    /// window.
    pub fn register_with_watchdog(
        self: &Arc<Self>,
        session: Arc<dyn Session>,
        requested: Duration,
    ) {
        let session_id = session.session_id();
        self.sessions.insert(session_id, Arc::clone(&session));
        let watchdog = requested.min(self.watchdog).max(Duration::from_millis(1));
        self.spawn_watcher(session_id, session, watchdog);
    }

    /// Look up a live session by id.
    pub fn get(&self, session_id: &SessionId) -> Option<Arc<dyn Session>> {
        self.sessions.get(session_id).map(|e| Arc::clone(&*e))
    }

    /// Remove (and return) a session explicitly. Normally the watcher does
    /// this; callers may use it for early teardown.
    pub fn remove(&self, session_id: &SessionId) -> Option<Arc<dyn Session>> {
        self.sessions.remove(session_id).map(|(_, s)| s)
    }

    /// Number of sessions currently parked.
    pub fn len(&self) -> usize {
        self.sessions.len()
    }

    /// Whether any sessions are parked.
    pub fn is_empty(&self) -> bool {
        self.sessions.is_empty()
    }

    fn spawn_watcher(
        self: &Arc<Self>,
        session_id: SessionId,
        session: Arc<dyn Session>,
        watchdog: Duration,
    ) {
        let manager = Arc::clone(self);
        self.runtime.spawn(async move {
            // Hold `session` for the watcher's lifetime so the map entry is
            // not the only thing keeping it alive while we watch.
            let mut lifecycle = session.lifecycle();
            let outcome: String = loop {
                match tokio::time::timeout(watchdog, lifecycle.next()).await {
                    Ok(Some(LifecycleEvent::Attached { .. })) => continue,
                    Ok(Some(LifecycleEvent::Detached { reason })) => {
                        session.close(Some(format!("session detached: {reason:?}")));
                        break format!("detached ({reason:?})");
                    }
                    Ok(Some(LifecycleEvent::Failed { reason })) => {
                        session.close(Some(format!("session failed: {reason}")));
                        break format!("failed ({reason})");
                    }
                    Ok(None) => {
                        session.close(Some("lifecycle stream ended".to_owned()));
                        break "lifecycle stream ended".to_string();
                    }
                    Err(_) => {
                        session.close(Some("session watchdog timeout".to_owned()));
                        break "watchdog timeout".to_string();
                    }
                }
            };
            while session.has_inflight_pulls() {
                tracing::debug!(
                    %session_id,
                    "SessionManager teardown quarantined for an in-flight pull"
                );
                tokio::time::sleep(watchdog).await;
            }
            manager.sessions.remove(&session_id);
            tracing::info!(%session_id, outcome, "SessionManager evicted session");
            drop(session);
        });
    }
}
