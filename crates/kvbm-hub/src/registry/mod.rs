// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Swappable peer registry backend for the hub.
//!
//! The HTTP handlers and the hub's own `velo::Velo` instance both delegate to
//! the same `Arc<dyn PeerRegistry>`. The default backend is
//! [`InMemoryRegistry`]; future backends (etcd, consul) implement the same
//! trait and plug into [`HubServerBuilder::registry`](crate::HubServerBuilder::registry).
//!
//! Because `PeerRegistry: PeerDiscovery`, an `Arc<dyn PeerRegistry>` coerces
//! to an `Arc<dyn PeerDiscovery>` via Rust stable trait upcasting (1.76+), so
//! the hub's Velo can share the same backend without a separate adapter.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use futures::future::BoxFuture;
use parking_lot::RwLock;
use tokio::task::JoinHandle;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use velo::discovery::PeerDiscovery;
use velo_ext::{InstanceId, PeerInfo, WorkerId};

/// Errors returned by [`PeerRegistry`] mutations.
#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    /// A different `InstanceId` already holds this `worker_id`.
    #[error("worker_id {worker_id} already claimed by instance {existing}")]
    Conflict {
        /// The `WorkerId` that is already claimed.
        worker_id: WorkerId,
        /// The existing `InstanceId` holding that `WorkerId`.
        existing: InstanceId,
    },

    /// Target instance is not registered.
    #[error("instance {0} not found")]
    NotFound(InstanceId),

    /// Opaque backend failure (e.g. etcd connection, network).
    #[error(transparent)]
    Backend(#[from] anyhow::Error),

    /// The authoritative hub removal hook may only be installed once.
    #[error("peer-removal hook is already installed")]
    RemovalHookAlreadyInstalled,

    /// The requested mutation targeted an older incarnation of the instance.
    #[error("instance {instance_id} incarnation changed (expected {expected}, current {current})")]
    StaleIncarnation {
        /// Instance whose registration changed.
        instance_id: InstanceId,
        /// Incarnation captured by the caller.
        expected: RegistryIncarnation,
        /// Incarnation currently stored by the registry.
        current: RegistryIncarnation,
    },
}

/// Opaque identity of one successful registry write.
///
/// Re-registering an `InstanceId` always creates a new incarnation, including
/// rollback writes that restore a prior [`PeerInfo`]. Callers must carry this
/// value through delayed work so it cannot mutate a replacement registration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RegistryIncarnation(u64);

impl RegistryIncarnation {
    /// Construct an incarnation for a custom registry backend. Backends must
    /// never reuse a value for successive writes of the same instance.
    pub const fn from_u64(value: u64) -> Self {
        Self(value)
    }

    /// Return the backend-defined numeric identity.
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for RegistryIncarnation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// One peer and the incarnation captured in the same registry snapshot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegisteredPeer {
    peer: PeerInfo,
    incarnation: RegistryIncarnation,
}

impl RegisteredPeer {
    /// Construct a snapshot entry for a custom registry backend.
    pub fn new(peer: PeerInfo, incarnation: RegistryIncarnation) -> Self {
        Self { peer, incarnation }
    }

    /// Registered peer information.
    pub fn peer(&self) -> &PeerInfo {
        &self.peer
    }

    /// Incarnation paired atomically with [`Self::peer`].
    pub fn incarnation(&self) -> RegistryIncarnation {
        self.incarnation
    }

    /// Consume the snapshot entry and return its peer information.
    pub fn into_peer(self) -> PeerInfo {
        self.peer
    }
}

/// Identity of the exact registry entry that was removed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegistryRemoval {
    instance_id: InstanceId,
    incarnation: RegistryIncarnation,
}

impl RegistryRemoval {
    /// Construct a removal event for a custom registry backend.
    pub const fn new(instance_id: InstanceId, incarnation: RegistryIncarnation) -> Self {
        Self {
            instance_id,
            incarnation,
        }
    }

    /// Removed instance.
    pub const fn instance_id(self) -> InstanceId {
        self.instance_id
    }

    /// Removed incarnation.
    pub const fn incarnation(self) -> RegistryIncarnation {
        self.incarnation
    }
}

/// Registry trait consumed by the hub HTTP handlers and the hub's own Velo.
///
/// Extends `velo::discovery::PeerDiscovery` so `Arc<dyn PeerRegistry>` upcasts
/// cleanly to `Arc<dyn PeerDiscovery>`.
#[async_trait]
pub trait PeerRegistry: PeerDiscovery + Send + Sync {
    /// Register or re-register a peer.
    ///
    /// Re-registering the same `instance_id` replaces its `PeerInfo`, resets
    /// liveness, and returns a fresh [`RegistryIncarnation`]. A different
    /// `instance_id` attempting to claim the same `worker_id` returns
    /// [`RegistryError::Conflict`].
    ///
    /// Once the hub installs its authoritative removal hook, writes must be
    /// routed through the hub registration transaction so credential and
    /// feature authority are committed with the returned incarnation.
    async fn register(&self, peer: PeerInfo) -> Result<RegistryIncarnation, RegistryError>;

    /// Remove a peer. Returns [`RegistryError::NotFound`] if the id was not
    /// registered.
    async fn unregister(
        &self,
        id: InstanceId,
        incarnation: RegistryIncarnation,
    ) -> Result<(), RegistryError>;

    /// Refresh the liveness timestamp for a registered peer. Returns
    /// [`RegistryError::NotFound`] if the id was not registered.
    async fn touch(
        &self,
        id: InstanceId,
        incarnation: RegistryIncarnation,
    ) -> Result<(), RegistryError>;

    /// Whether `incarnation` is still the current registration for `id`.
    fn is_current(&self, id: InstanceId, incarnation: RegistryIncarnation) -> bool;

    /// Capture the current incarnation for an instance, if registered.
    fn current_incarnation(&self, id: InstanceId) -> Option<RegistryIncarnation>;

    /// Is this instance currently registered?
    fn contains(&self, id: InstanceId) -> bool;

    /// Snapshot all registered peers.
    fn list(&self) -> Vec<PeerInfo>;

    /// Snapshot peers and incarnations atomically for delayed work.
    fn registrations(&self) -> Vec<RegisteredPeer>;

    /// Install the hub-owned hook that revokes all authority after a peer is
    /// removed. Installation and the returned snapshot must share one
    /// backend linearization point: every entry present at that point appears
    /// in the snapshot, and every later removal is reported to the callback.
    /// Implementations must invoke the hook exactly once after every later
    /// successful explicit removal and backend-driven expiry, outside locks.
    fn install_removal_hook(
        &self,
        callback: EvictionCallback,
    ) -> Result<Vec<RegisteredPeer>, RegistryError>;

    /// Spawn a backend-specific liveness/reaper task. Returns `None` for
    /// backends with native lease support (etcd). The caller owns the
    /// returned `JoinHandle` and is expected to cancel via the token at
    /// shutdown.
    fn spawn_liveness_task(self: Arc<Self>, _cancel: CancellationToken) -> Option<JoinHandle<()>> {
        None
    }
}

// ---------------------------------------------------------------------------
// In-memory backend
// ---------------------------------------------------------------------------

/// Default in-memory backend. Three maps behind one `parking_lot::RwLock`,
/// TTL-based eviction driven by [`InMemoryRegistry::spawn_liveness_task`].
///
/// Use [`InMemoryRegistry::protect`] to exempt an instance (e.g. the hub's
/// own self-entry) from TTL eviction.
pub struct InMemoryRegistry {
    inner: RwLock<RegistryInner>,
    ttl: Duration,
    prune_interval: Duration,
}

/// Callback invoked whenever an instance leaves the registry (explicit
/// unregister or reaper-driven TTL eviction). Always called **outside** the
/// registry lock. Idempotency is the callback's responsibility.
pub type EvictionCallback = Arc<dyn Fn(RegistryRemoval) + Send + Sync + 'static>;

#[derive(Default)]
struct RegistryInner {
    by_instance: HashMap<InstanceId, RegisteredPeer>,
    by_worker: HashMap<WorkerId, InstanceId>,
    last_seen: HashMap<InstanceId, Instant>,
    protected: HashSet<InstanceId>,
    last_incarnation: u64,
    eviction_callback: Option<EvictionCallback>,
}

impl std::fmt::Debug for InMemoryRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InMemoryRegistry")
            .field("peers", &self.inner.read().by_instance.len())
            .field("ttl", &self.ttl)
            .field("prune_interval", &self.prune_interval)
            .finish()
    }
}

impl InMemoryRegistry {
    /// Builder entry point.
    pub fn builder() -> InMemoryRegistryBuilder {
        InMemoryRegistryBuilder::default()
    }

    /// TTL after which an unfresh entry is eligible for eviction.
    pub fn ttl(&self) -> Duration {
        self.ttl
    }

    /// Interval between reaper ticks.
    pub fn prune_interval(&self) -> Duration {
        self.prune_interval
    }

    /// Protect an instance from TTL eviction. Used by the hub to pin its
    /// own self-entry so it stays discoverable regardless of touches.
    pub fn protect(&self, id: InstanceId) {
        self.inner.write().protected.insert(id);
    }

    /// Install the authoritative removal callback. Called for every instance
    /// removed via explicit unregister or the reaper's TTL pass. Installation
    /// returns the current registrations under the same lock, so callers can
    /// reserve their authority without an expiry gap. The callback is always
    /// invoked outside the registry lock and cannot be replaced.
    fn install_removal_hook(
        &self,
        callback: EvictionCallback,
    ) -> Result<Vec<RegisteredPeer>, RegistryError> {
        let mut inner = self.inner.write();
        if inner.eviction_callback.is_some() {
            return Err(RegistryError::RemovalHookAlreadyInstalled);
        }
        inner.eviction_callback = Some(callback);
        Ok(inner.by_instance.values().cloned().collect())
    }

    /// Immediately evict entries older than `ttl` that are not protected.
    /// Normally called from the reaper task; exposed for tests.
    pub fn prune_stale(&self) {
        let (evicted, callback) = {
            let mut w = self.inner.write();
            let now = Instant::now();
            let ttl = self.ttl;
            let stale: Vec<InstanceId> = w
                .last_seen
                .iter()
                .filter_map(|(id, seen)| {
                    if w.protected.contains(id) {
                        None
                    } else if now.saturating_duration_since(*seen) > ttl {
                        Some(*id)
                    } else {
                        None
                    }
                })
                .collect();
            let mut removals = Vec::with_capacity(stale.len());
            for id in stale {
                if let Some(registered) = w.by_instance.remove(&id) {
                    w.by_worker.retain(|_, owner| *owner != id);
                    removals.push(RegistryRemoval::new(id, registered.incarnation()));
                }
                w.last_seen.remove(&id);
            }
            let callback = w.eviction_callback.as_ref().map(Arc::clone);
            (removals, callback)
        };
        if let Some(cb) = callback {
            for removal in evicted {
                cb(removal);
            }
        }
    }
}

/// Builder for [`InMemoryRegistry`].
pub struct InMemoryRegistryBuilder {
    ttl: Duration,
    prune_interval: Duration,
}

impl Default for InMemoryRegistryBuilder {
    fn default() -> Self {
        Self {
            ttl: Duration::from_secs(30),
            prune_interval: Duration::from_secs(10),
        }
    }
}

impl InMemoryRegistryBuilder {
    /// Liveness TTL (default `30s`). Entries older than this are evicted.
    pub fn ttl(mut self, d: Duration) -> Self {
        self.ttl = d;
        self
    }

    /// Reaper tick interval (default `10s`).
    pub fn prune_interval(mut self, d: Duration) -> Self {
        self.prune_interval = d;
        self
    }

    /// Finalize the registry.
    pub fn build(self) -> InMemoryRegistry {
        InMemoryRegistry {
            inner: RwLock::new(RegistryInner::default()),
            ttl: self.ttl,
            prune_interval: self.prune_interval,
        }
    }
}

impl PeerDiscovery for InMemoryRegistry {
    fn discover_by_worker_id(&self, worker_id: WorkerId) -> BoxFuture<'_, Result<PeerInfo>> {
        let result = {
            let r = self.inner.read();
            match r.by_worker.get(&worker_id).copied() {
                Some(id) => r
                    .by_instance
                    .get(&id)
                    .map(|registered| registered.peer().clone())
                    .ok_or_else(|| {
                        anyhow!(
                            "registry inconsistency: worker {} mapped to missing instance {}",
                            worker_id.as_u64(),
                            id
                        )
                    }),
                None => Err(anyhow!("worker {} not registered", worker_id.as_u64())),
            }
        };
        Box::pin(async move { result })
    }

    fn discover_by_instance_id(&self, id: InstanceId) -> BoxFuture<'_, Result<PeerInfo>> {
        let result = self
            .inner
            .read()
            .by_instance
            .get(&id)
            .map(|registered| registered.peer().clone())
            .ok_or_else(|| anyhow!("instance {id} not registered"));
        Box::pin(async move { result })
    }
}

#[async_trait]
impl PeerRegistry for InMemoryRegistry {
    async fn register(&self, peer: PeerInfo) -> Result<RegistryIncarnation, RegistryError> {
        let id = peer.instance_id();
        let wid = peer.worker_id();
        let mut w = self.inner.write();
        if let Some(existing) = w.by_worker.get(&wid).copied()
            && existing != id
        {
            return Err(RegistryError::Conflict {
                worker_id: wid,
                existing,
            });
        }
        let next = w
            .last_incarnation
            .checked_add(1)
            .ok_or_else(|| RegistryError::Backend(anyhow!("registry incarnation exhausted")))?;
        let incarnation = RegistryIncarnation::from_u64(next);
        w.last_incarnation = next;

        // Re-registration replaces every index claim owned by the prior
        // incarnation. WorkerId is currently derived from InstanceId, but
        // retaining by owner keeps the secondary index correct if that
        // representation changes and repairs any stale rollback claim.
        w.by_worker.retain(|_, owner| *owner != id);
        w.by_worker.insert(wid, id);
        w.by_instance
            .insert(id, RegisteredPeer::new(peer, incarnation));
        w.last_seen.insert(id, Instant::now());
        Ok(incarnation)
    }

    async fn unregister(
        &self,
        id: InstanceId,
        incarnation: RegistryIncarnation,
    ) -> Result<(), RegistryError> {
        let (removal, callback) = {
            let mut w = self.inner.write();
            let registered = w.by_instance.get(&id).ok_or(RegistryError::NotFound(id))?;
            if registered.incarnation() != incarnation {
                return Err(RegistryError::StaleIncarnation {
                    instance_id: id,
                    expected: incarnation,
                    current: registered.incarnation(),
                });
            }
            w.by_instance.remove(&id);
            w.by_worker.retain(|_, owner| *owner != id);
            w.last_seen.remove(&id);
            w.protected.remove(&id);
            let callback = w.eviction_callback.as_ref().map(Arc::clone);
            (RegistryRemoval::new(id, incarnation), callback)
        };
        if let Some(cb) = callback {
            cb(removal);
        }
        Ok(())
    }

    async fn touch(
        &self,
        id: InstanceId,
        incarnation: RegistryIncarnation,
    ) -> Result<(), RegistryError> {
        let mut w = self.inner.write();
        let registered = w.by_instance.get(&id).ok_or(RegistryError::NotFound(id))?;
        if registered.incarnation() != incarnation {
            return Err(RegistryError::StaleIncarnation {
                instance_id: id,
                expected: incarnation,
                current: registered.incarnation(),
            });
        }
        w.last_seen.insert(id, Instant::now());
        Ok(())
    }

    fn is_current(&self, id: InstanceId, incarnation: RegistryIncarnation) -> bool {
        self.inner
            .read()
            .by_instance
            .get(&id)
            .is_some_and(|registered| registered.incarnation() == incarnation)
    }

    fn current_incarnation(&self, id: InstanceId) -> Option<RegistryIncarnation> {
        self.inner
            .read()
            .by_instance
            .get(&id)
            .map(RegisteredPeer::incarnation)
    }

    fn contains(&self, id: InstanceId) -> bool {
        self.inner.read().by_instance.contains_key(&id)
    }

    fn list(&self) -> Vec<PeerInfo> {
        self.inner
            .read()
            .by_instance
            .values()
            .map(|registered| registered.peer().clone())
            .collect()
    }

    fn registrations(&self) -> Vec<RegisteredPeer> {
        self.inner.read().by_instance.values().cloned().collect()
    }

    fn install_removal_hook(
        &self,
        callback: EvictionCallback,
    ) -> Result<Vec<RegisteredPeer>, RegistryError> {
        InMemoryRegistry::install_removal_hook(self, callback)
    }

    fn spawn_liveness_task(self: Arc<Self>, cancel: CancellationToken) -> Option<JoinHandle<()>> {
        let prune_interval = self.prune_interval;
        Some(tokio::spawn(async move {
            let mut ticker = tokio::time::interval(prune_interval);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            ticker.tick().await; // discard the immediate first tick
            loop {
                tokio::select! {
                    _ = cancel.cancelled() => break,
                    _ = ticker.tick() => self.prune_stale(),
                }
            }
        }))
    }
}

#[cfg(test)]
mod tests;
