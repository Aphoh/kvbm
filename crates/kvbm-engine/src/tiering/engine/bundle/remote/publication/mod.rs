// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Owner-local bundle publication renewal and timing.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, RwLock, Weak};
use std::time::Duration;

use futures::future::join_all;
use kvbm_protocols::cache_manifest::{
    BundleKey, BundleResourceLineage, CacheIdentity, RegistrationEpoch,
};
use tokio_util::sync::CancellationToken;

use super::super::BUNDLE_DIRECTORY_TTL_MS;
use crate::InstanceId;
use crate::remote::search::bundle::{BundleAdvertisement, BundleDirectoryError, unix_time_ms};
use crate::tiering::engine::local::LocalConnectorEngine;

type PublicationClock = dyn Fn() -> u64 + Send + Sync;
const BUNDLE_DIRECTORY_REFRESH_INTERVAL: Duration =
    Duration::from_millis(BUNDLE_DIRECTORY_TTL_MS / 2);
const MAX_CONCURRENT_BUNDLE_REFRESHES: usize = 16;

/// One weak-owned refresh loop and its exact current publication set.
pub(in crate::tiering::engine) struct BundlePublicationRuntime {
    publications: Mutex<HashMap<BundleKey, BundlePublication>>,
    clock: RwLock<Arc<PublicationClock>>,
    cancel: CancellationToken,
    started: OnceLock<()>,
}

impl BundlePublicationRuntime {
    pub(in crate::tiering::engine) fn new() -> Self {
        Self {
            publications: Mutex::new(HashMap::new()),
            clock: RwLock::new(Arc::new(unix_time_ms)),
            cancel: CancellationToken::new(),
            started: OnceLock::new(),
        }
    }

    pub(super) fn track(
        &self,
        publication: BundlePublication,
        engine: Weak<LocalConnectorEngine>,
        runtime: &tokio::runtime::Handle,
    ) {
        self.publications
            .lock()
            .expect("bundle-publication registry mutex poisoned")
            .insert(publication.key, publication);
        if self.started.set(()).is_ok() {
            runtime.spawn(refresh_loop(engine, self.cancel.clone()));
        }
    }

    pub(super) fn forget(&self, key: &BundleKey, generation: u64) {
        let mut publications = self
            .publications
            .lock()
            .expect("bundle-publication registry mutex poisoned");
        if publications
            .get(key)
            .is_some_and(|publication| publication.generation == generation)
        {
            publications.remove(key);
        }
    }

    pub(super) fn snapshot(&self) -> Vec<BundlePublication> {
        self.publications
            .lock()
            .expect("bundle-publication registry mutex poisoned")
            .values()
            .cloned()
            .collect()
    }

    pub(super) fn now_unix_ms(&self) -> u64 {
        self.clock
            .read()
            .expect("bundle-publication clock lock poisoned")()
    }

    #[cfg(test)]
    pub(in crate::tiering::engine) fn set_clock(&self, clock: Arc<PublicationClock>) {
        *self
            .clock
            .write()
            .expect("bundle-publication clock lock poisoned") = clock;
    }
}

impl Drop for BundlePublicationRuntime {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

/// Immutable data required to reconstruct one exact owner advertisement.
#[derive(Clone)]
pub(super) struct BundlePublication {
    identity: CacheIdentity,
    key: BundleKey,
    generation: u64,
    lineages: Vec<BundleResourceLineage>,
}

impl BundlePublication {
    pub(super) fn new(
        identity: CacheIdentity,
        key: BundleKey,
        generation: u64,
        lineages: Vec<BundleResourceLineage>,
    ) -> Self {
        Self {
            identity,
            key,
            generation,
            lineages,
        }
    }

    pub(super) const fn key(&self) -> &BundleKey {
        &self.key
    }

    pub(super) const fn generation(&self) -> u64 {
        self.generation
    }

    pub(super) fn advertisement(
        &self,
        owner: InstanceId,
        registration_epoch: RegistrationEpoch,
        expires_at_unix_ms: u64,
    ) -> Result<BundleAdvertisement, BundleDirectoryError> {
        BundleAdvertisement::new(
            self.identity.clone(),
            self.key,
            self.generation,
            owner,
            registration_epoch,
            expires_at_unix_ms,
            self.lineages.clone(),
        )
    }
}

async fn refresh_loop(engine: Weak<LocalConnectorEngine>, cancel: CancellationToken) {
    let first_refresh = tokio::time::Instant::now() + BUNDLE_DIRECTORY_REFRESH_INTERVAL;
    let mut ticker = tokio::time::interval_at(first_refresh, BUNDLE_DIRECTORY_REFRESH_INTERVAL);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        tokio::select! {
            _ = cancel.cancelled() => return,
            _ = ticker.tick() => {
                let publications = {
                    let Some(engine) = engine.upgrade() else {
                        return;
                    };
                    engine.bundle_publications.snapshot()
                };
                for batch in publications.chunks(MAX_CONCURRENT_BUNDLE_REFRESHES) {
                    if cancel.is_cancelled() {
                        return;
                    }
                    join_all(batch.iter().map(|publication| {
                        let engine = engine.clone();
                        async move {
                            let Some(engine) = engine.upgrade() else {
                                return;
                            };
                            engine.refresh_bundle_publication(publication).await;
                        }
                    }))
                    .await;
                }
            }
        }
    }
}
