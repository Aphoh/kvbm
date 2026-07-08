// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! High-level CPU fault harness for the production complete-bundle transaction.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use futures::future::BoxFuture;
use kvbm_common::{LogicalResourceId, SequenceHash};
use kvbm_protocols::cache_manifest::{BundleKey, CacheIdentity, CacheManifest, ModelIdentity};
use kvbm_protocols::connector::NoopWorkerSink;
use tokio_util::sync::CancellationToken;

use crate::InstanceId;
use crate::leader::{RemoteBlockDiscovery, RemoteCandidates};
use crate::remote::search::bundle::test_support::{
    BLOCK_SIZE, RESOURCES, RecordingParallelWorkers, build_loopback_bundle_fixture,
    pull_with_hanging_transfer,
};
use crate::remote::search::bundle::{
    BundleAdvertisement, BundleDiscoveryOutcome, BundleDiscoveryQuery, BundleMissReason,
    BundlePullOutcome, BundlePullTarget, RemoteBundleCandidate, pull_remote_bundle,
};
use crate::tiering::engine::LocalConnectorEngine;
use crate::worker::group::ParallelWorkers;

/// One injected failure in an otherwise complete production bundle pull.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BundleFault {
    MissingResource(LogicalResourceId),
    ManifestMismatch,
    RemoteTransfer(LogicalResourceId),
    OwnerLost,
    Timeout,
}

/// Scheduler-relevant visibility after a terminal production pull failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BundleFaultOutcome {
    terminal_reason: BundleMissReason,
    catalog_visible: bool,
    g1_visible: bool,
}

impl BundleFaultOutcome {
    pub const fn terminal_reason(self) -> BundleMissReason {
        self.terminal_reason
    }

    pub const fn catalog_visible(self) -> bool {
        self.catalog_visible
    }

    pub const fn g1_visible(self) -> bool {
        self.g1_visible
    }
}

/// Resource ids in the complete bundle exercised by this CPU fixture.
pub const fn manifest_resources() -> [LogicalResourceId; 3] {
    RESOURCES
}

/// Advertise, discover, and attempt one bundle through the production staged
/// pull and local atomic-catalog commit path.
pub async fn run_production_bundle_fault(fault: BundleFault) -> Result<BundleFaultOutcome> {
    validate_fault(fault)?;
    let omitted = match fault {
        BundleFault::MissingResource(resource) => Some(resource),
        _ => None,
    };
    let workers = Arc::new(RecordingParallelWorkers::default());
    let fixture = build_loopback_bundle_fixture(
        omitted,
        Some(Arc::clone(&workers) as Arc<dyn ParallelWorkers>),
    )
    .await?;
    let engine = LocalConnectorEngine::new(
        Arc::clone(&fixture.puller),
        NoopWorkerSink::new(),
        BLOCK_SIZE,
        true,
    );

    let expected_identity = match fault {
        BundleFault::ManifestMismatch => incompatible_identity(&fixture.identity)?,
        _ => fixture.identity.clone(),
    };
    let base_candidate = fixture.candidate();
    let advertisement = match fault {
        BundleFault::OwnerLost => {
            advertisement_with_owner(base_candidate.advertisement(), InstanceId::new_v4())?
        }
        _ => base_candidate.advertisement().clone(),
    };
    let advertised_identity = advertisement.identity().clone();
    let advertised_key = advertisement.key();
    let query_key = BundleKey::new(
        &expected_identity,
        advertised_key.boundary_hash(),
        advertised_key.boundary_tokens(),
    )?;
    let directory = Arc::new(RecordingBundleDirectory::new(matches!(
        fault,
        BundleFault::ManifestMismatch
    )));
    let directory_handle: Arc<dyn RemoteBlockDiscovery> = directory.clone();
    ensure!(
        fixture
            .puller
            .set_remote_discovery(Arc::clone(&directory_handle)),
        "bundle fault fixture could not install its directory"
    );
    directory_handle.advertise_bundle(advertisement).await?;
    let discovered = directory_handle
        .discover_bundle(BundleDiscoveryQuery::new(
            expected_identity.clone(),
            vec![query_key],
            unix_time_ms(),
        ))
        .await?;
    ensure!(
        directory.advertisements.load(Ordering::Acquire) == 1
            && directory.queries.load(Ordering::Acquire) == 1,
        "bundle fault fixture did not traverse advertisement and discovery"
    );
    let candidate = match discovered {
        BundleDiscoveryOutcome::Hit(candidate) => *candidate,
        BundleDiscoveryOutcome::Miss(reason) => {
            anyhow::bail!("fault fixture unexpectedly missed during discovery: {reason:?}")
        }
    };

    let _exhausted_destination = match fault {
        BundleFault::RemoteTransfer(resource) => {
            let manager = fixture
                .puller
                .g2_manager_for(resource)
                .context("fault resource has no destination manager")?;
            let available = manager.available_blocks();
            Some(
                manager
                    .allocate_blocks(available)
                    .context("could not reserve the fault resource destinations")?,
            )
        }
        _ => None,
    };
    let target: Arc<dyn BundlePullTarget> = engine.clone();
    let pull = match fault {
        BundleFault::Timeout => {
            pull_with_hanging_transfer(
                target,
                candidate,
                expected_identity.clone(),
                Duration::from_millis(25),
            )
            .await?
        }
        _ => {
            pull_remote_bundle(
                target,
                candidate,
                expected_identity.clone(),
                CancellationToken::new(),
                tokio::time::Instant::now() + Duration::from_secs(2),
            )
            .await?
        }
    };
    let terminal_reason = match pull {
        BundlePullOutcome::Miss(reason) => reason,
        BundlePullOutcome::Pulled(key) => {
            anyhow::bail!("fault fixture unexpectedly committed bundle {key:?}")
        }
    };

    let catalog_visible = engine.testing_bundle_visible(&advertised_identity, &advertised_key)
        || engine.testing_bundle_visible(&expected_identity, &query_key);
    let staged_visible = fixture.lineages.iter().any(|(&resource, hashes)| {
        fixture
            .puller
            .g2_manager_for(resource)
            .is_some_and(|manager| !manager.match_blocks(hashes).is_empty())
    });
    ensure!(
        !staged_visible,
        "failed bundle left a privately staged resource registry-visible"
    );

    Ok(BundleFaultOutcome {
        terminal_reason,
        catalog_visible,
        g1_visible: !workers.onboards().is_empty(),
    })
}

struct RecordingBundleDirectory {
    candidate: Mutex<Option<RemoteBundleCandidate>>,
    advertisements: AtomicUsize,
    queries: AtomicUsize,
    return_incompatible: bool,
}

impl RecordingBundleDirectory {
    fn new(return_incompatible: bool) -> Self {
        Self {
            candidate: Mutex::new(None),
            advertisements: AtomicUsize::new(0),
            queries: AtomicUsize::new(0),
            return_incompatible,
        }
    }
}

impl RemoteBlockDiscovery for RecordingBundleDirectory {
    fn discover(
        &self,
        _hashes: Vec<SequenceHash>,
    ) -> BoxFuture<'static, Result<Option<RemoteCandidates>>> {
        Box::pin(async { Ok(None) })
    }

    fn discover_bundle(
        &self,
        query: BundleDiscoveryQuery,
    ) -> BoxFuture<'static, Result<BundleDiscoveryOutcome>> {
        self.queries.fetch_add(1, Ordering::AcqRel);
        let candidate = self
            .candidate
            .lock()
            .expect("bundle-directory mutex poisoned")
            .clone();
        let outcome = candidate
            .filter(|candidate| {
                self.return_incompatible || candidate.advertisement().matches(&query)
            })
            .map(Box::new)
            .map(BundleDiscoveryOutcome::Hit)
            .unwrap_or(BundleDiscoveryOutcome::Miss(BundleMissReason::NotFound));
        Box::pin(async move { Ok(outcome) })
    }

    fn advertise_bundle(
        &self,
        advertisement: BundleAdvertisement,
    ) -> BoxFuture<'static, Result<()>> {
        self.advertisements.fetch_add(1, Ordering::AcqRel);
        let candidate = RemoteBundleCandidate::new(
            advertisement.clone(),
            uuid::Uuid::new_v4(),
            advertisement.expires_at_unix_ms(),
        );
        let result = candidate
            .map(|candidate| {
                *self
                    .candidate
                    .lock()
                    .expect("bundle-directory mutex poisoned") = Some(candidate);
            })
            .map_err(anyhow::Error::from);
        Box::pin(async move { result })
    }
}

fn validate_fault(fault: BundleFault) -> Result<()> {
    let resource = match fault {
        BundleFault::MissingResource(resource) | BundleFault::RemoteTransfer(resource) => {
            Some(resource)
        }
        BundleFault::ManifestMismatch | BundleFault::OwnerLost | BundleFault::Timeout => None,
    };
    ensure!(
        resource.is_none_or(|resource| RESOURCES.contains(&resource)),
        "bundle fault names a resource outside the fixture manifest"
    );
    Ok(())
}

fn incompatible_identity(identity: &CacheIdentity) -> Result<CacheIdentity> {
    Ok(CacheManifest::new(
        ModelIdentity::new("bundle-loopback", "incompatible", [0xA5; 32])?,
        "bundle-loopback-v1",
        identity.resources().to_vec(),
        BTreeMap::new(),
    )?
    .identity())
}

fn advertisement_with_owner(
    advertisement: &BundleAdvertisement,
    owner: InstanceId,
) -> Result<BundleAdvertisement> {
    Ok(BundleAdvertisement::new(
        advertisement.identity().clone(),
        advertisement.key(),
        advertisement.generation(),
        owner,
        advertisement.registration_epoch(),
        advertisement.expires_at_unix_ms(),
        advertisement.lineages().cloned(),
    )?)
}

fn unix_time_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0)
}
