// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::future::BoxFuture;
use kvbm_common::{LogicalResourceId, SequenceHash};
use kvbm_logical::BlockRegistry;
use kvbm_observability::KvbmObservability;
use kvbm_protocols::cache_manifest::{
    BundleKey, BundleResourceLineage, CacheManifest, ModelIdentity, RegistrationEpoch,
    ResourceRequirement, ResourceRole,
};
use kvbm_protocols::connector::{
    CacheScope, FindBlocksOutcome, FindBlocksRequest, LeaderEngine, LocalPrefillEstimate,
    NoopWorkerSink,
};

use super::*;
use crate::G2;
use crate::leader::{InstanceLeader, RemoteBlockDiscovery, RemoteCandidates};
use crate::p2p::session::{MockSessionFactory, SessionFactory};
use crate::remote::cd::budget::TierCell;
use crate::remote::cd::policy::SelectionPolicy;
use crate::remote::cd::{DisaggConfig, wire::PrefillPlane};
use crate::remote::search::bundle::{
    BundleAdvertisement, BundleDiscoveryOutcome, BundleDiscoveryQuery, BundleMissReason,
    RemoteBundleCandidate, unix_time_ms,
};
use crate::testing::managers::TestManagerBuilder;
use crate::testing::messenger::create_messenger_tcp;
use crate::tiering::engine::local::CdRuntime;
use crate::tiering::engine::offload::DisabledOffloadSubmit;

const BLOCK_SIZE: usize = 4;
const RESOURCE: LogicalResourceId = LogicalResourceId(71);

struct RecordingPlane {
    count: AtomicUsize,
    behavior: PlaneBehavior,
    last: Mutex<Option<PrefillDispatch>>,
}

#[derive(Clone, Copy)]
enum PlaneBehavior {
    Succeed,
    Fail,
    Pending,
}

impl RecordingPlane {
    fn new(behavior: PlaneBehavior) -> Arc<Self> {
        Arc::new(Self {
            count: AtomicUsize::new(0),
            behavior,
            last: Mutex::new(None),
        })
    }
}

impl PrefillPlane for RecordingPlane {
    fn dispatch(&self, req: PrefillDispatch) -> BoxFuture<'static, Result<()>> {
        self.count.fetch_add(1, Ordering::AcqRel);
        *self.last.lock().unwrap() = Some(req);
        let behavior = self.behavior;
        Box::pin(async move {
            match behavior {
                PlaneBehavior::Succeed => Ok(()),
                PlaneBehavior::Fail => anyhow::bail!("queue unavailable"),
                PlaneBehavior::Pending => std::future::pending().await,
            }
        })
    }
}

struct MissingDirectory;

struct ErrorDirectory;

struct BlockingMissingDirectory {
    started: Arc<std::sync::atomic::AtomicBool>,
    release: Arc<tokio::sync::Notify>,
}

struct HitDirectory {
    candidate: RemoteBundleCandidate,
    queries: AtomicUsize,
}

struct FailingThenMissingDirectory {
    candidate: Mutex<Option<RemoteBundleCandidate>>,
    queries: AtomicUsize,
}

impl RemoteBlockDiscovery for MissingDirectory {
    fn discover(
        &self,
        _hashes: Vec<SequenceHash>,
    ) -> BoxFuture<'static, Result<Option<RemoteCandidates>>> {
        Box::pin(async { Ok(None) })
    }
}

impl RemoteBlockDiscovery for ErrorDirectory {
    fn discover(
        &self,
        _hashes: Vec<SequenceHash>,
    ) -> BoxFuture<'static, Result<Option<RemoteCandidates>>> {
        Box::pin(async { Ok(None) })
    }

    fn discover_bundle(
        &self,
        _query: BundleDiscoveryQuery,
    ) -> BoxFuture<'static, Result<BundleDiscoveryOutcome>> {
        Box::pin(async { anyhow::bail!("directory unavailable") })
    }
}

impl RemoteBlockDiscovery for BlockingMissingDirectory {
    fn discover(
        &self,
        _hashes: Vec<SequenceHash>,
    ) -> BoxFuture<'static, Result<Option<RemoteCandidates>>> {
        Box::pin(async { Ok(None) })
    }

    fn discover_bundle(
        &self,
        _query: BundleDiscoveryQuery,
    ) -> BoxFuture<'static, Result<BundleDiscoveryOutcome>> {
        let started = Arc::clone(&self.started);
        let release = Arc::clone(&self.release);
        Box::pin(async move {
            started.store(true, Ordering::Release);
            release.notified().await;
            Ok(BundleDiscoveryOutcome::Miss(BundleMissReason::NotFound))
        })
    }
}

impl RemoteBlockDiscovery for HitDirectory {
    fn discover(
        &self,
        _hashes: Vec<SequenceHash>,
    ) -> BoxFuture<'static, Result<Option<RemoteCandidates>>> {
        Box::pin(async { Ok(None) })
    }

    fn discover_bundle(
        &self,
        _query: BundleDiscoveryQuery,
    ) -> BoxFuture<'static, Result<BundleDiscoveryOutcome>> {
        self.queries.fetch_add(1, Ordering::AcqRel);
        let candidate = self.candidate.clone();
        Box::pin(async move { Ok(BundleDiscoveryOutcome::Hit(Box::new(candidate))) })
    }
}

impl RemoteBlockDiscovery for FailingThenMissingDirectory {
    fn discover(
        &self,
        _hashes: Vec<SequenceHash>,
    ) -> BoxFuture<'static, Result<Option<RemoteCandidates>>> {
        Box::pin(async { Ok(None) })
    }

    fn discover_bundle(
        &self,
        _query: BundleDiscoveryQuery,
    ) -> BoxFuture<'static, Result<BundleDiscoveryOutcome>> {
        self.queries.fetch_add(1, Ordering::AcqRel);
        let outcome = self
            .candidate
            .lock()
            .unwrap()
            .take()
            .map(Box::new)
            .map(BundleDiscoveryOutcome::Hit)
            .unwrap_or(BundleDiscoveryOutcome::Miss(BundleMissReason::NotFound));
        Box::pin(async move { Ok(outcome) })
    }
}

fn identity() -> kvbm_protocols::cache_manifest::CacheIdentity {
    CacheManifest::new(
        ModelIdentity::new("bundle-disagg", "v1", [3; 32]).unwrap(),
        "bundle-disagg-v1",
        vec![ResourceRequirement::new(RESOURCE, ResourceRole::PrefixHistory, 4).unwrap()],
        BTreeMap::new(),
    )
    .unwrap()
    .identity()
}

fn lineage_hashes() -> [SequenceHash; 2] {
    let root = SequenceHash::root(1);
    [root, root.extend(2)]
}

async fn engine(
    plane: Arc<RecordingPlane>,
    configure: impl FnOnce(&mut DisaggConfig),
) -> Arc<LocalConnectorEngine> {
    let messenger = create_messenger_tcp().await.unwrap();
    let registry = BlockRegistry::new();
    let manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(8)
            .block_size(BLOCK_SIZE)
            .registry(registry.clone())
            .build(),
    );
    let leader = Arc::new(
        InstanceLeader::builder()
            .messenger(messenger)
            .registry(registry)
            .g2_manager(manager)
            .observability(Arc::new(KvbmObservability::default()))
            .build()
            .unwrap(),
    );
    let mut cfg = DisaggConfig {
        selection: SelectionPolicy::Always,
        max_inflight_remote_prefill_tokens: 64,
        bundle_prefill_timeout: std::time::Duration::from_millis(20),
        bundle_prefill_poll: std::time::Duration::from_millis(1),
        ..DisaggConfig::default()
    };
    configure(&mut cfg);
    let sessions: Arc<dyn SessionFactory> = MockSessionFactory::new();
    let cd = CdRuntime::new(cfg, Arc::new(TierCell::default()), sessions, plane, None);
    LocalConnectorEngine::with_offload_submit(
        leader,
        NoopWorkerSink::new(),
        BLOCK_SIZE,
        true,
        Arc::new(DisabledOffloadSubmit),
        Some(cd),
    )
}

async fn run(engine: Arc<LocalConnectorEngine>) -> Result<Option<BundleKey>> {
    run_with_directory(engine, Arc::new(MissingDirectory)).await
}

async fn run_with_directory(
    engine: Arc<LocalConnectorEngine>,
    directory: RemoteDiscoveryHandle,
) -> Result<Option<BundleKey>> {
    let identity = identity();
    let hashes: Arc<[SequenceHash]> = Arc::from(lineage_hashes());
    let target_key = target_key(&identity, &hashes);
    engine
        .run_bundle_prefill(
            directory,
            BundlePrefillRequest {
                request_id: "rq".to_owned(),
                identity,
                target: target_key,
                seed: None,
                num_computed_tokens: 0,
                total_tokens: 9,
                local_prefill_estimate: Some(LocalPrefillEstimate::from_rate(
                    std::time::Duration::from_secs(60),
                    1,
                )),
            },
            tokio_util::sync::CancellationToken::new(),
            tokio::time::Instant::now() + std::time::Duration::from_secs(1),
        )
        .await
}

fn target_key(
    identity: &kvbm_protocols::cache_manifest::CacheIdentity,
    hashes: &[SequenceHash],
) -> BundleKey {
    BundleKey::new(identity, hashes[1], 8).unwrap()
}

fn advertised_lineage(key: BundleKey, hashes: &[SequenceHash]) -> BundleResourceLineage {
    let count = usize::try_from(key.boundary_tokens() / BLOCK_SIZE as u64).unwrap();
    BundleResourceLineage::new(RESOURCE, hashes[..count].to_vec()).unwrap()
}

fn request(
    request_id: &str,
    identity: kvbm_protocols::cache_manifest::CacheIdentity,
    hashes: Arc<[SequenceHash]>,
    local_prefill: LocalPrefillEstimate,
) -> FindBlocksRequest {
    FindBlocksRequest {
        request_id: request_id.into(),
        cache: CacheScope::Manifest(identity),
        sequence_hashes: hashes,
        num_computed_tokens: 0,
        total_tokens: 9,
        transfer_params: None,
        local_prefill_estimate: Some(local_prefill),
    }
}

fn commit_seed(
    engine: &LocalConnectorEngine,
    identity: &kvbm_protocols::cache_manifest::CacheIdentity,
    hashes: &[SequenceHash],
) -> Result<BundleKey> {
    let seed = BundleKey::new(identity, hashes[0], 4)?;
    engine.bundle_catalog.lock().unwrap().index_mut().commit(
        identity,
        seed,
        1,
        [(RESOURCE, Vec::new())],
    )?;
    Ok(seed)
}

async fn terminal(
    engine: Arc<LocalConnectorEngine>,
    request: &FindBlocksRequest,
    search: &kvbm_protocols::connector::FindBlocksHandle,
) -> Result<FindBlocksOutcome> {
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        loop {
            let outcome = engine.clone().find_blocks(request, Some(search))?;
            if !matches!(outcome, FindBlocksOutcome::Searching { .. }) {
                return Ok::<_, kvbm_protocols::connector::LeaderEngineError>(outcome);
            }
            tokio::task::yield_now().await;
        }
    })
    .await?
    .map_err(anyhow::Error::from)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn local_seed_is_kept_while_remote_search_precedes_dispatch() -> Result<()> {
    let plane = RecordingPlane::new(PlaneBehavior::Fail);
    let engine = engine(Arc::clone(&plane), |_| {}).await;
    let identity = identity();
    let hashes: Arc<[SequenceHash]> = Arc::from(lineage_hashes());
    let seed = commit_seed(&engine, &identity, &hashes)?;
    assert!(
        engine
            .leader
            .set_remote_discovery(Arc::new(MissingDirectory))
    );
    let request = request(
        "seed-before-dispatch",
        identity,
        hashes,
        LocalPrefillEstimate::from_rate(std::time::Duration::from_secs(1), 1),
    );

    let FindBlocksOutcome::Searching {
        minted: Some(search),
    } = engine.clone().find_blocks(&request, None)?
    else {
        anyhow::bail!("the local seed bypassed remote search")
    };
    let FindBlocksOutcome::Resolved { matched_tokens, .. } =
        terminal(Arc::clone(&engine), &request, &search).await?
    else {
        anyhow::bail!("bundle placement did not reach a terminal result")
    };

    assert_eq!(matched_tokens, 4, "dispatch failure must return the seed");
    let dispatch_guard = plane.last.lock().unwrap();
    let dispatch = dispatch_guard.as_ref().expect("dispatch recorded");
    assert_eq!(dispatch.num_provided_tokens, 0);
    let context = dispatch.bundle.as_ref().expect("bundle context");
    assert_eq!(context.initial_bundle(), Some(seed));
    assert_eq!(context.target_bundle().boundary_tokens(), 8);
    assert_eq!(engine.cd.as_ref().unwrap().budget.available(), 64);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropping_pending_search_cancels_dispatch_and_releases_budget() -> Result<()> {
    let plane = RecordingPlane::new(PlaneBehavior::Pending);
    let engine = engine(Arc::clone(&plane), |_| {}).await;
    let identity = identity();
    let hashes: Arc<[SequenceHash]> = Arc::from(lineage_hashes());
    commit_seed(&engine, &identity, &hashes)?;
    assert!(
        engine
            .leader
            .set_remote_discovery(Arc::new(MissingDirectory))
    );
    let request = request(
        "drop-pending",
        identity,
        hashes,
        LocalPrefillEstimate::from_rate(std::time::Duration::from_secs(60), 1),
    );
    let FindBlocksOutcome::Searching {
        minted: Some(search),
    } = engine.clone().find_blocks(&request, None)?
    else {
        anyhow::bail!("remote search did not start")
    };
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        while plane.count.load(Ordering::Acquire) == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await?;
    assert_eq!(engine.cd.as_ref().unwrap().budget.available(), 60);

    drop(search);
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        while engine.cd.as_ref().unwrap().budget.available() != 64 {
            tokio::task::yield_now().await;
        }
    })
    .await?;
    assert!(engine.bundle_searches.is_empty());
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn target_published_locally_during_discovery_suppresses_dispatch() -> Result<()> {
    let plane = RecordingPlane::new(PlaneBehavior::Succeed);
    let engine = engine(Arc::clone(&plane), |_| {}).await;
    let identity = identity();
    let hashes: Arc<[SequenceHash]> = Arc::from(lineage_hashes());
    commit_seed(&engine, &identity, &hashes)?;
    let target = target_key(&identity, &hashes);
    let started = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let release = Arc::new(tokio::sync::Notify::new());
    assert!(
        engine
            .leader
            .set_remote_discovery(Arc::new(BlockingMissingDirectory {
                started: Arc::clone(&started),
                release: Arc::clone(&release),
            },))
    );
    let request = request(
        "local-target-race",
        identity.clone(),
        hashes,
        LocalPrefillEstimate::from_rate(std::time::Duration::from_secs(60), 1),
    );
    let FindBlocksOutcome::Searching {
        minted: Some(search),
    } = engine.clone().find_blocks(&request, None)?
    else {
        anyhow::bail!("remote discovery did not start")
    };
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        while !started.load(Ordering::Acquire) {
            tokio::task::yield_now().await;
        }
    })
    .await?;
    engine.bundle_catalog.lock().unwrap().index_mut().commit(
        &identity,
        target,
        2,
        [(RESOURCE, Vec::new())],
    )?;
    release.notify_waiters();

    let FindBlocksOutcome::Resolved { matched_tokens, .. } =
        terminal(Arc::clone(&engine), &request, &search).await?
    else {
        anyhow::bail!("local target did not resolve")
    };
    assert_eq!(matched_tokens, 8);
    assert_eq!(plane.count.load(Ordering::Acquire), 0);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn small_suffix_stays_local_after_remote_search() -> Result<()> {
    let plane = RecordingPlane::new(PlaneBehavior::Succeed);
    let engine = engine(Arc::clone(&plane), |cfg| {
        cfg.cost =
            crate::remote::cd::cost::CostModel::new(std::time::Duration::from_secs(1), 0, 1, None);
    })
    .await;
    let identity = identity();
    let hashes: Arc<[SequenceHash]> = Arc::from(lineage_hashes());
    commit_seed(&engine, &identity, &hashes)?;
    assert!(
        engine
            .leader
            .set_remote_discovery(Arc::new(MissingDirectory))
    );
    let request = request(
        "small-local",
        identity,
        hashes,
        LocalPrefillEstimate::from_rate(std::time::Duration::ZERO, 1_000_000),
    );

    let FindBlocksOutcome::Searching {
        minted: Some(search),
    } = engine.clone().find_blocks(&request, None)?
    else {
        anyhow::bail!("the remote lookup did not run before placement")
    };
    let FindBlocksOutcome::Resolved { matched_tokens, .. } =
        terminal(Arc::clone(&engine), &request, &search).await?
    else {
        anyhow::bail!("bundle placement did not resolve")
    };
    assert_eq!(matched_tokens, 4);
    assert_eq!(plane.count.load(Ordering::Acquire), 0);
    assert_eq!(engine.cd.as_ref().unwrap().budget.available(), 64);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn discovery_error_returns_the_pinned_seed_without_dispatch() -> Result<()> {
    let plane = RecordingPlane::new(PlaneBehavior::Succeed);
    let engine = engine(Arc::clone(&plane), |_| {}).await;
    let identity = identity();
    let hashes: Arc<[SequenceHash]> = Arc::from(lineage_hashes());
    commit_seed(&engine, &identity, &hashes)?;
    assert!(engine.leader.set_remote_discovery(Arc::new(ErrorDirectory)));
    let request = request(
        "directory-error",
        identity,
        hashes,
        LocalPrefillEstimate::from_rate(std::time::Duration::from_secs(1), 1),
    );

    let FindBlocksOutcome::Searching {
        minted: Some(search),
    } = engine.clone().find_blocks(&request, None)?
    else {
        anyhow::bail!("the remote lookup did not start")
    };
    let FindBlocksOutcome::Resolved { matched_tokens, .. } =
        terminal(Arc::clone(&engine), &request, &search).await?
    else {
        anyhow::bail!("bundle placement did not resolve")
    };
    assert_eq!(matched_tokens, 4);
    assert_eq!(plane.count.load(Ordering::Acquire), 0);
    assert_eq!(engine.cd.as_ref().unwrap().budget.available(), 64);
    Ok(())
}

#[tokio::test]
async fn queue_failure_falls_back_and_releases_the_budget() -> Result<()> {
    let plane = RecordingPlane::new(PlaneBehavior::Fail);
    let engine = engine(Arc::clone(&plane), |_| {}).await;

    assert!(run(Arc::clone(&engine)).await?.is_none());
    assert_eq!(plane.count.load(Ordering::Acquire), 1);
    assert_eq!(engine.cd.as_ref().unwrap().budget.available(), 64);
    Ok(())
}

#[tokio::test]
async fn publication_timeout_falls_back_and_releases_the_budget() -> Result<()> {
    let plane = RecordingPlane::new(PlaneBehavior::Succeed);
    let engine = engine(Arc::clone(&plane), |_| {}).await;

    assert!(run(Arc::clone(&engine)).await?.is_none());
    assert_eq!(plane.count.load(Ordering::Acquire), 1);
    let context = plane
        .last
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|dispatch| dispatch.bundle.clone())
        .expect("bundle context dispatched");
    assert_eq!(context.manifest(), identity().manifest());
    assert_eq!(context.target_bundle().boundary_tokens(), 8);
    assert_eq!(engine.cd.as_ref().unwrap().budget.available(), 64);
    Ok(())
}

#[tokio::test]
async fn cost_guard_suppresses_dispatch_before_reserving_budget() -> Result<()> {
    let plane = RecordingPlane::new(PlaneBehavior::Succeed);
    let engine = engine(Arc::clone(&plane), |cfg| {
        cfg.bundle_bytes_per_token = 1_000_000;
        cfg.cost = crate::remote::cd::cost::CostModel::new(
            std::time::Duration::from_millis(1),
            1_000_000,
            0,
            Some(std::time::Duration::from_millis(2)),
        );
    })
    .await;

    assert!(run(Arc::clone(&engine)).await?.is_none());
    assert_eq!(plane.count.load(Ordering::Acquire), 0);
    assert_eq!(engine.cd.as_ref().unwrap().budget.available(), 64);
    Ok(())
}

#[tokio::test]
async fn saved_token_threshold_suppresses_dispatch_before_reserving_budget() -> Result<()> {
    let plane = RecordingPlane::new(PlaneBehavior::Succeed);
    let engine = engine(Arc::clone(&plane), |cfg| {
        cfg.selection = SelectionPolicy::Threshold {
            min_remote_prefill_tokens: 10,
        };
    })
    .await;

    assert!(run(Arc::clone(&engine)).await?.is_none());
    assert_eq!(plane.count.load(Ordering::Acquire), 0);
    assert_eq!(engine.cd.as_ref().unwrap().budget.available(), 64);
    Ok(())
}

#[tokio::test]
async fn actual_cost_is_observed_without_revising_the_remote_decision() -> Result<()> {
    let plane = RecordingPlane::new(PlaneBehavior::Succeed);
    let engine = engine(Arc::clone(&plane), |_| {}).await;

    assert!(run(Arc::clone(&engine)).await?.is_none());
    let families = engine.leader.observability().unwrap().registry().gather();
    for name in [
        "kvbm_disagg_estimated_seconds",
        "kvbm_disagg_actual_seconds",
    ] {
        let family = families
            .iter()
            .find(|family| family.name() == name)
            .expect("disagg histogram registered");
        assert_eq!(family.get_metric()[0].get_histogram().get_sample_count(), 1);
    }
    let decisions = families
        .iter()
        .find(|family| family.name() == "kvbm_disagg_decision_total")
        .expect("disagg decision counter registered");
    let remote = decisions
        .get_metric()
        .iter()
        .find(|metric| {
            metric
                .get_label()
                .iter()
                .any(|label| label.name() == "decision" && label.value() == "remote")
        })
        .expect("remote decision recorded");
    assert_eq!(remote.get_counter().value(), 1.0);
    assert_eq!(plane.count.load(Ordering::Acquire), 1);
    Ok(())
}

#[tokio::test]
async fn queue_timeout_falls_back_without_publishing_a_bundle() -> Result<()> {
    let plane = RecordingPlane::new(PlaneBehavior::Pending);
    let engine = engine(Arc::clone(&plane), |_| {}).await;

    assert!(run(Arc::clone(&engine)).await?.is_none());
    let identity = identity();
    let hashes = lineage_hashes();
    assert!(
        engine
            .bundle_catalog
            .lock()
            .unwrap()
            .lease_exact(&identity, &target_key(&identity, &hashes))
            .is_none()
    );
    assert_eq!(plane.count.load(Ordering::Acquire), 1);
    assert_eq!(engine.cd.as_ref().unwrap().budget.available(), 64);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancellation_releases_budget_and_returns_the_seed() -> Result<()> {
    let plane = RecordingPlane::new(PlaneBehavior::Pending);
    let engine = engine(Arc::clone(&plane), |_| {}).await;
    let identity = identity();
    let hashes: Arc<[SequenceHash]> = Arc::from(lineage_hashes());
    let seed = BundleKey::new(&identity, hashes[0], 4)?;
    let target = target_key(&identity, &hashes);
    let cancel = tokio_util::sync::CancellationToken::new();
    let task = tokio::spawn({
        let engine = Arc::clone(&engine);
        let cancel = cancel.clone();
        async move {
            engine
                .run_bundle_prefill(
                    Arc::new(MissingDirectory),
                    BundlePrefillRequest {
                        request_id: "cancel".to_owned(),
                        identity,
                        target,
                        seed: Some(seed),
                        num_computed_tokens: 0,
                        total_tokens: 9,
                        local_prefill_estimate: Some(LocalPrefillEstimate::from_rate(
                            std::time::Duration::from_secs(60),
                            1,
                        )),
                    },
                    cancel,
                    tokio::time::Instant::now() + std::time::Duration::from_secs(2),
                )
                .await
        }
    });
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        while plane.count.load(Ordering::Acquire) == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await?;
    cancel.cancel();

    assert_eq!(task.await??, Some(seed));
    assert_eq!(engine.cd.as_ref().unwrap().budget.available(), 64);
    Ok(())
}

#[tokio::test]
async fn post_dispatch_directory_error_releases_budget_and_returns_the_seed() -> Result<()> {
    let plane = RecordingPlane::new(PlaneBehavior::Succeed);
    let engine = engine(Arc::clone(&plane), |_| {}).await;
    let identity = identity();
    let hashes: Arc<[SequenceHash]> = Arc::from(lineage_hashes());
    let seed = BundleKey::new(&identity, hashes[0], 4)?;
    let target = target_key(&identity, &hashes);

    let selected = engine
        .clone()
        .run_bundle_prefill(
            Arc::new(ErrorDirectory),
            BundlePrefillRequest {
                request_id: "directory-error-after-dispatch".to_owned(),
                identity,
                target,
                seed: Some(seed),
                num_computed_tokens: 0,
                total_tokens: 9,
                local_prefill_estimate: Some(LocalPrefillEstimate::from_rate(
                    std::time::Duration::from_secs(60),
                    1,
                )),
            },
            tokio_util::sync::CancellationToken::new(),
            tokio::time::Instant::now() + std::time::Duration::from_secs(1),
        )
        .await?;

    assert_eq!(selected, Some(seed));
    assert_eq!(plane.count.load(Ordering::Acquire), 1);
    assert_eq!(engine.cd.as_ref().unwrap().budget.available(), 64);
    Ok(())
}

#[tokio::test]
async fn local_bundle_hit_suppresses_conditional_dispatch() -> Result<()> {
    let plane = RecordingPlane::new(PlaneBehavior::Succeed);
    let engine = engine(Arc::clone(&plane), |_| {}).await;
    let identity = identity();
    let hashes: Arc<[SequenceHash]> = Arc::from(lineage_hashes());
    let key = target_key(&identity, &hashes);
    engine.bundle_catalog.lock().unwrap().index_mut().commit(
        &identity,
        key,
        1,
        [(RESOURCE, Vec::new())],
    )?;
    let request = FindBlocksRequest {
        request_id: "local-hit".into(),
        cache: CacheScope::Manifest(identity),
        sequence_hashes: hashes,
        num_computed_tokens: 0,
        total_tokens: 9,
        transfer_params: None,
        local_prefill_estimate: None,
    };

    let FindBlocksOutcome::Resolved { matched_tokens, .. } =
        engine.clone().find_blocks(&request, None)?
    else {
        anyhow::bail!("local bundle hit did not resolve synchronously");
    };
    assert_eq!(matched_tokens, 8);
    assert_eq!(plane.count.load(Ordering::Acquire), 0);
    Ok(())
}

#[tokio::test]
async fn transport_failure_falls_back_without_poisoning_the_bundle_index() -> Result<()> {
    let plane = RecordingPlane::new(PlaneBehavior::Succeed);
    let engine = engine(Arc::clone(&plane), |_| {}).await;
    let identity = identity();
    let hashes: Arc<[SequenceHash]> = Arc::from(lineage_hashes());
    let key = target_key(&identity, &hashes);
    let advertisement = BundleAdvertisement::new(
        identity.clone(),
        key,
        1,
        engine.leader.messenger().instance_id(),
        engine.leader.registration_epoch(),
        unix_time_ms() + 30_000,
        [advertised_lineage(key, &hashes)],
    )?;
    let directory = Arc::new(HitDirectory {
        candidate: RemoteBundleCandidate::new(
            advertisement,
            uuid::Uuid::new_v4(),
            unix_time_ms() + 20_000,
        )?,
        queries: AtomicUsize::new(0),
    });

    assert!(
        run_with_directory(
            Arc::clone(&engine),
            Arc::clone(&directory) as RemoteDiscoveryHandle,
        )
        .await?
        .is_none()
    );
    assert_eq!(directory.queries.load(Ordering::Acquire), 1);
    assert!(
        engine
            .bundle_catalog
            .lock()
            .unwrap()
            .lease_exact(&identity, &key)
            .is_none()
    );
    assert_eq!(plane.count.load(Ordering::Acquire), 1);
    assert_eq!(engine.cd.as_ref().unwrap().budget.available(), 64);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn failed_remote_candidate_stops_without_retry_or_dispatch() -> Result<()> {
    let plane = RecordingPlane::new(PlaneBehavior::Succeed);
    let engine = engine(Arc::clone(&plane), |_| {}).await;
    let identity = identity();
    let hashes: Arc<[SequenceHash]> = Arc::from(lineage_hashes());
    let key = target_key(&identity, &hashes);
    let advertisement = BundleAdvertisement::new(
        identity.clone(),
        key,
        1,
        crate::InstanceId::new_v4(),
        RegistrationEpoch::new(),
        unix_time_ms() + 30_000,
        [advertised_lineage(key, &hashes)],
    )?;
    let directory = Arc::new(FailingThenMissingDirectory {
        candidate: Mutex::new(Some(RemoteBundleCandidate::new(
            advertisement,
            uuid::Uuid::new_v4(),
            unix_time_ms() + 20_000,
        )?)),
        queries: AtomicUsize::new(0),
    });
    assert!(
        engine
            .leader
            .set_remote_discovery(Arc::clone(&directory) as Arc<dyn RemoteBlockDiscovery>)
    );
    let request = FindBlocksRequest {
        request_id: "failed-remote-stays-local".into(),
        cache: CacheScope::Manifest(identity),
        sequence_hashes: hashes,
        num_computed_tokens: 0,
        total_tokens: 9,
        transfer_params: None,
        local_prefill_estimate: None,
    };

    let FindBlocksOutcome::Searching {
        minted: Some(search),
    } = engine.clone().find_blocks(&request, None)?
    else {
        anyhow::bail!("remote lookup did not start")
    };
    let terminal = tokio::time::timeout(std::time::Duration::from_secs(2), async {
        loop {
            let outcome = engine.clone().find_blocks(&request, Some(&search))?;
            if !matches!(outcome, FindBlocksOutcome::Searching { .. }) {
                return Ok::<_, kvbm_protocols::connector::LeaderEngineError>(outcome);
            }
            tokio::task::yield_now().await;
        }
    })
    .await??;

    let FindBlocksOutcome::Resolved { matched_tokens, .. } = terminal else {
        anyhow::bail!("remote lookup did not resolve")
    };
    assert_eq!(matched_tokens, 0);
    assert_eq!(
        directory.queries.load(Ordering::Acquire),
        1,
        "an integrity/owner/transfer failure is not a clean NotFound"
    );
    assert_eq!(
        plane.count.load(Ordering::Acquire),
        0,
        "a failed remote candidate must force local recompute for the request"
    );
    Ok(())
}

#[tokio::test]
async fn non_target_advertisement_fails_closed_without_polling_forever() -> Result<()> {
    let plane = RecordingPlane::new(PlaneBehavior::Succeed);
    let engine = engine(Arc::clone(&plane), |_| {}).await;
    let identity = identity();
    let hashes: Arc<[SequenceHash]> = Arc::from(lineage_hashes());
    let earlier = BundleKey::new(&identity, hashes[0], 4)?;
    let advertisement = BundleAdvertisement::new(
        identity,
        earlier,
        1,
        engine.leader.messenger().instance_id(),
        engine.leader.registration_epoch(),
        unix_time_ms() + 30_000,
        [advertised_lineage(earlier, &hashes)],
    )?;
    let directory = Arc::new(HitDirectory {
        candidate: RemoteBundleCandidate::new(
            advertisement,
            uuid::Uuid::new_v4(),
            unix_time_ms() + 20_000,
        )?,
        queries: AtomicUsize::new(0),
    });

    assert!(
        run_with_directory(
            Arc::clone(&engine),
            Arc::clone(&directory) as RemoteDiscoveryHandle,
        )
        .await?
        .is_none()
    );
    assert_eq!(directory.queries.load(Ordering::Acquire), 1);
    assert_eq!(plane.count.load(Ordering::Acquire), 1);
    assert_eq!(engine.cd.as_ref().unwrap().budget.available(), 64);
    Ok(())
}
