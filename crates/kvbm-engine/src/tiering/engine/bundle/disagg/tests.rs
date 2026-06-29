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
    BundleKey, CacheManifest, ModelIdentity, ResourceRequirement, ResourceRole,
};
use kvbm_protocols::connector::{
    CacheScope, FindBlocksOutcome, FindBlocksRequest, LeaderEngine, NoopWorkerSink,
};

use super::*;
use crate::G2;
use crate::leader::{InstanceLeader, RemoteBlockDiscovery, RemoteCandidates};
use crate::p2p::session::{MockSessionFactory, SessionFactory};
use crate::remote::cd::budget::TierCell;
use crate::remote::cd::policy::SelectionPolicy;
use crate::remote::cd::{DisaggConfig, wire::PrefillPlane};
use crate::remote::search::bundle::{
    BundleAdvertisement, BundleDiscoveryOutcome, BundleDiscoveryQuery, RemoteBundleCandidate,
    unix_time_ms,
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

struct HitDirectory {
    candidate: RemoteBundleCandidate,
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
    let hashes: Arc<[SequenceHash]> =
        Arc::from([SequenceHash::new(1, None, 1), SequenceHash::new(2, None, 2)]);
    let target_key = target_key(&identity, &hashes);
    engine
        .run_bundle_prefill(
            directory,
            BundlePrefillRequest::new("rq".to_owned(), identity, target_key, hashes, 0, 9),
        )
        .await
}

fn target_key(
    identity: &kvbm_protocols::cache_manifest::CacheIdentity,
    hashes: &[SequenceHash],
) -> BundleKey {
    BundleKey::new(identity, hashes[1], 8).unwrap()
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
    let hashes = [SequenceHash::new(1, None, 1), SequenceHash::new(2, None, 2)];
    assert!(
        engine
            .bundle_index
            .lock()
            .unwrap()
            .lease_exact(&identity, &target_key(&identity, &hashes))
            .is_none()
    );
    assert_eq!(plane.count.load(Ordering::Acquire), 1);
    assert_eq!(engine.cd.as_ref().unwrap().budget.available(), 64);
    Ok(())
}

#[tokio::test]
async fn local_bundle_hit_suppresses_conditional_dispatch() -> Result<()> {
    let plane = RecordingPlane::new(PlaneBehavior::Succeed);
    let engine = engine(Arc::clone(&plane), |_| {}).await;
    let identity = identity();
    let hashes: Arc<[SequenceHash]> =
        Arc::from([SequenceHash::new(1, None, 1), SequenceHash::new(2, None, 2)]);
    let key = target_key(&identity, &hashes);
    engine
        .bundle_index
        .lock()
        .unwrap()
        .commit(&identity, key, 1, [(RESOURCE, Vec::new())])?;
    let request = FindBlocksRequest {
        request_id: "local-hit".into(),
        cache: CacheScope::Manifest(identity),
        sequence_hashes: hashes,
        num_computed_tokens: 0,
        total_tokens: 9,
        transfer_params: None,
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
    let hashes: Arc<[SequenceHash]> =
        Arc::from([SequenceHash::new(1, None, 1), SequenceHash::new(2, None, 2)]);
    let key = target_key(&identity, &hashes);
    let advertisement = BundleAdvertisement::new(
        identity.clone(),
        key,
        1,
        engine.leader.messenger().instance_id(),
        unix_time_ms() + 30_000,
        [RESOURCE],
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
            .bundle_index
            .lock()
            .unwrap()
            .lease_exact(&identity, &key)
            .is_none()
    );
    assert_eq!(plane.count.load(Ordering::Acquire), 1);
    assert_eq!(engine.cd.as_ref().unwrap().budget.available(), 64);
    Ok(())
}

#[tokio::test]
async fn non_target_advertisement_is_ignored_until_the_fallback_timeout() -> Result<()> {
    let plane = RecordingPlane::new(PlaneBehavior::Succeed);
    let engine = engine(Arc::clone(&plane), |_| {}).await;
    let identity = identity();
    let hashes: Arc<[SequenceHash]> =
        Arc::from([SequenceHash::new(1, None, 1), SequenceHash::new(2, None, 2)]);
    let earlier = BundleKey::new(&identity, hashes[0], 4)?;
    let advertisement = BundleAdvertisement::new(
        identity,
        earlier,
        1,
        engine.leader.messenger().instance_id(),
        unix_time_ms() + 30_000,
        [RESOURCE],
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
    assert!(directory.queries.load(Ordering::Acquire) > 1);
    assert_eq!(plane.count.load(Ordering::Acquire), 1);
    assert_eq!(engine.cd.as_ref().unwrap().budget.available(), 64);
    Ok(())
}
