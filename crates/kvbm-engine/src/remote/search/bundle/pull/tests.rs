// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use futures::future::BoxFuture;
use kvbm_logical::{BlockManagerSet, BlockRegistry, KvbmSequenceHashProvider};
use kvbm_protocols::cache_manifest::{
    CacheManifest, ModelIdentity, ResourceRequirement, ResourceRole,
};
use kvbm_protocols::disagg::SessionEndpoint;
use velo::transports::tcp::TcpTransportBuilder;

use crate::G2;
use crate::p2p::session::{MockSessionFactory, SessionFactory};
use crate::testing::managers::TestManagerBuilder;
use crate::testing::token_blocks::create_token_sequence;

use super::*;

const BLOCK_SIZE: usize = 4;
const RESOURCES: [LogicalResourceId; 3] = [
    LogicalResourceId(10),
    LogicalResourceId(11),
    LogicalResourceId(12),
];

fn new_transport() -> Arc<velo::transports::tcp::TcpTransport> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    Arc::new(
        TcpTransportBuilder::new()
            .from_listener(listener)
            .unwrap()
            .build()
            .unwrap(),
    )
}

async fn new_velo() -> Arc<velo::Velo> {
    velo::Velo::builder()
        .add_transport(new_transport())
        .build()
        .await
        .unwrap()
}

fn manifest() -> CacheIdentity {
    CacheManifest::new(
        ModelIdentity::new("bundle-loopback", "v1", [8; 32]).unwrap(),
        "bundle-loopback-v1",
        vec![
            ResourceRequirement::new(RESOURCES[0], ResourceRole::PrefixHistory, 4).unwrap(),
            ResourceRequirement::new(RESOURCES[1], ResourceRole::PrefixHistory, 4).unwrap(),
            ResourceRequirement::new(RESOURCES[2], ResourceRole::BoundaryCapsule, 8).unwrap(),
        ],
        BTreeMap::new(),
    )
    .unwrap()
    .identity()
}

fn manager_set(
    populated: bool,
    omit: Option<LogicalResourceId>,
) -> (Arc<BlockManagerSet<G2>>, BlockRegistry, Arc<[SequenceHash]>) {
    let sequence = create_token_sequence(2, BLOCK_SIZE, 900);
    let hashes = sequence
        .blocks()
        .iter()
        .map(KvbmSequenceHashProvider::kvbm_sequence_hash)
        .collect::<Arc<[_]>>();
    let mut managers = BlockManagerSet::new();
    let mut primary_registry = None;
    for resource in RESOURCES {
        let registry = BlockRegistry::new();
        let manager = Arc::new(
            TestManagerBuilder::<G2>::new()
                .block_count(16)
                .block_size(BLOCK_SIZE)
                .registry(registry.clone())
                .build(),
        );
        if populated && omit != Some(resource) {
            let blocks = if resource == RESOURCES[2] {
                vec![sequence.blocks()[1].clone()]
            } else {
                sequence.blocks().to_vec()
            };
            let completed = manager
                .allocate_blocks(blocks.len())
                .unwrap()
                .into_iter()
                .zip(blocks)
                .map(|(block, tokens)| block.complete(&tokens).unwrap())
                .collect();
            manager.register_blocks(completed);
        }
        if resource == RESOURCES[0] {
            primary_registry = Some(registry);
        }
        managers.insert(resource, manager).unwrap();
    }
    (Arc::new(managers), primary_registry.unwrap(), hashes)
}

async fn leaders(
    omit: Option<LogicalResourceId>,
) -> (
    Arc<InstanceLeader>,
    Arc<InstanceLeader>,
    Arc<[SequenceHash]>,
) {
    let holder_velo = new_velo().await;
    let puller_velo = new_velo().await;
    holder_velo.register_peer(puller_velo.peer_info()).unwrap();
    puller_velo.register_peer(holder_velo.peer_info()).unwrap();
    let (holder_factory, puller_factory) = MockSessionFactory::make_paired();
    let (holder_managers, holder_registry, hashes) = manager_set(true, omit);
    let holder = Arc::new(
        InstanceLeader::builder()
            .messenger(holder_velo.messenger().clone())
            .registry(holder_registry)
            .g2_manager_set(holder_managers, RESOURCES[0])
            .workers(vec![])
            .build()
            .unwrap(),
    );
    holder.set_session_factory(holder_factory as Arc<dyn SessionFactory>);
    let _control = holder.register_control_plane(false, false).unwrap();
    puller_velo
        .messenger()
        .refresh_handlers(holder_velo.instance_id())
        .await
        .unwrap();

    let (puller_managers, puller_registry, _) = manager_set(false, None);
    let puller = Arc::new(
        InstanceLeader::builder()
            .messenger(puller_velo.messenger().clone())
            .registry(puller_registry)
            .g2_manager_set(puller_managers, RESOURCES[0])
            .workers(vec![])
            .build()
            .unwrap(),
    );
    puller.set_session_factory(puller_factory as Arc<dyn SessionFactory>);
    (holder, puller, hashes)
}

struct RecordingTarget {
    leader: Arc<InstanceLeader>,
    committed: AtomicBool,
}

impl BundlePullTarget for RecordingTarget {
    fn instance_leader(&self) -> Arc<InstanceLeader> {
        Arc::clone(&self.leader)
    }

    fn commit_pulled_bundle(
        &self,
        _identity: CacheIdentity,
        _key: BundleKey,
        _generation: u64,
        lineages: BTreeMap<LogicalResourceId, Vec<SequenceHash>>,
    ) -> BoxFuture<'static, Result<()>> {
        let leader = Arc::clone(&self.leader);
        let complete = lineages.into_iter().all(|(resource, hashes)| {
            leader
                .g2_manager_for(resource)
                .is_some_and(|manager| manager.match_blocks(&hashes).len() == hashes.len())
        });
        self.committed.store(complete, Ordering::Release);
        Box::pin(async move {
            anyhow::ensure!(
                complete,
                "bundle became visible before every local resource"
            );
            Ok(())
        })
    }
}

fn candidate(
    holder: &InstanceLeader,
    identity: &CacheIdentity,
    hashes: &[SequenceHash],
) -> RemoteBundleCandidate {
    let key = BundleKey::new(identity, hashes[1], 8).unwrap();
    let advertisement = super::super::BundleAdvertisement::new(
        identity.clone(),
        key,
        4,
        holder.messenger().instance_id(),
        unix_time_ms() + 30_000,
        RESOURCES,
    )
    .unwrap();
    RemoteBundleCandidate::new(advertisement, uuid::Uuid::new_v4(), unix_time_ms() + 20_000)
        .unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn loopback_pulls_every_resource_before_local_commit() -> Result<()> {
    let (holder, puller, hashes) = leaders(None).await;
    let identity = manifest();
    let target = Arc::new(RecordingTarget {
        leader: puller,
        committed: AtomicBool::new(false),
    });
    let expected = candidate(&holder, &identity, &hashes).advertisement().key();

    let result = pull_remote_bundle(
        Arc::clone(&target) as Arc<dyn BundlePullTarget>,
        candidate(&holder, &identity, &hashes),
        Arc::clone(&hashes),
        BLOCK_SIZE,
    )
    .await?;

    assert_eq!(result, BundlePullOutcome::Pulled(expected));
    assert!(target.committed.load(Ordering::Acquire));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn omitted_remote_resource_aborts_without_local_bundle_commit() -> Result<()> {
    let (holder, puller, hashes) = leaders(Some(RESOURCES[1])).await;
    let identity = manifest();
    let target = Arc::new(RecordingTarget {
        leader: puller,
        committed: AtomicBool::new(false),
    });

    let result = pull_remote_bundle(
        Arc::clone(&target) as Arc<dyn BundlePullTarget>,
        candidate(&holder, &identity, &hashes),
        hashes,
        BLOCK_SIZE,
    )
    .await?;

    assert_eq!(
        result,
        BundlePullOutcome::Miss(BundleMissReason::Incomplete)
    );
    assert!(!target.committed.load(Ordering::Acquire));
    Ok(())
}

struct OwnerLossTransfer {
    pulls: AtomicUsize,
}

impl BundleTransfer for OwnerLossTransfer {
    fn open(
        &self,
        resource: LogicalResourceId,
        hashes: Vec<SequenceHash>,
    ) -> BoxFuture<'_, Result<OpenTransferSessionResponse>> {
        Box::pin(async move {
            Ok(OpenTransferSessionResponse::Sync {
                capability: TransferSessionCapability {
                    session_id: uuid::Uuid::new_v4(),
                    instance_id: uuid::Uuid::new_v4().into(),
                    endpoint: SessionEndpoint {
                        kind: "fault".to_owned(),
                        payload: serde_json::Value::Null,
                    },
                    resource,
                },
                committed: hashes,
                breakdown: Default::default(),
            })
        })
    }

    fn pull(&self, _resource: &OpenedResource) -> BoxFuture<'_, Result<()>> {
        let attempt = self.pulls.fetch_add(1, Ordering::AcqRel);
        Box::pin(async move {
            anyhow::ensure!(attempt == 0, "owner lost during bundle pull");
            Ok(())
        })
    }

    fn close(&self, _session_id: uuid::Uuid, _reason: &str) -> BoxFuture<'_, ()> {
        Box::pin(async {})
    }
}

#[tokio::test]
async fn owner_loss_during_pull_aborts_without_local_bundle_commit() -> Result<()> {
    let (holder, puller, hashes) = leaders(None).await;
    let identity = manifest();
    let target = Arc::new(RecordingTarget {
        leader: puller,
        committed: AtomicBool::new(false),
    });
    let transfer = Arc::new(OwnerLossTransfer {
        pulls: AtomicUsize::new(0),
    });

    let result = pull_remote_bundle_with_transfer(
        Arc::clone(&target) as Arc<dyn BundlePullTarget>,
        candidate(&holder, &identity, &hashes),
        hashes,
        BLOCK_SIZE,
        Arc::clone(&transfer) as Arc<dyn BundleTransfer>,
    )
    .await;

    assert!(result.is_err());
    assert_eq!(transfer.pulls.load(Ordering::Acquire), RESOURCES.len());
    assert!(!target.committed.load(Ordering::Acquire));
    Ok(())
}
