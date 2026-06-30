// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::disallowed_macros)]

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow};
use futures::future::BoxFuture;
use kvbm_common::{BlockId, LogicalLayoutHandle, LogicalResourceId, SequenceHash};
use kvbm_logical::manager::InactiveBackendConfig;
use kvbm_logical::{BlockManager, BlockManagerSet, BlockRegistry};
use kvbm_physical::TransferOptions;
use kvbm_physical::transfer::TransferCompleteNotification;
use kvbm_protocols::cache_manifest::{
    BundleKey, CacheManifest, ModelIdentity, ResourceRequirement, ResourceRole,
};
use kvbm_protocols::connector::{
    BundleOffloadPlan, CacheScope, FindBlocksOutcome, FindBlocksRequest, LeaderEngine, LoadOutcome,
    OffloadMode, ResourceDestination, ResourceOffload, ResourceOnboard, SaveOutcome,
};

use super::super::local::LocalConnectorEngine;
use super::super::offload::{OffloadSubmit, OffloadTransfer};
use crate::leader::{InstanceLeader, RemoteBlockDiscovery, RemoteCandidates};
use crate::object::ObjectBlockOps;
use crate::offload::{ExternalBlock, TransferStatus};
use crate::remote::search::bundle::{
    BundleAdvertisement, BundleDiscoveryOutcome, BundleDiscoveryQuery, BundleInvalidation,
    BundleMissReason,
};
use crate::testing::{managers::TestManagerBuilder, messenger::create_messenger_tcp};
use crate::tiering::policy::{ResourcePolicies, ResourcePolicy};
use crate::worker::group::ParallelWorkers;
use crate::worker::{
    ConnectRemoteResponse, ImportMetadataResponse, InstanceId, RemoteDescriptor, SerializedLayout,
    SerializedLayoutResponse, Worker, WorkerTransfers,
};
use crate::{G1, G2};

const BLOCK_SIZE: usize = 4;
const RESOURCES: [LogicalResourceId; 3] = [
    LogicalResourceId(10),
    LogicalResourceId(11),
    LogicalResourceId(12),
];

#[derive(Default)]
struct RecordingBundleDirectory {
    advertisements: Mutex<Vec<BundleAdvertisement>>,
    invalidations: Mutex<Vec<BundleInvalidation>>,
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
        _query: BundleDiscoveryQuery,
    ) -> BoxFuture<'static, Result<BundleDiscoveryOutcome>> {
        Box::pin(async { Ok(BundleDiscoveryOutcome::Miss(BundleMissReason::NotFound)) })
    }

    fn advertise_bundle(
        &self,
        advertisement: BundleAdvertisement,
    ) -> BoxFuture<'static, Result<()>> {
        self.advertisements.lock().unwrap().push(advertisement);
        Box::pin(async { Ok(()) })
    }

    fn invalidate_bundle(
        &self,
        invalidation: BundleInvalidation,
    ) -> BoxFuture<'static, Result<()>> {
        self.invalidations.lock().unwrap().push(invalidation);
        Box::pin(async { Ok(()) })
    }
}

struct RegisteringOffloadSubmit {
    managers: BTreeMap<LogicalResourceId, Arc<BlockManager<G2>>>,
}

impl OffloadSubmit for RegisteringOffloadSubmit {
    fn supports_resource(&self, resource: LogicalResourceId) -> bool {
        self.managers.contains_key(&resource)
    }

    fn submit_g1_to_g2(
        &self,
        resource: Option<LogicalResourceId>,
        blocks: Vec<ExternalBlock<G1>>,
        _precondition: Option<velo::EventHandle>,
    ) -> Result<Box<dyn OffloadTransfer>> {
        let resource = resource.ok_or_else(|| anyhow!("bundle child requires a resource"))?;
        let manager = self
            .managers
            .get(&resource)
            .ok_or_else(|| anyhow!("no mock manager for {resource:?}"))?;
        let allocated = manager
            .allocate_blocks(blocks.len())
            .ok_or_else(|| anyhow!("mock G2 manager is full"))?;
        let pins = allocated
            .into_iter()
            .zip(blocks)
            .map(|(block, source)| -> Result<_> {
                let complete = block.stage(source.sequence_hash, manager.block_size())?;
                Ok(manager.register_block(complete))
            })
            .collect::<Result<Vec<_>>>()?;
        drop(pins);
        Ok(Box::new(CompletedTransfer))
    }
}

struct CompletedTransfer;

impl OffloadTransfer for CompletedTransfer {
    fn status(&self) -> TransferStatus {
        TransferStatus::Complete
    }

    fn completed_blocks(&self) -> Vec<BlockId> {
        Vec::new()
    }

    fn failed_blocks(&self) -> Vec<BlockId> {
        Vec::new()
    }

    fn wait_terminal(&self) -> BoxFuture<'static, ()> {
        Box::pin(async {})
    }
}

struct CompletedParallelWorkers;

impl WorkerTransfers for CompletedParallelWorkers {
    fn execute_local_transfer(
        &self,
        _src: LogicalLayoutHandle,
        _dst: LogicalLayoutHandle,
        _src_block_ids: Arc<[BlockId]>,
        _dst_block_ids: Arc<[BlockId]>,
        _options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        Ok(TransferCompleteNotification::completed())
    }

    fn execute_local_transfer_for_resource(
        &self,
        _resource: LogicalResourceId,
        _src: LogicalLayoutHandle,
        _dst: LogicalLayoutHandle,
        _src_block_ids: Arc<[BlockId]>,
        _dst_block_ids: Arc<[BlockId]>,
        _options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        Ok(TransferCompleteNotification::completed())
    }

    fn execute_remote_onboard(
        &self,
        _src: RemoteDescriptor,
        _dst: LogicalLayoutHandle,
        _dst_block_ids: Arc<[BlockId]>,
        _options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        anyhow::bail!("remote onboard is outside this local transaction test")
    }

    fn execute_remote_offload(
        &self,
        _src: LogicalLayoutHandle,
        _src_block_ids: Arc<[BlockId]>,
        _dst: RemoteDescriptor,
        _options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        anyhow::bail!("remote offload is outside this local transaction test")
    }

    fn connect_remote(
        &self,
        _instance_id: InstanceId,
        _metadata: Vec<SerializedLayout>,
    ) -> Result<ConnectRemoteResponse> {
        Ok(ConnectRemoteResponse::ready())
    }

    fn has_remote_metadata(&self, _instance_id: InstanceId) -> bool {
        false
    }

    fn execute_remote_onboard_for_instance(
        &self,
        _instance_id: InstanceId,
        _remote_logical_type: LogicalLayoutHandle,
        _src_block_ids: Vec<BlockId>,
        _dst: LogicalLayoutHandle,
        _dst_block_ids: Arc<[BlockId]>,
        _options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        anyhow::bail!("remote instance onboard is outside this local transaction test")
    }
}

impl ObjectBlockOps for CompletedParallelWorkers {
    fn has_blocks(
        &self,
        keys: Vec<SequenceHash>,
    ) -> BoxFuture<'static, Vec<(SequenceHash, Option<usize>)>> {
        Box::pin(async move { keys.into_iter().map(|key| (key, None)).collect() })
    }

    fn put_blocks(
        &self,
        keys: Vec<SequenceHash>,
        _layout: LogicalLayoutHandle,
        _block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        Box::pin(async move { keys.into_iter().map(Err).collect() })
    }

    fn get_blocks(
        &self,
        keys: Vec<SequenceHash>,
        _layout: LogicalLayoutHandle,
        _block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        Box::pin(async move { keys.into_iter().map(Err).collect() })
    }
}

impl ParallelWorkers for CompletedParallelWorkers {
    fn export_metadata(&self) -> Result<Vec<SerializedLayoutResponse>> {
        Ok(Vec::new())
    }

    fn import_metadata(
        &self,
        _metadata: Vec<SerializedLayout>,
    ) -> Result<Vec<ImportMetadataResponse>> {
        Ok(Vec::new())
    }

    fn worker_count(&self) -> usize {
        0
    }

    fn workers(&self) -> &[Arc<dyn Worker>] {
        &[]
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bundle_offload_publishes_once_and_onboard_requires_its_exact_lease() -> Result<()> {
    let (leader, managers) = build_resource_test_leader().await?;
    let leader = Arc::new(leader);
    let directory = Arc::new(RecordingBundleDirectory::default());
    assert!(leader.set_remote_discovery(Arc::clone(&directory) as Arc<dyn RemoteBlockDiscovery>));
    let engine = LocalConnectorEngine::with_offload_submit(
        leader,
        kvbm_protocols::connector::NoopWorkerSink::new(),
        BLOCK_SIZE,
        true,
        Arc::new(RegisteringOffloadSubmit { managers }),
        None,
    );
    let manifest = manifest()?;
    let identity = manifest.identity();
    let key = BundleKey::new(&identity, hash(1), BLOCK_SIZE as u64)?;
    let offload = engine.clone().offload_bundle(
        &"bundle-rq".into(),
        BundleOffloadPlan {
            identity: identity.clone(),
            key,
            generation: 3,
            mode: OffloadMode::Move,
            resources: RESOURCES
                .iter()
                .enumerate()
                .map(|(index, &resource)| ResourceOffload {
                    resource,
                    blocks: vec![(hash(1), 20 + index)],
                })
                .collect(),
        },
    )?;

    assert!(!offload.is_complete());
    assert!(directory.advertisements.lock().unwrap().is_empty());
    assert!(
        engine
            .bundle_index
            .lock()
            .unwrap()
            .lease_exact(&identity, &key)
            .is_none(),
        "buffering cannot publish a partial bundle"
    );
    kvbm_protocols::connector::WorkerEngineDriver::finish_forward_pass(engine.as_ref(), 0);
    wait_until(|| offload.is_complete()).await;
    assert_eq!(offload.outcome(), Some(SaveOutcome::Done));
    wait_until(|| !directory.advertisements.lock().unwrap().is_empty()).await;
    let advertisements = directory.advertisements.lock().unwrap();
    assert_eq!(advertisements.len(), 1);
    assert_eq!(advertisements[0].key(), key);
    assert_eq!(advertisements[0].resources().collect::<Vec<_>>(), RESOURCES);
    drop(advertisements);
    assert_eq!(
        engine
            .bundle_dependencies
            .lock()
            .unwrap()
            .dependents(RESOURCES[0], hash(1)),
        vec![key],
        "publication installs resource-lineage invalidation before visibility"
    );

    let request = FindBlocksRequest {
        request_id: "bundle-rq".into(),
        cache: CacheScope::Manifest(identity),
        sequence_hashes: Arc::from([hash(1)]),
        num_computed_tokens: 0,
        total_tokens: BLOCK_SIZE + 1,
        transfer_params: None,
    };
    let FindBlocksOutcome::Resolved {
        matched_tokens,
        minted: Some(search),
        ..
    } = engine.clone().find_blocks(&request, None)?
    else {
        anyhow::bail!("committed bundle must resolve through manifest search")
    };
    assert_eq!(matched_tokens, BLOCK_SIZE);
    engine.invalidate_resource_blocks(RESOURCES[2], &[hash(1)]);
    wait_until(|| !directory.invalidations.lock().unwrap().is_empty()).await;
    assert_eq!(directory.invalidations.lock().unwrap()[0].key, key);

    let duplicate = engine.clone().onboard_bundle(
        &search,
        vec![
            ResourceDestination {
                resource: RESOURCES[0],
                block_ids: vec![100],
            },
            ResourceDestination {
                resource: RESOURCES[0],
                block_ids: vec![999],
            },
            ResourceDestination {
                resource: RESOURCES[1],
                block_ids: vec![101],
            },
            ResourceDestination {
                resource: RESOURCES[2],
                block_ids: vec![102],
            },
        ],
        matched_tokens,
    );
    assert!(matches!(
        duplicate,
        Err(kvbm_protocols::connector::LeaderEngineError::InvalidBundleTransfer { .. })
    ));

    let onboard = engine.clone().onboard_bundle(
        &search,
        RESOURCES
            .into_iter()
            .enumerate()
            .map(|(index, resource)| ResourceDestination {
                resource,
                block_ids: vec![100 + index],
            })
            .collect(),
        matched_tokens,
    )?;
    wait_until(|| onboard.is_complete()).await;
    assert_eq!(onboard.outcome(), Some(LoadOutcome::Done));
    assert!(engine.inflight.lock().unwrap().overlaps(&[hash(1)]));
    drop(onboard);
    assert!(!engine.inflight.lock().unwrap().overlaps(&[hash(1)]));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bundle_offload_rejects_incomplete_or_wrong_lineage_children() -> Result<()> {
    let (leader, managers) = build_resource_test_leader().await?;
    let engine = LocalConnectorEngine::with_offload_submit(
        Arc::new(leader),
        kvbm_protocols::connector::NoopWorkerSink::new(),
        BLOCK_SIZE,
        true,
        Arc::new(RegisteringOffloadSubmit { managers }),
        None,
    );
    let manifest = manifest()?;
    let identity = manifest.identity();
    let key = BundleKey::new(&identity, hash(2), (2 * BLOCK_SIZE) as u64)?;
    let result = engine.clone().offload_bundle(
        &"incomplete-bundle".into(),
        BundleOffloadPlan {
            identity: identity.clone(),
            key,
            generation: 1,
            mode: OffloadMode::Move,
            resources: RESOURCES
                .iter()
                .enumerate()
                .map(|(index, &resource)| ResourceOffload {
                    resource,
                    blocks: vec![(hash(2), 20 + index)],
                })
                .collect(),
        },
    );

    assert!(matches!(
        result,
        Err(kvbm_protocols::connector::LeaderEngineError::InvalidBundleTransfer { .. })
    ));
    assert!(engine.offload_buffer.lock().unwrap().is_empty());

    let wrong_lineage = engine.clone().offload_bundle(
        &"wrong-lineage-bundle".into(),
        BundleOffloadPlan {
            identity,
            key,
            generation: 1,
            mode: OffloadMode::Move,
            resources: RESOURCES
                .iter()
                .enumerate()
                .map(|(index, &resource)| ResourceOffload {
                    resource,
                    blocks: if index < 2 {
                        vec![(hash(1), 30 + index), (hash(99), 40 + index)]
                    } else {
                        vec![(hash(99), 40 + index)]
                    },
                })
                .collect(),
        },
    );
    assert!(matches!(
        wrong_lineage,
        Err(kvbm_protocols::connector::LeaderEngineError::InvalidBundleTransfer { .. })
    ));
    assert!(engine.offload_buffer.lock().unwrap().is_empty());
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn explicit_same_request_resource_restore_does_not_require_a_bundle_identity() -> Result<()> {
    let (leader, _) = build_resource_test_leader().await?;
    let engine = LocalConnectorEngine::new(
        Arc::new(leader),
        kvbm_protocols::connector::NoopWorkerSink::new(),
        BLOCK_SIZE,
        false,
    );
    let resources = RESOURCES
        .into_iter()
        .enumerate()
        .map(|(index, resource)| ResourceOnboard {
            resource,
            source_block_ids: vec![index],
            destination_block_ids: vec![100 + index],
        })
        .collect();

    let onboard = engine
        .clone()
        .onboard_resource_blocks(&"same-request".to_owned(), resources)?;
    wait_until(|| onboard.is_complete()).await;
    assert_eq!(onboard.outcome(), Some(LoadOutcome::Done));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn resource_policy_role_must_agree_with_the_manifest() -> Result<()> {
    let (leader, managers) = build_resource_test_leader().await?;
    let mut policies = ResourcePolicies::new();
    policies
        .insert(
            RESOURCES[0],
            ResourcePolicy::new(
                ResourceRole::BoundaryCapsule,
                InactiveBackendConfig::Lru,
                InactiveBackendConfig::Lru,
            ),
        )
        .unwrap();
    let engine = LocalConnectorEngine::with_offload_submit_and_policies(
        Arc::new(leader),
        kvbm_protocols::connector::NoopWorkerSink::new(),
        BLOCK_SIZE,
        true,
        Arc::new(RegisteringOffloadSubmit { managers }),
        None,
        policies,
    );
    let identity = manifest()?.identity();
    let key = BundleKey::new(&identity, hash(1), BLOCK_SIZE as u64)?;

    let result = engine.clone().offload_bundle(
        &"policy-role-mismatch".into(),
        BundleOffloadPlan {
            identity,
            key,
            generation: 1,
            mode: OffloadMode::Move,
            resources: RESOURCES
                .iter()
                .enumerate()
                .map(|(index, &resource)| ResourceOffload {
                    resource,
                    blocks: vec![(hash(1), 50 + index)],
                })
                .collect(),
        },
    );

    assert!(matches!(
        result,
        Err(kvbm_protocols::connector::LeaderEngineError::InvalidBundleTransfer { .. })
    ));
    assert!(engine.offload_buffer.lock().unwrap().is_empty());
    Ok(())
}

async fn build_resource_test_leader() -> Result<(
    InstanceLeader,
    BTreeMap<LogicalResourceId, Arc<BlockManager<G2>>>,
)> {
    let messenger = create_messenger_tcp().await?;
    let mut managers = BTreeMap::new();
    let mut set = BlockManagerSet::new();
    for resource in RESOURCES {
        let manager = Arc::new(
            TestManagerBuilder::<G2>::new()
                .block_count(4)
                .block_size(BLOCK_SIZE)
                .registry(BlockRegistry::new())
                .build(),
        );
        set.insert(resource, Arc::clone(&manager))?;
        managers.insert(resource, manager);
    }
    let leader = InstanceLeader::builder()
        .messenger(messenger)
        .registry(BlockRegistry::new())
        .g2_manager_set(Arc::new(set), RESOURCES[0])
        .parallel_worker(Arc::new(CompletedParallelWorkers))
        .build()?;
    Ok((leader, managers))
}

fn manifest() -> Result<CacheManifest> {
    Ok(CacheManifest::new(
        ModelIdentity::new("hybrid-cache", "revision-a", [5; 32])?,
        "hybrid-cache-v1",
        vec![
            ResourceRequirement::new(RESOURCES[0], ResourceRole::PrefixHistory, 4)?,
            ResourceRequirement::new(RESOURCES[1], ResourceRole::PrefixHistory, 4)?,
            ResourceRequirement::new(RESOURCES[2], ResourceRole::BoundaryCapsule, 4)?,
        ],
        Default::default(),
    )?)
}

fn hash(index: u64) -> SequenceHash {
    SequenceHash::new(index, None, index)
}

async fn wait_until(done: impl Fn() -> bool) {
    for _ in 0..200 {
        if done() {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    panic!("bundle action did not reach a terminal state");
}
