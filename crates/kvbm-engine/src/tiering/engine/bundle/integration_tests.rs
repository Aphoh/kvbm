// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::disallowed_macros)]

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use futures::future::BoxFuture;
use kvbm_common::{BlockId, LogicalLayoutHandle, LogicalResourceId, SequenceHash};
use kvbm_logical::{BlockManager, BlockManagerSet, BlockRegistry};
use kvbm_physical::TransferOptions;
use kvbm_physical::transfer::TransferCompleteNotification;
use kvbm_protocols::cache_manifest::{
    BundleKey, CacheManifest, ModelIdentity, ResourceRequirement, ResourceRole,
};
use kvbm_protocols::connector::{
    BundleOffloadPlan, BundleOnboardPlan, LeaderEngine, LoadOutcome, OffloadMode, ResourceOffload,
    ResourceOnboard, SaveOutcome,
};

use super::super::local::LocalConnectorEngine;
use super::super::offload::{OffloadSubmit, OffloadTransfer};
use crate::leader::InstanceLeader;
use crate::object::ObjectBlockOps;
use crate::offload::{ExternalBlock, TransferStatus};
use crate::testing::{managers::TestManagerBuilder, messenger::create_messenger_tcp};
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

    let lease = engine
        .bundle_index
        .lock()
        .unwrap()
        .lease_exact(&identity, &key)
        .expect("all three completed children publish one bundle");
    assert_eq!(lease.resources().len(), 3);
    let resources = lease
        .resources()
        .iter()
        .enumerate()
        .map(|(index, (&resource, pins))| ResourceOnboard {
            resource,
            source_block_ids: pins.iter().map(|pin| pin.block_id()).collect(),
            destination_block_ids: vec![100 + index],
        })
        .collect();
    drop(lease);

    let onboard = engine.clone().onboard_resources(
        &"bundle-rq".into(),
        BundleOnboardPlan {
            identity,
            key,
            resources,
        },
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
