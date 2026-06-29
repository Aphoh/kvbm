// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::disallowed_macros)]

use std::sync::Arc;

use anyhow::Result;
use kvbm_common::{LogicalResourceId, SequenceHash};
use kvbm_logical::{BlockRegistry, ImmutableBlock};
use kvbm_protocols::cache_manifest::{
    BundleKey, CacheIdentity, CacheManifest, ModelIdentity, ResourceRequirement, ResourceRole,
};
use kvbm_protocols::connector::{
    ActionId, CacheScope, FindBlocksHandle, FindBlocksOutcome, FindBlocksRequest, LeaderEngine,
    LeaderEngineError, NoopWorkerSink,
};

use crate::G2;
use crate::leader::InstanceLeader;
use crate::testing::managers::TestManagerBuilder;
use crate::testing::messenger::create_messenger_tcp;
use crate::testing::token_blocks::create_token_sequence;
use crate::tiering::engine::inflight::InflightKey;
use crate::tiering::engine::local::LocalConnectorEngine;

const BLOCK_SIZE: usize = 4;
const RESOURCES: [LogicalResourceId; 2] = [LogicalResourceId(90), LogicalResourceId(91)];

struct FindRig {
    engine: Arc<LocalConnectorEngine>,
    hashes: Vec<SequenceHash>,
    held: Vec<ImmutableBlock<G2>>,
}

async fn find_rig(start: u32) -> Result<FindRig> {
    let messenger = create_messenger_tcp().await?;
    let registry = BlockRegistry::new();
    let manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(7)
            .block_size(BLOCK_SIZE)
            .registry(registry.clone())
            .build(),
    );
    let sequence = create_token_sequence(3, BLOCK_SIZE, start);
    let complete = manager
        .allocate_blocks(3)
        .unwrap()
        .into_iter()
        .zip(sequence.blocks())
        .map(|(block, tokens)| block.complete(tokens).unwrap())
        .collect();
    let held = manager.register_blocks(complete);
    let hashes = held.iter().map(ImmutableBlock::sequence_hash).collect();
    let leader = InstanceLeader::builder()
        .messenger(messenger)
        .registry(registry)
        .g2_manager(manager)
        .build()?;
    Ok(FindRig {
        engine: LocalConnectorEngine::new(
            Arc::new(leader),
            NoopWorkerSink::new(),
            BLOCK_SIZE,
            false,
        ),
        hashes,
        held,
    })
}

fn manifest(revision: &str, resources: &[LogicalResourceId]) -> CacheManifest {
    CacheManifest::new(
        ModelIdentity::new("bundle-router-test", revision, [4; 32]).unwrap(),
        "bundle-router-v1",
        resources
            .iter()
            .enumerate()
            .map(|(index, &resource)| {
                ResourceRequirement::new(
                    resource,
                    if index + 1 == resources.len() {
                        ResourceRole::BoundaryCapsule
                    } else {
                        ResourceRole::PrefixHistory
                    },
                    BLOCK_SIZE as u32,
                )
                .unwrap()
            })
            .collect(),
        Default::default(),
    )
    .unwrap()
}

fn find_request(
    request_id: &str,
    identity: CacheIdentity,
    hashes: Vec<SequenceHash>,
) -> FindBlocksRequest {
    FindBlocksRequest {
        request_id: request_id.to_owned(),
        cache: CacheScope::Manifest(identity),
        sequence_hashes: Arc::from(hashes),
        num_computed_tokens: 0,
        total_tokens: 3 * BLOCK_SIZE + 1,
        transfer_params: None,
    }
}

fn commit(rig: &FindRig, identity: &CacheIdentity) -> BundleKey {
    commit_at(rig, identity, 2)
}

fn commit_at(rig: &FindRig, identity: &CacheIdentity, boundary_blocks: usize) -> BundleKey {
    let key = BundleKey::new(
        identity,
        rig.hashes[boundary_blocks - 1],
        (boundary_blocks * BLOCK_SIZE) as u64,
    )
    .unwrap();
    rig.engine
        .bundle_index
        .lock()
        .unwrap()
        .commit(
            identity,
            key,
            1,
            RESOURCES
                .into_iter()
                .enumerate()
                .map(|(index, resource)| (resource, vec![rig.held[index].clone()])),
        )
        .unwrap();
    key
}

fn resolved(outcome: FindBlocksOutcome) -> (usize, Option<FindBlocksHandle>, bool) {
    match outcome {
        FindBlocksOutcome::Resolved {
            matched_tokens,
            minted,
            release_parked,
        } => (matched_tokens, minted, release_parked),
        other => panic!("expected resolved bundle find, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn changed_window_reconciles_to_one_new_common_boundary() -> Result<()> {
    let rig = find_rig(4_350).await?;
    let identity = manifest("v1", &RESOURCES).identity();
    commit_at(&rig, &identity, 2);
    let later = commit_at(&rig, &identity, 3);
    let mut initial = find_request("rq", identity.clone(), rig.hashes.clone());
    initial.total_tokens = 3 * BLOCK_SIZE;

    let (matched, minted, _) = resolved(rig.engine.clone().find_blocks(&initial, None)?);
    assert_eq!(
        matched,
        2 * BLOCK_SIZE,
        "aligned total excludes block three"
    );
    let live = minted.unwrap();

    let mut changed = find_request("rq", identity, rig.hashes.clone());
    changed.num_computed_tokens = 2 * BLOCK_SIZE;
    let (matched, minted, release) =
        resolved(rig.engine.clone().find_blocks(&changed, Some(&live))?);
    assert_eq!(matched, BLOCK_SIZE);
    assert!(minted.is_none());
    assert!(!release);
    assert_eq!(
        rig.engine
            .bundle_searches
            .get(&live.search_id().unwrap())
            .unwrap()
            .lease
            .key(),
        &later
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn repoll_keeps_exact_lease_but_cross_request_needs_index_visibility() -> Result<()> {
    let rig = find_rig(4_400).await?;
    let identity = manifest("v1", &RESOURCES).identity();
    let key = commit(&rig, &identity);
    let request = find_request("same-request", identity.clone(), rig.hashes.clone());

    let (matched, minted, _) = resolved(rig.engine.clone().find_blocks(&request, None)?);
    assert_eq!(matched, 2 * BLOCK_SIZE);
    let live = minted.expect("bundle hit mints the existing handle type");
    assert_eq!(rig.engine.bundle_searches.len(), 1);

    rig.engine.bundle_index.lock().unwrap().invalidate(key);
    let (matched, minted, release) =
        resolved(rig.engine.clone().find_blocks(&request, Some(&live))?);
    assert_eq!(matched, 2 * BLOCK_SIZE, "the parked lease stays stable");
    assert!(minted.is_none(), "identical repolls reuse the handle");
    assert!(!release);

    let cross = find_request("cross-request", identity, rig.hashes.clone());
    let (matched, minted, _) = resolved(rig.engine.clone().find_blocks(&cross, None)?);
    assert_eq!(matched, 0, "removed bundles are not cross-request hits");
    assert!(minted.is_none());
    drop(live);
    assert!(rig.engine.bundle_searches.is_empty());
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn changed_manifest_or_resource_set_desyncs() -> Result<()> {
    let rig = find_rig(4_500).await?;
    let identity = manifest("v1", &RESOURCES).identity();
    commit(&rig, &identity);
    let request = find_request("rq", identity, rig.hashes.clone());
    let (_, minted, _) = resolved(rig.engine.clone().find_blocks(&request, None)?);
    let live = minted.unwrap();

    for changed in [
        manifest("v2", &RESOURCES).identity(),
        manifest("v1", &[RESOURCES[0], LogicalResourceId(92)]).identity(),
    ] {
        let changed_request = find_request("rq", changed, rig.hashes.clone());
        assert!(matches!(
            rig.engine
                .clone()
                .find_blocks(&changed_request, Some(&live)),
            Err(LeaderEngineError::FindBlocksDesync)
        ));
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn overlapping_onboard_defers_bundle_find() -> Result<()> {
    let rig = find_rig(4_600).await?;
    let identity = manifest("v1", &RESOURCES).identity();
    commit(&rig, &identity);
    let inflight = InflightKey::Action(ActionId::new());
    rig.engine
        .inflight
        .lock()
        .unwrap()
        .record(inflight.clone(), vec![rig.hashes[0]]);

    let request = find_request("rq", identity, rig.hashes.clone());
    assert!(matches!(
        rig.engine.clone().find_blocks(&request, None)?,
        FindBlocksOutcome::Deferred
    ));
    assert!(rig.engine.bundle_searches.is_empty());
    rig.engine.inflight.lock().unwrap().clear(&inflight);
    Ok(())
}
