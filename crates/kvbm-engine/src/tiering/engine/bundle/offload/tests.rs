// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::disallowed_macros)]

use std::collections::BTreeMap;
use std::sync::{Arc, Weak};

use kvbm_common::LogicalResourceId;

use super::{BundleOffload, BundleOffloadState, OffloadTransition};
use crate::tiering::engine::bundle::test_support::{CAPSULE, CSA, HCA, bundle_key, identity};
use kvbm_protocols::connector::OffloadMode;

fn pins() -> (BTreeMap<LogicalResourceId, Arc<()>>, Vec<Weak<()>>) {
    let pins = [CSA, HCA, CAPSULE]
        .into_iter()
        .map(|resource| (resource, Arc::new(())))
        .collect::<BTreeMap<_, _>>();
    let weak = pins.values().map(Arc::downgrade).collect();
    (pins, weak)
}

#[test]
fn offload_publishes_once_only_after_every_resource_completes() {
    let (sources, _) = pins();
    let mut offload =
        BundleOffload::new(identity(), bundle_key(512), 7, OffloadMode::Mirror, sources).unwrap();
    assert_eq!(offload.start(), BundleOffloadState::Transferring);

    assert!(matches!(
        offload.complete(CSA, Ok(Arc::new(()))).unwrap(),
        OffloadTransition::Pending
    ));
    assert!(matches!(
        offload.complete(CAPSULE, Ok(Arc::new(()))).unwrap(),
        OffloadTransition::Pending
    ));
    let OffloadTransition::Commit(commit) = offload.complete(HCA, Ok(Arc::new(()))).unwrap() else {
        panic!("the final resource must publish one commit");
    };
    assert_eq!(commit.resources().len(), 3);
    assert_eq!(commit.key(), &bundle_key(512));
    assert_eq!(commit.generation(), 7);
    assert_eq!(commit.retained_sources().unwrap().len(), 3);

    assert_eq!(
        offload.complete(HCA, Ok(Arc::new(()))).unwrap(),
        OffloadTransition::Settled(BundleOffloadState::Committed),
        "a duplicate completion cannot publish twice"
    );
}

#[test]
fn failed_resource_aborts_staging_and_retains_every_source_pin() {
    let (sources, source_weak) = pins();
    let staged = Arc::new(());
    let staged_weak = Arc::downgrade(&staged);
    let mut offload =
        BundleOffload::new(identity(), bundle_key(512), 9, OffloadMode::Move, sources).unwrap();
    offload.start();
    assert!(matches!(
        offload.complete(CSA, Ok(staged)).unwrap(),
        OffloadTransition::Pending
    ));

    let OffloadTransition::Abort(abort) = offload.fail(HCA, Some(vec![4, 8])).unwrap() else {
        panic!("one failed child must abort the bundle");
    };
    assert_eq!(abort.failure().resource(), HCA);
    assert_eq!(abort.failure().failed_blocks(), Some(&[4, 8][..]));
    assert!(
        staged_weak.upgrade().is_none(),
        "staged pins must be dropped"
    );
    assert!(source_weak.iter().all(|pin| pin.upgrade().is_some()));
    assert_eq!(abort.retained_sources().len(), 3);
}

#[test]
fn move_releases_sources_only_after_commit_while_mirror_retains_them() {
    let (move_sources, move_weak) = pins();
    let mut moving = BundleOffload::new(
        identity(),
        bundle_key(256),
        1,
        OffloadMode::Move,
        move_sources,
    )
    .unwrap();
    moving.start();
    moving.complete(CSA, Ok(Arc::new(()))).unwrap();
    moving.complete(HCA, Ok(Arc::new(()))).unwrap();
    assert!(move_weak.iter().all(|pin| pin.upgrade().is_some()));
    let OffloadTransition::Commit(commit) = moving.complete(CAPSULE, Ok(Arc::new(()))).unwrap()
    else {
        panic!("all resources should commit");
    };
    assert!(commit.retained_sources().is_none());
    assert!(move_weak.iter().all(|pin| pin.upgrade().is_none()));

    let (mirror_sources, mirror_weak) = pins();
    let mut mirroring = BundleOffload::new(
        identity(),
        bundle_key(256),
        2,
        OffloadMode::Mirror,
        mirror_sources,
    )
    .unwrap();
    mirroring.start();
    mirroring.complete(CSA, Ok(Arc::new(()))).unwrap();
    mirroring.complete(HCA, Ok(Arc::new(()))).unwrap();
    let OffloadTransition::Commit(commit) = mirroring.complete(CAPSULE, Ok(Arc::new(()))).unwrap()
    else {
        panic!("all resources should commit");
    };
    assert!(mirror_weak.iter().all(|pin| pin.upgrade().is_some()));
    drop(commit);
    assert!(mirror_weak.iter().all(|pin| pin.upgrade().is_none()));
}
