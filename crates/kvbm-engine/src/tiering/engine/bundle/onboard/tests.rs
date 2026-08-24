// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::disallowed_macros)]

use std::collections::BTreeMap;
use std::sync::{Arc, Weak};

use kvbm_common::LogicalResourceId;

use super::{BundleOnboard, BundleOnboardState, OnboardTransition};
use crate::tiering::engine::bundle::test_support::{CAPSULE, CSA, HCA, bundle_key, identity};

fn reservations() -> (BTreeMap<LogicalResourceId, Arc<()>>, Vec<Weak<()>>) {
    let reservations = [CSA, HCA, CAPSULE]
        .into_iter()
        .map(|resource| (resource, Arc::new(())))
        .collect::<BTreeMap<_, _>>();
    let weak = reservations.values().map(Arc::downgrade).collect();
    (reservations, weak)
}

#[test]
fn onboard_exposes_computed_boundary_only_at_all_resource_barrier() {
    let (reservations, _) = reservations();
    let mut onboard = BundleOnboard::new(identity(), bundle_key(512), reservations).unwrap();
    assert_eq!(onboard.start(), BundleOnboardState::Transferring);

    assert_eq!(onboard.complete(CSA).unwrap(), OnboardTransition::Pending);
    assert_eq!(onboard.complete(HCA).unwrap(), OnboardTransition::Pending);
    let OnboardTransition::Visible(visible) = onboard.complete(CAPSULE).unwrap() else {
        panic!("the final child must cross the visibility barrier");
    };
    assert_eq!(visible.computed_tokens(), 512);
    assert_eq!(visible.reservations().len(), 3);
}

#[test]
fn destination_pressure_aborts_and_releases_every_reservation() {
    let (reservations, weak) = reservations();
    let mut onboard = BundleOnboard::new(identity(), bundle_key(256), reservations).unwrap();
    onboard.start();
    onboard.complete(CSA).unwrap();

    let OnboardTransition::Aborted(failure) = onboard.fail(HCA, None).unwrap() else {
        panic!("destination pressure must abort the bundle");
    };
    assert_eq!(failure.resource(), HCA);
    assert!(
        weak.iter()
            .all(|reservation| reservation.upgrade().is_none()),
        "an aborted bundle must release all destination reservations"
    );
}

#[test]
fn duplicate_child_completion_is_idempotent() {
    let (reservations, _) = reservations();
    let mut onboard = BundleOnboard::new(identity(), bundle_key(256), reservations).unwrap();
    onboard.start();
    assert_eq!(onboard.complete(CSA).unwrap(), OnboardTransition::Pending);
    assert_eq!(onboard.complete(CSA).unwrap(), OnboardTransition::Pending);
    onboard.complete(HCA).unwrap();
    assert!(matches!(
        onboard.complete(CAPSULE).unwrap(),
        OnboardTransition::Visible(_)
    ));
    assert_eq!(
        onboard.complete(CAPSULE).unwrap(),
        OnboardTransition::Settled(BundleOnboardState::Visible)
    );
}
