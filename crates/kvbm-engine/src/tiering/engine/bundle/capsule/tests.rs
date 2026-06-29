// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::disallowed_macros)]

use super::{CapsuleDescriptor, CapsulePoolCopy, CapsuleStorage};
use crate::tiering::engine::bundle::test_support::bundle_key;

#[test]
fn fixed_slot_capsule_builds_one_atomic_multi_pool_copy_plan() {
    let descriptor = CapsuleDescriptor::new(
        CapsuleStorage::RequestSlots { generation: 4 },
        vec![
            CapsulePoolCopy::new(vec![1, 2], vec![11, 12]).unwrap(),
            CapsulePoolCopy::new(vec![3], vec![13]).unwrap(),
        ],
    )
    .unwrap();

    let plan = descriptor.copy_plan();
    assert_eq!(plan.generation(), 4);
    assert_eq!(plan.pools().len(), 2);
    assert_eq!(plan.total_slots(), 3);
}

#[test]
fn shared_capsule_is_scoped_to_bundle_key_and_generation() {
    let key = bundle_key(512);
    let descriptor = CapsuleDescriptor::new(
        CapsuleStorage::SharedPrefix { key, generation: 8 },
        vec![CapsulePoolCopy::new(vec![7], vec![17]).unwrap()],
    )
    .unwrap();

    assert_eq!(descriptor.storage().bundle_key(), Some(&key));
    assert_eq!(descriptor.copy_plan().generation(), 8);
}

#[test]
fn capsule_rejects_partial_or_mismatched_pool_shapes() {
    assert!(CapsulePoolCopy::new(vec![], vec![]).is_err());
    assert!(CapsulePoolCopy::new(vec![1, 2], vec![3]).is_err());
    assert!(
        CapsuleDescriptor::new(CapsuleStorage::RequestSlots { generation: 1 }, vec![]).is_err()
    );
}
