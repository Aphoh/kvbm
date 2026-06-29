// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::num::{NonZeroU64, NonZeroUsize};

use kvbm_common::LogicalResourceId;
use kvbm_logical::manager::{InactiveBackendConfig, LineageEviction};
use kvbm_protocols::cache_manifest::ResourceRole;

use super::{
    AdmissionScoreInputs, ResourcePolicies, ResourcePolicy, RetentionDecision, RetentionReason,
};

fn bytes(value: u64) -> NonZeroU64 {
    NonZeroU64::new(value).unwrap()
}

#[test]
fn resources_can_select_different_serializable_inactive_backends() {
    let csa = LogicalResourceId(40);
    let hca = LogicalResourceId(41);
    let mut policies = ResourcePolicies::new();
    policies
        .insert(
            csa,
            ResourcePolicy::new(
                ResourceRole::PrefixHistory,
                InactiveBackendConfig::MultiLru {
                    frequency_thresholds: [3, 8, 15],
                },
                InactiveBackendConfig::Lineage {
                    eviction: LineageEviction::Fifo,
                },
            ),
        )
        .unwrap();
    policies
        .insert(
            hca,
            ResourcePolicy::new(
                ResourceRole::PrefixHistory,
                InactiveBackendConfig::Lru,
                InactiveBackendConfig::Lru,
            ),
        )
        .unwrap();

    let json = serde_json::to_string(&policies).unwrap();
    let decoded: ResourcePolicies = serde_json::from_str(&json).unwrap();

    assert_ne!(decoded.get(csa).unwrap(), decoded.get(hca).unwrap());
    assert!(matches!(
        decoded.get(csa).unwrap().g1_inactive(),
        InactiveBackendConfig::MultiLru { .. }
    ));
    assert_eq!(
        decoded.get(hca).unwrap().g2_inactive(),
        &InactiveBackendConfig::Lru
    );
}

#[test]
fn byte_normalized_scores_do_not_treat_equal_hits_as_equal_value() {
    let small = AdmissionScoreInputs::complete(8, [bytes(4 * 1024)]).unwrap();
    let large = AdmissionScoreInputs::complete(8, [bytes(64 * 1024)]).unwrap();
    let policy = ResourcePolicy::new(
        ResourceRole::PrefixHistory,
        InactiveBackendConfig::Lru,
        InactiveBackendConfig::Lru,
    )
    .with_minimum_hits_per_mib(1024);

    assert!(small.score() > large.score());
    assert_ne!(small.score(), large.score());
    assert_eq!(
        policy.evaluate(&small).decision(),
        RetentionDecision::Retain
    );
    assert_eq!(policy.evaluate(&large).decision(), RetentionDecision::Drop);
}

#[test]
fn atomic_resource_cannot_admit_only_one_component() {
    let policy = ResourcePolicy::new(
        ResourceRole::PrefixHistory,
        InactiveBackendConfig::Lru,
        InactiveBackendConfig::Lru,
    )
    .with_atomic_components(NonZeroUsize::new(2).unwrap());
    let partial = AdmissionScoreInputs::complete(8, [bytes(4096)]).unwrap();
    let outcome = policy.evaluate(&partial);

    assert_eq!(outcome.decision(), RetentionDecision::Drop);
    assert_eq!(outcome.reason(), RetentionReason::IncompleteAtomicResource);
}

#[test]
fn orphan_capsule_is_never_admitted() {
    let policy = ResourcePolicy::new(
        ResourceRole::BoundaryCapsule,
        InactiveBackendConfig::Lru,
        InactiveBackendConfig::Lru,
    );
    let orphan = AdmissionScoreInputs::orphan(10, [bytes(1024)]).unwrap();
    let outcome = policy.evaluate(&orphan);

    assert_eq!(outcome.decision(), RetentionDecision::Drop);
    assert_eq!(outcome.reason(), RetentionReason::MissingHistoryDependency);
}
