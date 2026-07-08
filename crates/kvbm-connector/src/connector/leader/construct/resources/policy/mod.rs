// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Resolve manifest resource geometry into engine admission defaults.

use std::collections::BTreeMap;
use std::num::{NonZeroU64, NonZeroUsize};

use anyhow::{Result, ensure};
use kvbm_common::LogicalResourceId;
use kvbm_engine::tiering::policy::{ResourceComponentBytes, ResourcePolicies, ResourcePolicy};
use kvbm_logical::manager::InactiveBackendConfig;
use kvbm_physical::layout::LayoutConfig;
use kvbm_protocols::cache_manifest::CacheIdentity;

/// Admission inputs resolved once from the registered manifest and layouts.
pub(in crate::connector::leader) struct ResourceAdmissionPlan {
    policies: ResourcePolicies,
    component_bytes: ResourceComponentBytes,
}

impl ResourceAdmissionPlan {
    pub(in crate::connector::leader) fn into_engine_config(
        self,
        block_size: usize,
        remote: kvbm_engine::RemoteOps,
    ) -> kvbm_engine::ConnectorEngineConfig {
        kvbm_engine::ConnectorEngineConfig {
            block_size,
            remote,
            resource_policies: self.policies,
            resource_component_bytes: self.component_bytes,
        }
    }

    pub(super) fn build(
        identity: Option<&CacheIdentity>,
        layouts: &BTreeMap<LogicalResourceId, LayoutConfig>,
        parallelism: &BTreeMap<LogicalResourceId, kvbm_config::ParallelismMode>,
        worker_count: usize,
    ) -> Result<Self> {
        let Some(identity) = identity else {
            return Ok(Self {
                policies: ResourcePolicies::new(),
                component_bytes: ResourceComponentBytes::new(),
            });
        };
        ensure!(
            identity.resources().len() == layouts.len()
                && identity
                    .resources()
                    .iter()
                    .all(|requirement| layouts.contains_key(&requirement.resource())),
            "cache identity and registered worker layouts have different resource sets"
        );
        ensure!(
            worker_count > 0,
            "cannot derive admission components without workers"
        );

        let mut policies = ResourcePolicies::new();
        let mut component_bytes = ResourceComponentBytes::new();
        for requirement in identity.resources() {
            let resource = requirement.resource();
            let layout = layouts
                .get(&resource)
                .expect("resource-set equality was checked");
            let bytes = u64::try_from(layout.bytes_per_block())?;
            let bytes = NonZeroU64::new(bytes).ok_or_else(|| {
                anyhow::anyhow!("logical resource {resource:?} has a zero-byte block layout")
            })?;
            let component_count = match parallelism.get(&resource).ok_or_else(|| {
                anyhow::anyhow!("logical resource {resource:?} has no parallelism policy")
            })? {
                kvbm_config::ParallelismMode::TensorParallel => worker_count,
                kvbm_config::ParallelismMode::ReplicatedData => 1,
            };
            component_bytes.insert(resource, std::iter::repeat_n(bytes, component_count))?;
            policies.insert(
                resource,
                ResourcePolicy::new(
                    requirement.role(),
                    InactiveBackendConfig::default(),
                    InactiveBackendConfig::default(),
                )
                .with_atomic_components(
                    NonZeroUsize::new(component_count)
                        .expect("worker/component count was checked positive"),
                ),
            )?;
        }
        Ok(Self {
            policies,
            component_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kvbm_protocols::cache_manifest::{
        CacheManifest, ModelIdentity, ResourceRequirement, ResourceRole,
    };

    fn layout(bytes_per_block: usize) -> LayoutConfig {
        LayoutConfig::builder()
            .num_blocks(4)
            .num_layers(1)
            .outer_dim(1)
            .page_size(1)
            .inner_dim(bytes_per_block)
            .dtype_width_bytes(1)
            .build()
            .unwrap()
    }

    #[test]
    fn manifest_defaults_have_exact_checked_resource_coverage() {
        let history = LogicalResourceId(4);
        let capsule = LogicalResourceId(7);
        let identity = CacheManifest::new(
            ModelIdentity::new("policy", "v1", [3; 32]).unwrap(),
            "policy-v1",
            vec![
                ResourceRequirement::new(history, ResourceRole::PrefixHistory, 1).unwrap(),
                ResourceRequirement::new(capsule, ResourceRole::BoundaryCapsule, 1).unwrap(),
            ],
            Default::default(),
        )
        .unwrap()
        .identity();
        let plan = ResourceAdmissionPlan::build(
            Some(&identity),
            &BTreeMap::from([(history, layout(32)), (capsule, layout(8))]),
            &BTreeMap::from([
                (history, kvbm_config::ParallelismMode::TensorParallel),
                (capsule, kvbm_config::ParallelismMode::ReplicatedData),
            ]),
            2,
        )
        .unwrap();

        assert_eq!(plan.policies.iter().count(), 2);
        assert_eq!(
            plan.policies.get(history).unwrap().role(),
            ResourceRole::PrefixHistory
        );
        assert_eq!(
            plan.component_bytes
                .get(history)
                .unwrap()
                .iter()
                .map(|bytes| bytes.get())
                .collect::<Vec<_>>(),
            vec![32, 32]
        );
        assert_eq!(plan.component_bytes.get(capsule).unwrap()[0].get(), 8);
    }

    #[test]
    fn manifest_defaults_reject_resource_set_drift() {
        let resource = LogicalResourceId(4);
        let identity = CacheManifest::new(
            ModelIdentity::new("policy", "v1", [3; 32]).unwrap(),
            "policy-v1",
            vec![ResourceRequirement::new(resource, ResourceRole::PrefixHistory, 1).unwrap()],
            Default::default(),
        )
        .unwrap()
        .identity();

        assert!(
            ResourceAdmissionPlan::build(Some(&identity), &BTreeMap::new(), &BTreeMap::new(), 1,)
                .is_err()
        );
    }
}
