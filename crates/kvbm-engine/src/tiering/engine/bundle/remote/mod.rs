// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Remote publication, invalidation, and local commit of pulled bundles.

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use futures::future::BoxFuture;
use kvbm_common::{LogicalResourceId, SequenceHash};
use kvbm_protocols::cache_manifest::{BundleKey, CacheIdentity};

use super::BundleCommitMetadata;
use crate::leader::InstanceLeader;
use crate::remote::search::bundle::{
    BundleAdvertisement, BundleInvalidation, BundlePullTarget, unix_time_ms,
};
use crate::tiering::engine::local::LocalConnectorEngine;
use crate::tiering::policy::ResourceLineage;

impl BundlePullTarget for LocalConnectorEngine {
    fn instance_leader(&self) -> Arc<InstanceLeader> {
        Arc::clone(&self.leader)
    }

    fn commit_pulled_bundle(
        &self,
        identity: CacheIdentity,
        key: BundleKey,
        generation: u64,
        lineages: BTreeMap<LogicalResourceId, Vec<SequenceHash>>,
    ) -> BoxFuture<'static, Result<()>> {
        let engine = self.weak_self.upgrade();
        Box::pin(async move {
            let engine = engine.ok_or_else(|| anyhow!("bundle pull target was dropped"))?;
            let mut pins = Vec::with_capacity(identity.resources().len());
            let mut dependency_lineages = Vec::with_capacity(identity.resources().len());
            for requirement in identity.resources() {
                let resource = requirement.resource();
                let hashes = lineages
                    .get(&resource)
                    .ok_or_else(|| anyhow!("pulled bundle is missing lineage for {resource:?}"))?;
                let manager = engine
                    .leader
                    .g2_manager_for(resource)
                    .ok_or_else(|| anyhow!("no local G2 manager for {resource:?}"))?;
                let resource_pins = manager.match_blocks(hashes);
                anyhow::ensure!(
                    resource_pins.len() == hashes.len(),
                    "pulled bundle resource {resource:?} did not commit every block locally"
                );
                pins.push((resource, resource_pins));
                dependency_lineages.push(ResourceLineage::new(
                    resource,
                    requirement.role(),
                    hashes.clone(),
                ));
            }

            {
                let mut bundles = engine
                    .bundle_index
                    .lock()
                    .expect("bundle-index mutex poisoned");
                bundles
                    .commit(&identity, key, generation, pins)
                    .map_err(anyhow::Error::from)?;
                let tracked = engine
                    .bundle_dependencies
                    .lock()
                    .expect("bundle-dependencies mutex poisoned")
                    .track(key, dependency_lineages);
                if let Err(error) = tracked {
                    bundles.invalidate(key);
                    return Err(anyhow::Error::from(error));
                }
            }
            engine.advertise_committed_bundle(BundleCommitMetadata {
                identity,
                key,
                generation,
            });
            Ok(())
        })
    }
}

impl LocalConnectorEngine {
    pub(in crate::tiering::engine) fn advertise_committed_bundle(
        &self,
        metadata: BundleCommitMetadata,
    ) {
        let Some(directory) = self.leader.remote_discovery() else {
            return;
        };
        const ADVERTISEMENT_TTL_MS: u64 = 300_000;
        let owner = self.leader.messenger().instance_id();
        let expires_at_unix_ms = unix_time_ms().saturating_add(ADVERTISEMENT_TTL_MS);
        let resources = metadata
            .identity
            .resources()
            .iter()
            .map(|requirement| requirement.resource())
            .collect::<Vec<_>>();
        let advertisement = match BundleAdvertisement::new(
            metadata.identity,
            metadata.key,
            metadata.generation,
            owner,
            expires_at_unix_ms,
            resources,
        ) {
            Ok(advertisement) => advertisement,
            Err(error) => {
                tracing::error!(%error, "committed bundle could not be advertised");
                return;
            }
        };
        self.leader.runtime().spawn(async move {
            if let Err(error) = directory.advertise_bundle(advertisement).await {
                tracing::warn!(%error, "bundle advertisement publish failed");
            }
        });
    }

    pub(in crate::tiering::engine) fn invalidate_resource_blocks(
        &self,
        resource: LogicalResourceId,
        hashes: &[SequenceHash],
    ) {
        let mut invalidated = Vec::new();
        {
            let mut dependencies = self
                .bundle_dependencies
                .lock()
                .expect("bundle-dependencies mutex poisoned");
            for &hash in hashes {
                dependencies.invalidate(resource, hash, |event| invalidated.push(event));
            }
        }
        if invalidated.is_empty() {
            return;
        }
        let mut bundles = self
            .bundle_index
            .lock()
            .expect("bundle-index mutex poisoned");
        for event in invalidated {
            let generation = bundles.remove(event.key());
            tracing::debug!(
                resource = ?event.resource(),
                hash = ?event.hash(),
                boundary_tokens = event.key().boundary_tokens(),
                "resource eviction invalidated a dependent bundle"
            );
            if let Some(generation) = generation {
                self.invalidate_bundle_advertisement(event.key(), generation);
            }
        }
    }

    fn invalidate_bundle_advertisement(&self, key: BundleKey, generation: u64) {
        let Some(directory) = self.leader.remote_discovery() else {
            return;
        };
        let invalidation = BundleInvalidation {
            key,
            generation,
            owner: self.leader.messenger().instance_id(),
        };
        self.leader.runtime().spawn(async move {
            if let Err(error) = directory.invalidate_bundle(invalidation).await {
                tracing::warn!(%error, "bundle advertisement invalidation failed");
            }
        });
    }
}
