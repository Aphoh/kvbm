// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Atomic multi-resource offload transaction.

mod driver;

use std::collections::{BTreeMap, BTreeSet};

use kvbm_common::{BlockId, LogicalResourceId};
use kvbm_protocols::cache_manifest::{BundleKey, CacheIdentity};
use kvbm_protocols::connector::OffloadMode;

use super::barrier::{BarrierError, BarrierState, BarrierUpdate, ResourceBarrier};

/// Atomic offload transaction. Source pins stay owned until commit or abort;
/// staged destination pins are published only by the final success transition.
pub(in crate::tiering::engine) struct BundleOffload<S, P> {
    identity: CacheIdentity,
    key: BundleKey,
    generation: u64,
    mode: OffloadMode,
    barrier: ResourceBarrier,
    sources: Option<BTreeMap<LogicalResourceId, S>>,
    staged: BTreeMap<LogicalResourceId, P>,
}

impl<S, P> BundleOffload<S, P> {
    pub(in crate::tiering::engine) fn new(
        identity: CacheIdentity,
        key: BundleKey,
        generation: u64,
        mode: OffloadMode,
        sources: BTreeMap<LogicalResourceId, S>,
    ) -> Result<Self, BundleOffloadError> {
        if !key.is_compatible_with(&identity) {
            return Err(BundleOffloadError::IncompatibleKey);
        }
        validate_resources(&identity, sources.keys().copied())?;
        Ok(Self {
            identity,
            key,
            generation,
            mode,
            barrier: ResourceBarrier::new(sources.keys().copied())?,
            sources: Some(sources),
            staged: BTreeMap::new(),
        })
    }

    pub(in crate::tiering::engine) fn start(&mut self) -> BundleOffloadState {
        self.barrier.start().into()
    }

    pub(in crate::tiering::engine) fn complete(
        &mut self,
        resource: LogicalResourceId,
        result: Result<P, BundleResourceFailure>,
    ) -> Result<OffloadTransition<S, P>, BundleOffloadError> {
        let pin = match result {
            Ok(pin) => pin,
            Err(failure) => return self.abort(failure),
        };
        if self.barrier.is_pending(resource) {
            self.staged.insert(resource, pin);
        }
        match self.barrier.succeed(resource)? {
            BarrierUpdate::Pending => Ok(OffloadTransition::Pending),
            BarrierUpdate::Complete => {
                let resources = std::mem::take(&mut self.staged);
                let retained_sources = match self.mode {
                    OffloadMode::Mirror => self.sources.take(),
                    OffloadMode::Move => {
                        drop(self.sources.take());
                        None
                    }
                };
                Ok(OffloadTransition::Commit(BundleCommit {
                    identity: self.identity.clone(),
                    key: self.key,
                    generation: self.generation,
                    resources,
                    retained_sources,
                }))
            }
            BarrierUpdate::Settled(state) => {
                Ok(OffloadTransition::Settled(BundleOffloadState::from(state)))
            }
            BarrierUpdate::Aborted => Err(BundleOffloadError::InvalidBarrierTransition),
        }
    }

    pub(in crate::tiering::engine) fn fail(
        &mut self,
        resource: LogicalResourceId,
        failed_blocks: Option<Vec<BlockId>>,
    ) -> Result<OffloadTransition<S, P>, BundleOffloadError> {
        self.abort(BundleResourceFailure::new(resource, failed_blocks))
    }

    fn abort(
        &mut self,
        failure: BundleResourceFailure,
    ) -> Result<OffloadTransition<S, P>, BundleOffloadError> {
        match self.barrier.fail(failure.resource)? {
            BarrierUpdate::Aborted => {
                self.staged.clear();
                Ok(OffloadTransition::Abort(BundleAbort {
                    failure,
                    retained_sources: self.sources.take().unwrap_or_default(),
                }))
            }
            BarrierUpdate::Pending => Ok(OffloadTransition::Pending),
            BarrierUpdate::Settled(state) => {
                Ok(OffloadTransition::Settled(BundleOffloadState::from(state)))
            }
            BarrierUpdate::Complete => Err(BundleOffloadError::InvalidBarrierTransition),
        }
    }
}

fn validate_resources(
    identity: &CacheIdentity,
    resources: impl IntoIterator<Item = LogicalResourceId>,
) -> Result<(), BundleOffloadError> {
    let expected = identity
        .resources()
        .iter()
        .map(|requirement| requirement.resource())
        .collect::<BTreeSet<_>>();
    let actual = resources.into_iter().collect::<BTreeSet<_>>();
    if expected == actual {
        Ok(())
    } else {
        Err(BundleOffloadError::ResourceSetMismatch)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::tiering::engine) enum BundleOffloadState {
    Prepared,
    Transferring,
    Committed,
    Aborted,
}

impl From<BarrierState> for BundleOffloadState {
    fn from(state: BarrierState) -> Self {
        match state {
            BarrierState::Prepared => Self::Prepared,
            BarrierState::Transferring => Self::Transferring,
            BarrierState::Complete => Self::Committed,
            BarrierState::Aborted => Self::Aborted,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(in crate::tiering::engine) enum OffloadTransition<S, P> {
    Pending,
    Commit(BundleCommit<S, P>),
    Abort(BundleAbort<S>),
    Settled(BundleOffloadState),
}

#[derive(Debug, PartialEq, Eq)]
pub(in crate::tiering::engine) struct BundleCommit<S, P> {
    identity: CacheIdentity,
    key: BundleKey,
    generation: u64,
    resources: BTreeMap<LogicalResourceId, P>,
    retained_sources: Option<BTreeMap<LogicalResourceId, S>>,
}

impl<S, P> BundleCommit<S, P> {
    #[cfg(test)]
    pub(in crate::tiering::engine) const fn key(&self) -> &BundleKey {
        &self.key
    }

    #[cfg(test)]
    pub(in crate::tiering::engine) const fn generation(&self) -> u64 {
        self.generation
    }

    #[cfg(test)]
    pub(in crate::tiering::engine) const fn resources(&self) -> &BTreeMap<LogicalResourceId, P> {
        &self.resources
    }

    #[cfg(test)]
    pub(in crate::tiering::engine) const fn retained_sources(
        &self,
    ) -> Option<&BTreeMap<LogicalResourceId, S>> {
        self.retained_sources.as_ref()
    }

    pub(in crate::tiering::engine) fn into_publication(
        self,
    ) -> (BundlePublication<P>, Option<BTreeMap<LogicalResourceId, S>>) {
        (
            BundlePublication {
                identity: self.identity,
                key: self.key,
                generation: self.generation,
                resources: self.resources,
            },
            self.retained_sources,
        )
    }
}

pub(in crate::tiering::engine) struct BundlePublication<P> {
    identity: CacheIdentity,
    key: BundleKey,
    generation: u64,
    resources: BTreeMap<LogicalResourceId, P>,
}

impl<P> BundlePublication<P> {
    pub(in crate::tiering::engine) fn into_parts(
        self,
    ) -> (
        CacheIdentity,
        BundleKey,
        u64,
        BTreeMap<LogicalResourceId, P>,
    ) {
        (self.identity, self.key, self.generation, self.resources)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(in crate::tiering::engine) struct BundleAbort<S> {
    failure: BundleResourceFailure,
    retained_sources: BTreeMap<LogicalResourceId, S>,
}

impl<S> BundleAbort<S> {
    pub(in crate::tiering::engine) const fn failure(&self) -> &BundleResourceFailure {
        &self.failure
    }

    #[cfg(test)]
    pub(in crate::tiering::engine) const fn retained_sources(
        &self,
    ) -> &BTreeMap<LogicalResourceId, S> {
        &self.retained_sources
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::tiering::engine) struct BundleResourceFailure {
    resource: LogicalResourceId,
    failed_blocks: Option<Vec<BlockId>>,
}

impl BundleResourceFailure {
    pub(in crate::tiering::engine) fn new(
        resource: LogicalResourceId,
        failed_blocks: Option<Vec<BlockId>>,
    ) -> Self {
        Self {
            resource,
            failed_blocks,
        }
    }

    pub(in crate::tiering::engine) const fn resource(&self) -> LogicalResourceId {
        self.resource
    }

    pub(in crate::tiering::engine) fn failed_blocks(&self) -> Option<&[BlockId]> {
        self.failed_blocks.as_deref()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub(in crate::tiering::engine) enum BundleOffloadError {
    #[error("bundle key is incompatible with its cache identity")]
    IncompatibleKey,
    #[error("bundle transaction resources do not match its cache identity")]
    ResourceSetMismatch,
    #[error("resource barrier produced an invalid offload transition")]
    InvalidBarrierTransition,
    #[error(transparent)]
    Barrier(#[from] BarrierError),
}

#[cfg(test)]
mod tests;
