// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Remote-block discovery seam for the leader's remote-search path.
//!
//! The leader's `find_matches_with_options(search_remote)` path needs to know
//! *which* remote instances hold a request's locally-uncached blocks before it
//! can pin + pull them over the transfer control plane. That mapping lives in
//! the hub's KV indexer — but `kvbm-engine` must not depend on `kvbm-hub`, so
//! discovery is expressed as this trait and injected by the connector (which
//! implements it over the hub's indexer client + peer resolver).

use std::sync::Arc;

use anyhow::Result;
use futures::future::BoxFuture;

use super::bundle::{
    BundleAdvertisement, BundleDiscoveryOutcome, BundleDiscoveryQuery, BundleInvalidation,
    BundleMissReason,
};
use crate::{InstanceId, SequenceHash};

/// Outcome of a successful remote-block discovery.
#[derive(Debug, Clone)]
pub struct RemoteCandidates {
    /// Deepest queried hash the index could place. Defines the inclusive end
    /// of the contiguous prefix the remote-search driver will try to pull.
    pub deepest: SequenceHash,

    /// Instances reported to hold `deepest`. The implementation is responsible
    /// for having peer-resolved and registered these for velo reachability +
    /// RDMA metadata before returning them, so the driver can open a transfer
    /// session and pull without further resolution. Always non-empty.
    pub instances: Vec<InstanceId>,
}

/// Resolves which remote instances hold a set of locally-uncached blocks.
///
/// Injected into the [`InstanceLeader`](crate::leader::InstanceLeader) at
/// construction. When absent, the leader performs no remote search.
pub trait RemoteBlockDiscovery: Send + Sync {
    /// Resolve candidate holders for the (locally-uncached, ascending-position)
    /// `hashes`.
    ///
    /// `Ok(None)` means nothing was indexed for any of the hashes (a full
    /// miss). `Err` is reserved for infrastructure failures (the driver
    /// degrades to the local match on error). The returned instances must be
    /// reachable (peer-resolved/registered) by the time the future resolves.
    fn discover(
        &self,
        hashes: Vec<SequenceHash>,
    ) -> BoxFuture<'static, Result<Option<RemoteCandidates>>>;

    /// Resolve one complete manifest-scoped bundle. Legacy discovery
    /// implementations safely report no bundle support.
    fn discover_bundle(
        &self,
        _query: BundleDiscoveryQuery,
    ) -> BoxFuture<'static, Result<BundleDiscoveryOutcome>> {
        Box::pin(async { Ok(BundleDiscoveryOutcome::Miss(BundleMissReason::NotFound)) })
    }

    /// Publish a locally committed complete bundle. Legacy directories no-op.
    fn advertise_bundle(
        &self,
        _advertisement: BundleAdvertisement,
    ) -> BoxFuture<'static, Result<()>> {
        Box::pin(async { Ok(()) })
    }

    /// Remove one exact owner generation after local invalidation.
    fn invalidate_bundle(
        &self,
        _invalidation: BundleInvalidation,
    ) -> BoxFuture<'static, Result<()>> {
        Box::pin(async { Ok(()) })
    }
}

/// Convenience alias for the injected, optional discovery handle.
pub type RemoteDiscoveryHandle = Arc<dyn RemoteBlockDiscovery>;
