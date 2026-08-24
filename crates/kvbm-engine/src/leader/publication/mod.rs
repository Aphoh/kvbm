// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Owner-lifetime bundle publication sequencing.

use std::sync::atomic::{AtomicU64, Ordering};

/// One checked monotonic sequence shared by all publication paths for an
/// [`InstanceLeader`](super::InstanceLeader) owner.
#[derive(Default)]
pub(super) struct BundlePublicationSequence {
    last: AtomicU64,
}

impl BundlePublicationSequence {
    pub(super) fn reserve(&self) -> Result<u64, PublicationGenerationExhausted> {
        self.last
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |last| {
                last.checked_add(1)
            })
            .map(|last| last + 1)
            .map_err(|_| PublicationGenerationExhausted)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("bundle publication generation space is exhausted")]
pub(crate) struct PublicationGenerationExhausted;

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicU64;

    use super::BundlePublicationSequence;

    #[test]
    fn sequence_reserves_the_last_generation_then_fails_closed() {
        let sequence = BundlePublicationSequence {
            last: AtomicU64::new(u64::MAX - 1),
        };

        assert_eq!(sequence.reserve(), Ok(u64::MAX));
        assert!(sequence.reserve().is_err());
        assert!(sequence.reserve().is_err());
    }
}
