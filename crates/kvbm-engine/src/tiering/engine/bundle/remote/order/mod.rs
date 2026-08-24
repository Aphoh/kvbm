// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded per-key ordering for asynchronous directory mutations.

use std::hash::{DefaultHasher, Hash, Hasher};

use kvbm_protocols::cache_manifest::BundleKey;

const ORDER_STRIPES: usize = 64;

pub(in crate::tiering::engine) struct BundleDirectoryOrder {
    stripes: [tokio::sync::Mutex<()>; ORDER_STRIPES],
}

impl BundleDirectoryOrder {
    pub(in crate::tiering::engine) fn new() -> Self {
        Self {
            stripes: std::array::from_fn(|_| tokio::sync::Mutex::new(())),
        }
    }

    pub(in crate::tiering::engine) async fn enter(
        &self,
        key: &BundleKey,
    ) -> tokio::sync::MutexGuard<'_, ()> {
        self.stripes[stripe(key)].lock().await
    }
}

fn stripe(key: &BundleKey) -> usize {
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    let stripe_count = u64::try_from(ORDER_STRIPES).expect("ordering stripe count fits u64");
    usize::try_from(hasher.finish() % stripe_count)
        .expect("directory ordering stripe always fits usize")
}
