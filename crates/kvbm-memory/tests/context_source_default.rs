// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! §5.5 regression test: `set_cuda_context_provider` must be rejected after
//! the *default* cache path has already run once -- not just after a prior
//! provider install. Both origins share one `OnceLock`.
//!
//! Hardware-free: `DeviceStorage::new` claims the `ContextSource` slot with
//! the default (empty) cache *before* it calls `CudaContext::new`, so this
//! holds even on a GPU-less box where the allocation itself fails. Runs in
//! its own process (a separate integration-test binary), isolated from
//! `CONTEXT_SOURCE`'s use by any other test.

use kvbm_memory::{CudaContextProvider, DeviceStorage, set_cuda_context_provider};

#[test]
fn default_cache_use_occupies_the_slot_too() {
    // Ignore the result: on a box without device 0, this fails downstream at
    // `CudaContext::new`, but the `ContextSource` slot is claimed by the
    // `Default` cache before that call -- which is the property under test.
    let _ = DeviceStorage::new(4096, 0);

    let unreachable_provider: Box<CudaContextProvider> = Box::new(|_| unreachable!());
    let result = set_cuda_context_provider(unreachable_provider);
    assert!(
        result.is_err(),
        "installing a provider after a default-path allocation attempt must be rejected"
    );
}
