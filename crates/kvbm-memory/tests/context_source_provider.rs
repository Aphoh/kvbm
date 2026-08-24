// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! §5.5 regression test: an installed `CudaContextProvider` is actually
//! consulted by `DeviceStorage::new`, and a second `set_cuda_context_provider`
//! call is rejected once the process-wide slot is occupied.
//!
//! Hardware-free: the installed provider always returns an error, so no real
//! CUDA context is ever created. Runs in its own process (a separate
//! integration-test binary), isolated from `CONTEXT_SOURCE`'s use by any
//! other test.

use kvbm_memory::{CudaContextProvider, DeviceStorage, StorageError, set_cuda_context_provider};

#[test]
fn provider_is_consulted_and_a_second_install_is_rejected() {
    const SENTINEL: &str = "sentinel-context-error-9f3c9a";

    set_cuda_context_provider(Box::new(|_device_id| {
        Err(StorageError::OperationFailed(SENTINEL.to_string()))
    }))
    .expect("first install should succeed");

    let err = DeviceStorage::new(4096, 0).expect_err("the installed provider should be consulted");
    assert!(
        err.to_string().contains(SENTINEL),
        "DeviceStorage::new should surface the provider's error, got: {err}"
    );

    let unreachable_provider: Box<CudaContextProvider> = Box::new(|_| unreachable!());
    let second = set_cuda_context_provider(unreachable_provider);
    assert!(
        second.is_err(),
        "installing a provider after one is already installed must be rejected"
    );
}
