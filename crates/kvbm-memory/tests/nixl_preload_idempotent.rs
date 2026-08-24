// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! §5.2 regression test: the first successful `NixlAgent::preload_runtime`
//! call wins for the process lifetime -- later calls (even with a
//! different, invalid path) are no-ops that return `Ok`.
//!
//! This deliberately preloads `libc`, not a real NIXL runtime: `libc` is
//! guaranteed dlopen-able (already loaded) in any process, which lets this
//! test exercise the `OnceLock` caching semantics without depending on a
//! NIXL install being present. Runs in its own process (a separate
//! integration-test binary), isolated from `NIXL_CAPI`'s use by any other
//! test.

use kvbm_memory::nixl::NixlAgent;
use std::path::Path;

#[test]
fn first_success_wins_for_the_process_lifetime() {
    let already_loadable = if cfg!(target_os = "macos") {
        "libSystem.B.dylib"
    } else {
        "libc.so.6"
    };
    NixlAgent::preload_runtime(Path::new(already_loadable))
        .expect("preloading libc should succeed");

    let bogus = Path::new("/definitely/not/a/real/path/libbogus.so");
    assert!(
        NixlAgent::preload_runtime(bogus).is_ok(),
        "a later call must be a no-op once the slot is claimed, even with an invalid path"
    );
}
