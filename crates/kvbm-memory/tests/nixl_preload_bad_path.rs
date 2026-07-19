// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! §5.2 regression test: a failed `NixlAgent::preload_runtime` attempt must
//! return an error (with the path in context) and must not poison the
//! process-wide cache into a false "loaded" state. This test is
//! hardware-free -- it never attempts to load a real NIXL runtime -- and
//! runs in its own process (a separate integration-test binary), isolated
//! from `NIXL_CAPI`'s use by any other test.

use kvbm_memory::nixl::NixlAgent;
use std::path::Path;

#[test]
fn bad_path_returns_err_and_does_not_poison_the_cache() {
    let bogus = Path::new("/definitely/not/a/real/path/libnixl_capi_bogus.so");

    let first = NixlAgent::preload_runtime(bogus);
    assert!(first.is_err(), "loading a nonexistent library must fail");
    let msg = first.unwrap_err().to_string();
    assert!(
        msg.contains("libnixl_capi_bogus.so"),
        "error should mention the offending path: {msg}"
    );

    // A failed attempt must not be cached as success; a second failing
    // attempt (even with the exact same bogus path) must fail again rather
    // than silently reporting success.
    let second = NixlAgent::preload_runtime(bogus);
    assert!(
        second.is_err(),
        "a failed preload must not be cached as a successful load"
    );
}
