// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Small always-compiled helpers shared across gated and ungated modules.
//!
//! Lives outside `resources/` (which is gated behind the `host-pools`
//! feature) so that unconditional modules such as [`crate::hugepage`] can
//! keep calling it with the feature off.

/// Format a byte count as a human-readable string (`KiB`/`MiB`/`GiB`/`TiB`).
pub(crate) fn format_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    const TIB: f64 = GIB * 1024.0;
    let b = bytes as f64;
    if b >= TIB {
        format!("{:.1} TiB", b / TIB)
    } else if b >= GIB {
        format!("{:.1} GiB", b / GIB)
    } else if b >= MIB {
        format!("{:.1} MiB", b / MIB)
    } else if b >= KIB {
        format!("{:.1} KiB", b / KIB)
    } else {
        format!("{} B", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_bytes_human_readable() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(2048), "2.0 KiB");
        assert_eq!(format_bytes(8 * 1024 * 1024 * 1024), "8.0 GiB");
    }
}
