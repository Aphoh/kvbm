// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deadline enforcement for advisory bundle-directory mutations.

use std::future::Future;
use std::time::Duration;

const BUNDLE_DIRECTORY_CALL_TIMEOUT: Duration = Duration::from_secs(1);

pub(super) enum DirectoryCallError {
    Failed(anyhow::Error),
    TimedOut,
}

pub(super) async fn run_directory_call(
    call: impl Future<Output = anyhow::Result<()>>,
) -> Result<(), DirectoryCallError> {
    match tokio::time::timeout(BUNDLE_DIRECTORY_CALL_TIMEOUT, call).await {
        Ok(Ok(())) => Ok(()),
        Ok(Err(error)) => Err(DirectoryCallError::Failed(error)),
        Err(_) => Err(DirectoryCallError::TimedOut),
    }
}
