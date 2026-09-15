# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""This module provides strict native runtime stubs for source-only Python tests."""

from __future__ import annotations

import sys
from types import ModuleType


def _native_runtime_unavailable(*_args: object, **_kwargs: object) -> None:
    raise ImportError("the native KVBM runtime is unavailable in source-only tests")


def _is_available() -> bool:
    return False


_core = ModuleType("kvbm._core")
for _name in (
    "ConnectorLeader",
    "ConnectorWorker",
    "KvbmRequest",
    "KvbmRuntime",
    "KvbmVllmConfig",
    "SchedulerOutput",
    "Tensor",
):
    setattr(_core, _name, _native_runtime_unavailable)

_core.__version__ = "0.0.0+source-only"
_core.is_available = _is_available
sys.modules.setdefault("kvbm._core", _core)
