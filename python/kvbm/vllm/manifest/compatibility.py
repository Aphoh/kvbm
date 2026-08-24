# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Narrow probes for optional vLLM hybrid-memory APIs."""

from __future__ import annotations

import json
from typing import Any

from .manifest import CacheManifest

_MLA_SPEC_TYPES = frozenset({"MLAAttentionSpec", "SlidingWindowMLASpec"})


def all_group_block_ids(blocks: Any) -> tuple[tuple[int, ...], ...]:
    """Copy every vLLM KV-cache group's block IDs without flattening."""
    raw_groups = blocks.get_block_ids()
    if not isinstance(raw_groups, (list, tuple)):
        raise TypeError("KVCacheBlocks.get_block_ids() must return a sequence")
    groups: list[tuple[int, ...]] = []
    for group in raw_groups:
        if not isinstance(group, (list, tuple)):
            raise TypeError("each KV-cache block-ID group must be a sequence")
        groups.append(tuple(int(block_id) for block_id in group))
    return tuple(groups)


def is_mla_cache_spec(spec: Any) -> bool:
    """Classify MLA from the vLLM cache spec, never from tensor shape."""
    try:
        from vllm.v1.kv_cache_interface import (
            MLAAttentionSpec,
            SlidingWindowMLASpec,
        )
    except (ImportError, AttributeError):
        return type(spec).__name__ in _MLA_SPEC_TYPES
    return isinstance(spec, (MLAAttentionSpec, SlidingWindowMLASpec)) or (
        type(spec).__name__ in _MLA_SPEC_TYPES
    )


def manifest_from_extra_config(extra_config: Any) -> CacheManifest | None:
    """Read the one supported manifest config location and validate it."""
    if not isinstance(extra_config, dict):
        return None
    raw = extra_config.get("cache_manifest")
    if raw is None and isinstance(extra_config.get("default"), dict):
        raw = extra_config["default"].get("cache_manifest")
    if raw is None:
        return None
    if isinstance(raw, dict):
        raw = json.dumps(raw)
    if not isinstance(raw, str):
        raise TypeError("cache_manifest must be a JSON object or string")
    return CacheManifest.from_json(raw)
