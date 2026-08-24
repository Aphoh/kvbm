# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated cache manifests and vLLM resource mapping."""

from .manifest import CacheManifest, ManifestResource
from .compatibility import (
    all_group_block_ids,
    is_mla_cache_spec,
    manifest_from_extra_config,
)
from .mapping import (
    ResourceMapping,
    ResourceTensorPlan,
    build_resource_tensor_plans,
    map_kv_cache_resources,
)

__all__ = [
    "CacheManifest",
    "ManifestResource",
    "ResourceMapping",
    "ResourceTensorPlan",
    "all_group_block_ids",
    "build_resource_tensor_plans",
    "is_mla_cache_spec",
    "manifest_from_extra_config",
    "map_kv_cache_resources",
]
