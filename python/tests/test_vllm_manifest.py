# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the vLLM-to-KVBM cache manifest adapter."""

from __future__ import annotations

import json
import unittest
from dataclasses import dataclass

from kvbm.vllm.manifest import (
    CacheManifest,
    all_group_block_ids,
    build_resource_tensor_plans,
    is_mla_cache_spec,
    map_kv_cache_resources,
)


@dataclass(frozen=True)
class FakeSpec:
    name: str


@dataclass(frozen=True)
class FakeGroup:
    layer_names: tuple[str, ...]
    kv_cache_spec: object


@dataclass(frozen=True)
class FakeConfig:
    kv_cache_groups: tuple[FakeGroup, ...]
    kv_cache_tensors: tuple[object, ...] = ()


@dataclass(frozen=True)
class FakeTensorConfig:
    shared_by: tuple[str, ...]


@dataclass(frozen=True)
class FakeTensor:
    shape: tuple[int, ...]


class MainBackend:
    pass


class IndexerBackend:
    pass


def manifest_json(*, duplicate_resource: bool = False) -> str:
    bindings = [
        {
            "resource": 10,
            "groups": [
                {
                    "layers": ["model.layers.0.main", "model.layers.1.main"],
                    "spec_type": "FakeSpec",
                    "backend": "MainBackend",
                }
            ],
        },
        {
            "resource": 11,
            "groups": [
                {
                    "layers": ["model.layers.0.indexer"],
                    "spec_type": "FakeSpec",
                    "backend": "IndexerBackend",
                }
            ],
        },
    ]
    resources = [
        {
            "resource": 10,
            "role": "prefix_history",
            "native_block_tokens": 4,
        },
        {
            "resource": 11,
            "role": "prefix_history",
            "native_block_tokens": 8,
        },
    ]
    if duplicate_resource:
        resources[1]["resource"] = 10
    return json.dumps(
        {
            "schema_version": 1,
            "model": {
                "architecture": "fake_dsv4",
                "revision": "test",
                "weights_digest": [7] * 32,
            },
            "cache_abi": "test-hybrid-v1",
            "resources": resources,
            "attributes": {"vllm_bindings": json.dumps(bindings)},
        }
    )


def cache_config(*, reverse_groups: bool = False) -> FakeConfig:
    groups = (
        FakeGroup(
            ("model.layers.0.main", "model.layers.1.main"),
            FakeSpec("main"),
        ),
        FakeGroup(("model.layers.0.indexer",), FakeSpec("indexer")),
    )
    return FakeConfig(tuple(reversed(groups)) if reverse_groups else groups)


def backends(*, reverse: bool = False) -> dict[str, type]:
    items = [
        ("model.layers.0.main", MainBackend),
        ("model.layers.1.main", MainBackend),
        ("model.layers.0.indexer", IndexerBackend),
    ]
    if reverse:
        items.reverse()
    return dict(items)


class CacheManifestTests(unittest.TestCase):
    def test_mla_probe_uses_each_groups_spec_type(self) -> None:
        mla_spec = type("MLAAttentionSpec", (), {})()
        sliding_mla_spec = type("SlidingWindowMLASpec", (), {})()

        self.assertTrue(is_mla_cache_spec(mla_spec))
        self.assertTrue(is_mla_cache_spec(sliding_mla_spec))
        self.assertFalse(is_mla_cache_spec(FakeSpec("standard")))

    def test_two_and_three_group_block_ids_are_never_flattened(self) -> None:
        class FakeBlocks:
            def __init__(self, groups: tuple[list[int], ...]) -> None:
                self.groups = groups

            def get_block_ids(self) -> tuple[list[int], ...]:
                return self.groups

        self.assertEqual(
            all_group_block_ids(FakeBlocks(([1, 2], [30]))),
            ((1, 2), (30,)),
        )
        self.assertEqual(
            all_group_block_ids(FakeBlocks(([1], [20, 21], [300]))),
            ((1,), (20, 21), (300,)),
        )

    def test_rejects_duplicate_resources(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate logical resource 10"):
            CacheManifest.from_json(manifest_json(duplicate_resource=True))

    def test_binding_encoding_is_canonical(self) -> None:
        parsed = CacheManifest.from_json(manifest_json())
        encoded = json.loads(parsed.to_json())
        binding_text = encoded["attributes"]["vllm_bindings"]

        self.assertEqual(
            binding_text,
            json.dumps(json.loads(binding_text), sort_keys=True, separators=(",", ":")),
        )

    def test_group_mapping_is_independent_of_input_order(self) -> None:
        parsed = CacheManifest.from_json(manifest_json())
        first = map_kv_cache_resources(parsed, cache_config(), backends())
        second = map_kv_cache_resources(
            parsed,
            cache_config(reverse_groups=True),
            backends(reverse=True),
        )

        self.assertEqual(first.layer_resources, second.layer_resources)
        self.assertEqual(
            first.layer_resources,
            {
                "model.layers.0.main": 10,
                "model.layers.1.main": 10,
                "model.layers.0.indexer": 11,
            },
        )
        self.assertEqual(first.group_resources, (10, 11))
        self.assertEqual(second.group_resources, (11, 10))

    def test_all_resource_destinations_are_preserved(self) -> None:
        mapping = map_kv_cache_resources(
            CacheManifest.from_json(manifest_json()), cache_config(), backends()
        )
        self.assertEqual(
            mapping.allocations(((1, 2), (30,))),
            ((10, [1, 2]), (11, [30])),
        )

    def test_resource_plans_accept_manifest_declared_divergent_shapes(self) -> None:
        base = cache_config()
        config = FakeConfig(
            base.kv_cache_groups,
            (
                FakeTensorConfig(("model.layers.0.main",)),
                FakeTensorConfig(("model.layers.1.main",)),
                FakeTensorConfig(("model.layers.0.indexer",)),
            ),
        )
        tensors = {
            "model.layers.0.main": FakeTensor((16, 2, 4)),
            "model.layers.1.main": FakeTensor((16, 2, 4)),
            "model.layers.0.indexer": FakeTensor((8, 3, 9)),
        }

        plans = build_resource_tensor_plans(
            CacheManifest.from_json(manifest_json()),
            config,
            tensors,
            lambda layers: MainBackend if layers[0].endswith("main") else IndexerBackend,
        )

        self.assertEqual([plan.resource for plan in plans], [10, 11])
        self.assertEqual([plan.primary for plan in plans], [True, False])
        self.assertEqual(plans[0].tensors[0].shape, (16, 2, 4))
        self.assertEqual(plans[1].tensors[0].shape, (8, 3, 9))

        reversed_plans = build_resource_tensor_plans(
            CacheManifest.from_json(manifest_json()),
            FakeConfig(tuple(reversed(config.kv_cache_groups)), config.kv_cache_tensors),
            tensors,
            lambda layers: MainBackend if layers[0].endswith("main") else IndexerBackend,
        )
        self.assertEqual([plan.resource for plan in reversed_plans], [10, 11])
        self.assertEqual([plan.primary for plan in reversed_plans], [True, False])

    def test_unknown_layer_fails_registration(self) -> None:
        parsed = CacheManifest.from_json(manifest_json())
        config = FakeConfig(
            (
                *cache_config().kv_cache_groups,
                FakeGroup(("model.layers.9.unknown",), FakeSpec("unknown")),
            )
        )
        layer_backends = {**backends(), "model.layers.9.unknown": MainBackend}

        with self.assertRaisesRegex(ValueError, "no manifest binding matches"):
            map_kv_cache_resources(parsed, config, layer_backends)

    def test_backend_mismatch_fails_registration(self) -> None:
        parsed = CacheManifest.from_json(manifest_json())
        layer_backends = {**backends(), "model.layers.0.indexer": MainBackend}

        with self.assertRaisesRegex(ValueError, "no manifest binding matches"):
            map_kv_cache_resources(parsed, cache_config(), layer_backends)

    def test_every_manifest_resource_requires_a_vllm_binding(self) -> None:
        value = json.loads(manifest_json())
        bindings = json.loads(value["attributes"]["vllm_bindings"])
        value["attributes"]["vllm_bindings"] = json.dumps(bindings[:1])

        with self.assertRaisesRegex(ValueError, r"resources without vLLM bindings: \[11\]"):
            map_kv_cache_resources(
                CacheManifest.from_json(json.dumps(value)), cache_config(), backends()
            )


if __name__ == "__main__":
    unittest.main()
