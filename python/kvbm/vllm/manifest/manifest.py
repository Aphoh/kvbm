# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable Python view of the KVBM cache manifest wire format."""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

_SCHEMA_VERSION = 1
_VLLM_BINDINGS_ATTRIBUTE = "vllm_bindings"


@dataclass(frozen=True, slots=True)
class ManifestResource:
    """One required logical cache resource."""

    resource: int
    role: str
    native_block_tokens: int


@dataclass(frozen=True, slots=True)
class CacheManifest:
    """Validated manifest plus its canonical JSON representation."""

    schema_version: int
    model: Mapping[str, Any]
    cache_abi: str
    resources: tuple[ManifestResource, ...]
    attributes: Mapping[str, str]

    @classmethod
    def from_json(cls, document: str) -> "CacheManifest":
        """Parse and validate the JSON accepted by Rust ``CacheManifest``."""
        try:
            value = json.loads(document)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid cache manifest JSON: {error}") from error
        if not isinstance(value, dict):
            raise ValueError("cache manifest must be a JSON object")

        schema_version = _required_int(value, "schema_version")
        if schema_version != _SCHEMA_VERSION:
            raise ValueError(f"unsupported cache manifest schema {schema_version}")
        model = _parse_model(value.get("model"))
        cache_abi = _required_text(value, "cache_abi")
        resources = _parse_resources(value.get("resources"))
        attributes = _parse_attributes(value.get("attributes", {}))
        return cls(
            schema_version=schema_version,
            model=MappingProxyType(model),
            cache_abi=cache_abi,
            resources=resources,
            attributes=MappingProxyType(attributes),
        )

    def to_json(self) -> str:
        """Return deterministic JSON suitable for ``register_manifest``."""
        value = {
            "schema_version": self.schema_version,
            "model": dict(self.model),
            "cache_abi": self.cache_abi,
            "resources": [
                {
                    "resource": resource.resource,
                    "role": resource.role,
                    "native_block_tokens": resource.native_block_tokens,
                }
                for resource in self.resources
            ],
            "attributes": dict(self.attributes),
        }
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    @property
    def resource_ids(self) -> frozenset[int]:
        return frozenset(resource.resource for resource in self.resources)

    def vllm_bindings(self) -> tuple[dict[str, Any], ...]:
        """Decode the cache-identity-scoped vLLM adapter bindings."""
        encoded = self.attributes.get(_VLLM_BINDINGS_ATTRIBUTE)
        if encoded is None:
            raise ValueError("cache manifest has no vllm_bindings attribute")
        value = json.loads(encoded)
        if not isinstance(value, list):
            raise ValueError("vllm_bindings must encode a JSON list")
        return tuple(value)


def _parse_model(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("cache manifest model must be an object")
    architecture = _required_text(value, "architecture")
    revision = _required_text(value, "revision")
    weights_digest = value.get("weights_digest")
    if (
        not isinstance(weights_digest, list)
        or len(weights_digest) != 32
        or any(not isinstance(byte, int) or not 0 <= byte <= 255 for byte in weights_digest)
    ):
        raise ValueError("model weights_digest must contain exactly 32 bytes")
    return {
        "architecture": architecture,
        "revision": revision,
        "weights_digest": tuple(weights_digest),
    }


def _parse_resources(value: Any) -> tuple[ManifestResource, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("cache manifest must contain at least one resource")
    resources: list[ManifestResource] = []
    seen: set[int] = set()
    for raw in value:
        if not isinstance(raw, dict):
            raise ValueError("cache manifest resource must be an object")
        resource = _required_int(raw, "resource")
        if resource in seen:
            raise ValueError(f"duplicate logical resource {resource}")
        seen.add(resource)
        role = _required_text(raw, "role")
        if role not in {"prefix_history", "boundary_capsule"}:
            raise ValueError(f"unknown logical resource role {role!r}")
        native_block_tokens = _required_int(raw, "native_block_tokens")
        if native_block_tokens <= 0:
            raise ValueError(
                f"logical resource {resource} has zero native_block_tokens"
            )
        resources.append(ManifestResource(resource, role, native_block_tokens))
    return tuple(sorted(resources, key=lambda item: item.resource))


def _parse_attributes(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("cache manifest attributes must be an object")
    attributes: dict[str, str] = {}
    for key, raw in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("cache manifest attribute names must be non-empty strings")
        if not isinstance(raw, str):
            raise ValueError(f"cache manifest attribute {key!r} must be a string")
        if key == _VLLM_BINDINGS_ATTRIBUTE:
            try:
                raw = json.dumps(
                    json.loads(raw), sort_keys=True, separators=(",", ":")
                )
            except json.JSONDecodeError as error:
                raise ValueError("vllm_bindings is not valid JSON") from error
        attributes[key] = raw
    return dict(sorted(attributes.items()))


def _required_text(value: Mapping[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ValueError(f"cache manifest {key} must be a non-empty string")
    return result


def _required_int(value: Mapping[str, Any], key: str) -> int:
    result = value.get(key)
    if not isinstance(result, int) or isinstance(result, bool):
        raise ValueError(f"cache manifest {key} must be an integer")
    return result
