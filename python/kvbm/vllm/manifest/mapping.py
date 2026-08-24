# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Map vLLM cache groups to manifest logical resources."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from collections.abc import Callable
from typing import Any, Mapping

from .compatibility import is_mla_cache_spec
from .manifest import CacheManifest


@dataclass(frozen=True, slots=True)
class ResourceMapping:
    """Validated mapping in vLLM group order and by stable layer name."""

    group_resources: tuple[int, ...]
    layer_resources: Mapping[str, int]

    def allocations(
        self, block_ids_by_group: tuple[tuple[int, ...], ...]
    ) -> tuple[tuple[int, list[int]], ...]:
        """Pair complete group allocations with their logical resources."""
        if len(block_ids_by_group) != len(self.group_resources):
            raise ValueError(
                f"vLLM returned {len(block_ids_by_group)} block-ID groups for "
                f"{len(self.group_resources)} manifest mappings"
            )
        if len(set(self.group_resources)) != len(self.group_resources):
            raise ValueError(
                "multiple vLLM backing groups for one logical resource require "
                "a multi-pool manifest registration"
            )
        return tuple(
            (resource, list(block_ids))
            for resource, block_ids in zip(
                self.group_resources, block_ids_by_group, strict=True
            )
        )


@dataclass(frozen=True, slots=True)
class ResourceTensorPlan:
    """One resource's ordered vLLM tensors and attention backend."""

    resource: int
    primary: bool
    tensors: tuple[Any, ...]
    backend: type
    use_mla: bool


def build_resource_tensor_plans(
    manifest: CacheManifest,
    kv_cache_config: Any,
    kv_caches: Mapping[str, Any],
    resolve_backend: Callable[[list[str]], type],
) -> tuple[ResourceTensorPlan, ...]:
    """Build deterministic per-resource registration plans."""
    groups = kv_cache_config.kv_cache_groups
    backends = [resolve_backend(list(group.layer_names)) for group in groups]
    layer_backends = {
        str(layer): backend
        for group, backend in zip(groups, backends, strict=True)
        for layer in group.layer_names
    }
    mapping = map_kv_cache_resources(manifest, kv_cache_config, layer_backends)
    primary = manifest.resources[0].resource
    plans: list[ResourceTensorPlan] = []
    seen_resources: set[int] = set()
    for group_index in sorted(
        range(len(groups)),
        key=lambda index: (
            mapping.group_resources[index] != primary,
            mapping.group_resources[index],
        ),
    ):
        group = groups[group_index]
        resource = mapping.group_resources[group_index]
        if resource in seen_resources:
            raise ValueError(
                f"logical resource {resource} has multiple vLLM backing groups; "
                "declare each backing pool as a separate manifest resource"
            )
        seen_resources.add(resource)
        group_layers = set(group.layer_names)
        tensors: list[Any] = []
        for tensor_config in kv_cache_config.kv_cache_tensors:
            shared_by = set(tensor_config.shared_by)
            if not shared_by & group_layers:
                continue
            if not shared_by <= group_layers:
                raise ValueError(
                    f"KV cache tensor spans manifest groups: {sorted(shared_by)}"
                )
            layer_name = tensor_config.shared_by[0]
            try:
                tensors.append(kv_caches[layer_name])
            except KeyError as error:
                raise ValueError(
                    f"vLLM did not register tensor for layer {layer_name!r}"
                ) from error
        if not tensors:
            raise ValueError(f"manifest resource {resource} has no registered KV tensors")
        plans.append(
            ResourceTensorPlan(
                resource=resource,
                primary=resource == primary,
                tensors=tuple(tensors),
                backend=backends[group_index],
                use_mla=is_mla_cache_spec(group.kv_cache_spec),
            )
        )
    return tuple(plans)


@dataclass(frozen=True, slots=True)
class _GroupFacts:
    layers: frozenset[str]
    spec_type: str
    backend: str


@dataclass(frozen=True, slots=True)
class _Binding:
    resource: int
    facts: _GroupFacts


def map_kv_cache_resources(
    manifest: CacheManifest,
    kv_cache_config: Any,
    layer_backends: Mapping[str, type] | None = None,
) -> ResourceMapping:
    """Validate and map every vLLM group without depending on input order."""
    bindings = _parse_bindings(manifest)
    group_resources: list[int] = []
    layer_resources: dict[str, int] = {}
    for group in kv_cache_config.kv_cache_groups:
        facts = _group_facts(group, layer_backends)
        matches = [
            binding
            for binding in bindings
            if binding.facts.layers == facts.layers
            and binding.facts.spec_type == facts.spec_type
            and (layer_backends is None or binding.facts.backend == facts.backend)
        ]
        if len(matches) != 1:
            layers = sorted(facts.layers)
            raise ValueError(
                "no manifest binding matches vLLM cache group "
                f"layers={layers}, spec_type={facts.spec_type!r}, "
                f"backend={facts.backend!r}"
                if not matches
                else "multiple manifest bindings match vLLM cache group "
                f"layers={layers}"
            )
        resource = matches[0].resource
        group_resources.append(resource)
        for layer in facts.layers:
            previous = layer_resources.setdefault(layer, resource)
            if previous != resource:
                raise ValueError(
                    f"vLLM layer {layer!r} maps to resources {previous} and {resource}"
                )

    bound_layers = {layer for binding in bindings for layer in binding.facts.layers}
    observed_layers = set(layer_resources)
    missing = sorted(bound_layers - observed_layers)
    if missing:
        raise ValueError(f"manifest bindings name unknown vLLM layers: {missing}")
    return ResourceMapping(tuple(group_resources), MappingProxyType(layer_resources))


def _parse_bindings(manifest: CacheManifest) -> tuple[_Binding, ...]:
    bindings: list[_Binding] = []
    for resource_entry in manifest.vllm_bindings():
        if not isinstance(resource_entry, dict):
            raise ValueError("vllm binding entry must be an object")
        resource = resource_entry.get("resource")
        if resource not in manifest.resource_ids:
            raise ValueError(f"vllm binding names unknown logical resource {resource!r}")
        groups = resource_entry.get("groups")
        if not isinstance(groups, list) or not groups:
            raise ValueError(f"logical resource {resource} has no vLLM group bindings")
        for group in groups:
            if not isinstance(group, dict):
                raise ValueError("vllm group binding must be an object")
            layers = group.get("layers")
            if (
                not isinstance(layers, list)
                or not layers
                or any(not isinstance(layer, str) or not layer for layer in layers)
            ):
                raise ValueError("vllm group binding layers must be non-empty strings")
            if len(set(layers)) != len(layers):
                raise ValueError("vllm group binding contains a duplicate layer")
            spec_type = group.get("spec_type")
            backend = group.get("backend")
            if not isinstance(spec_type, str) or not spec_type:
                raise ValueError("vllm group binding spec_type must be a string")
            if not isinstance(backend, str) or not backend:
                raise ValueError("vllm group binding backend must be a string")
            bindings.append(
                _Binding(
                    resource,
                    _GroupFacts(frozenset(layers), spec_type, backend),
                )
            )
    if len({binding.facts for binding in bindings}) != len(bindings):
        raise ValueError("vllm_bindings contains duplicate group selectors")
    missing_resources = sorted(
        manifest.resource_ids - {binding.resource for binding in bindings}
    )
    if missing_resources:
        raise ValueError(
            f"manifest resources without vLLM bindings: {missing_resources}"
        )
    return tuple(bindings)


def _group_facts(
    group: Any, layer_backends: Mapping[str, type] | None
) -> _GroupFacts:
    layers = frozenset(str(layer) for layer in group.layer_names)
    if not layers:
        raise ValueError("vLLM cache group contains no layers")
    backend_names: set[str] = set()
    if layer_backends is not None:
        try:
            backend_names = {layer_backends[layer].__name__ for layer in layers}
        except KeyError as error:
            raise ValueError(
                f"vLLM layer {error.args[0]!r} has no attention backend"
            ) from error
        if len(backend_names) != 1:
            raise ValueError(
                f"vLLM cache group mixes attention backends: {sorted(backend_names)}"
            )
    spec = group.kv_cache_spec
    return _GroupFacts(
        layers,
        type(spec).__name__,
        backend_names.pop() if backend_names else "",
    )
