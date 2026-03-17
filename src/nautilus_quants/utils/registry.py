# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Metadata renderer registry for configuration-driven injection."""

from typing import Callable

from nautilus_quants.utils.protocols import BaseMetadataRenderer, MetadataRenderer

RendererFactory = Callable[[], MetadataRenderer]


class RendererRegistry:
    """Registry for MetadataRenderer implementations.

    Provides configuration-driven renderer lookup by name.
    """

    _renderers: dict[str, RendererFactory] = {}

    @classmethod
    def register(cls, name: str, factory: RendererFactory) -> None:
        cls._renderers[name] = factory

    @classmethod
    def register_as(cls, name: str):
        def decorator(renderer_cls):
            cls._renderers[name] = renderer_cls
            return renderer_cls
        return decorator

    @classmethod
    def get(cls, name: str | None) -> MetadataRenderer:
        if name is None or name == "base":
            return BaseMetadataRenderer()

        if name not in cls._renderers:
            available = ", ".join(sorted(cls._renderers.keys()))
            raise ValueError(
                f"Unknown metadata_renderer: '{name}'. "
                f"Available: {available or 'none (use base)'}"
            )

        return cls._renderers[name]()

    @classmethod
    def list_available(cls) -> list[str]:
        return ["base"] + sorted(cls._renderers.keys())
