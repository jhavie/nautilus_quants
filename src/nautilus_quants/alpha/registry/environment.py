# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Environment resolution for multi-environment registry databases."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from nautilus_quants.alpha.registry.models import RegistryConfig

VALID_ENVS = {"test", "test_rc", "dev", "dev_rc", "prod"}
DEFAULT_ENV = "test"
DEFAULT_DB_DIR = "logs/registry"


def resolve_env(
    explicit: str | None = None,
    yaml_env: str | None = None,
) -> str:
    """Resolve environment from explicit arg, YAML, or env var.

    Priority: explicit > yaml_env > NAUTILUS_QUANTS_ENV > "test"
    """
    env = (
        explicit
        or yaml_env
        or os.environ.get("NAUTILUS_QUANTS_ENV")
        or DEFAULT_ENV
    )
    if env not in VALID_ENVS:
        raise ValueError(
            f"Invalid env: {env!r}. Must be one of {sorted(VALID_ENVS)}"
        )
    return env


def resolve_db_path(
    env: str = DEFAULT_ENV,
    db_dir: str | Path = DEFAULT_DB_DIR,
) -> Path:
    """Resolve database file path for given environment."""
    return Path(db_dir) / f"{env}.duckdb"


def parse_registry_config(raw: dict[str, Any] | None) -> RegistryConfig:
    """Parse a ``registry`` YAML section into RegistryConfig."""
    if not raw:
        return RegistryConfig()
    return RegistryConfig(
        env=raw.get("env", DEFAULT_ENV),
        db_dir=raw.get("db_dir", DEFAULT_DB_DIR),
        enabled=raw.get("enabled", True),
    )
