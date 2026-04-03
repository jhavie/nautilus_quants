# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for environment resolution — env priority, db path, config parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from nautilus_quants.alpha.registry.environment import (
    DEFAULT_DB_DIR,
    DEFAULT_ENV,
    parse_registry_config,
    resolve_db_path,
    resolve_env,
)
from nautilus_quants.alpha.registry.models import RegistryConfig


# ---------------------------------------------------------------------------
# resolve_env
# ---------------------------------------------------------------------------


class TestResolveEnv:
    def test_explicit_takes_priority(self) -> None:
        result = resolve_env(explicit="prod", yaml_env="dev")
        assert result == "prod"

    def test_yaml_env_second_priority(self) -> None:
        result = resolve_env(explicit=None, yaml_env="dev")
        assert result == "dev"

    def test_env_var_third_priority(self, monkeypatch) -> None:
        monkeypatch.setenv("NAUTILUS_QUANTS_ENV", "prod")
        result = resolve_env(explicit=None, yaml_env=None)
        assert result == "prod"

    def test_default_is_test(self, monkeypatch) -> None:
        monkeypatch.delenv("NAUTILUS_QUANTS_ENV", raising=False)
        result = resolve_env(explicit=None, yaml_env=None)
        assert result == DEFAULT_ENV
        assert result == "test"

    def test_explicit_overrides_env_var(self, monkeypatch) -> None:
        monkeypatch.setenv("NAUTILUS_QUANTS_ENV", "prod")
        result = resolve_env(explicit="dev")
        assert result == "dev"

    def test_yaml_overrides_env_var(self, monkeypatch) -> None:
        monkeypatch.setenv("NAUTILUS_QUANTS_ENV", "prod")
        result = resolve_env(explicit=None, yaml_env="dev")
        assert result == "dev"

    def test_all_valid_envs(self) -> None:
        for env in ("test", "dev", "prod"):
            assert resolve_env(explicit=env) == env

    def test_invalid_env_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid env"):
            resolve_env(explicit="staging")

    def test_invalid_yaml_env_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid env"):
            resolve_env(yaml_env="production")

    def test_invalid_env_var_raises(self, monkeypatch) -> None:
        monkeypatch.setenv("NAUTILUS_QUANTS_ENV", "staging")
        with pytest.raises(ValueError, match="Invalid env"):
            resolve_env()

    def test_no_args_no_env_var(self, monkeypatch) -> None:
        monkeypatch.delenv("NAUTILUS_QUANTS_ENV", raising=False)
        assert resolve_env() == "test"


# ---------------------------------------------------------------------------
# resolve_db_path
# ---------------------------------------------------------------------------


class TestResolveDbPath:
    def test_default_path(self) -> None:
        result = resolve_db_path()
        assert result == Path(DEFAULT_DB_DIR) / "test.duckdb"

    def test_custom_env(self) -> None:
        result = resolve_db_path(env="prod")
        assert result == Path(DEFAULT_DB_DIR) / "prod.duckdb"

    def test_custom_db_dir(self) -> None:
        result = resolve_db_path(db_dir="/custom/dir")
        assert result == Path("/custom/dir") / "test.duckdb"

    def test_custom_env_and_dir(self) -> None:
        result = resolve_db_path(env="dev", db_dir="/data/registry")
        assert result == Path("/data/registry") / "dev.duckdb"

    def test_returns_path_object(self) -> None:
        result = resolve_db_path()
        assert isinstance(result, Path)

    def test_path_with_string_dir(self) -> None:
        result = resolve_db_path(env="test", db_dir="logs/registry")
        assert result == Path("logs/registry/test.duckdb")

    def test_path_with_path_dir(self) -> None:
        result = resolve_db_path(env="test", db_dir=Path("logs/registry"))
        assert result == Path("logs/registry/test.duckdb")


# ---------------------------------------------------------------------------
# parse_registry_config
# ---------------------------------------------------------------------------


class TestParseRegistryConfig:
    def test_none_returns_defaults(self) -> None:
        config = parse_registry_config(None)
        assert config == RegistryConfig()
        assert config.env == "test"
        assert config.db_dir == "logs/registry"
        assert config.enabled is True

    def test_empty_dict_returns_defaults(self) -> None:
        config = parse_registry_config({})
        assert config.env == "test"
        assert config.db_dir == "logs/registry"
        assert config.enabled is True

    def test_full_config(self) -> None:
        raw = {
            "env": "prod",
            "db_dir": "/data/registry",
            "enabled": False,
        }
        config = parse_registry_config(raw)
        assert config.env == "prod"
        assert config.db_dir == "/data/registry"
        assert config.enabled is False

    def test_partial_config_env_only(self) -> None:
        config = parse_registry_config({"env": "dev"})
        assert config.env == "dev"
        assert config.db_dir == DEFAULT_DB_DIR
        assert config.enabled is True

    def test_partial_config_db_dir_only(self) -> None:
        config = parse_registry_config({"db_dir": "/custom"})
        assert config.env == DEFAULT_ENV
        assert config.db_dir == "/custom"
        assert config.enabled is True

    def test_partial_config_enabled_only(self) -> None:
        config = parse_registry_config({"enabled": False})
        assert config.env == DEFAULT_ENV
        assert config.db_dir == DEFAULT_DB_DIR
        assert config.enabled is False

    def test_returns_frozen_dataclass(self) -> None:
        config = parse_registry_config({"env": "dev"})
        assert isinstance(config, RegistryConfig)
        with pytest.raises(AttributeError):
            config.env = "prod"  # type: ignore[misc]

    def test_unknown_keys_ignored(self) -> None:
        raw = {"env": "dev", "unknown_key": "value", "extra": 42}
        config = parse_registry_config(raw)
        assert config.env == "dev"
