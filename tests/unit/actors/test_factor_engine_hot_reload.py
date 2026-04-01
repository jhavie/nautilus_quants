# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for FactorEngineActor hot-reload mechanism (Clock Timer based)."""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from nautilus_quants.actors.factor_engine import (
    FactorEngineActorConfig,
    _HOT_RELOAD_TIMER,
    _try_reload_factors,
)
from nautilus_quants.factors.config import FactorConfig, FactorDefinition


def _write_factors_yaml(path: Path, factors: dict[str, str]) -> None:
    """Write a minimal factors.yaml."""
    doc = {
        "metadata": {"name": "test", "version": "1.0"},
        "factors": {k: {"expression": v} for k, v in factors.items()},
    }
    with open(path, "w") as f:
        yaml.dump(doc, f)


# ---------------------------------------------------------------------------
# Config fields
# ---------------------------------------------------------------------------


class TestConfigFields:
    def test_defaults(self) -> None:
        config = FactorEngineActorConfig(factor_config_path="dummy.yaml")
        assert config.enable_hot_reload is False
        assert config.hot_reload_interval_secs == 300

    def test_custom_values(self) -> None:
        config = FactorEngineActorConfig(
            factor_config_path="dummy.yaml",
            enable_hot_reload=True,
            hot_reload_interval_secs=60,
        )
        assert config.enable_hot_reload is True
        assert config.hot_reload_interval_secs == 60


# ---------------------------------------------------------------------------
# _try_reload_factors (extracted logic, no Nautilus dependency)
# ---------------------------------------------------------------------------


class TestTryReloadFactors:
    def test_mtime_unchanged_returns_none(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "factors.yaml"
        _write_factors_yaml(yaml_path, {"f1": "close"})
        current_mtime = yaml_path.stat().st_mtime

        result = _try_reload_factors(str(yaml_path), current_mtime)
        assert result is None  # No change detected

    def test_mtime_changed_returns_new_config(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "factors.yaml"
        _write_factors_yaml(yaml_path, {"f1": "close"})

        # Pass an old mtime so the function detects a change.
        result = _try_reload_factors(str(yaml_path), 0.0)
        assert result is not None
        new_config, new_mtime = result
        assert isinstance(new_config, FactorConfig)
        assert len(new_config.factors) == 1
        assert new_mtime > 0.0

    def test_file_not_found_returns_none(self) -> None:
        result = _try_reload_factors("/nonexistent/path.yaml", 0.0)
        assert result is None

    def test_invalid_yaml_returns_none(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "factors.yaml"
        yaml_path.write_text("invalid: yaml: content: [[[")

        result = _try_reload_factors(str(yaml_path), 0.0)
        assert result is None

    def test_validation_failure_returns_none(self, tmp_path: Path) -> None:
        """Factor with empty expression should fail validation."""
        yaml_path = tmp_path / "factors.yaml"
        doc = {
            "metadata": {"name": "test", "version": "1.0"},
            "factors": {"bad": {"expression": ""}},
        }
        with open(yaml_path, "w") as f:
            yaml.dump(doc, f)

        result = _try_reload_factors(str(yaml_path), 0.0)
        assert result is None

    def test_variables_included_in_config(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "factors.yaml"
        doc = {
            "metadata": {"name": "test", "version": "1.0"},
            "variables": {"ret": "delta(close, 1)"},
            "factors": {"f1": {"expression": "cs_rank(ret)"}},
        }
        with open(yaml_path, "w") as f:
            yaml.dump(doc, f)

        result = _try_reload_factors(str(yaml_path), 0.0)
        assert result is not None
        new_config, _ = result
        assert new_config.variables == {"ret": "delta(close, 1)"}

    def test_detects_actual_file_modification(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "factors.yaml"
        _write_factors_yaml(yaml_path, {"f1": "close"})
        mtime1 = yaml_path.stat().st_mtime

        # Ensure mtime changes (some filesystems have 1s resolution).
        time.sleep(0.05)
        _write_factors_yaml(yaml_path, {"f1": "open", "f2": "volume"})

        result = _try_reload_factors(str(yaml_path), mtime1)
        assert result is not None
        new_config, new_mtime = result
        assert len(new_config.factors) == 2
        assert new_mtime > mtime1


# ---------------------------------------------------------------------------
# Timer constant
# ---------------------------------------------------------------------------


class TestTimerConstant:
    def test_timer_name_defined(self) -> None:
        assert _HOT_RELOAD_TIMER == "factor_config_hot_reload"
