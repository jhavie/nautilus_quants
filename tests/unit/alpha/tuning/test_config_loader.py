# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for config_loader.py — YAML → TuneConfig coercion + overrides."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from nautilus_quants.alpha.tuning.config_loader import build_tune_config, load_tune_config


def _write_yaml(tmp_path: Path, contents: dict) -> Path:
    path = tmp_path / "tune.yaml"
    path.write_text(yaml.safe_dump(contents))
    return path


class TestBuildTuneConfig:
    def test_empty_mapping_yields_defaults(self) -> None:
        cfg = build_tune_config({})
        assert cfg.enabled is False
        assert cfg.dimensions.numeric is True
        assert cfg.dimensions.operators is False
        assert cfg.by_prototype is True
        assert cfg.trials == 50

    def test_dimensions_overrides_apply(self) -> None:
        cfg = build_tune_config(
            {"dimensions": {"numeric": False, "operators": True}},
        )
        assert cfg.dimensions.numeric is False
        assert cfg.dimensions.operators is True

    def test_eligibility_parses_nested(self) -> None:
        cfg = build_tune_config(
            {
                "candidates": {
                    "status": "active",
                    "tags": ["volume", "mean_reversion"],
                    "eligibility": {
                        "icir_abs_min": 0.05,
                        "coverage_min": 0.5,
                        "n_samples_min": 1000,
                    },
                }
            }
        )
        assert cfg.candidates.status == "active"
        assert cfg.candidates.tags == ("volume", "mean_reversion")
        assert cfg.candidates.eligibility.icir_abs_min == 0.05
        assert cfg.candidates.eligibility.coverage_min == 0.5

    def test_forward_horizon_and_ic_weight_default_to_legacy(self) -> None:
        """Missing fields default to legacy behaviour (1-bar horizon, pure ICIR)."""
        cfg = build_tune_config({})
        assert cfg.forward_horizon_bars == 1
        assert cfg.ic_mean_weight == 0.0

    def test_forward_horizon_and_ic_weight_parsed_from_yaml(self) -> None:
        """YAML values feed straight through to TuneConfig."""
        cfg = build_tune_config(
            {"forward_horizon_bars": 2, "ic_mean_weight": 0.5}
        )
        assert cfg.forward_horizon_bars == 2
        assert cfg.ic_mean_weight == 0.5

    def test_cli_overrides_win_over_yaml(self) -> None:
        cfg = build_tune_config(
            {"trials": 10, "dimensions": {"operators": False}},
            overrides={
                "trials": 75,
                "dimensions.operators": True,
            },
        )
        assert cfg.trials == 75
        assert cfg.dimensions.operators is True


class TestLoadTuneConfig:
    def test_reads_top_level_tune_section(self, tmp_path: Path) -> None:
        p = _write_yaml(
            tmp_path,
            {"tune": {"enabled": True, "trials": 20}},
        )
        cfg = load_tune_config(p)
        assert cfg.enabled is True
        assert cfg.trials == 20

    def test_falls_back_to_root_when_no_tune_key(self, tmp_path: Path) -> None:
        p = _write_yaml(
            tmp_path,
            {"enabled": True, "trials": 15, "dimensions": {"numeric": False}},
        )
        cfg = load_tune_config(p)
        assert cfg.enabled is True
        assert cfg.trials == 15
        assert cfg.dimensions.numeric is False

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_tune_config(tmp_path / "nope.yaml")

    def test_non_mapping_root_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "bogus.yaml"
        p.write_text("- just\n- a\n- list\n")
        with pytest.raises(ValueError, match="must be a mapping"):
            load_tune_config(p)
