# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for alpha CLI registry commands: register, list, inspect, status, export-factors."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from nautilus_quants.alpha.cli import cli


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def factors_yaml(tmp_path: Path) -> Path:
    """Write a small test factors.yaml."""
    doc = {
        "metadata": {"name": "test_config", "version": "1.0"},
        "variables": {"ret": "delta(close, 1)"},
        "parameters": {"window": 24},
        "factors": {
            "alpha001": {
                "expression": "rank(close)",
                "description": "Test factor 1",
                "category": "momentum",
            },
            "alpha002": {
                "expression": "ts_std(close, 24)",
                "description": "Test factor 2",
                "category": "volatility",
            },
            "composite": {
                "expression": "0.5 * cs_rank(alpha001) + 0.5 * cs_rank(alpha002)",
            },
        },
    }
    path = tmp_path / "factors.yaml"
    with open(path, "w") as f:
        yaml.dump(doc, f)
    return path


@pytest.fixture()
def db_path(tmp_path: Path) -> str:
    return str(tmp_path / "test_registry.duckdb")


class TestRegister:
    def test_register_new_factors(
        self, runner: CliRunner, factors_yaml: Path, db_path: str,
    ) -> None:
        result = runner.invoke(cli, [
            "register", str(factors_yaml), "--source", "test", "--db", db_path,
        ])
        assert result.exit_code == 0
        assert "3" in result.output  # 3 factors (including composite)
        assert "新增" in result.output or "new" in result.output.lower()

    def test_register_idempotent(
        self, runner: CliRunner, factors_yaml: Path, db_path: str,
    ) -> None:
        runner.invoke(cli, ["register", str(factors_yaml), "--db", db_path])
        result = runner.invoke(cli, ["register", str(factors_yaml), "--db", db_path])
        assert result.exit_code == 0
        assert "无变化" in result.output or "unchanged" in result.output.lower()

    def test_register_nonexistent_file(
        self, runner: CliRunner, db_path: str,
    ) -> None:
        result = runner.invoke(cli, [
            "register", "/nonexistent.yaml", "--db", db_path,
        ])
        assert result.exit_code != 0


class TestList:
    def test_list_all(
        self, runner: CliRunner, factors_yaml: Path, db_path: str,
    ) -> None:
        runner.invoke(cli, ["register", str(factors_yaml), "--db", db_path])
        result = runner.invoke(cli, ["list", "--db", db_path])
        assert result.exit_code == 0
        assert "alpha001" in result.output
        assert "alpha002" in result.output

    def test_list_filter_status(
        self, runner: CliRunner, factors_yaml: Path, db_path: str,
    ) -> None:
        runner.invoke(cli, ["register", str(factors_yaml), "--db", db_path])
        result = runner.invoke(cli, ["list", "--status", "active", "--db", db_path])
        assert result.exit_code == 0
        # No active factors yet — should still succeed with empty output.
        assert "alpha001" not in result.output

    def test_list_with_limit(
        self, runner: CliRunner, factors_yaml: Path, db_path: str,
    ) -> None:
        runner.invoke(cli, ["register", str(factors_yaml), "--db", db_path])
        result = runner.invoke(cli, ["list", "--limit", "1", "--db", db_path])
        assert result.exit_code == 0


class TestInspect:
    def test_inspect_existing(
        self, runner: CliRunner, factors_yaml: Path, db_path: str,
    ) -> None:
        runner.invoke(cli, ["register", str(factors_yaml), "--db", db_path])
        result = runner.invoke(cli, ["inspect", "alpha001", "--db", db_path])
        assert result.exit_code == 0
        assert "alpha001" in result.output
        assert "rank(close)" in result.output

    def test_inspect_nonexistent(
        self, runner: CliRunner, db_path: str,
    ) -> None:
        result = runner.invoke(cli, ["inspect", "nonexistent", "--db", db_path])
        assert result.exit_code != 0


class TestStatus:
    def test_status_transition(
        self, runner: CliRunner, factors_yaml: Path, db_path: str,
    ) -> None:
        runner.invoke(cli, ["register", str(factors_yaml), "--db", db_path])
        result = runner.invoke(cli, [
            "status", "alpha001", "active", "--db", db_path,
        ])
        assert result.exit_code == 0
        assert "active" in result.output

    def test_status_invalid_transition(
        self, runner: CliRunner, factors_yaml: Path, db_path: str,
    ) -> None:
        runner.invoke(cli, ["register", str(factors_yaml), "--db", db_path])
        result = runner.invoke(cli, [
            "status", "alpha001", "archived", "--db", db_path,
        ])
        assert result.exit_code != 0


class TestExportFactors:
    def test_export_creates_file(
        self, runner: CliRunner, factors_yaml: Path, db_path: str, tmp_path: Path,
    ) -> None:
        runner.invoke(cli, ["register", str(factors_yaml), "--db", db_path])
        # Set factors as active first.
        runner.invoke(cli, ["status", "alpha001", "active", "--db", db_path])
        runner.invoke(cli, ["status", "alpha002", "active", "--db", db_path])

        out = tmp_path / "exported.yaml"
        result = runner.invoke(cli, [
            "export-factors", "--top", "10", "-o", str(out), "--db", db_path,
        ])
        assert result.exit_code == 0
        assert out.exists()

    def test_export_round_trip(
        self, runner: CliRunner, factors_yaml: Path, db_path: str, tmp_path: Path,
    ) -> None:
        runner.invoke(cli, ["register", str(factors_yaml), "--db", db_path])
        runner.invoke(cli, ["status", "alpha001", "active", "--db", db_path])
        runner.invoke(cli, ["status", "alpha002", "active", "--db", db_path])

        out = tmp_path / "exported.yaml"
        runner.invoke(cli, [
            "export-factors", "--top", "10", "-o", str(out), "--db", db_path,
        ])

        from nautilus_quants.factors.config import load_factor_config
        config = load_factor_config(out)
        factor_names = {f.name for f in config.factors}
        assert "alpha001" in factor_names
        assert "composite" in factor_names
