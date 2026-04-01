# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""End-to-end test: register → list → inspect → status → export → load_factor_config."""

from __future__ import annotations

from pathlib import Path

import yaml
from click.testing import CliRunner

from nautilus_quants.alpha.cli import cli
from nautilus_quants.factors.config import load_factor_config


def test_full_registry_workflow(tmp_path: Path) -> None:
    """Exercise the complete CLI workflow in sequence."""
    runner = CliRunner()
    db_path = str(tmp_path / "registry.duckdb")

    # 1. Create a test factors.yaml (mimics real fmz config).
    factors_yaml = tmp_path / "factors.yaml"
    doc = {
        "metadata": {"name": "e2e_test", "version": "1.0"},
        "variables": {"returns": "delta(close, 1) / delay(close, 1)"},
        "parameters": {"short_window": 24},
        "factors": {
            "volume": {
                "expression": "volume",
                "description": "Quote volume",
            },
            "momentum_3h": {
                "expression": "(close - delay(close, 3)) / delay(close, 3)",
                "description": "3-hour momentum",
                "category": "momentum",
            },
            "volatility": {
                "expression": "ts_std(close / open, 24)",
                "description": "24-hour volatility",
                "category": "volatility",
            },
            "composite": {
                "expression": "0.5 * cs_rank(volume) + 0.5 * cs_rank(momentum_3h)",
            },
        },
    }
    with open(factors_yaml, "w") as f:
        yaml.dump(doc, f)

    # 2. Register.
    result = runner.invoke(cli, [
        "register", str(factors_yaml), "--source", "e2e", "--db", db_path,
    ])
    assert result.exit_code == 0, result.output
    assert "4" in result.output  # 4 factors total

    # 3. List — all should be candidates.
    result = runner.invoke(cli, ["list", "--db", db_path])
    assert result.exit_code == 0
    assert "candidate" in result.output
    assert "momentum_3h" in result.output

    # 4. Inspect one factor.
    result = runner.invoke(cli, ["inspect", "momentum_3h", "--db", db_path])
    assert result.exit_code == 0
    assert "momentum" in result.output
    assert "v1" in result.output

    # 5. Set statuses to active.
    for fid in ("volume", "momentum_3h", "volatility"):
        result = runner.invoke(cli, ["status", fid, "active", "--db", db_path])
        assert result.exit_code == 0, result.output

    # 6. List active only.
    result = runner.invoke(cli, ["list", "--status", "active", "--db", db_path])
    assert result.exit_code == 0
    assert "volume" in result.output
    assert "composite" not in result.output  # composite is still candidate

    # 7. Export.
    export_path = tmp_path / "exported_factors.yaml"
    result = runner.invoke(cli, [
        "export-factors",
        "--method", "equal",
        "--top", "10",
        "--transform", "cs_rank",
        "-o", str(export_path),
        "--db", db_path,
    ])
    assert result.exit_code == 0, result.output
    assert export_path.exists()

    # 8. Round-trip: load_factor_config must succeed.
    config = load_factor_config(export_path)
    factor_names = {f.name for f in config.factors}
    assert "volume" in factor_names
    assert "momentum_3h" in factor_names
    assert "volatility" in factor_names
    assert "composite" in factor_names

    # Variables and parameters must be preserved from context.
    assert config.variables == {"returns": "delta(close, 1) / delay(close, 1)"}
    assert config.parameters == {"short_window": 24}

    # Composite expression should contain equal-weight terms.
    composite = config.get_factor("composite")
    assert composite is not None
    assert "cs_rank(volume)" in composite.expression
    assert "cs_rank(momentum_3h)" in composite.expression
