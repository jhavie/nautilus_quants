# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""End-to-end test: register → list → inspect → status → export."""

from __future__ import annotations

from pathlib import Path

import yaml
from click.testing import CliRunner

from nautilus_quants.alpha.cli import cli
from nautilus_quants.factors.config import load_factor_config


def test_full_registry_workflow(tmp_path: Path) -> None:
    """Exercise the complete CLI workflow in sequence."""
    runner = CliRunner()
    db_dir = str(tmp_path / "registry")

    # 1. Create a test factors.yaml with v2 fields
    factors_yaml = tmp_path / "factors.yaml"
    doc = {
        "metadata": {
            "name": "e2e_test",
            "source": "e2e",
            "version": "1.0",
        },
        "variables": {"returns": "delta(close, 1) / delay(close, 1)"},
        "parameters": {"short_window": 24},
        "factors": {
            "volume": {
                "expression": "volume",
                "description": "Quote volume",
                "tags": ["volume"],
                "prototype": "volume",
            },
            "momentum_3h": {
                "expression": "(close - delay(close, 3)) / delay(close, 3)",
                "description": "3-hour momentum",
                "tags": ["momentum"],
                "prototype": "momentum",
            },
            "volatility": {
                "expression": "ts_std(close / open, 24)",
                "description": "24-hour volatility",
                "tags": ["volatility"],
                "prototype": "volatility",
            },
            "composite": {
                "expression": (
                    "0.5 * cs_rank(volume) + 0.5 * cs_rank(momentum_3h)"
                ),
                "tags": ["composite"],
                "prototype": "composite",
            },
        },
    }
    with open(factors_yaml, "w") as f:
        yaml.dump(doc, f)

    # 2. Register
    result = runner.invoke(cli, [
        "register", str(factors_yaml), "--db-dir", db_dir,
    ])
    assert result.exit_code == 0, result.output
    assert "4" in result.output

    # 3. List — all should be candidates
    result = runner.invoke(cli, ["list", "--db-dir", db_dir])
    assert result.exit_code == 0
    assert "candidate" in result.output
    assert "e2e_momentum_3h" in result.output  # factor_id = source_key

    # 4. Inspect one factor
    result = runner.invoke(cli, [
        "inspect", "e2e_momentum_3h", "--db-dir", db_dir,
    ])
    assert result.exit_code == 0
    assert "momentum" in result.output

    # 5. Set statuses to active
    for fid in ("e2e_volume", "e2e_momentum_3h", "e2e_volatility"):
        result = runner.invoke(cli, [
            "status", fid, "active", "--db-dir", db_dir,
        ])
        assert result.exit_code == 0, result.output

    # 6. List active only
    result = runner.invoke(cli, [
        "list", "--status", "active", "--db-dir", db_dir,
    ])
    assert result.exit_code == 0
    assert "e2e_volume" in result.output
    # composite is still candidate
    assert "e2e_composite" not in result.output

    # 7. Export
    export_path = tmp_path / "exported_factors.yaml"
    result = runner.invoke(cli, [
        "export-factors",
        "--method", "equal",
        "--top", "10",
        "--transform", "cs_rank",
        "-o", str(export_path),
        "--db-dir", db_dir,
    ])
    assert result.exit_code == 0, result.output
    assert export_path.exists()

    # 8. Round-trip: load_factor_config must succeed
    config = load_factor_config(export_path)
    factor_names = {f.name for f in config.factors}
    assert "composite" in factor_names
    assert len(factor_names) >= 4  # 3 active + composite
