# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for export_factors_yaml — round-trip, composite generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from nautilus_quants.alpha.registry.database import RegistryDatabase
from nautilus_quants.alpha.registry.export import export_factors_yaml
from nautilus_quants.alpha.registry.models import FactorRecord
from nautilus_quants.alpha.registry.repository import FactorRepository
from nautilus_quants.factors.config import load_factor_config


@pytest.fixture()
def repo() -> FactorRepository:
    db = RegistryDatabase(":memory:")
    return FactorRepository(db)


def _seed_factors(repo: FactorRepository, n: int = 3) -> None:
    for i in range(n):
        repo.upsert_factor(FactorRecord(
            factor_id=f"alpha{i:03d}",
            expression=f"rank(close, {i + 1})",
            prototype=f"alpha{i:03d}",
            source="test",
            status="active",
            tags=["momentum"],
            parameters={"w": i + 1},
            variables={},
        ))


def test_export_creates_file(repo: FactorRepository, tmp_path: Path) -> None:
    _seed_factors(repo)
    out = tmp_path / "factors.yaml"
    export_factors_yaml(repo, out, status="active")
    assert out.exists()


def test_export_loadable(repo: FactorRepository, tmp_path: Path) -> None:
    _seed_factors(repo)
    out = tmp_path / "factors.yaml"
    export_factors_yaml(repo, out, status="active")
    config = load_factor_config(out)
    assert len(config.factors) >= 3  # 3 factors + composite


def test_export_composite_present(
    repo: FactorRepository, tmp_path: Path,
) -> None:
    _seed_factors(repo)
    out = tmp_path / "factors.yaml"
    export_factors_yaml(repo, out, status="active")
    config = load_factor_config(out)
    # composite is in all_factors (auto-generated from composite section)
    names = {f.name for f in config.all_factors}
    assert "composite" in names


def test_export_equal_weights(
    repo: FactorRepository, tmp_path: Path,
) -> None:
    _seed_factors(repo, 2)
    out = tmp_path / "factors.yaml"
    export_factors_yaml(
        repo, out, status="active", composite_method="equal",
    )
    config = load_factor_config(out)
    composite = config.get_factor("composite")
    # composite is now in all_factors, not factors
    assert composite is None  # not in base factors
    all_names = {f.name for f in config.all_factors}
    assert "composite" in all_names


def test_export_empty_registry(
    repo: FactorRepository, tmp_path: Path,
) -> None:
    out = tmp_path / "factors.yaml"
    export_factors_yaml(repo, out, status="active")
    config = load_factor_config(out)
    # No factors, no composite section generated
    assert len(config.factors) == 0
    assert len(config.all_factors) == 0


def test_export_with_context(
    repo: FactorRepository, tmp_path: Path,
) -> None:
    _seed_factors(repo)
    # Save a config snapshot as context
    cid = repo.save_config_snapshot(
        {
            "metadata": {"name": "test", "source": "alpha101"},
            "parameters": {"w": 5},
            "variables": {"returns": "delta(close,1)/delay(close,1)"},
        },
        "factors",
        config_name="test",
    )
    out = tmp_path / "factors.yaml"
    export_factors_yaml(repo, out, context_id=cid, status="active")
    config = load_factor_config(out)
    assert config.parameters.get("w") == 5
    assert "returns" in config.variables


def test_export_preserves_tags(
    repo: FactorRepository, tmp_path: Path,
) -> None:
    repo.upsert_factor(FactorRecord(
        factor_id="f1",
        expression="rank(close)",
        tags=["reversal", "volume"],
        status="active",
    ))
    out = tmp_path / "factors.yaml"
    export_factors_yaml(repo, out, status="active")
    config = load_factor_config(out)
    f1 = config.get_factor("f1")
    assert f1 is not None
    assert "reversal" in f1.tags
