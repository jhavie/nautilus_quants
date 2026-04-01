# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for export_factors_yaml — round-trip, composite generation, context inclusion."""

from __future__ import annotations

from pathlib import Path

import pytest

from nautilus_quants.alpha.registry.database import RegistryDatabase
from nautilus_quants.alpha.registry.export import export_factors_yaml
from nautilus_quants.alpha.registry.models import ConfigContext, FactorRecord
from nautilus_quants.alpha.registry.repository import FactorRepository
from nautilus_quants.factors.config import load_factor_config


@pytest.fixture()
def repo() -> FactorRepository:
    db = RegistryDatabase(":memory:")
    repository = FactorRepository(db)
    yield repository
    db.close()


def _seed_active_factors(repo: FactorRepository, n: int = 5) -> None:
    """Insert n active factors."""
    for i in range(n):
        repo.upsert_factor(
            FactorRecord(
                factor_id=f"alpha{i:03d}",
                expression=f"ts_rank(close, {10 + i})",
                description=f"Test factor {i}",
                category="momentum" if i % 2 == 0 else "volume",
                source="test",
                status="active",
            )
        )


class TestExportBasic:
    def test_export_creates_file(self, repo: FactorRepository, tmp_path: Path) -> None:
        _seed_active_factors(repo, 3)
        out = tmp_path / "factors.yaml"
        export_factors_yaml(repo, out)
        assert out.exists()

    def test_round_trip_load(self, repo: FactorRepository, tmp_path: Path) -> None:
        """Exported YAML must be loadable by load_factor_config."""
        _seed_active_factors(repo, 3)
        out = tmp_path / "factors.yaml"
        export_factors_yaml(repo, out)
        config = load_factor_config(out)
        # 3 individual factors + 1 composite
        assert len(config.factors) == 4
        factor_names = {f.name for f in config.factors}
        assert "composite" in factor_names
        assert "alpha000" in factor_names

    def test_equal_weights(self, repo: FactorRepository, tmp_path: Path) -> None:
        _seed_active_factors(repo, 4)
        out = tmp_path / "factors.yaml"
        export_factors_yaml(repo, out, composite_method="equal")
        config = load_factor_config(out)
        composite = config.get_factor("composite")
        assert composite is not None
        # Each factor should have weight 0.25
        assert "0.2500" in composite.expression

    def test_top_n_limits(self, repo: FactorRepository, tmp_path: Path) -> None:
        _seed_active_factors(repo, 10)
        out = tmp_path / "factors.yaml"
        export_factors_yaml(repo, out, composite_top_n=3)
        config = load_factor_config(out)
        # 3 individual + 1 composite
        assert len(config.factors) == 4


class TestExportWithContext:
    def test_variables_preserved(self, repo: FactorRepository, tmp_path: Path) -> None:
        _seed_active_factors(repo, 2)
        repo.upsert_context(ConfigContext(
            context_id="test_ctx",
            variables={"returns": "delta(close, 1) / delay(close, 1)"},
        ))
        out = tmp_path / "factors.yaml"
        export_factors_yaml(repo, out, context_id="test_ctx")
        config = load_factor_config(out)
        assert config.variables == {"returns": "delta(close, 1) / delay(close, 1)"}

    def test_parameters_preserved(self, repo: FactorRepository, tmp_path: Path) -> None:
        _seed_active_factors(repo, 2)
        repo.upsert_context(ConfigContext(
            context_id="test_ctx",
            parameters={"short_window": 24, "long_window": 96},
        ))
        out = tmp_path / "factors.yaml"
        export_factors_yaml(repo, out, context_id="test_ctx")
        config = load_factor_config(out)
        assert config.parameters == {"short_window": 24, "long_window": 96}

    def test_no_context_still_works(self, repo: FactorRepository, tmp_path: Path) -> None:
        """Export without context_id should work (empty variables/parameters)."""
        _seed_active_factors(repo, 2)
        out = tmp_path / "factors.yaml"
        export_factors_yaml(repo, out)
        config = load_factor_config(out)
        assert config.variables == {}
        assert config.parameters == {}


class TestExportTransform:
    def test_cs_rank_transform(self, repo: FactorRepository, tmp_path: Path) -> None:
        _seed_active_factors(repo, 2)
        out = tmp_path / "factors.yaml"
        export_factors_yaml(repo, out, composite_transform="cs_rank")
        config = load_factor_config(out)
        composite = config.get_factor("composite")
        assert "cs_rank(alpha000)" in composite.expression

    def test_raw_transform(self, repo: FactorRepository, tmp_path: Path) -> None:
        _seed_active_factors(repo, 2)
        out = tmp_path / "factors.yaml"
        export_factors_yaml(repo, out, composite_transform="raw")
        config = load_factor_config(out)
        composite = config.get_factor("composite")
        # raw = no wrapping function
        assert "alpha000" in composite.expression
        assert "cs_rank" not in composite.expression


class TestExportExcludesComposite:
    """Fix P2: existing 'composite' factor must not be overwritten in export."""

    def test_registered_composite_excluded(
        self, repo: FactorRepository, tmp_path: Path,
    ) -> None:
        repo.upsert_factor(FactorRecord(
            factor_id="composite",
            expression="original_composite_expr",
            status="active",
            source="user",
        ))
        _seed_active_factors(repo, 3)
        out = tmp_path / "factors.yaml"
        export_factors_yaml(repo, out)
        config = load_factor_config(out)
        composite = config.get_factor("composite")
        assert composite is not None
        # Should be the auto-generated expression, NOT the user's original.
        assert "original_composite_expr" not in composite.expression
        assert "cs_rank" in composite.expression

    def test_composite_not_counted_in_top_n(
        self, repo: FactorRepository, tmp_path: Path,
    ) -> None:
        repo.upsert_factor(FactorRecord(
            factor_id="composite", expression="old", status="active",
        ))
        _seed_active_factors(repo, 5)
        out = tmp_path / "factors.yaml"
        export_factors_yaml(repo, out, composite_top_n=3)
        config = load_factor_config(out)
        # 3 individual + 1 synthetic composite = 4
        assert len(config.factors) == 4


class TestExportEmpty:
    def test_no_active_factors(self, repo: FactorRepository, tmp_path: Path) -> None:
        """Should still produce a valid YAML with empty composite."""
        out = tmp_path / "factors.yaml"
        export_factors_yaml(repo, out)
        config = load_factor_config(out)
        # Only composite (with value 0)
        assert len(config.factors) >= 1
