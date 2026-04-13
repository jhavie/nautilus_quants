# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for variant_registration.py.

The tests avoid requiring real alphalens data — we stub ``analyze_variant``
via monkeypatching when exercising the registration path, and call the real
implementation only on the small synthetic panels in the smoke test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nautilus_quants.alpha.registry.database import RegistryDatabase
from nautilus_quants.alpha.registry.models import FactorRecord
from nautilus_quants.alpha.registry.repository import FactorRepository
from nautilus_quants.alpha.tuning import variant_registration
from nautilus_quants.alpha.tuning.config import TrialResult, TuneConfig, TuneResult
from nautilus_quants.alpha.tuning.variant_registration import (
    RegisteredVariant,
    RegistrationSummary,
    _make_variant_id,
    register_tuned_variants,
)


@pytest.fixture()
def repo() -> FactorRepository:
    db = RegistryDatabase(":memory:")
    yield FactorRepository(db)
    db.close()


def _source_factor() -> FactorRecord:
    return FactorRecord(
        factor_id="alpha101_alpha044_8h",
        expression="-correlation(high, rank(volume), 5)",
        prototype="alpha044",
        source="alpha101",
        status="candidate",
        tags=["volume", "reversal"],
        parameters={"w4h": 5},
        variables={},
    )


def _make_trial(
    rank: int,
    expression: str,
    mean_icir: float = 0.15,
) -> TrialResult:
    return TrialResult(
        trial_number=rank,
        params={},
        expression=expression,
        cv_icir=(mean_icir,) * 3,
        mean_icir=mean_icir,
        objective_value=abs(mean_icir),
    )


def _make_tune_result(
    expressions: list[str],
) -> TuneResult:
    top_k = tuple(_make_trial(i + 1, expr, mean_icir=0.2 - i * 0.01) for i, expr in enumerate(expressions))
    return TuneResult(
        template="p0 * correlation(high, rank(volume), p1)",
        original_expression="-correlation(high, rank(volume), 5)",
        original_params={"p0": -1.0, "p1": 5.0},
        best_params={"p0": -1.0, "p1": 20.0},
        best_expression=top_k[0].expression if top_k else "",
        best_icir_cv=top_k[0].mean_icir if top_k else float("nan"),
        top_k=top_k,
        n_trials=len(top_k) * 2,
    )


def _synthetic_panel(T: int = 60, N: int = 10) -> tuple[dict, pd.DataFrame]:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2025-01-01", periods=T, freq="4h")
    cols = [f"I{i}" for i in range(N)]
    close = pd.DataFrame(
        100 + rng.standard_normal((T, N)).cumsum(axis=0) * 0.3,
        index=idx,
        columns=cols,
    )
    volume = pd.DataFrame(
        np.abs(rng.standard_normal((T, N))) * 1000 + 5000,
        index=idx,
        columns=cols,
    )
    high = close * (1 + np.abs(rng.standard_normal((T, N))) * 1e-3)
    low = close * (1 - np.abs(rng.standard_normal((T, N))) * 1e-3)
    open_ = close.shift(1).fillna(close)
    panel = {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }
    # Tests now pass ``pricing`` (close panel) directly to
    # register_tuned_variants — alphalens forward returns are derived
    # internally from prices.
    fwd = close  # backward-compatible name; actual arg is ``pricing=fwd``
    return panel, fwd


# ── Pure helpers ────────────────────────────────────────────────────────────


class TestMakeVariantId:
    def test_deterministic_hash(self) -> None:
        a = _make_variant_id("alpha044", "-1 * correlation(high, volume, 20)", 1)
        b = _make_variant_id("alpha044", "-1 * correlation(high, volume, 20)", 1)
        assert a == b

    def test_rank_is_embedded(self) -> None:
        a = _make_variant_id("alpha044", "x", 1)
        b = _make_variant_id("alpha044", "x", 2)
        assert "_tune1_" in a and "_tune2_" in b

    def test_unsafe_characters_are_stripped(self) -> None:
        vid = _make_variant_id("alpha/044.raw", "x", 1)
        assert "/" not in vid and "." not in vid


# ── register_tuned_variants (mocked analysis) ──────────────────────────────


class TestRegisterTunedVariants:
    def test_registers_top_k_with_tuned_tag_and_source(self, repo: FactorRepository, monkeypatch: pytest.MonkeyPatch) -> None:
        source = _source_factor()
        repo.upsert_factor(source)

        tune_result = _make_tune_result(
            [
                "-1 * correlation(high, rank(volume), 20)",
                "-1 * correlation(high, rank(volume), 30)",
            ]
        )

        monkeypatch.setattr(
            variant_registration,
            "analyze_variant",
            lambda *args, **kwargs: None,  # skip metrics, still registers record
        )

        panel, fwd = _synthetic_panel()
        summary = register_tuned_variants(
            tune_result=tune_result,
            source_factor=source,
            repo=repo,
            panel_fields=panel,
            pricing=fwd,
            tune_config=TuneConfig(source="tune_unit"),
            run_id="20260412_000000",
            timeframe="4h",
        )
        assert summary.n_registered == 2
        assert len(summary.factor_ids) == 2
        first = repo.get_factor(summary.factor_ids[0])
        assert first is not None
        assert first.source == "tune_unit"
        assert "tuned" in first.tags
        assert first.prototype == "alpha044"
        # Parameters are extracted from the *tuned* expression (not copied
        # from source), so non-p* keys persist but p* keys reflect the
        # tuned numeric literals.
        assert first.parameters["w4h"] == source.parameters["w4h"]
        # p0/p1 come from expression_template on the tuned expression
        # (e.g. ``-1 * correlation(high, rank(volume), 20)``).
        assert "p0" in first.parameters
        assert first.parameters["p0"] == -1.0  # sign
        assert first.parameters["p1"] == 20.0  # tuned window

    def test_skips_when_expression_hash_already_exists(self, repo: FactorRepository, monkeypatch: pytest.MonkeyPatch) -> None:
        source = _source_factor()
        repo.upsert_factor(source)

        existing = FactorRecord(
            factor_id="existing_variant",
            expression="-1 * correlation(high, rank(volume), 20)",
            prototype="alpha044",
            source="llm",
            status="candidate",
        )
        repo.upsert_factor(existing)

        tune_result = _make_tune_result(
            [
                "-1 * correlation(high, rank(volume), 20)",  # same expr as existing
            ]
        )

        monkeypatch.setattr(
            variant_registration,
            "analyze_variant",
            lambda *args, **kwargs: None,
        )

        panel, fwd = _synthetic_panel()
        summary = register_tuned_variants(
            tune_result=tune_result,
            source_factor=source,
            repo=repo,
            panel_fields=panel,
            pricing=fwd,
            tune_config=TuneConfig(source="tune_unit"),
            run_id="20260412_000000",
            timeframe="4h",
        )
        assert summary.n_skipped == 1
        assert summary.n_registered == 0
        assert summary.variants[0].outcome == "skipped_duplicate"
        assert summary.variants[0].factor_id == "existing_variant"

    def test_empty_top_k_yields_empty_summary(self, repo: FactorRepository, monkeypatch: pytest.MonkeyPatch) -> None:
        source = _source_factor()
        repo.upsert_factor(source)
        tune_result = _make_tune_result([])

        monkeypatch.setattr(
            variant_registration,
            "analyze_variant",
            lambda *args, **kwargs: None,
        )

        panel, fwd = _synthetic_panel()
        summary = register_tuned_variants(
            tune_result=tune_result,
            source_factor=source,
            repo=repo,
            panel_fields=panel,
            pricing=fwd,
            tune_config=TuneConfig(),
            run_id="20260412_000000",
            timeframe="4h",
        )
        assert summary.variants == ()
        assert summary.n_registered == 0
        assert summary.n_skipped == 0
