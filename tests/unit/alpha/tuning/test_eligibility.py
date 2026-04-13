# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for pre-tune eligibility filtering."""

from __future__ import annotations

import pytest

from nautilus_quants.alpha.registry.database import RegistryDatabase
from nautilus_quants.alpha.registry.models import AnalysisMetrics, FactorRecord
from nautilus_quants.alpha.registry.repository import FactorRepository
from nautilus_quants.alpha.tuning.config import CandidatesConfig, EligibilityConfig
from nautilus_quants.alpha.tuning.eligibility import (
    filter_tune_eligible,
    group_eligible_by_prototype,
)


@pytest.fixture()
def repo() -> FactorRepository:
    db = RegistryDatabase(":memory:")
    yield FactorRepository(db)
    db.close()


def _register_factor(
    repo: FactorRepository,
    factor_id: str,
    expression: str,
    prototype: str = "proto_x",
    source: str = "unit",
    status: str = "candidate",
) -> None:
    repo.upsert_factor(
        FactorRecord(
            factor_id=factor_id,
            expression=expression,
            prototype=prototype,
            source=source,
            status=status,
        )
    )


def _save_metrics(
    repo: FactorRepository,
    factor_id: str,
    *,
    icir: float = 0.1,
    t_stat_nw: float = 3.0,
    coverage: float = 0.8,
    n_samples: int = 5000,
    period: str = "4h",
    timeframe: str = "4h",
    run_id: str = "20260412_000000",
    created_at: str = "",
) -> None:
    repo.save_metrics(
        [
            AnalysisMetrics(
                run_id=run_id,
                factor_id=factor_id,
                period=period,
                icir=icir,
                t_stat_nw=t_stat_nw,
                coverage=coverage,
                n_samples=n_samples,
                timeframe=timeframe,
                created_at=created_at,
            )
        ]
    )


# ── Individual checks ──────────────────────────────────────────────────────


class TestFilterTuneEligible:
    def test_accepts_factor_meeting_all_thresholds(self, repo: FactorRepository) -> None:
        _register_factor(repo, "f1", "ts_mean(close, 5)")
        _save_metrics(repo, "f1", icir=0.15, coverage=0.8, n_samples=5000)

        report = filter_tune_eligible(repo, CandidatesConfig())
        assert report.n_eligible == 1
        assert report.eligible_factors[0].factor_id == "f1"

    def test_rejects_factor_without_metrics(self, repo: FactorRepository) -> None:
        _register_factor(repo, "f1", "ts_mean(close, 5)")
        report = filter_tune_eligible(repo, CandidatesConfig())
        assert report.n_eligible == 0
        assert report.n_no_metrics == 1

    def test_rejects_factor_with_low_icir(self, repo: FactorRepository) -> None:
        _register_factor(repo, "f1", "ts_mean(close, 5)")
        _save_metrics(repo, "f1", icir=0.005, coverage=0.8, n_samples=5000)

        report = filter_tune_eligible(repo, CandidatesConfig())
        assert report.n_eligible == 0
        assert report.n_low_icir == 1

    def test_rejects_factor_with_low_t_stat_nw(self, repo: FactorRepository) -> None:
        # |ICIR| above 0.05 but t_stat_nw below 1.5 → spurious peak,
        # likely from short / autocorrelated samples. Should be rejected.
        _register_factor(repo, "f1", "ts_mean(close, 5)")
        _save_metrics(repo, "f1", icir=0.08, t_stat_nw=0.5, coverage=0.8, n_samples=5000)

        report = filter_tune_eligible(repo, CandidatesConfig())
        assert report.n_eligible == 0

    def test_rejects_factor_with_low_coverage(self, repo: FactorRepository) -> None:
        _register_factor(repo, "f1", "ts_mean(close, 5)")
        _save_metrics(repo, "f1", icir=0.15, coverage=0.1, n_samples=5000)

        report = filter_tune_eligible(repo, CandidatesConfig())
        assert report.n_eligible == 0
        assert report.n_low_coverage == 1

    def test_rejects_factor_with_too_few_samples(self, repo: FactorRepository) -> None:
        _register_factor(repo, "f1", "ts_mean(close, 5)")
        _save_metrics(repo, "f1", icir=0.15, coverage=0.8, n_samples=100)

        report = filter_tune_eligible(repo, CandidatesConfig())
        assert report.n_eligible == 0
        assert report.n_low_samples == 1

    def test_any_passing_period_is_sufficient_by_default(self, repo: FactorRepository) -> None:
        _register_factor(repo, "f1", "ts_mean(close, 5)")
        _save_metrics(
            repo,
            "f1",
            period="4h",
            icir=0.005,
            coverage=0.8,
            n_samples=5000,
            run_id="r1",
        )
        _save_metrics(
            repo,
            "f1",
            period="8h",
            icir=0.15,
            coverage=0.8,
            n_samples=5000,
            run_id="r1",
        )

        report = filter_tune_eligible(repo, CandidatesConfig())
        assert report.n_eligible == 1

    def test_min_valid_periods_stricter_than_one(self, repo: FactorRepository) -> None:
        _register_factor(repo, "f1", "ts_mean(close, 5)")
        _save_metrics(
            repo,
            "f1",
            period="4h",
            icir=0.005,
            coverage=0.8,
            n_samples=5000,
            run_id="r1",
        )
        _save_metrics(
            repo,
            "f1",
            period="8h",
            icir=0.15,
            coverage=0.8,
            n_samples=5000,
            run_id="r1",
        )

        cfg = CandidatesConfig(eligibility=EligibilityConfig(min_valid_periods=2))
        report = filter_tune_eligible(repo, cfg)
        assert report.n_eligible == 0

    def test_prototype_filter_narrows_scope(self, repo: FactorRepository) -> None:
        _register_factor(repo, "f1", "ts_mean(close, 5)", prototype="alpha044")
        _save_metrics(repo, "f1", icir=0.15)
        _register_factor(repo, "f2", "ts_mean(close, 10)", prototype="alpha022")
        _save_metrics(repo, "f2", icir=0.15)

        cfg = CandidatesConfig(prototype="alpha044")
        report = filter_tune_eligible(repo, cfg)
        assert [f.factor_id for f in report.eligible_factors] == ["f1"]

    def test_dedupes_latest_run_per_period(self, repo: FactorRepository) -> None:
        """Regression: prior implementation accepted a factor whose newest
        run failed because an older successful run was still in
        ``alpha_analysis_metrics`` and got counted as a passing period.
        """
        _register_factor(repo, "f1", "ts_mean(close, 5)")
        # Older run: passes all gates.
        _save_metrics(
            repo,
            "f1",
            period="4h",
            icir=0.20,
            coverage=0.8,
            n_samples=5000,
            run_id="r_old",
            created_at="2026-04-10T00:00:00",
        )
        # Newest run for the same (timeframe, period): fails ICIR gate.
        _save_metrics(
            repo,
            "f1",
            period="4h",
            icir=0.005,
            coverage=0.8,
            n_samples=5000,
            run_id="r_new",
            created_at="2026-04-13T00:00:00",
        )

        report = filter_tune_eligible(repo, CandidatesConfig())
        # Eligibility must be driven by the most recent run per period only.
        assert report.n_eligible == 0
        assert report.n_low_icir == 1

    def test_duplicate_runs_dont_inflate_passing_count(self, repo: FactorRepository) -> None:
        """Regression: same (timeframe, period) saved twice must not count as
        two passing periods (would let ``min_valid_periods=2`` pass via a
        single passing period).
        """
        _register_factor(repo, "f1", "ts_mean(close, 5)")
        # Two runs of the same period — both pass; should still count as 1.
        _save_metrics(
            repo,
            "f1",
            period="4h",
            icir=0.20,
            coverage=0.8,
            n_samples=5000,
            run_id="r1",
            created_at="2026-04-12T00:00:00",
        )
        _save_metrics(
            repo,
            "f1",
            period="4h",
            icir=0.18,
            coverage=0.8,
            n_samples=5000,
            run_id="r2",
            created_at="2026-04-13T00:00:00",
        )

        cfg = CandidatesConfig(eligibility=EligibilityConfig(min_valid_periods=2))
        report = filter_tune_eligible(repo, cfg)
        assert report.n_eligible == 0
        assert report.n_too_few_periods == 1

    def test_source_filter_narrows_scope(self, repo: FactorRepository) -> None:
        _register_factor(repo, "f1", "ts_mean(close, 5)", source="llm")
        _save_metrics(repo, "f1", icir=0.15)
        _register_factor(repo, "f2", "ts_mean(close, 10)", source="alpha101")
        _save_metrics(repo, "f2", icir=0.15)

        cfg = CandidatesConfig(source="llm")
        report = filter_tune_eligible(repo, cfg)
        assert [f.factor_id for f in report.eligible_factors] == ["f1"]


class TestGroupByPrototype:
    def test_groups_by_prototype(self, repo: FactorRepository) -> None:
        _register_factor(repo, "a1", "ts_mean(close, 5)", prototype="A")
        _save_metrics(repo, "a1", icir=0.2)
        _register_factor(repo, "a2", "ts_mean(close, 20)", prototype="A")
        _save_metrics(repo, "a2", icir=0.1)
        _register_factor(repo, "b1", "ts_std(close, 5)", prototype="B")
        _save_metrics(repo, "b1", icir=0.15)

        report = filter_tune_eligible(repo, CandidatesConfig())
        groups = group_eligible_by_prototype(report)
        assert set(groups.keys()) == {"A", "B"}
        assert len(groups["A"]) == 2
        assert len(groups["B"]) == 1

    def test_missing_prototype_each_gets_solo_bucket(self, repo: FactorRepository) -> None:
        # Two factors without a prototype must NOT collapse into one bucket
        # — each is a unique structure (e.g. ``llm_sanapi_social_lead_oi_lag``)
        # and must be tuned independently.
        _register_factor(repo, "llm_lead_oi_lag", "ts_mean(close, 5)", prototype="")
        _save_metrics(repo, "llm_lead_oi_lag", icir=0.15)
        _register_factor(repo, "llm_social_burst", "ts_std(close, 8)", prototype="")
        _save_metrics(repo, "llm_social_burst", icir=0.12)

        report = filter_tune_eligible(repo, CandidatesConfig())
        groups = group_eligible_by_prototype(report)

        assert "_solo_llm_lead_oi_lag" in groups
        assert "_solo_llm_social_burst" in groups
        assert len(groups["_solo_llm_lead_oi_lag"]) == 1
        assert len(groups["_solo_llm_social_burst"]) == 1

    def test_mixed_prototype_and_solo_factors(self, repo: FactorRepository) -> None:
        # Real prototypes still collapse (true parameter variants); solo
        # factors stay independent. This is the realistic post-backfill state.
        _register_factor(repo, "alpha044_v1", "ts_mean(close, 5)", prototype="alpha044")
        _save_metrics(repo, "alpha044_v1", icir=0.15)
        _register_factor(repo, "alpha044_v2", "ts_mean(close, 20)", prototype="alpha044")
        _save_metrics(repo, "alpha044_v2", icir=0.12)
        _register_factor(repo, "llm_unique_signal", "ts_skew(close, 30)", prototype="")
        _save_metrics(repo, "llm_unique_signal", icir=0.18)

        report = filter_tune_eligible(repo, CandidatesConfig())
        groups = group_eligible_by_prototype(report)

        assert "alpha044" in groups
        assert len(groups["alpha044"]) == 2  # parameter variants collapse
        assert "_solo_llm_unique_signal" in groups
        assert len(groups["_solo_llm_unique_signal"]) == 1
