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


# ── Resume support (--skip-already-tuned / --resume-from-dir) ───────────────


def _register_tuned_variant(
    repo: FactorRepository,
    factor_id: str,
    *,
    prototype: str = "",
    source: str = "alpha_tune_test",
) -> None:
    """Register a tuned variant FactorRecord (tag='tuned') for resume tests."""
    repo.upsert_factor(
        FactorRecord(
            factor_id=factor_id,
            expression=f"expr_of_{factor_id}",
            prototype=prototype,
            source=source,
            status="candidate",
            tags=["tuned"],
        )
    )


class TestFilterAlreadyTuned:
    """filter_already_tuned — resume support."""

    def test_skips_prototype_with_full_register_top_k(self, repo: FactorRepository) -> None:
        from nautilus_quants.alpha.tuning.eligibility import filter_already_tuned

        # Register 3 tune variants for prototype "alpha044" (top_k=3 → complete)
        for rank in (1, 2, 3):
            _register_tuned_variant(
                repo,
                f"alpha044_v1_tune{rank}_abcdef01",
                prototype="alpha044",
            )
        # Source candidate
        eligible = [
            FactorRecord(
                factor_id="alpha044_v1",
                expression="ts_mean(close, 5)",
                prototype="alpha044",
                source="alpha101",
            ),
        ]
        kept, skipped = filter_already_tuned(
            eligible,
            repo,
            by_prototype=True,
            register_top_k=3,
        )
        assert kept == []
        assert skipped == ["alpha044"]

    def test_keeps_prototype_with_partial_variants(self, repo: FactorRepository) -> None:
        """Prototype with fewer than top_k variants should be retained (allow retry)."""
        from nautilus_quants.alpha.tuning.eligibility import filter_already_tuned

        # Only 1 out of 3 expected variants registered — crashed before finishing
        _register_tuned_variant(
            repo,
            "alpha044_v1_tune1_abcdef01",
            prototype="alpha044",
        )
        eligible = [
            FactorRecord(
                factor_id="alpha044_v1",
                expression="ts_mean(close, 5)",
                prototype="alpha044",
                source="alpha101",
            ),
        ]
        kept, skipped = filter_already_tuned(
            eligible,
            repo,
            by_prototype=True,
            register_top_k=3,
        )
        assert len(kept) == 1
        assert skipped == []

    def test_by_factor_mode_matches_source_prefix(self, repo: FactorRepository) -> None:
        from nautilus_quants.alpha.tuning.eligibility import filter_already_tuned

        # 3 tune variants for source factor "llm_solo_foo"
        for rank in (1, 2, 3):
            _register_tuned_variant(
                repo,
                f"llm_solo_foo_tune{rank}_deadbeef",
                prototype="",
            )
        eligible = [
            FactorRecord(
                factor_id="llm_solo_foo",
                expression="ts_mean(close, 5)",
                prototype="",
                source="llm_mining",
            ),
            FactorRecord(
                factor_id="llm_solo_bar",  # not tuned yet
                expression="ts_std(close, 10)",
                prototype="",
                source="llm_mining",
            ),
        ]
        kept, skipped = filter_already_tuned(
            eligible,
            repo,
            by_prototype=False,
            register_top_k=3,
        )
        assert [f.factor_id for f in kept] == ["llm_solo_bar"]
        assert skipped == ["llm_solo_foo"]

    def test_empty_db_keeps_all(self, repo: FactorRepository) -> None:
        from nautilus_quants.alpha.tuning.eligibility import filter_already_tuned

        eligible = [
            FactorRecord(
                factor_id="foo",
                expression="x",
                prototype="P",
                source="src",
            ),
            FactorRecord(
                factor_id="bar",
                expression="y",
                prototype="Q",
                source="src",
            ),
        ]
        kept, skipped = filter_already_tuned(
            eligible,
            repo,
            by_prototype=True,
            register_top_k=3,
        )
        assert len(kept) == 2
        assert skipped == []

    def test_by_prototype_mode_skips_solo_factor_via_id_prefix(
        self, repo: FactorRepository
    ) -> None:
        """Solo factors (no prototype) must still be skippable in by_prototype mode.

        Tune variants derived from a solo source factor inherit an empty
        prototype field, so pure prototype-match would miss them entirely.
        The implementation falls back to factor_id-prefix matching for the
        solo case and emits the ``_solo_{factor_id}`` label used by
        ``group_eligible_by_prototype``.
        """
        from nautilus_quants.alpha.tuning.eligibility import filter_already_tuned

        # 3 tune variants for solo source factor "llm_solo_foo" (no prototype)
        for rank in (1, 2, 3):
            _register_tuned_variant(
                repo,
                f"llm_solo_foo_tune{rank}_deadbeef",
                prototype="",  # inherited from solo source factor
            )
        eligible = [
            FactorRecord(
                factor_id="llm_solo_foo",
                expression="ts_mean(close, 5)",
                prototype="",
                source="llm_mining",
            ),
            FactorRecord(
                factor_id="llm_solo_bar",  # not tuned yet
                expression="ts_std(close, 10)",
                prototype="",
                source="llm_mining",
            ),
        ]
        kept, skipped = filter_already_tuned(
            eligible, repo, by_prototype=True, register_top_k=3,
        )
        assert [f.factor_id for f in kept] == ["llm_solo_bar"]
        # Skip label uses the "_solo_{factor_id}" form to match the key
        # produced by group_eligible_by_prototype.
        assert skipped == ["_solo_llm_solo_foo"]

    def test_by_prototype_mode_solo_partial_variants_retained(
        self, repo: FactorRepository
    ) -> None:
        """Solo factor with fewer than register_top_k variants must be retained."""
        from nautilus_quants.alpha.tuning.eligibility import filter_already_tuned

        # Only 1 of 3 variants registered — crashed mid-registration
        _register_tuned_variant(repo, "llm_solo_foo_tune1_deadbeef", prototype="")
        eligible = [
            FactorRecord(
                factor_id="llm_solo_foo",
                expression="ts_mean(close, 5)",
                prototype="",
                source="llm_mining",
            ),
        ]
        kept, skipped = filter_already_tuned(
            eligible, repo, by_prototype=True, register_top_k=3,
        )
        assert len(kept) == 1
        assert skipped == []


class TestLabelsCompletedInDir:
    """labels_completed_in_dir — resume from run_dir primary evidence."""

    def test_prototype_named(self, tmp_path):
        from nautilus_quants.alpha.tuning.eligibility import labels_completed_in_dir

        proto_dir = tmp_path / "proto_001_alpha044"
        proto_dir.mkdir()
        (proto_dir / "registration_summary.json").write_text("{}")
        assert labels_completed_in_dir(tmp_path) == {"alpha044"}

    def test_solo_factor_retains_leading_underscore(self, tmp_path):
        from nautilus_quants.alpha.tuning.eligibility import labels_completed_in_dir

        # On-disk looks like proto_002__solo_foo because label = "_solo_foo"
        proto_dir = tmp_path / "proto_002__solo_foo"
        proto_dir.mkdir()
        (proto_dir / "registration_summary.json").write_text("{}")
        assert labels_completed_in_dir(tmp_path) == {"_solo_foo"}

    def test_skips_crashed_without_summary(self, tmp_path):
        from nautilus_quants.alpha.tuning.eligibility import labels_completed_in_dir

        proto_dir = tmp_path / "proto_003_alpha_crashed"
        proto_dir.mkdir()
        # No registration_summary.json -> treat as incomplete
        assert labels_completed_in_dir(tmp_path) == set()

    def test_mixed_completed_and_crashed(self, tmp_path):
        from nautilus_quants.alpha.tuning.eligibility import labels_completed_in_dir

        (tmp_path / "proto_001_alpha044").mkdir()
        (tmp_path / "proto_001_alpha044" / "registration_summary.json").write_text("{}")
        (tmp_path / "proto_002__solo_foo").mkdir()
        (tmp_path / "proto_002__solo_foo" / "registration_summary.json").write_text("{}")
        (tmp_path / "proto_003_crashed").mkdir()  # no summary.json
        assert labels_completed_in_dir(tmp_path) == {"alpha044", "_solo_foo"}

    def test_missing_dir_returns_empty(self, tmp_path):
        from nautilus_quants.alpha.tuning.eligibility import labels_completed_in_dir

        assert labels_completed_in_dir(tmp_path / "does_not_exist") == set()

    def test_ignores_non_proto_entries(self, tmp_path):
        from nautilus_quants.alpha.tuning.eligibility import labels_completed_in_dir

        (tmp_path / "summary.json").write_text("{}")  # file, not a dir
        (tmp_path / "some_other_dir").mkdir()  # dir but wrong prefix
        (tmp_path / "proto_001_keep").mkdir()
        (tmp_path / "proto_001_keep" / "registration_summary.json").write_text("{}")
        assert labels_completed_in_dir(tmp_path) == {"keep"}
