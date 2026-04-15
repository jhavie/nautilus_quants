# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Pre-tune eligibility filter.

Decides which registered factors are *worth* tuning. This is intentionally
much looser than the promote ``HardFilterConfig`` — we want to eliminate
obviously hopeless cases (no signal at all, catastrophic data coverage) but
leave marginal factors in, because those are exactly the ones parameter
tuning can rescue.

The eligibility check reads each factor's latest ``AnalysisMetrics`` rows
and decides per-factor / per-prototype whether to proceed. It **never** runs
IC analysis — the check is pure metadata lookup over the registry.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from nautilus_quants.alpha.registry.models import AnalysisMetrics, FactorRecord
from nautilus_quants.alpha.registry.repository import FactorRepository
from nautilus_quants.alpha.tuning.config import CandidatesConfig, EligibilityConfig

# Matches a tune run dir's per-prototype subdir: ``proto_NNN_{label}``.
# ``build_factor_dir`` uses ONE underscore between the zero-padded index and
# the label; what looks like a double underscore in ``proto_001__solo_foo`` is
# really ``proto_001_`` + ``_solo_foo`` (label starts with ``_`` for solo
# factors keyed as ``_solo_{factor_id}`` by ``group_eligible_by_prototype``).
_PROTO_DIR_RE = re.compile(r"^proto_\d{3}_(?P<label>.+)$")

# Matches variant factor_ids produced by ``_make_variant_id``:
# ``{safe_source_factor_id}_tune{rank}_{hash[:8]}``.
_TUNED_ID_RE = re.compile(r"^(?P<src>.+)_tune\d+_[0-9a-f]{8}$")


@dataclass(frozen=True)
class EligibilityReason:
    """Why a factor was accepted or rejected for tuning."""

    factor_id: str
    eligible: bool
    reason: str
    best_icir_abs: float | None = None
    best_coverage: float | None = None
    max_n_samples: int | None = None
    n_valid_periods: int = 0


@dataclass(frozen=True)
class EligibilityReport:
    """Summary of the filter pass over the candidate set."""

    eligible_factors: tuple[FactorRecord, ...]
    rejected_factors: tuple[FactorRecord, ...]
    reasons: tuple[EligibilityReason, ...]
    n_total: int
    n_eligible: int
    n_rejected: int
    n_no_metrics: int
    n_low_icir: int
    n_low_coverage: int
    n_low_samples: int
    n_too_few_periods: int

    def eligible_prototypes(self) -> tuple[str, ...]:
        """Unique prototypes present in ``eligible_factors``."""
        protos = {f.prototype for f in self.eligible_factors if f.prototype}
        return tuple(sorted(protos))


# ── Factor-level check ─────────────────────────────────────────────────────


def _check_single_factor(
    factor_id: str,
    metrics: Iterable[AnalysisMetrics],
    cfg: EligibilityConfig,
) -> EligibilityReason:
    """Decide whether one factor is worth tuning.

    Accept iff at least ``cfg.min_valid_periods`` of the factor's *latest run
    per (timeframe, period)* satisfy ALL gates simultaneously: |ICIR|,
    |t_stat_nw|, coverage, and n_samples. Cross-period rule is OR (any
    qualifying period passes the factor) but per-period rule is AND (each
    candidate period must clear every gate).

    Multiple historical runs of the same (timeframe, period) are deduped to
    the latest one — otherwise stale-good runs would inflate
    ``passing_periods`` above ``min_valid_periods`` even when the most
    recent run failed.
    """
    # ``repo.get_metrics`` returns rows ordered by ``created_at DESC``, so
    # the first occurrence of a given (timeframe, period) is the most
    # recent run; subsequent duplicates can be ignored safely.
    seen: set[tuple[str | None, str | None]] = set()
    metrics_list: list[AnalysisMetrics] = []
    for m in metrics:
        if m is None:
            continue
        key = (m.timeframe, m.period)
        if key in seen:
            continue
        seen.add(key)
        metrics_list.append(m)

    if not metrics_list:
        return EligibilityReason(
            factor_id=factor_id,
            eligible=False,
            reason="no metrics registered — run `alpha analyze` first",
        )

    best_icir_abs: float = 0.0
    best_t_stat_abs: float = 0.0
    best_coverage: float = 0.0
    max_n_samples: int = 0
    passing_periods = 0

    for m in metrics_list:
        icir = abs(m.icir) if m.icir is not None else 0.0
        t_stat = abs(m.t_stat_nw) if m.t_stat_nw is not None else 0.0
        coverage = m.coverage if m.coverage is not None else 0.0
        n_samples = m.n_samples if m.n_samples is not None else 0

        best_icir_abs = max(best_icir_abs, icir)
        best_t_stat_abs = max(best_t_stat_abs, t_stat)
        best_coverage = max(best_coverage, coverage)
        max_n_samples = max(max_n_samples, n_samples)

        if (
            icir >= cfg.icir_abs_min
            and t_stat >= cfg.t_stat_nw_abs_min
            and coverage >= cfg.coverage_min
            and n_samples >= cfg.n_samples_min
        ):
            passing_periods += 1

    if passing_periods < cfg.min_valid_periods:
        failures: list[str] = []
        if best_icir_abs < cfg.icir_abs_min:
            failures.append(f"|icir|={best_icir_abs:.3f} < {cfg.icir_abs_min}")
        if best_t_stat_abs < cfg.t_stat_nw_abs_min:
            failures.append(f"|t_stat_nw|={best_t_stat_abs:.2f} < {cfg.t_stat_nw_abs_min}")
        if best_coverage < cfg.coverage_min:
            failures.append(f"coverage={best_coverage:.2f} < {cfg.coverage_min}")
        if max_n_samples < cfg.n_samples_min:
            failures.append(f"n_samples={max_n_samples} < {cfg.n_samples_min}")
        failures.append(f"passing_periods={passing_periods} < {cfg.min_valid_periods}")
        return EligibilityReason(
            factor_id=factor_id,
            eligible=False,
            reason="; ".join(failures),
            best_icir_abs=best_icir_abs,
            best_coverage=best_coverage,
            max_n_samples=max_n_samples,
            n_valid_periods=passing_periods,
        )

    return EligibilityReason(
        factor_id=factor_id,
        eligible=True,
        reason="pass",
        best_icir_abs=best_icir_abs,
        best_coverage=best_coverage,
        max_n_samples=max_n_samples,
        n_valid_periods=passing_periods,
    )


# ── Batch filter ───────────────────────────────────────────────────────────


def filter_tune_eligible(
    repo: FactorRepository,
    candidates: CandidatesConfig,
) -> EligibilityReport:
    """Load candidates from ``repo`` and apply the eligibility check.

    The candidate selection honours ``candidates.prototype``,
    ``candidates.source``, ``candidates.status`` and ``candidates.tags`` as
    AND filters. Each surviving factor then has its latest metrics looked up
    and checked against ``candidates.eligibility``.
    """
    factors = _load_candidates(repo, candidates)

    eligible: list[FactorRecord] = []
    rejected: list[FactorRecord] = []
    reasons: list[EligibilityReason] = []

    counts = {
        "no_metrics": 0,
        "low_icir": 0,
        "low_coverage": 0,
        "low_samples": 0,
        "too_few_periods": 0,
    }

    for factor in factors:
        metrics = repo.get_metrics(factor.factor_id)
        reason = _check_single_factor(factor.factor_id, metrics, candidates.eligibility)
        reasons.append(reason)
        if reason.eligible:
            eligible.append(factor)
            continue

        rejected.append(factor)
        if "no metrics" in reason.reason:
            counts["no_metrics"] += 1
        else:
            if (
                reason.best_icir_abs is not None
                and reason.best_icir_abs < candidates.eligibility.icir_abs_min
            ):
                counts["low_icir"] += 1
            if (
                reason.best_coverage is not None
                and reason.best_coverage < candidates.eligibility.coverage_min
            ):
                counts["low_coverage"] += 1
            if (
                reason.max_n_samples is not None
                and reason.max_n_samples < candidates.eligibility.n_samples_min
            ):
                counts["low_samples"] += 1
            if reason.n_valid_periods < candidates.eligibility.min_valid_periods:
                counts["too_few_periods"] += 1

    return EligibilityReport(
        eligible_factors=tuple(eligible),
        rejected_factors=tuple(rejected),
        reasons=tuple(reasons),
        n_total=len(factors),
        n_eligible=len(eligible),
        n_rejected=len(rejected),
        n_no_metrics=counts["no_metrics"],
        n_low_icir=counts["low_icir"],
        n_low_coverage=counts["low_coverage"],
        n_low_samples=counts["low_samples"],
        n_too_few_periods=counts["too_few_periods"],
    )


def _load_candidates(
    repo: FactorRepository,
    candidates: CandidatesConfig,
) -> list[FactorRecord]:
    """Apply the coarse registry filters before per-factor checks."""
    factors = repo.list_factors(
        status=candidates.status,
        source=candidates.source,
        tag=candidates.tags[0] if candidates.tags else None,
    )
    if candidates.prototype is not None:
        factors = [f for f in factors if f.prototype == candidates.prototype]
    if len(candidates.tags) > 1:
        tag_set = set(candidates.tags)
        factors = [f for f in factors if tag_set.issubset(set(f.tags or ()))]
    return factors


# ── Prototype grouping ─────────────────────────────────────────────────────


def group_eligible_by_prototype(
    report: EligibilityReport,
) -> dict[str, list[FactorRecord]]:
    """Group eligible factors by their prototype.

    Factors with a non-empty prototype share parameter-variant semantics —
    they collapse into a single group whose representative is tuned once
    on behalf of the entire family.

    Factors **without** a prototype (typical of LLM-mined one-off
    expressions like ``llm_sanapi_social_lead_oi_lag``) are *not*
    parameter variants of each other; lumping them together would mean
    only the single highest-ICIR factor in the no-prototype set ever gets
    tuned. Instead, every prototype-less factor receives its own synthetic
    bucket keyed by its ``factor_id`` so the downstream loop tunes each
    structure independently.
    """
    grouped: dict[str, list[FactorRecord]] = defaultdict(list)
    for factor in report.eligible_factors:
        if factor.prototype:
            grouped[factor.prototype].append(factor)
        else:
            # Synthetic key — unique per factor, distinguishable from real
            # prototype names by the leading underscore prefix.
            grouped[f"_solo_{factor.factor_id}"].append(factor)
    return dict(grouped)


# ── Resume support ────────────────────────────────────────────────────────


def filter_already_tuned(
    eligible_factors: Iterable[FactorRecord],
    repo: FactorRepository,
    *,
    by_prototype: bool,
    register_top_k: int,
) -> tuple[list[FactorRecord], list[str]]:
    """Drop source factors whose tune variants already fill ``register_top_k``.

    Returns ``(kept, skipped_labels)``. Used by ``alpha tune
    --skip-already-tuned`` to resume after a crashed batch without rerunning
    Optuna on prototypes that have already been fully processed.

    Identification uses ``tag="tuned"`` (hardcoded by ``register_tuned_variants``)
    rather than ``source=...`` because the latter is user-configurable via
    ``tune_config.source`` and therefore unreliable.

    ``by_prototype=True``
        Group registered tune variants by their ``prototype`` field (which is
        stably inherited from the source factor) and skip any eligible factor
        whose prototype already has ``>= register_top_k`` variants registered.
        Representative-choice drift between runs is irrelevant: prototype is
        stable, so the partition survives.

    ``by_prototype=False``
        Reverse-match the ``{safe_source_id}_tune{rank}_{hash[:8]}`` pattern
        produced by ``_make_variant_id`` to recover the originating source
        factor_id. Skip when that bucket has ``>= register_top_k`` entries.

    Partially-registered prototypes (count ``< register_top_k``) are retained:
    rerunning lets ``find_by_expression_hash`` dedup the already-written
    variants and only fill in the missing ranks. This protects against the
    edge case where a crash left rank-1/rank-2 written but rank-3 missing.
    """
    from nautilus_quants.alpha.tuning.variant_registration import _SAFE_NAME_RE

    tuned = repo.list_factors(tag="tuned")
    kept: list[FactorRecord] = []
    skipped_labels: list[str] = []

    # Build both maps upfront — needed for the solo-factor fallback in
    # by_prototype mode as well as the full by_factor mode.
    proto_counts: Counter[str] = Counter()
    tuned_by_source: Counter[str] = Counter()
    for t in tuned:
        if t.prototype:
            proto_counts[t.prototype] += 1
        m = _TUNED_ID_RE.match(t.factor_id)
        if m is not None:
            tuned_by_source[m.group("src")] += 1

    if by_prototype:
        complete_protos = {p for p, c in proto_counts.items() if c >= register_top_k}
        for f in eligible_factors:
            if f.prototype and f.prototype in complete_protos:
                skipped_labels.append(f.prototype)
                continue
            # Solo factor (no prototype): tune variants inherit an empty
            # prototype, so proto_counts will never catch them. Fall back to
            # factor_id-prefix matching and emit the "_solo_{id}" label that
            # `group_eligible_by_prototype` uses as the grouped-dict key.
            if not f.prototype:
                safe = _SAFE_NAME_RE.sub("_", f.factor_id)
                if tuned_by_source.get(safe, 0) >= register_top_k:
                    skipped_labels.append(f"_solo_{f.factor_id}")
                    continue
            kept.append(f)
        return kept, sorted(set(skipped_labels))

    # by_factor mode: reverse-match the _make_variant_id pattern.
    for f in eligible_factors:
        safe = _SAFE_NAME_RE.sub("_", f.factor_id)
        if tuned_by_source.get(safe, 0) >= register_top_k:
            skipped_labels.append(f.factor_id)
        else:
            kept.append(f)
    return kept, skipped_labels


def labels_completed_in_dir(run_dir: Path) -> set[str]:
    """Scan a prior tune run dir for prototype/factor labels already completed.

    Each fully-processed label produces a ``proto_NNN_{label}/`` subdir
    containing ``registration_summary.json`` — this file is written by
    ``write_factor_artefacts`` *after* ``register_tuned_variants`` returns
    successfully (see ``alpha.cli.tune`` batch loop). Its absence therefore
    identifies a prototype that either crashed mid-registration or never
    started. Those labels are NOT returned so the caller still reprocesses
    them.

    Returns a set of label strings that match the keys produced by
    ``group_eligible_by_prototype``:

        - Real prototypes: ``"alpha044"``
        - Solo factors:    ``"_solo_llm_claude_foo"`` (keeps the ``_`` prefix)

    The regex captures the label verbatim from ``proto_\\d{3}_(.+)``; since
    ``build_factor_dir`` joins the format ``proto_{idx:03d}_{label}`` with a
    single underscore, a label that itself starts with ``_`` naturally yields
    the ``"__"`` double-underscore pattern seen in on-disk directory names.
    """
    labels: set[str] = set()
    if not run_dir.is_dir():
        return labels
    for child in run_dir.iterdir():
        if not child.is_dir():
            continue
        m = _PROTO_DIR_RE.match(child.name)
        if m is None:
            continue
        if not (child / "registration_summary.json").exists():
            continue
        labels.add(m.group("label"))
    return labels
