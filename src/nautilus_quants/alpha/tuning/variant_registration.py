# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Register tuned factor variants into the registry.

Takes the ``TuneResult.top_k`` trials, rebuilds each variant's expression,
runs a full alphalens-compatible analysis over the supplied panel, and
upserts the resulting ``FactorRecord`` + ``AnalysisMetrics`` rows into the
source registry (``test.duckdb`` by default).

Design notes
------------
* Only the top-k trials get registered — the inner loop already skipped
  pruned / duplicate expressions via ``_unique_top_k``.
* The ``source`` field is set from ``TuneConfig.source`` (default
  ``"tune"``), keeping LLM-mined and tuned factors distinguishable in the
  registry.
* ``prototype`` is inherited from the source factor so downstream audit
  tools continue to group variants correctly.
* Expression-hash collisions with an existing factor are a no-op (the
  existing record wins); collisions on ``factor_id`` trigger an auto-rename
  by appending ``_{hash[:8]}`` — this mirrors
  ``FactorRepository.register_factors_from_config``'s contract.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import pandas as pd

from nautilus_quants.alpha.analysis.evaluator import run_alphalens_with_forward_returns
from nautilus_quants.alpha.analysis.report import (
    build_analysis_metrics,
    compute_all_factor_metrics,
    compute_ic_summary,
)
from nautilus_quants.alpha.registry.models import AnalysisMetrics, FactorRecord
from nautilus_quants.alpha.registry.repository import FactorRepository
from nautilus_quants.alpha.tuning.config import TuneConfig, TuneResult
from nautilus_quants.alpha.tuning.objective import evaluate_expression_panel
from nautilus_quants.factors.expression.normalize import expression_hash

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ── Data model for registration output ─────────────────────────────────────


@dataclass(frozen=True)
class RegisteredVariant:
    """One tuned variant that has been written to the registry."""

    factor_id: str
    expression: str
    status: str
    outcome: str  # "new" | "updated" | "unchanged" | "skipped_duplicate"
    source_factor_id: str
    metrics: tuple[AnalysisMetrics, ...]


@dataclass(frozen=True)
class RegistrationSummary:
    """Aggregate result of registering a ``TuneResult``'s top_k variants."""

    source_factor_id: str
    variants: tuple[RegisteredVariant, ...]
    n_registered: int
    n_updated: int
    n_skipped: int

    @property
    def factor_ids(self) -> tuple[str, ...]:
        return tuple(v.factor_id for v in self.variants)


# ── Helpers ────────────────────────────────────────────────────────────────


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_]+")


def _make_variant_id(
    source_factor_id: str,
    expression: str,
    rank: int,
) -> str:
    """Create a stable factor_id for a tuned variant.

    Pattern: ``{source_id}_tune{rank}_{hash[:8]}`` — short enough to avoid
    excessive length in the registry, deterministic enough to survive
    repeated tune runs.
    """
    h = expression_hash(expression)[:8]
    base = _SAFE_NAME_RE.sub("_", source_factor_id)
    return f"{base}_tune{rank}_{h}"


def _panel_to_alphalens(
    factor_panel: pd.DataFrame,
) -> pd.Series:
    """Stack a ``T × N`` panel into a MultiIndex(date, asset) Series.

    Mirrors the conversion step in ``FactorEvaluator._compute_all_factors``
    so the result plugs straight into ``run_alphalens_with_forward_returns``.
    """
    if factor_panel.empty:
        return pd.Series(dtype=float)
    stacked = factor_panel.stack(future_stack=True).dropna().astype(float)
    stacked.index.names = ["date", "asset"]
    return stacked


def _format_bar_period(n_bars: int, bar_td: pd.Timedelta) -> str:
    """Format ``n_bars * bar_td`` as a human-readable period label.

    Mirrors ``alpha.analysis.evaluator._format_bar_period``. Examples:
      n_bars=1, bar_td=4h → "4h"
      n_bars=6, bar_td=4h → "1d"
      n_bars=18, bar_td=4h → "3d"
    """
    total_seconds = int((bar_td * n_bars).total_seconds())
    if total_seconds % 86400 == 0:
        return f"{total_seconds // 86400}d"
    if total_seconds % 3600 == 0:
        return f"{total_seconds // 3600}h"
    if total_seconds % 60 == 0:
        return f"{total_seconds // 60}m"
    return f"{total_seconds}s"


def _rename_period_columns(
    df: pd.DataFrame,
    periods: tuple[int, ...],
) -> pd.DataFrame:
    """Rename alphalens Timedelta-style period column names.

    alphalens-reloaded names columns based on its internal
    ``diff_custom_calendar_timedeltas``, producing labels like
    ``"1D4h"`` even when the caller's intent is just "1 bar of 4h".
    Mirrors ``FactorEvaluator._rename_period_columns`` so that
    registered ``AnalysisMetrics.period`` rows look identical to the ones
    produced by ``alpha analyze``.
    """
    inferred_freq = df.index.levels[0].freq if hasattr(df.index, "levels") else None
    if inferred_freq is None:
        return df
    try:
        bar_td = pd.Timedelta(inferred_freq)
    except (ValueError, TypeError):
        try:
            bar_td = pd.Timedelta(
                pd.tseries.frequencies.to_offset(inferred_freq).nanos,
                unit="ns",
            )
        except Exception:
            return df

    period_cols = [c for c in df.columns if c not in ("factor", "factor_quantile")]
    sorted_periods = sorted(periods)
    if len(period_cols) != len(sorted_periods):
        return df

    rename_map = {col: _format_bar_period(n, bar_td) for col, n in zip(period_cols, sorted_periods)}
    return df.rename(columns=rename_map)


# ── Per-variant analysis ───────────────────────────────────────────────────


@dataclass(frozen=True)
class VariantAnalysis:
    """Alphalens outputs plus pre-computed metrics for one variant."""

    expression: str
    ic_summary: pd.DataFrame
    factor_metrics: Any  # ``FactorMetricsResult`` from analysis.report
    n_samples: int


def analyze_variant(
    expression: str,
    panel_fields: dict[str, pd.DataFrame | float],
    pricing: pd.DataFrame,
    *,
    periods: tuple[int, ...] = (1, 2, 6, 18),
    quantiles: int = 5,
    max_loss: float = 0.35,
    filter_zscore: float | None = 20.0,
    parameters: dict[str, Any] | None = None,
) -> VariantAnalysis | None:
    """Evaluate ``expression`` and run a full multi-period alphalens analysis.

    Mirrors what ``alpha analyze`` does for a single factor: build the
    factor series, derive proper alphalens forward-returns frame from
    ``pricing`` over the requested ``periods``, then call
    ``compute_ic_summary`` + ``compute_all_factor_metrics``. The output is
    written verbatim into the registry, so a tuned variant ends up with the
    same metric coverage (one row per period) as a hand-written factor.

    Returns ``None`` when the variant produces an empty / unusable panel —
    callers should skip registration for that variant.
    """
    factor_panel = evaluate_expression_panel(expression, panel_fields, parameters)
    if not isinstance(factor_panel, pd.DataFrame) or factor_panel.empty:
        logger.debug("Variant '%s' produced empty panel", expression)
        return None

    factor_series = _panel_to_alphalens(factor_panel)
    if factor_series.empty:
        logger.debug("Variant '%s' has no valid MultiIndex rows", expression)
        return None

    # Build alphalens-format forward returns from pricing, mirroring
    # ``FactorEvaluator.compute_forward_returns``. Doing it inside this
    # function keeps the call-site simple — caller only passes ``pricing``.
    #
    # Tuning produces variants whose distributions can be highly degenerate
    # (e.g. ``clip_quantile`` with aggressive bounds collapses 50%+ of the
    # mass onto two constants → quantile binning loses most of the data).
    # The CV inner loop uses Spearman IC which is rank-tie-tolerant, so
    # such variants still survive selection. Here we tolerate the
    # binning-induced data loss by retrying alphalens with ``max_loss=1.0``
    # if the configured threshold rejects the variant — the IC
    # series itself is still meaningful even when quantile-based stats
    # degrade.
    import alphalens.utils as al_utils

    inferred = pd.infer_freq(pricing.index) if len(pricing) > 2 else "h"
    freq_offset = pd.tseries.frequencies.to_offset(inferred or "h")
    original_infer = al_utils.infer_trading_calendar

    def _crypto_calendar(_factor_idx, _prices_idx):
        return freq_offset

    al_utils.infer_trading_calendar = _crypto_calendar
    result: dict | None = None
    used_max_loss = max_loss
    try:
        try:
            forward_returns = al_utils.compute_forward_returns(
                factor=factor_series,
                prices=pricing,
                periods=tuple(periods),
                filter_zscore=filter_zscore,
            )
            # Rename alphalens internal Timedelta column names ("1D4h",
            # "2D8h", ...) to human-readable bar-count labels ("4h", "8h",
            # ...) BEFORE calling alphalens — this way every downstream
            # frame (factor_data, ic_df, ic_summary) inherits the right
            # period labels. Mirrors ``FactorEvaluator.compute_forward_returns``.
            forward_returns = _rename_period_columns(forward_returns, periods)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "compute_forward_returns failed on variant '%s': %s",
                expression,
                exc,
            )
            return None

        # First try the strict max_loss; on failure retry once with 1.0 so
        # heavily-clipped variants still register IC metrics.
        for attempt_max_loss in (max_loss, 1.0):
            try:
                result = run_alphalens_with_forward_returns(
                    factor_series,
                    forward_returns,
                    quantiles,
                    attempt_max_loss,
                )
                used_max_loss = attempt_max_loss
                if attempt_max_loss != max_loss:
                    logger.warning(
                        "Variant '%s' triggered max_loss fallback " "(retried with max_loss=1.0)",
                        expression,
                    )
                break
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "Alphalens failed on variant '%s' at max_loss=%.2f: %s",
                    expression,
                    attempt_max_loss,
                    exc,
                )
                continue

    finally:
        al_utils.infer_trading_calendar = original_infer

    # Degenerate-distribution fallback: alphalens's quantile binning may
    # drop 100% of the data when the factor distribution has heavy ties
    # (e.g. ``clip_quantile`` with aggressive bounds). In that case either
    # ``factor_data`` is empty or alphalens raises internally (e.g.
    # ``mean_return_by_quantile`` "No objects to concatenate"). We still
    # derive IC from the raw factor + forward returns via vectorised
    # Spearman rank correlation — that's the metric the inner CV loop
    # optimises against, so it MUST end up in the registry for downstream
    # ``promote`` filtering to evaluate the variant fairly.
    factor_data = result["factor_data"] if result is not None else pd.DataFrame()
    ic_df = result["ic"] if result is not None else pd.DataFrame()

    if result is None or factor_data.empty or ic_df.empty:
        ic_df = _ic_fallback_from_forward_returns(factor_panel, forward_returns)
        if ic_df.empty:
            logger.debug(
                "Variant '%s' has no usable IC even after Spearman fallback",
                expression,
            )
            return None
        # Reject low-n IC series. With < FALLBACK_MIN_IC_ROWS valid rows the
        # ICIR ratio (mean / std) is a small-sample artefact — typical
        # cause: aggressive clip / normalize bounds collapse the
        # cross-section, so almost every timestamp's rank correlation is
        # NaN-out from ties. The remaining 3-10 rows produce wildly inflated
        # ICIR (e.g. 1.37 from 3 samples) that fool downstream `promote`.
        # Better to write nothing than write a misleading metric.
        FALLBACK_MIN_IC_ROWS = 100  # ~17 days of 4h bars
        max_per_period_rows = max(int(ic_df[col].notna().sum()) for col in ic_df.columns)
        if max_per_period_rows < FALLBACK_MIN_IC_ROWS:
            logger.warning(
                "Variant '%s': fallback produced only %d valid IC rows "
                "(< %d required); skipping registration to avoid noisy "
                "small-sample ICIR (likely degenerate clip/normalize bounds)",
                expression,
                max_per_period_rows,
                FALLBACK_MIN_IC_ROWS,
            )
            return None
        logger.warning(
            "Variant '%s': alphalens binning lost all data; "
            "using Spearman-IC fallback (quantile metrics will be null)",
            expression,
        )
        ic_summary = compute_ic_summary(ic_df)
        return VariantAnalysis(
            expression=expression,
            ic_summary=ic_summary,
            factor_metrics=None,
            n_samples=int(ic_df.notna().sum().sum()),
        )

    # Top-level guard: a single pathological variant (e.g. sparse IC with
    # only 1-2 valid rows per period) must not kill the whole tune batch.
    # compute_ic_summary can raise from downstream scipy/pandas quirks we
    # don't control; if that happens we skip registration for this variant
    # rather than propagate the exception up to the CLI batch loop.
    try:
        ic_summary = compute_ic_summary(ic_df)
        total_timestamps = len(pricing.index)
        total_assets = len(pricing.columns)
        try:
            factor_metrics = compute_all_factor_metrics(
                factor_data,
                ic_df,
                total_timestamps,
                total_assets,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Extended metrics failed on variant '%s': %s",
                expression,
                exc,
            )
            factor_metrics = None

        return VariantAnalysis(
            expression=expression,
            ic_summary=ic_summary,
            factor_metrics=factor_metrics,
            n_samples=len(factor_data),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Variant '%s': ic_summary computation failed (%s); " "skipping registration",
            expression,
            exc,
        )
        return None


def _ic_fallback_from_forward_returns(
    factor_panel: pd.DataFrame,
    forward_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-period IC series via Spearman rank when alphalens fails.

    ``factor_panel`` is ``index=date, columns=instrument``; ``forward_returns``
    is alphalens-format ``MultiIndex(date, asset)`` with one column per
    period. The output mirrors ``alphalens.performance.factor_information_coefficient``
    — index by date, one column per period — so ``compute_ic_summary``
    can ingest it without changes.
    """
    if factor_panel.empty or forward_returns.empty:
        return pd.DataFrame()

    period_cols = [c for c in forward_returns.columns if c not in ("factor", "factor_quantile")]
    if not period_cols:
        return pd.DataFrame()

    factor_ranks = factor_panel.rank(axis=1, method="average")
    ic_dict: dict[str, pd.Series] = {}
    for period in period_cols:
        # Pivot the period column into a date-by-asset frame.
        period_panel = forward_returns[period].unstack(level="asset")
        common_idx = factor_panel.index.intersection(period_panel.index)
        common_cols = factor_panel.columns.intersection(period_panel.columns)
        if len(common_idx) == 0 or len(common_cols) == 0:
            continue
        f = factor_ranks.loc[common_idx, common_cols]
        r = period_panel.loc[common_idx, common_cols].rank(axis=1, method="average")
        # Mask rows where fewer than 5 instruments have valid pairs.
        valid = (~f.isna() & ~r.isna()).sum(axis=1) >= 5
        if not valid.any():
            continue
        ic_dict[period] = f.loc[valid].corrwith(r.loc[valid], axis=1).dropna()

    if not ic_dict:
        return pd.DataFrame()
    return pd.concat(ic_dict, axis=1)


# ── Top-level registration ─────────────────────────────────────────────────


def register_tuned_variants(
    *,
    tune_result: TuneResult,
    source_factor: FactorRecord,
    repo: FactorRepository,
    panel_fields: dict[str, pd.DataFrame | float],
    pricing: pd.DataFrame,
    tune_config: TuneConfig,
    run_id: str,
    timeframe: str,
    periods: tuple[int, ...] = (1, 2, 6, 18),
    quantiles: int = 5,
    max_loss: float = 0.35,
    filter_zscore: float | None = 20.0,
    output_dir: str = "",
    status: str = "candidate",
) -> RegistrationSummary:
    """Register the top-k variants from ``tune_result`` into ``repo``.

    Each variant goes through a full ``alpha analyze`` pass (multi-period
    alphalens IC + extended factor metrics) before being upserted, so the
    resulting ``AnalysisMetrics`` rows look identical to those produced by a
    standalone ``alpha analyze`` run for the same expression.

    The ``source_factor`` passes through its ``prototype``, ``tags``, and
    ``variables`` fields to every new variant so the usual registry tooling
    (``alpha inspect --prototype X``, ``alpha audit``) keeps working.
    """
    variants: list[RegisteredVariant] = []
    n_new = n_updated = n_skipped = 0

    for rank, trial in enumerate(tune_result.top_k, start=1):
        expression = trial.expression
        factor_id = _make_variant_id(source_factor.factor_id, expression, rank)
        h = expression_hash(expression)

        # Expression-dedup: if another factor already has this hash, link
        # to it and skip registration.
        existing_by_hash = repo.find_by_expression_hash(h)
        if existing_by_hash is not None and existing_by_hash.factor_id != factor_id:
            variants.append(
                RegisteredVariant(
                    factor_id=existing_by_hash.factor_id,
                    expression=expression,
                    status=existing_by_hash.status,
                    outcome="skipped_duplicate",
                    source_factor_id=source_factor.factor_id,
                    metrics=(),
                )
            )
            n_skipped += 1
            continue

        # Compute metrics for the new variant — full multi-period alphalens.
        analysis = analyze_variant(
            expression,
            panel_fields,
            pricing,
            periods=periods,
            quantiles=quantiles,
            max_loss=max_loss,
            filter_zscore=filter_zscore,
            parameters=source_factor.parameters,
        )

        # Upsert FactorRecord.
        description = (
            f"Tuned variant (rank {rank}) of {source_factor.factor_id} — "
            f"ICIR CV {trial.mean_icir:+.4f}"
        )
        tags = list(dict.fromkeys([*source_factor.tags, "tuned"]))

        # Build parameters from the *tuned* expression so they stay in sync
        # with the actual numeric literals baked into ``expression``.
        # Without this, FactorRecord.parameters would just be a copy of the
        # source factor's params (e.g. ``{p0: 20}``) while ``expression``
        # uses the tuned value (e.g. ``ts_mean(..., 80)``) — confusing
        # in ``alpha inspect`` and only fixed downstream by ``alpha backfill``.
        # Mirrors ``audit.backfill_factors`` so the two paths agree.
        try:
            from nautilus_quants.factors.expression.normalize import expression_template

            _, vals = expression_template(expression)
            extracted = {f"p{i}": float(v) for i, v in enumerate(vals)}
            non_p = {
                k: v
                for k, v in source_factor.parameters.items()
                if not (k.startswith("p") and k[1:].isdigit())
            }
            tuned_parameters: dict[str, Any] = {**non_p, **extracted}
        except Exception:
            # Fall back to source params on parse error — backfill will
            # repair later if needed.
            tuned_parameters = dict(source_factor.parameters)

        record = FactorRecord(
            factor_id=factor_id,
            expression=expression,
            expression_hash=h,
            prototype=source_factor.prototype,
            description=description,
            source=tune_config.source,
            status=status,
            tags=tags,
            parameters=tuned_parameters,
            variables=dict(source_factor.variables),
        )
        outcome = repo.upsert_factor(record)
        if outcome == "new":
            n_new += 1
        elif outcome == "updated":
            n_updated += 1

        metrics: list[AnalysisMetrics] = []
        if analysis is not None:
            metrics = list(
                build_analysis_metrics(
                    run_id=run_id,
                    factor_id=factor_id,
                    timeframe=timeframe,
                    ic_summary=analysis.ic_summary,
                    metrics_result=analysis.factor_metrics,
                    output_dir=output_dir,
                )
            )
            if metrics:
                repo.save_metrics(metrics)

        variants.append(
            RegisteredVariant(
                factor_id=factor_id,
                expression=expression,
                status=status,
                outcome=outcome,
                source_factor_id=source_factor.factor_id,
                metrics=tuple(metrics),
            )
        )

    return RegistrationSummary(
        source_factor_id=source_factor.factor_id,
        variants=tuple(variants),
        n_registered=n_new,
        n_updated=n_updated,
        n_skipped=n_skipped,
    )
