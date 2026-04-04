# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Factor scoring, deduplication, decorrelation, and promotion engine.

Implements the three-phase factor promotion pipeline:
  Phase 1: Hard filtering + 5-dimension scoring
  Phase 2: Fingerprint dedup + Spearman correlation dedup + greedy selection
  Phase 3: Migration from source env to target env
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata

from nautilus_quants.alpha.registry.database import RegistryDatabase

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class HardFilterConfig:
    """Hard filter thresholds — any period failing all is excluded."""

    icir_abs_min: float = 0.05
    t_stat_nw_abs_min: float = 2.0
    p_value_nw_max: float = 0.05
    win_rate_lower: float = 0.45
    win_rate_upper: float = 0.55
    ic_linearity_min: float = 0.85
    coverage_min: float = 0.50
    n_samples_min: int = 3000
    min_valid_periods: int = 2


@dataclass(frozen=True)
class ScoringWeights:
    """Dimension weights for the 5-dimension scoring system."""

    predictiveness: float = 0.30
    stability: float = 0.25
    monotonicity: float = 0.10
    consistency: float = 0.15
    turnover_friendliness: float = 0.20


@dataclass(frozen=True)
class SubWeights:
    """Sub-dimension weights within each dimension."""

    pred_icir: float = 0.6
    pred_t_stat_nw: float = 0.4
    stab_ic_linearity: float = 0.6
    stab_win_rate: float = 0.4


@dataclass(frozen=True)
class DedupConfig:
    """Deduplication configuration."""

    fingerprint_threshold: float = 1e-4
    max_corr: float = 0.30


@dataclass(frozen=True)
class PromoteConfig:
    """Promotion configuration."""

    max_factors: int = 50
    target_status: str = "active"


@dataclass(frozen=True)
class DataConfig:
    """Data paths for correlation computation."""

    catalog_path: str = ""
    bar_spec: str = "4h"
    factor_configs: list[str] = field(default_factory=list)
    instrument_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ScoringConfig:
    """Complete scoring configuration."""

    periods: list[str] = field(
        default_factory=lambda: ["4h", "8h", "12h", "1d"],
    )
    hard_filters: HardFilterConfig = field(default_factory=HardFilterConfig)
    weights: ScoringWeights = field(default_factory=ScoringWeights)
    sub_weights: SubWeights = field(default_factory=SubWeights)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    promote: PromoteConfig = field(default_factory=PromoteConfig)
    data: DataConfig = field(default_factory=DataConfig)


def load_scoring_config(path: str | Path) -> ScoringConfig:
    """Load scoring configuration from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scoring config not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    hf = raw.get("hard_filters", {})
    sc = raw.get("scoring", {})
    sw = sc.get("sub_weights", {})
    dd = raw.get("dedup", {})
    pr = raw.get("promote", {})
    da = raw.get("data", {})

    return ScoringConfig(
        periods=raw.get("periods", ["4h", "8h", "12h", "1d"]),
        hard_filters=HardFilterConfig(
            icir_abs_min=hf.get("icir_abs_min", 0.05),
            t_stat_nw_abs_min=hf.get("t_stat_nw_abs_min", 2.0),
            p_value_nw_max=hf.get("p_value_nw_max", 0.05),
            win_rate_lower=hf.get("win_rate_lower", 0.45),
            win_rate_upper=hf.get("win_rate_upper", 0.55),
            ic_linearity_min=hf.get("ic_linearity_min", 0.85),
            coverage_min=hf.get("coverage_min", 0.50),
            n_samples_min=hf.get("n_samples_min", 3000),
            min_valid_periods=hf.get("min_valid_periods", 2),
        ),
        weights=ScoringWeights(
            **{k: v for k, v in sc.get("weights", {}).items()
               if k in ScoringWeights.__dataclass_fields__},
        ) if sc.get("weights") else ScoringWeights(),
        sub_weights=SubWeights(
            pred_icir=sw.get("predictiveness", {}).get("icir", 0.6),
            pred_t_stat_nw=sw.get("predictiveness", {}).get("t_stat_nw", 0.4),
            stab_ic_linearity=sw.get("stability", {}).get("ic_linearity", 0.6),
            stab_win_rate=sw.get("stability", {}).get("win_rate", 0.4),
        ),
        dedup=DedupConfig(
            fingerprint_threshold=dd.get("fingerprint_threshold", 1e-4),
            max_corr=dd.get("max_corr", 0.30),
        ),
        promote=PromoteConfig(
            max_factors=pr.get("max_factors", 50),
            target_status=pr.get("target_status", "active"),
        ),
        data=DataConfig(
            catalog_path=da.get("catalog_path", ""),
            bar_spec=da.get("bar_spec", "4h"),
            factor_configs=da.get("factor_configs", []),
            instrument_ids=da.get("instrument_ids", []),
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1: Load data + Hard filters + Scoring
# ──────────────────────────────────────────────────────────────────────────────


def load_scoring_data(
    db: RegistryDatabase,
    periods: list[str],
) -> pd.DataFrame:
    """Load analysis metrics from the registry for scoring.

    Queries the latest run per factor_id and pivots on ``period`` so
    each row is a factor_id with columns like
    ``icir_4h``, ``ic_mean_4h``, ``t_stat_nw_4h``, etc.

    The ``periods`` parameter corresponds to the ``period`` column
    (forward-return horizons like 4h, 8h, 12h, 1d), NOT ``timeframe``.

    Returns a wide DataFrame indexed by ``factor_id``.
    """
    # Get latest run per factor_id
    sql = """
        WITH latest_runs AS (
            SELECT factor_id,
                   MAX(created_at) AS max_created
            FROM alpha_analysis_metrics
            GROUP BY factor_id
        )
        SELECT m.*
        FROM alpha_analysis_metrics m
        INNER JOIN latest_runs lr
            ON m.factor_id = lr.factor_id
           AND m.created_at = lr.max_created
    """
    conn = db.connection
    df = conn.execute(sql).fetchdf()

    if df.empty:
        return pd.DataFrame()

    # Filter to requested periods
    df = df[df["period"].isin(periods)]

    if df.empty:
        return pd.DataFrame()

    # Pivot: one row per factor_id, columns suffixed by period
    metric_cols = [
        "icir", "ic_mean", "ic_std", "t_stat_nw", "p_value_nw",
        "win_rate", "monotonicity", "ic_linearity", "ic_ar1",
        "coverage", "n_samples",
    ]

    pivot_frames = []
    for period in periods:
        p_df = df[df["period"] == period].copy()
        if p_df.empty:
            continue
        p_df = p_df.drop_duplicates(subset=["factor_id"], keep="first")
        p_df = p_df.set_index("factor_id")
        rename = {col: f"{col}_{period}" for col in metric_cols if col in p_df.columns}
        pivot_frames.append(p_df[list(rename.keys())].rename(columns=rename))

    if not pivot_frames:
        return pd.DataFrame()

    wide = pd.concat(pivot_frames, axis=1)
    return wide


def apply_hard_filters(
    df: pd.DataFrame,
    config: HardFilterConfig,
    periods: list[str],
) -> pd.DataFrame:
    """Apply hard filters per period.

    Each period is checked independently. A factor is eliminated only if
    fewer than ``min_valid_periods`` pass all checks.

    Returns a DataFrame with an added ``valid_periods`` column listing
    which periods passed, and a ``n_valid_periods`` count.
    """
    if df.empty:
        return df

    valid_periods_map: dict[str, list[str]] = {}

    for factor_id in df.index:
        valid = []
        for period in periods:
            suffix = f"_{period}"
            try:
                icir = df.loc[factor_id, f"icir{suffix}"]
                t_nw = df.loc[factor_id, f"t_stat_nw{suffix}"]
                p_nw = df.loc[factor_id, f"p_value_nw{suffix}"]
                wr = df.loc[factor_id, f"win_rate{suffix}"]
                lin = df.loc[factor_id, f"ic_linearity{suffix}"]
                cov = df.loc[factor_id, f"coverage{suffix}"]
                n = df.loc[factor_id, f"n_samples{suffix}"]
            except KeyError:
                continue

            # Skip if any metric is NaN
            vals = [icir, t_nw, p_nw, wr, lin, cov, n]
            if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in vals):
                continue

            passes = (
                abs(icir) >= config.icir_abs_min
                and abs(t_nw) >= config.t_stat_nw_abs_min
                and p_nw <= config.p_value_nw_max
                and (wr > config.win_rate_upper or wr < config.win_rate_lower)
                and lin >= config.ic_linearity_min
                and cov >= config.coverage_min
                and n >= config.n_samples_min
            )
            if passes:
                valid.append(period)

        valid_periods_map[factor_id] = valid

    df = df.copy()
    df["valid_periods"] = df.index.map(
        lambda fid: valid_periods_map.get(fid, []),
    )
    df["n_valid_periods"] = df["valid_periods"].apply(len)

    # Filter: need at least min_valid_periods
    mask = df["n_valid_periods"] >= config.min_valid_periods
    return df[mask]


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    """Compute percentile rank (0~1) for an array, handling NaN."""
    valid_mask = ~np.isnan(values)
    result = np.full_like(values, np.nan)
    if valid_mask.sum() > 0:
        ranks = rankdata(values[valid_mask], method="average")
        result[valid_mask] = (ranks - 1) / max(len(ranks) - 1, 1)
    return result


def score_factors(
    df: pd.DataFrame,
    config: ScoringConfig,
) -> pd.DataFrame:
    """Compute the 5-dimension composite score for each factor.

    Scoring is done across valid periods using percentile rank
    normalization (0~1).

    Returns DataFrame with added score columns:
      - ``pred_score``, ``stab_score``, ``mono_score`` (per-period averages)
      - ``consistency``, ``turnover_friendliness``
      - ``final_score``
    """
    if df.empty:
        return df

    weights = config.weights
    sub_w = config.sub_weights
    periods = config.periods

    # Collect per-factor, per-period sub-scores
    records: list[dict[str, Any]] = []

    # First, collect all raw values for percentile ranking
    all_icir: list[float] = []
    all_t_nw: list[float] = []
    all_lin: list[float] = []
    all_wr_dev: list[float] = []
    all_mono: list[float] = []

    factor_period_data: dict[str, dict[str, dict[str, float]]] = {}

    for factor_id in df.index:
        valid_pds = df.loc[factor_id, "valid_periods"]
        if not valid_pds:
            continue
        factor_period_data[factor_id] = {}
        for period in valid_pds:
            sfx = f"_{period}"
            icir_val = abs(df.loc[factor_id, f"icir{sfx}"])
            t_nw_val = abs(df.loc[factor_id, f"t_stat_nw{sfx}"])
            lin_val = df.loc[factor_id, f"ic_linearity{sfx}"]
            wr_val = df.loc[factor_id, f"win_rate{sfx}"]
            mono_val = abs(df.loc[factor_id, f"monotonicity{sfx}"])

            d = {
                "icir": icir_val,
                "t_nw": t_nw_val,
                "lin": lin_val,
                "wr_dev": abs(wr_val - 0.5),
                "mono": mono_val,
            }
            factor_period_data[factor_id][period] = d

            all_icir.append(icir_val)
            all_t_nw.append(t_nw_val)
            all_lin.append(lin_val)
            all_wr_dev.append(abs(wr_val - 0.5))
            all_mono.append(mono_val)

    if not factor_period_data:
        return df

    # Compute percentile ranks across all factor-period observations
    arr_icir = np.array(all_icir)
    arr_t_nw = np.array(all_t_nw)
    arr_lin = np.array(all_lin)
    arr_wr_dev = np.array(all_wr_dev)
    arr_mono = np.array(all_mono)

    rank_icir = _percentile_rank(arr_icir)
    rank_t_nw = _percentile_rank(arr_t_nw)
    rank_lin = _percentile_rank(arr_lin)
    rank_wr_dev = _percentile_rank(arr_wr_dev)
    rank_mono = _percentile_rank(arr_mono)

    # Map back to factor-period
    idx = 0
    factor_scores: dict[str, dict[str, Any]] = {}

    for factor_id, period_data in factor_period_data.items():
        per_period_scores = []
        pred_scores = []
        stab_scores = []
        mono_scores = []
        raw_icir_vals = []
        raw_t_nw_vals = []
        raw_wr_vals = []
        raw_lin_vals = []
        raw_mono_vals = []

        for period, d in period_data.items():
            r_icir = rank_icir[idx]
            r_t_nw = rank_t_nw[idx]
            r_lin = rank_lin[idx]
            r_wr = rank_wr_dev[idx]
            r_mono = rank_mono[idx]

            pred = sub_w.pred_icir * r_icir + sub_w.pred_t_stat_nw * r_t_nw
            stab = sub_w.stab_ic_linearity * r_lin + sub_w.stab_win_rate * r_wr
            mono = r_mono

            pred_scores.append(pred)
            stab_scores.append(stab)
            mono_scores.append(mono)

            # per_period weighted by config (pred + stab + mono dimensions)
            pp_score = (
                weights.predictiveness * pred
                + weights.stability * stab
                + weights.monotonicity * mono
            )
            per_period_scores.append(pp_score)
            idx += 1

            # Collect raw values for CSV transparency
            raw_icir_vals.append(d["icir"])
            raw_t_nw_vals.append(d["t_nw"])
            raw_wr_vals.append(d["wr_dev"] + 0.5)  # restore original win_rate
            raw_lin_vals.append(d["lin"])
            raw_mono_vals.append(d["mono"])

        # Consistency: min/max ratio
        if per_period_scores:
            max_pp = max(per_period_scores)
            min_pp = min(per_period_scores)
            consistency = min_pp / max_pp if max_pp > 0 else 0.0
        else:
            consistency = 0.0

        avg_period_score = np.mean(per_period_scores) if per_period_scores else 0.0

        factor_scores[factor_id] = {
            "avg_period_score": avg_period_score,
            "pred_score": float(np.mean(pred_scores)) if pred_scores else 0.0,
            "stab_score": float(np.mean(stab_scores)) if stab_scores else 0.0,
            "mono_score": float(np.mean(mono_scores)) if mono_scores else 0.0,
            "consistency": consistency,
            "per_period_scores": per_period_scores,
            # Raw metric averages
            "avg_icir": float(np.mean(raw_icir_vals)),
            "avg_t_stat_nw": float(np.mean(raw_t_nw_vals)),
            "avg_win_rate": float(np.mean(raw_wr_vals)),
            "avg_ic_linearity": float(np.mean(raw_lin_vals)),
            "avg_monotonicity": float(np.mean(raw_mono_vals)),
        }

    # Turnover friendliness: mean(|ic_ar1|) across valid periods
    ar1_means: dict[str, float] = {}
    for factor_id in factor_period_data:
        valid_pds = df.loc[factor_id, "valid_periods"]
        ar1_vals = []
        for period in valid_pds:
            sfx = f"_{period}"
            col = f"ic_ar1{sfx}"
            if col in df.columns:
                v = df.loc[factor_id, col]
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    ar1_vals.append(abs(v))
        ar1_means[factor_id] = np.mean(ar1_vals) if ar1_vals else 0.0

    # Percentile rank the ar1 means
    ar1_arr = np.array([ar1_means.get(fid, 0.0) for fid in factor_scores])
    ar1_ranks = _percentile_rank(ar1_arr)
    ar1_rank_map = dict(zip(factor_scores.keys(), ar1_ranks))

    # Final score: per-period dimensions + cross-period dimensions
    # per_period_weight = sum of pred + stab + mono weights
    pp_weight = weights.predictiveness + weights.stability + weights.monotonicity
    cons_weight = weights.consistency
    turn_weight = weights.turnover_friendliness

    df = df.copy()
    final_scores = {}
    for factor_id, fs in factor_scores.items():
        turnover_rank = ar1_rank_map.get(factor_id, 0.0)
        final = (
            pp_weight * fs["avg_period_score"]
            + cons_weight * fs["consistency"]
            + turn_weight * turnover_rank
        )
        final_scores[factor_id] = {
            "final_score": final,
            "avg_period_score": fs["avg_period_score"],
            "pred_score": fs["pred_score"],
            "stab_score": fs["stab_score"],
            "mono_score": fs["mono_score"],
            "consistency": fs["consistency"],
            "turnover_friendliness": turnover_rank,
            # Raw metric averages
            "avg_icir": fs["avg_icir"],
            "avg_t_stat_nw": fs["avg_t_stat_nw"],
            "avg_win_rate": fs["avg_win_rate"],
            "avg_ic_linearity": fs["avg_ic_linearity"],
            "avg_monotonicity": fs["avg_monotonicity"],
        }

    score_df = pd.DataFrame.from_dict(final_scores, orient="index")
    for col in score_df.columns:
        df[col] = score_df[col]

    # Sort by final_score descending
    df = df.sort_values("final_score", ascending=False)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: Dedup & Decorrelation
# ──────────────────────────────────────────────────────────────────────────────


def dedup_by_fingerprint(
    df: pd.DataFrame,
    periods: list[str],
    threshold: float = 1e-4,
) -> pd.DataFrame:
    """Remove exact duplicates based on indicator fingerprints.

    Fingerprint = (icir_4h, ic_mean_4h, icir_8h, ic_mean_8h,
                   icir_12h, ic_mean_12h, icir_1d, ic_mean_1d)

    Factors with Euclidean distance < threshold are duplicates.
    Keeps the factor with the lexicographically smallest factor_id.
    """
    if df.empty or len(df) < 2:
        return df

    # Build fingerprint matrix
    fp_cols = []
    for period in periods:
        for metric in ["icir", "ic_mean"]:
            col = f"{metric}_{period}"
            if col in df.columns:
                fp_cols.append(col)

    if not fp_cols:
        return df

    fp_matrix = df[fp_cols].fillna(0.0).values

    # Compute pairwise Euclidean distances
    if len(fp_matrix) < 2:
        return df

    dists = squareform(pdist(fp_matrix, metric="euclidean"))

    # Find duplicate pairs
    to_remove: set[str] = set()
    factor_ids = list(df.index)

    for i in range(len(factor_ids)):
        if factor_ids[i] in to_remove:
            continue
        for j in range(i + 1, len(factor_ids)):
            if factor_ids[j] in to_remove:
                continue
            if dists[i, j] < threshold:
                # Keep lexicographically smaller
                remove_id = max(factor_ids[i], factor_ids[j])
                keep_id = min(factor_ids[i], factor_ids[j])
                to_remove.add(remove_id)
                logger.info(
                    "Dedup: %s ≡ %s (dist=%.2e), keeping %s",
                    factor_ids[i], factor_ids[j], dists[i, j], keep_id,
                )

    return df.drop(index=list(to_remove), errors="ignore")


def compute_factor_correlation(
    factor_ids: list[str],
    config: ScoringConfig,
    registry_db: RegistryDatabase | None = None,
    registry_dbs: list[RegistryDatabase] | None = None,
) -> pd.DataFrame:
    """Compute cross-sectional Spearman rank correlation matrix.

    1. Load OHLCV data (40 coins × 4 years × 4h bars)
    2. Compute factor values for each source config
    3. Fallback: load missing factors from registry DB(s)
    4. Per cross-section: Spearman rank correlation
    5. Time-average → correlation matrix

    Args:
        registry_db: Single registry DB for fallback (convenience).
        registry_dbs: Multiple registry DBs to search (tried in order).

    Returns correlation matrix DataFrame (factor_id × factor_id).
    """
    from nautilus_quants.alpha.analysis.evaluator import FactorEvaluator
    from nautilus_quants.alpha.data_loader import CatalogDataLoader
    from nautilus_quants.factors.config import (
        FactorConfig,
        FactorDefinition,
        generate_factor_id,
        load_factor_config,
    )

    data_cfg = config.data
    if not data_cfg.catalog_path:
        raise ValueError(
            "data.catalog_path is required for correlation computation.",
        )
    if not data_cfg.factor_configs and registry_db is None and not registry_dbs:
        raise ValueError(
            "data.factor_configs or registry_db/registry_dbs is required "
            "for correlation computation.",
        )

    # Load bar data
    loader = CatalogDataLoader(data_cfg.catalog_path, data_cfg.bar_spec)
    bars_by_instrument = loader.load_bars(data_cfg.instrument_ids)

    # Compute factor values from each config
    all_factor_panels: dict[str, pd.DataFrame] = {}

    for config_path in data_cfg.factor_configs:
        try:
            factor_config = load_factor_config(config_path)
            evaluator = FactorEvaluator(factor_config)
            factor_series, _ = evaluator.evaluate(bars_by_instrument)

            for fname, series in factor_series.items():
                fid = generate_factor_id(factor_config.source, fname)
                if fid in factor_ids:
                    # Unstack to panel (timestamps × instruments)
                    panel = series.unstack(level="asset")
                    all_factor_panels[fid] = panel
        except Exception as e:
            logger.warning("Failed to compute factors from %s: %s", config_path, e)

    # ── Registry fallback for factors not covered by YAML configs ──
    all_dbs: list[RegistryDatabase] = []
    if registry_dbs:
        all_dbs.extend(registry_dbs)
    elif registry_db is not None:
        all_dbs.append(registry_db)

    if all_dbs:
        missing_ids = set(factor_ids) - set(all_factor_panels.keys())
        if missing_ids:
            logger.info(
                "Registry lookup: %d of %d factors missing from YAML configs",
                len(missing_ids), len(factor_ids),
            )
            from nautilus_quants.alpha.registry.repository import FactorRepository

            # Search across all registry DBs (first match wins)
            repos = [FactorRepository(db) for db in all_dbs]
            # (source, vars_key, params_key) → [(fid, key, expression), ...]
            groups: dict[tuple, list[tuple[str, str, str]]] = {}
            for fid in missing_ids:
                record = None
                for repo in repos:
                    record = repo.get_factor(fid)
                    if record is not None:
                        break
                if record is None:
                    logger.warning("Registry: factor %s not found", fid)
                    continue
                source = record.source or ""
                key = fid[len(source) + 1:] if source and fid.startswith(f"{source}_") else fid
                if not key:
                    key = fid
                vars_key = tuple(sorted((record.variables or {}).items()))
                params_key = tuple(sorted(
                    (k, v) for k, v in (record.parameters or {}).items()
                    if k != "promote_score"
                ))
                group_key = (source, vars_key, params_key)
                groups.setdefault(group_key, []).append(
                    (fid, key, record.expression),
                )

            # Merge groups with same source into larger batches to
            # avoid redundant bars→panel conversion per group.
            merged: dict[str, tuple[dict[str, str], dict[str, Any], list[tuple[str, str, str]]]] = {}
            for (source, vars_tup, params_tup), group_records in groups.items():
                if source not in merged:
                    merged[source] = (dict(vars_tup), dict(params_tup), list(group_records))
                else:
                    existing_vars, existing_params, existing_records = merged[source]
                    # Merge variables (union; later values win on conflict)
                    existing_vars.update(dict(vars_tup))
                    existing_params.update(dict(params_tup))
                    existing_records.extend(group_records)

            loaded_count = 0
            for source, (variables, parameters, all_records) in merged.items():
                key_to_fid = {key: orig_fid for orig_fid, key, _ in all_records}
                batch_config = FactorConfig(
                    name=f"registry_{source}" if source else "registry",
                    source=source,
                    variables=variables,
                    parameters=parameters,
                    factors=[
                        FactorDefinition(name=key, expression=expr)
                        for _, key, expr in all_records
                    ],
                )
                try:
                    logger.info(
                        "Registry: evaluating %d factors for source=%s",
                        len(all_records), source,
                    )
                    eval_instance = FactorEvaluator(batch_config)
                    factor_series, _ = eval_instance.evaluate(bars_by_instrument)
                    for fname, series in factor_series.items():
                        orig_fid = key_to_fid.get(fname)
                        if orig_fid is not None and orig_fid in factor_ids:
                            all_factor_panels[orig_fid] = series.unstack(
                                level="asset",
                            )
                            loaded_count += 1
                except Exception as e:
                    logger.warning(
                        "Failed to compute registry factors for source=%s: %s",
                        source, e,
                    )

            logger.info(
                "Registry: loaded %d factors in %d groups",
                loaded_count, len(groups),
            )

    if len(all_factor_panels) < 2:
        logger.warning(
            "Only %d factors computed for correlation (need ≥ 2)",
            len(all_factor_panels),
        )
        return pd.DataFrame()

    # Align all panels to common timestamps
    common_idx = None
    for panel in all_factor_panels.values():
        if common_idx is None:
            common_idx = panel.index
        else:
            common_idx = common_idx.intersection(panel.index)

    if common_idx is None or len(common_idx) == 0:
        return pd.DataFrame()

    matched_ids = [fid for fid in factor_ids if fid in all_factor_panels]
    n_factors = len(matched_ids)

    # Cross-sectional Spearman correlation, averaged over time
    from scipy.stats import spearmanr

    corr_sum = np.zeros((n_factors, n_factors))
    n_valid = np.zeros((n_factors, n_factors))

    for t in common_idx:
        # Get cross-section at time t for each factor
        cs_data = {}
        for fid in matched_ids:
            row = all_factor_panels[fid].loc[t].dropna()
            cs_data[fid] = row

        # Need at least 5 common assets
        common_assets = None
        for fid, row in cs_data.items():
            if common_assets is None:
                common_assets = set(row.index)
            else:
                common_assets &= set(row.index)

        if common_assets is None or len(common_assets) < 5:
            continue

        common_assets_list = sorted(common_assets)

        # Build matrix: factors × assets
        cs_matrix = np.column_stack([
            cs_data[fid].loc[common_assets_list].values
            for fid in matched_ids
        ])

        # Spearman rank correlation for this cross-section
        rho, _ = spearmanr(cs_matrix)
        if rho.ndim == 0:
            # Only 2 factors
            rho = np.array([[1.0, float(rho)], [float(rho), 1.0]])

        valid_mask = ~np.isnan(rho)
        corr_sum[valid_mask] += rho[valid_mask]
        n_valid[valid_mask] += 1

    # Average
    with np.errstate(invalid="ignore"):
        avg_corr = np.where(n_valid > 0, corr_sum / n_valid, 0.0)

    corr_df = pd.DataFrame(avg_corr, index=matched_ids, columns=matched_ids)
    return corr_df


def greedy_select(
    scores: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    max_corr: float = 0.30,
    max_factors: int = 50,
    existing_ids: list[str] | None = None,
) -> list[str]:
    """Greedy factor selection: pick top-scored, reject if corr > max_corr.

    If ``existing_ids`` is provided, those factors are pre-seeded into the
    selected set (they act as gatekeepers but are not themselves returned).

    ```python
    selected = [*existing_ids_in_corr_matrix]   # pre-seeded
    for factor_id in sorted_by_score_descending:
        if all(|corr(factor_id, s)| <= max_corr for s in selected):
            selected.append(factor_id)
    return [s for s in selected if s not in existing_ids]
    ```
    """
    sorted_ids = scores.sort_values("final_score", ascending=False).index.tolist()

    # Pre-seed with existing factors from target DB
    existing_set: set[str] = set()
    selected: list[str] = []
    if existing_ids:
        for eid in existing_ids:
            if eid in corr_matrix.index:
                selected.append(eid)
                existing_set.add(eid)
        if existing_set:
            logger.info(
                "Greedy: %d existing factors pre-seeded as gatekeepers",
                len(existing_set),
            )

    n_new = 0
    for factor_id in sorted_ids:
        if factor_id in existing_set:
            continue
        if n_new >= max_factors:
            break
        if factor_id not in corr_matrix.index:
            logger.warning(
                "Greedy skip: %s not in correlation matrix", factor_id,
            )
            continue

        correlated = False
        for s in selected:
            if s in corr_matrix.columns:
                c = abs(corr_matrix.loc[factor_id, s])
                if c > max_corr:
                    correlated = True
                    source = "existing" if s in existing_set else "selected"
                    logger.info(
                        "Greedy reject: %s corr with %s (%s) = %.3f > %.3f",
                        factor_id, s, source, c, max_corr,
                    )
                    break
        if not correlated:
            selected.append(factor_id)
            n_new += 1

    return [s for s in selected if s not in existing_set]


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3: Migration
# ──────────────────────────────────────────────────────────────────────────────


def migrate_factors(
    source_db: RegistryDatabase,
    target_db: RegistryDatabase,
    factor_ids: list[str],
    target_status: str = "active",
    scores: pd.DataFrame | None = None,
) -> dict[str, int]:
    """Copy factors, metrics, and config snapshots from source to target.

    If ``scores`` is provided, ``final_score`` is stored in each
    factor's ``parameters["promote_score"]`` for display in ``list``.

    Returns dict with counts: {"factors", "metrics", "configs"}.
    """
    import json

    from nautilus_quants.alpha.registry.models import AnalysisMetrics, FactorRecord
    from nautilus_quants.alpha.registry.repository import FactorRepository, _now_iso

    src_repo = FactorRepository(source_db)
    tgt_repo = FactorRepository(target_db)

    counts = {"factors": 0, "metrics": 0, "configs": 0}

    for fid in factor_ids:
        # 1. Copy factor record
        factor = src_repo.get_factor(fid)
        if factor is None:
            logger.warning("Factor not found in source: %s", fid)
            continue

        # Clone with target status + inject score
        params = dict(factor.parameters) if factor.parameters else {}
        if scores is not None and fid in scores.index:
            params["promote_score"] = round(float(scores.loc[fid, "final_score"]), 4)

        promoted = FactorRecord(
            factor_id=factor.factor_id,
            expression=factor.expression,
            prototype=factor.prototype,
            description=factor.description,
            source=factor.source,
            status=target_status,
            tags=factor.tags,
            parameters=params,
            variables=factor.variables,
        )
        result = tgt_repo.upsert_factor(promoted)
        counts["factors"] += 1

        # Always force status + parameters (upsert may not update these)
        try:
            import json as _json

            tgt_repo._db.execute(
                "UPDATE factors SET status = ?, parameters = ?, updated_at = ? "
                "WHERE factor_id = ?",
                [target_status, _json.dumps(params), _now_iso(), fid],
            )
        except Exception:
            pass

        # 2. Copy metrics
        metrics = src_repo.get_metrics(fid)
        if metrics:
            tgt_repo.save_metrics(metrics)
            counts["metrics"] += len(metrics)

        # 3. Copy referenced config snapshots
        config_ids_seen: set[str] = set()
        for m in metrics:
            for cfg_id in [m.factor_config_id, m.analysis_config_id]:
                if cfg_id and cfg_id not in config_ids_seen:
                    config_ids_seen.add(cfg_id)
                    snap = src_repo.get_config_snapshot(cfg_id)
                    if snap is not None:
                        tgt_repo.save_config_snapshot(
                            snap.config_json,
                            snap.type,
                            config_name=snap.config_name,
                            file_path=snap.file_path,
                        )
                        counts["configs"] += 1

    return counts
