# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Regime analysis matplotlib charts.

Generates comparison visualizations for Jump Model vs EMA regime detection:
  1. Regime timeline — BTC price with colored backgrounds
  2. Per-factor regime ICIR grouped bar chart
  3. L/S equity curves — equal-weight vs regime-adjusted
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from nautilus_quants.alpha.regime.regime_ic_analysis import (
        ComparativeRegimeReport,
    )

logger = logging.getLogger(__name__)

_REGIME_COLORS = {
    "bull": "#2ca02c",
    "bear": "#d62728",
    "neutral": "#cccccc",
}
_BG_ALPHA = 0.15


def _add_regime_background(
    ax: plt.Axes,
    regime: pd.Series,
    skip_neutral: bool = True,
) -> None:
    """Add colored background spans for regime periods."""
    if regime.empty:
        return
    prev_label = regime.iloc[0]
    span_start = regime.index[0]

    for i in range(1, len(regime)):
        label = regime.iloc[i]
        if label != prev_label:
            if not (skip_neutral and prev_label == "neutral"):
                color = _REGIME_COLORS.get(prev_label)
                if color:
                    ax.axvspan(
                        span_start, regime.index[i - 1],
                        color=color, alpha=_BG_ALPHA, linewidth=0,
                    )
            span_start = regime.index[i]
            prev_label = label

    if not (skip_neutral and prev_label == "neutral"):
        color = _REGIME_COLORS.get(prev_label)
        if color:
            ax.axvspan(
                span_start, regime.index[-1],
                color=color, alpha=_BG_ALPHA, linewidth=0,
            )


def _regime_stats_label(regime: pd.Series, name: str) -> str:
    counts = regime.value_counts(normalize=True)
    switches = int((regime != regime.shift()).sum())
    parts = [f"{name}:"]
    for label in ["bear", "neutral", "bull"]:
        if label in counts.index:
            parts.append(f"{label}={counts[label]:.0%}")
    parts.append(f"sw={switches}")
    return " ".join(parts)


def chart_regime_timeline(
    report: ComparativeRegimeReport,
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Regime timeline: BTC price with colored backgrounds (JM vs EMA)."""
    fig, (ax_jm, ax_ema) = plt.subplots(
        2, 1, figsize=(18, 9), sharex=True,
    )

    btc = report.btc_close

    _add_regime_background(ax_jm, report.jm_regime, skip_neutral=True)
    ax_jm.plot(btc.index, btc.values, color="black", linewidth=0.8)
    ax_jm.set_ylabel("BTC Close")
    ax_jm.set_title(
        _regime_stats_label(report.jm_regime, "Jump Model"),
        fontsize=11, loc="left",
    )
    ax_jm.grid(True, alpha=0.3)

    _add_regime_background(ax_ema, report.ema_regime, skip_neutral=True)
    ax_ema.plot(btc.index, btc.values, color="black", linewidth=0.8)
    ax_ema.set_ylabel("BTC Close")
    ax_ema.set_title(
        _regime_stats_label(report.ema_regime, "EMA"),
        fontsize=11, loc="left",
    )
    ax_ema.grid(True, alpha=0.3)

    fig.suptitle(
        "Regime Detection: Jump Model vs EMA",
        fontsize=13, fontweight="bold",
    )
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved regime timeline: %s", output_path)


def chart_regime_icir_bars(
    report: ComparativeRegimeReport,
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Grouped bar chart: per-factor regime ICIR (JM vs EMA)."""
    names = report.factor_names
    n = len(names)

    jm_regimes = ["bear", "neutral", "bull"]
    ema_regimes = ["bear", "bull"]
    jm_colors = [_REGIME_COLORS[r] for r in jm_regimes]
    ema_colors = [_REGIME_COLORS[r] for r in ema_regimes]

    group_width = 0.7
    n_bars = len(jm_regimes) + len(ema_regimes) + 1
    bar_w = group_width / n_bars
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(max(14, n * 1.2), 7))

    for j, regime in enumerate(jm_regimes):
        vals = [
            next(
                (r.icir(regime) for r in report.jm_results
                 if r.factor_name == name), 0.0,
            )
            for name in names
        ]
        offset = (j - (n_bars - 1) / 2) * bar_w
        ax.bar(
            x + offset, vals, bar_w * 0.9,
            color=jm_colors[j], alpha=0.85,
            label=f"JM {regime}",
        )

    for j, regime in enumerate(ema_regimes):
        vals = [
            next(
                (r.icir(regime) for r in report.ema_results
                 if r.factor_name == name), 0.0,
            )
            for name in names
        ]
        offset = (j + len(jm_regimes) + 1 - (n_bars - 1) / 2) * bar_w
        ax.bar(
            x + offset, vals, bar_w * 0.9,
            color=ema_colors[j], alpha=0.5, hatch="//",
            label=f"EMA {regime}",
        )

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(0.2, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axhline(-0.2, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("ICIR")
    ax.set_title(
        "Per-Factor Regime ICIR: Jump Model (solid) vs EMA (hatched)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved regime ICIR bars: %s", output_path)


def _compute_ls_returns(
    factor_dfs: dict[str, pd.DataFrame],
    fwd_returns: pd.DataFrame,
    weights: dict[str, float],
    quintile_frac: float = 0.2,
    min_obs: int = 20,
) -> pd.Series:
    """Compute L/S spread returns using signed weighted composite."""
    from nautilus_quants.alpha.regime.regime_ic_analysis import _cs_normalize

    factor_names = list(weights.keys())
    norm_dfs = {n: _cs_normalize(factor_dfs[n]) for n in factor_names}
    composite = sum(weights[n] * norm_dfs[n] for n in factor_names)

    common_dates = composite.index.intersection(fwd_returns.index)
    ls_returns = {}

    for dt in common_dates:
        comp = composite.loc[dt].dropna()
        ret = fwd_returns.loc[dt].reindex(comp.index).dropna()
        common = comp.index.intersection(ret.index)
        if len(common) < min_obs:
            continue
        comp_vals = comp[common]
        ret_vals = ret[common]
        k = max(1, int(len(common) * quintile_frac))
        top_idx = comp_vals.nlargest(k).index
        bot_idx = comp_vals.nsmallest(k).index
        ls_returns[dt] = float(ret_vals[top_idx].mean() - ret_vals[bot_idx].mean())

    return pd.Series(ls_returns, name="ls_return")


def chart_ls_equity_curves(
    report: ComparativeRegimeReport,
    factor_dfs: dict[str, pd.DataFrame],
    fwd_returns: pd.DataFrame,
    output_path: Path,
    equal_weights: dict[str, float] | None = None,
    dpi: int = 150,
) -> None:
    """L/S equity curves: equal-weight vs JM vs EMA regime-adjusted."""
    names = report.factor_names
    if equal_weights is None:
        equal_weights = {n: 1.0 / len(names) for n in names}

    ls_equal = _compute_ls_returns(factor_dfs, fwd_returns, equal_weights)

    def _regime_ls(
        regime: pd.Series,
        weight_map: dict[str, dict[str, float]],
    ) -> pd.Series:
        ls_parts: dict[pd.Timestamp, float] = {}
        common_dates = fwd_returns.index.intersection(regime.index)
        for dt in common_dates:
            label = regime.loc[dt]
            w = weight_map.get(label, equal_weights)
            comp_vals: dict[str, float] = {}
            for n in names:
                if n not in factor_dfs or dt not in factor_dfs[n].index:
                    continue
                row = factor_dfs[n].loc[dt].dropna()
                mean = row.mean()
                std = row.std()
                if std > 0:
                    for inst in row.index:
                        comp_vals.setdefault(inst, 0.0)
                        comp_vals[inst] += (
                            w.get(n, 0) * (row[inst] - mean) / std
                        )

            if dt not in fwd_returns.index:
                continue
            ret = fwd_returns.loc[dt].dropna()
            common_insts = set(comp_vals.keys()) & set(ret.index)
            if len(common_insts) < 20:
                continue

            comp_s = pd.Series(
                {i: comp_vals[i] for i in common_insts},
            )
            ret_s = ret.reindex(comp_s.index)

            k = max(1, int(len(common_insts) * 0.2))
            top = comp_s.nlargest(k).index
            bot = comp_s.nsmallest(k).index
            ls_parts[dt] = float(ret_s[top].mean() - ret_s[bot].mean())

        return pd.Series(ls_parts, name="ls_return")

    ls_jm = _regime_ls(report.jm_regime, report.jm_weights)
    ls_ema = _regime_ls(report.ema_regime, report.ema_weights)

    common_idx = ls_equal.index.intersection(
        ls_jm.index,
    ).intersection(ls_ema.index).sort_values()

    cum_equal = (1 + ls_equal.reindex(common_idx).fillna(0)).cumprod()
    cum_jm = (1 + ls_jm.reindex(common_idx).fillna(0)).cumprod()
    cum_ema = (1 + ls_ema.reindex(common_idx).fillna(0)).cumprod()

    def _sharpe(s: pd.Series, bars_per_day: float = 6.0) -> float:
        if s.std() == 0:
            return 0.0
        return float(s.mean() / s.std() * np.sqrt(365 * bars_per_day))

    sh_equal = _sharpe(ls_equal.reindex(common_idx).fillna(0))
    sh_jm = _sharpe(ls_jm.reindex(common_idx).fillna(0))
    sh_ema = _sharpe(ls_ema.reindex(common_idx).fillna(0))

    fig, ax = plt.subplots(figsize=(18, 8))

    _add_regime_background(ax, report.jm_regime, skip_neutral=True)

    ax.plot(
        cum_equal.index, cum_equal.values,
        color="#1f77b4", linewidth=1.5,
        label=f"Equal-weight L/S (Sharpe={sh_equal:.2f})",
    )
    ax.plot(
        cum_jm.index, cum_jm.values,
        color="#d62728", linewidth=1.5,
        label=f"JM regime-adjusted L/S (Sharpe={sh_jm:.2f})",
    )
    ax.plot(
        cum_ema.index, cum_ema.values,
        color="#ff7f0e", linewidth=1.5, linestyle="--",
        label=f"EMA regime-adjusted L/S (Sharpe={sh_ema:.2f})",
    )

    ax.set_ylabel("Cumulative Return")
    ax.set_title(
        "Long/Short Equity Curves: Equal vs Regime-Adjusted",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved L/S equity curves: %s", output_path)
