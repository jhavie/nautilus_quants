"""Analysis report generator.

Generates alphalens tearsheet charts, IC/ICIR summary, and extended
factor metrics (signal quality + portfolio performance).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for chart generation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.optimize import curve_fit

if TYPE_CHECKING:
    from nautilus_quants.alpha.analysis.config import AlphaAnalysisConfig

logger = logging.getLogger(__name__)


def _render_dataframe_as_table(df: pd.DataFrame, title: str = "") -> None:
    """Render a DataFrame as a matplotlib table on the current figure."""
    fig = plt.gcf()
    fig.clf()
    ax = fig.add_subplot(111)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    cell_text = df.map(
        lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)
    ).values
    table = ax.table(
        cellText=cell_text,
        colLabels=df.columns.tolist(),
        rowLabels=df.index.tolist(),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)


def _infer_bars_per_day(index: pd.Index) -> int:
    """Infer the number of observations per day from a DatetimeIndex.

    Falls back to 1 (daily) when frequency cannot be determined.
    """
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return 1
    freq = index.freq or pd.infer_freq(index[:min(len(index), 100)])
    if freq is not None:
        td = pd.Timedelta(pd.tseries.frequencies.to_offset(freq))  # type: ignore[arg-type]
        if td.total_seconds() > 0:
            return max(int(pd.Timedelta(days=1) / td), 1)
    # Fallback: median diff
    median_diff = pd.Series(index).diff().dropna().median()
    if pd.notna(median_diff) and median_diff.total_seconds() > 0:
        return max(int(pd.Timedelta(days=1) / median_diff), 1)
    return 1


def _newey_west_tstat(series: pd.Series) -> tuple[float, float, int]:
    """Compute Newey-West adjusted t-statistic for H0: mean=0.

    Uses Bartlett kernel with frequency-aware lag selection:
    - Base lag from Newey-West (1994): floor(4 * (T/100)^(2/9))
    - Minimum lag: bars_per_day * 10  (covers ~10 days of autocorrelation)

    For sub-daily data the classic formula severely underestimates the
    required bandwidth, producing inflated t-statistics.

    Returns:
        (t_stat, p_value, n_eff) tuple.  (NaN, NaN, 0) when data is
        insufficient.  *n_eff* is the effective sample size after
        accounting for serial correlation: ``N * gamma_0 / nw_var``.
    """
    x = series.dropna().values
    n = len(x)
    if n < 2:
        return np.nan, np.nan, 0

    x_bar = x.mean()
    # Automatic bandwidth selection (Newey-West 1994)
    nw_lag = max(int(np.floor(4 * (n / 100) ** (2 / 9))), 1)
    # Frequency-aware minimum: cover at least 10 days of autocorrelation
    bars_per_day = _infer_bars_per_day(series.index)
    freq_lag = bars_per_day * 10
    max_lag = min(max(nw_lag, freq_lag), n // 3)

    # Newey-West HAC variance estimate
    gamma_0 = np.mean((x - x_bar) ** 2)
    nw_var = gamma_0
    for j in range(1, max_lag + 1):
        weight = 1 - j / (max_lag + 1)  # Bartlett kernel
        gamma_j = np.mean((x[j:] - x_bar) * (x[:-j] - x_bar))
        nw_var += 2 * weight * gamma_j

    # Effective sample size: N / VIF where VIF = nw_var / gamma_0
    if nw_var > 1e-15:
        n_eff = min(max(int(round(n * gamma_0 / nw_var)), 1), n)
    else:
        n_eff = n

    se = np.sqrt(nw_var / n)
    if se < 1e-15:
        return np.nan, np.nan, 0

    t_stat = x_bar / se
    # Two-sided p-value using t-distribution with n_eff-1 degrees of freedom;
    # more conservative than normal for small n_eff, avoids sf underflow.
    df = max(n_eff - 1, 1)
    p_value = float(2 * scipy_stats.t.sf(abs(t_stat), df))
    return t_stat, p_value, n_eff


def compute_ic_summary(ic_df: pd.DataFrame) -> pd.DataFrame:
    """Compute IC statistics with proper NaN handling.

    Unlike alphalens ``plot_information_table`` which propagates NaN
    through ``scipy.stats.ttest_1samp``, this function drops NaN values
    **per column** before computing statistics.  This avoids cross-column
    contamination where long-period NaN tails truncate short-period samples.

    Includes both raw t-stat (assumes independence) and Newey-West
    adjusted t-stat (accounts for serial correlation in high-frequency
    IC series).

    Returns a DataFrame indexed by period with columns:
        IC Mean, IC Std., Risk-Adjusted IC, t-stat(IC), p-value(IC),
        t-stat(NW), p-value(NW), N_eff, IC Skew, IC Kurtosis, N
    """
    table = pd.DataFrame()
    for col in ic_df.columns:
        col_clean = ic_df[col].dropna()
        n = len(col_clean)
        ic_mean = col_clean.mean() if n > 0 else np.nan
        ic_std = col_clean.std() if n > 1 else np.nan
        table.loc[col, "IC Mean"] = ic_mean
        table.loc[col, "IC Std."] = ic_std
        table.loc[col, "Risk-Adjusted IC"] = (
            ic_mean / ic_std if ic_std is not None and ic_std > 1e-15 else np.nan
        )
        if n >= 2:
            t_val, p_val = scipy_stats.ttest_1samp(col_clean, 0)
            table.loc[col, "t-stat(IC)"] = t_val
            table.loc[col, "p-value(IC)"] = p_val
        else:
            table.loc[col, "t-stat(IC)"] = np.nan
            table.loc[col, "p-value(IC)"] = np.nan
        nw_t, nw_p, n_eff = _newey_west_tstat(col_clean)
        table.loc[col, "t-stat(NW)"] = nw_t
        table.loc[col, "p-value(NW)"] = nw_p
        table.loc[col, "N_eff"] = n_eff
        table.loc[col, "IC Skew"] = (
            float(scipy_stats.skew(col_clean)) if n >= 3 else np.nan
        )
        table.loc[col, "IC Kurtosis"] = (
            float(scipy_stats.kurtosis(col_clean)) if n >= 4 else np.nan
        )
        table.loc[col, "N"] = n
    nan_count = ic_df.isna().sum()
    if nan_count.any():
        table["NaN Count"] = nan_count
    return table


# ── Factor Signal Quality Metrics ──


def compute_win_rate(ic_df: pd.DataFrame) -> pd.Series:
    """Fraction of periods with positive IC per forward-return period.

    Args:
        ic_df: IC DataFrame (index=date, columns=period labels)

    Returns:
        Series[period] with values in [0, 1].
    """
    result = {}
    for col in ic_df.columns:
        clean = ic_df[col].dropna()
        result[col] = (clean > 0).sum() / max(len(clean), 1)
    return pd.Series(result)


def compute_coverage(
    factor_data: pd.DataFrame,
    total_timestamps: int,
    total_assets: int,
) -> float:
    """Ratio of valid factor observations to total possible.

    factor_data has already been cleaned by alphalens (NaN rows dropped),
    so len(factor_data) is the actual non-NaN observation count.

    Args:
        factor_data: alphalens factor_data (MultiIndex[date, asset])
        total_timestamps: Total time steps in pricing data
        total_assets: Total instruments in pricing data

    Returns:
        Float in [0, 1].
    """
    possible = max(total_timestamps * total_assets, 1)
    return len(factor_data) / possible


def compute_ic_linearity(ic_df: pd.DataFrame) -> pd.Series:
    """R² of linear fit to cumulative IC (IC stability).

    Measures how consistently the factor accumulates predictive power.
    R² ≈ 1.0 = steady IC, no regime breaks.  R² ≈ 0.0 = erratic.

    Args:
        ic_df: IC DataFrame (index=date, columns=period labels)

    Returns:
        Series[period] with R² values in [0, 1].
    """
    result = {}
    for col in ic_df.columns:
        clean = ic_df[col].dropna()
        n = len(clean)
        if n < 3:
            result[col] = np.nan
            continue
        cum = clean.cumsum().values
        t = np.arange(n, dtype=float)
        ss_res = np.sum((cum - np.polyval(np.polyfit(t, cum, 1), t)) ** 2)
        ss_tot = np.sum((cum - cum.mean()) ** 2)
        result[col] = 1 - ss_res / ss_tot if ss_tot > 1e-15 else np.nan
    return pd.Series(result)


def compute_ic_ar1(ic_df: pd.DataFrame) -> pd.Series:
    """Lag-1 autocorrelation of IC series per period.

    Scalar summary of IC persistence.  AR(1) > 0.3 = persistent signal,
    ≈ 0 = no memory, < 0 = reverting (unusual).

    Args:
        ic_df: IC DataFrame (index=date, columns=period labels)

    Returns:
        Series[period] with autocorrelation values.
    """
    result = {}
    for col in ic_df.columns:
        clean = ic_df[col].dropna()
        result[col] = clean.autocorr(lag=1) if len(clean) >= 3 else np.nan
    return pd.Series(result)


def compute_ic_half_life(ic_df: pd.DataFrame) -> pd.Series:
    """Half-life of IC autocorrelation decay in bars.

    For each period column, computes the autocorrelation at increasing
    lags and fits an exponential decay ``a * exp(-t / tau)``.
    Half-life = ``tau * ln(2)``.

    Args:
        ic_df: IC DataFrame (index=date, columns=period labels)

    Returns:
        Series[period] in bars.  NaN if fit fails or insufficient data.
    """
    result = {}
    for col in ic_df.columns:
        result[col] = _fit_ic_half_life(ic_df[col].dropna())
    return pd.Series(result)


def _fit_ic_half_life(ic_series: pd.Series) -> float:
    """Fit exponential decay to IC autocorrelation."""
    n = len(ic_series)
    if n < 20:
        return np.nan

    max_lag = min(n // 4, 100)
    lags = np.arange(1, max_lag + 1, dtype=float)
    autocorrs = np.array([ic_series.autocorr(lag=int(k)) for k in lags])

    # Drop NaN autocorrelations
    valid = np.isfinite(autocorrs)
    if valid.sum() < 3:
        return np.nan

    lags_clean = lags[valid]
    ac_clean = autocorrs[valid]

    # Only fit if first autocorrelation is positive (decaying signal)
    if ac_clean[0] <= 0:
        return np.nan

    def _exp_decay(t: np.ndarray, a: float, tau: float) -> np.ndarray:
        return a * np.exp(-t / tau)

    try:
        popt, _ = curve_fit(
            _exp_decay, lags_clean, ac_clean,
            p0=[ac_clean[0], 10.0],
            bounds=([0, 0.1], [2.0, max_lag * 5]),
            maxfev=2000,
        )
        tau = popt[1]
        return tau * np.log(2)
    except (RuntimeError, ValueError):
        return np.nan


def compute_monotonicity(factor_data: pd.DataFrame) -> pd.Series:
    """Spearman correlation between quantile rank and mean return.

    A score of +1 means higher quantiles have strictly higher returns
    (positive factor).  -1 means perfect reversal factor.

    Args:
        factor_data: alphalens factor_data (MultiIndex[date, asset])

    Returns:
        Series[period] with values in [-1, 1].
    """
    import alphalens.performance as perf

    mean_ret, _ = perf.mean_return_by_quantile(factor_data, by_date=False)
    period_cols = [c for c in mean_ret.columns if c not in ("factor", "factor_quantile")]
    result = {}
    for col in period_cols:
        returns = mean_ret[col].values
        ranks = np.arange(1, len(returns) + 1)
        if len(ranks) < 3:
            result[col] = np.nan
        else:
            rho, _ = scipy_stats.spearmanr(ranks, returns)
            result[col] = rho
    return pd.Series(result)




# ── Result Dataclasses ──


@dataclass(frozen=True)
class FactorMetricsResult:
    """Result container for factor signal quality metrics."""

    win_rate: pd.Series        # period → float [0, 1]
    coverage: float            # [0, 1]
    ic_half_life: pd.Series    # period → float (bars), NaN if fit fails
    monotonicity: pd.Series    # period → float [-1, 1]
    ic_linearity: pd.Series    # period → float [0, 1] (R² of cumulative IC)
    ic_ar1: pd.Series          # period → float (lag-1 autocorrelation)



def compute_all_factor_metrics(
    factor_data: pd.DataFrame,
    ic_df: pd.DataFrame,
    total_timestamps: int,
    total_assets: int,
) -> FactorMetricsResult:
    """Compute all factor signal quality metrics.

    Args:
        factor_data: alphalens factor_data (MultiIndex[date, asset])
        ic_df: IC DataFrame (index=date, columns=period labels)
        total_timestamps: Total time steps in pricing data
        total_assets: Total instruments in pricing data
    """
    return FactorMetricsResult(
        win_rate=compute_win_rate(ic_df),
        coverage=compute_coverage(factor_data, total_timestamps, total_assets),
        ic_half_life=compute_ic_half_life(ic_df),
        monotonicity=compute_monotonicity(factor_data),
        ic_linearity=compute_ic_linearity(ic_df),
        ic_ar1=compute_ic_ar1(ic_df),
    )



def build_analysis_metrics(
    run_id: str,
    factor_id: str,
    timeframe: str,
    ic_summary: pd.DataFrame,
    metrics_result: FactorMetricsResult | None = None,
    factor_config_id: str = "",
    analysis_config_id: str = "",
    output_dir: str = "",
) -> list:
    """Build AnalysisMetrics from IC summary and FactorMetricsResult.

    Returns a list of AnalysisMetrics (one per period).
    Imported lazily to avoid circular imports.
    """
    from nautilus_quants.alpha.registry.models import AnalysisMetrics

    now = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    result = []

    for period_label in ic_summary.index:
        row = ic_summary.loc[period_label]

        # Signal quality metrics (per-period from FactorMetricsResult)
        win_rate_val = None
        mono_val = None
        hl_val = None
        lin_val = None
        ar1_val = None
        cov_val = None
        if metrics_result is not None:
            if period_label in metrics_result.win_rate.index:
                win_rate_val = _safe_float(
                    metrics_result.win_rate[period_label],
                )
            if period_label in metrics_result.monotonicity.index:
                mono_val = _safe_float(
                    metrics_result.monotonicity[period_label],
                )
            if period_label in metrics_result.ic_half_life.index:
                hl_val = _safe_float(
                    metrics_result.ic_half_life[period_label],
                )
            if period_label in metrics_result.ic_linearity.index:
                lin_val = _safe_float(
                    metrics_result.ic_linearity[period_label],
                )
            if period_label in metrics_result.ic_ar1.index:
                ar1_val = _safe_float(
                    metrics_result.ic_ar1[period_label],
                )
            cov_val = _safe_float(metrics_result.coverage)

        result.append(AnalysisMetrics(
            run_id=run_id,
            factor_id=factor_id,
            period=str(period_label),
            ic_mean=_safe_float(row.get("IC Mean")),
            ic_std=_safe_float(row.get("IC Std.")),
            icir=_safe_float(row.get("Risk-Adjusted IC")),
            t_stat_ic=_safe_float(row.get("t-stat(IC)")),
            p_value_ic=_safe_float(row.get("p-value(IC)")),
            t_stat_nw=_safe_float(row.get("t-stat(NW)")),
            p_value_nw=_safe_float(row.get("p-value(NW)")),
            n_eff=_safe_int(row.get("N_eff")),
            ic_skew=_safe_float(row.get("IC Skew")),
            ic_kurtosis=_safe_float(row.get("IC Kurtosis")),
            n_samples=_safe_int(row.get("N")),
            win_rate=win_rate_val,
            monotonicity=mono_val,
            ic_half_life=hl_val,
            ic_linearity=lin_val,
            ic_ar1=ar1_val,
            coverage=cov_val,
            factor_config_id=factor_config_id,
            analysis_config_id=analysis_config_id,
            output_dir=output_dir,
            timeframe=timeframe,
            created_at=now,
        ))

    return result


def _safe_float(val: Any) -> float | None:
    """Convert to float, returning None for NaN/None."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if np.isnan(f) else f
    except (ValueError, TypeError):
        return None


def _safe_int(val: Any) -> int | None:
    """Convert to int, returning None for NaN/None."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if np.isnan(f) else int(f)
    except (ValueError, TypeError):
        return None


def _chart_quantile_returns_bar(factor_data: pd.DataFrame, period: str, **kwargs: Any) -> None:
    import alphalens.performance as perf
    import alphalens.plotting as plotting

    mean_ret, _ = perf.mean_return_by_quantile(factor_data, by_date=False)
    plotting.plot_quantile_returns_bar(mean_ret)


def _chart_quantile_returns_violin(factor_data: pd.DataFrame, period: str, **kwargs: Any) -> None:
    import alphalens.performance as perf
    import alphalens.plotting as plotting

    mean_ret, _ = perf.mean_return_by_quantile(factor_data, by_date=True)
    plotting.plot_quantile_returns_violin(mean_ret)


def _chart_cumulative_returns(factor_data: pd.DataFrame, period: str, **kwargs: Any) -> None:
    import alphalens.performance as perf
    import alphalens.plotting as plotting

    mean_ret, _ = perf.mean_return_by_quantile(factor_data, by_date=True)
    plotting.plot_cumulative_returns_by_quantile(mean_ret, period=period)


def _chart_quantile_spread(factor_data: pd.DataFrame, period: str, **kwargs: Any) -> None:
    import alphalens.performance as perf
    import alphalens.plotting as plotting

    mean_ret, std_err = perf.mean_return_by_quantile(factor_data, by_date=True)
    spread, spread_std = perf.compute_mean_returns_spread(
        mean_ret, upper_quant=factor_data["factor_quantile"].max(),
        lower_quant=factor_data["factor_quantile"].min(),
        std_err=std_err,
    )
    plotting.plot_mean_quantile_returns_spread_time_series(spread, std_err=spread_std)


def _chart_ic_time_series(factor_data: pd.DataFrame, period: str, **kwargs: Any) -> None:
    import alphalens.performance as perf
    import alphalens.plotting as plotting

    ic = perf.factor_information_coefficient(factor_data)
    plotting.plot_ic_ts(ic)


def _chart_ic_histogram(factor_data: pd.DataFrame, period: str, **kwargs: Any) -> None:
    import alphalens.performance as perf

    ic = perf.factor_information_coefficient(factor_data)

    num_plots = len(ic.columns)
    v_spaces = ((num_plots - 1) // 3) + 1
    fig, axes = plt.subplots(v_spaces, 3, figsize=(18, v_spaces * 6))
    axes = np.atleast_1d(axes).flatten()

    for ax, (period_num, col) in zip(axes, ic.items()):
        data = col.fillna(0.0)
        ax.hist(data, bins=50, edgecolor="black", alpha=0.7)
        ax.set(title=f"{period_num} Period IC", xlabel="IC")
        ax.set_xlim([-1, 1])
        ax.text(
            0.05, 0.95,
            f"Mean {col.mean():.3f} \n Std. {col.std():.3f}",
            fontsize=16,
            bbox={"facecolor": "white", "alpha": 1, "pad": 5},
            transform=ax.transAxes, verticalalignment="top",
        )
        ax.axvline(col.mean(), color="w", linestyle="dashed", linewidth=2)

    for ax in axes[num_plots:]:
        ax.set_visible(False)


def _chart_ic_qq(factor_data: pd.DataFrame, period: str, **kwargs: Any) -> None:
    import alphalens.performance as perf
    import alphalens.plotting as plotting

    ic = perf.factor_information_coefficient(factor_data)
    plotting.plot_ic_qq(ic)


def _chart_turnover(factor_data: pd.DataFrame, period: str, **kwargs: Any) -> None:
    import alphalens.performance as perf
    import alphalens.plotting as plotting

    quantile_factor = factor_data["factor_quantile"]
    quantiles = quantile_factor.sort_values().unique()
    turnover = pd.DataFrame(
        {q: perf.quantile_turnover(quantile_factor, q) for q in quantiles}
    )
    plotting.plot_top_bottom_quantile_turnover(turnover)


def _chart_factor_rank_autocorrelation(factor_data: pd.DataFrame, period: str, **kwargs: Any) -> None:
    import alphalens.performance as perf
    import alphalens.plotting as plotting

    autocorr = perf.factor_rank_autocorrelation(factor_data)
    plotting.plot_factor_rank_auto_correlation(autocorr)


# ── New charts: alphalens full API coverage ──


def _chart_monthly_ic_heatmap(factor_data: pd.DataFrame, period: str, **kwargs: Any) -> None:
    import alphalens.performance as perf
    import alphalens.plotting as plotting

    mean_monthly_ic = perf.mean_information_coefficient(
        factor_data, by_time="M",
    )
    plotting.plot_monthly_ic_heatmap(mean_monthly_ic)


def _chart_cumulative_returns_long_short(factor_data: pd.DataFrame, period: str, **kwargs: Any) -> None:
    import alphalens.performance as perf
    import alphalens.plotting as plotting

    factor_ret = perf.factor_returns(factor_data, demeaned=True)
    plotting.plot_cumulative_returns(factor_ret, period=period)


def _chart_returns_table(factor_data: pd.DataFrame, period: str, **kwargs: Any) -> None:
    import alphalens.performance as perf
    import alphalens.plotting as plotting

    alpha_beta = perf.factor_alpha_beta(factor_data)
    mean_ret, _ = perf.mean_return_by_quantile(factor_data, by_date=False)
    mean_ret_by_date, std_err = perf.mean_return_by_quantile(factor_data, by_date=True)
    spread, _ = perf.compute_mean_returns_spread(
        mean_ret_by_date,
        upper_quant=factor_data["factor_quantile"].max(),
        lower_quant=factor_data["factor_quantile"].min(),
        std_err=std_err,
    )
    table_df = plotting.plot_returns_table(alpha_beta, mean_ret, spread, return_df=True)
    _render_dataframe_as_table(table_df, title="Returns Analysis")


def _chart_event_study(factor_data: pd.DataFrame, period: str, **kwargs: Any) -> None:
    import alphalens.performance as perf
    import alphalens.plotting as plotting

    pricing = kwargs.get("pricing")
    if pricing is None:
        logger.warning("event_study requires pricing data, skipping")
        return

    avg_cum_ret = perf.average_cumulative_return_by_quantile(
        factor_data, returns=pricing,
        periods_before=10, periods_after=15,
    )
    plotting.plot_quantile_average_cumulative_return(
        avg_cum_ret, by_quantile=True, std_bar=True,
    )


def _chart_quantile_statistics_table(factor_data: pd.DataFrame, period: str, **kwargs: Any) -> None:
    import alphalens.plotting as plotting

    table_df = plotting.plot_quantile_statistics_table(factor_data, return_df=True)
    _render_dataframe_as_table(table_df, title="Quantile Statistics")


def _chart_events_distribution(factor_data: pd.DataFrame, period: str, **kwargs: Any) -> None:
    import alphalens.plotting as plotting

    plotting.plot_events_distribution(factor_data["factor"], num_bars=50)


def _chart_turnover_table(factor_data: pd.DataFrame, period: str, **kwargs: Any) -> None:
    import alphalens.performance as perf
    import alphalens.plotting as plotting

    period_cols = [c for c in factor_data.columns if c not in ("factor", "factor_quantile")]
    quantile_factor = factor_data["factor_quantile"]
    quantiles = quantile_factor.sort_values().unique()

    # plot_turnover_table expects {period: {quantile: Series}} and {period: Series}
    qt: dict[str, dict] = {}
    ac: dict[str, pd.Series] = {}
    for p in period_cols:
        qt[p] = {q: perf.quantile_turnover(quantile_factor, q) for q in quantiles}
        # Derive lag from period label (e.g., "4h" with 1h bars → lag=4)
        try:
            bar_freq = factor_data.index.levels[0].freq or "1h"
            period_bars = int(pd.Timedelta(p) / pd.Timedelta(bar_freq))
            period_bars = max(period_bars, 1)
        except (ValueError, TypeError):
            period_bars = 1
        ac[p] = perf.factor_rank_autocorrelation(factor_data, period=period_bars)

    turnover_df, auto_corr_df = plotting.plot_turnover_table(ac, qt, return_df=True)
    combined = pd.concat([turnover_df, auto_corr_df])
    _render_dataframe_as_table(combined, title="Turnover Analysis")


CHART_REGISTRY: dict[str, Callable[..., None]] = {
    # Returns
    "quantile_returns_bar": _chart_quantile_returns_bar,
    "quantile_returns_violin": _chart_quantile_returns_violin,
    "cumulative_returns": _chart_cumulative_returns,
    "cumulative_returns_long_short": _chart_cumulative_returns_long_short,
    "quantile_spread": _chart_quantile_spread,
    "returns_table": _chart_returns_table,
    # IC
    "ic_time_series": _chart_ic_time_series,
    "ic_histogram": _chart_ic_histogram,
    "ic_qq": _chart_ic_qq,
    "monthly_ic_heatmap": _chart_monthly_ic_heatmap,
    # Turnover
    "turnover": _chart_turnover,
    "turnover_table": _chart_turnover_table,
    "factor_rank_autocorrelation": _chart_factor_rank_autocorrelation,
    # Event study & distribution
    "event_study": _chart_event_study,
    "events_distribution": _chart_events_distribution,
    "quantile_statistics_table": _chart_quantile_statistics_table,
}


class AnalysisReportGenerator:
    """Generate analysis reports for evaluated factors."""

    def __init__(self, config: AlphaAnalysisConfig) -> None:
        self._config = config

    def generate_factor_charts(
        self,
        factor_name: str,
        factor_data: pd.DataFrame,
        output_dir: Path,
        pricing: pd.DataFrame | None = None,
    ) -> list[Path]:
        """Generate configured charts for a single factor.

        Args:
            factor_name: Name of the factor
            factor_data: alphalens factor_data DataFrame
            output_dir: Run output directory
            pricing: Price DataFrame for charts that need it (e.g. event_study)

        Returns:
            List of generated chart file paths
        """
        factor_dir = output_dir / factor_name
        factor_dir.mkdir(parents=True, exist_ok=True)

        # Use first period column as default (alphalens uses Timedelta-style names)
        period_cols = [c for c in factor_data.columns if c not in ("factor", "factor_quantile")]
        period = period_cols[0] if period_cols else "1h"

        generated: list[Path] = []

        for chart_name in self._config.charts:
            chart_func = CHART_REGISTRY.get(chart_name)
            if chart_func is None:
                logger.warning(f"Unknown chart type: {chart_name}")
                continue

            try:
                plt.figure(figsize=(12, 6))
                chart_func(factor_data, period, pricing=pricing)
                for fmt in self._config.output_format:
                    path = factor_dir / f"{chart_name}.{fmt}"
                    plt.savefig(path, bbox_inches="tight", dpi=100)
                    generated.append(path)
                plt.close("all")
            except Exception as e:
                logger.warning(f"Failed to generate {chart_name} for {factor_name}: {e}")
                plt.close("all")

        return generated

    def generate_summary(
        self,
        ic_results: dict[str, pd.DataFrame],
        output_dir: Path,
        factor_series: dict[str, pd.Series] | None = None,
        skipped_factors: dict[str, str] | None = None,
    ) -> Path:
        """Generate IC/ICIR summary text file.

        Uses ``compute_ic_summary`` for IC Mean, IC Std, Risk-Adjusted IC
        (ICIR), raw t-stat, **Newey-West adjusted t-stat**, p-value, Skew,
        Kurtosis.  The NW t-stat corrects for serial correlation in
        high-frequency IC series.

        Args:
            ic_results: {factor_name: IC DataFrame}
            output_dir: Run output directory
            factor_series: {factor_name: Series(MultiIndex[date, asset])} for correlation
            skipped_factors: {factor_name: reason} for factors that failed analysis

        Returns:
            Path to summary file
        """
        lines = ["Factor Analysis Summary", "=" * 60, ""]

        for factor_name, ic_df in ic_results.items():
            lines.append(f"Factor: {factor_name}")
            lines.append("-" * 40)

            ic_summary = compute_ic_summary(ic_df)
            for period in ic_summary.index:
                row = ic_summary.loc[period]
                n = int(row["N"])
                n_eff = int(row["N_eff"])
                lines.append(
                    f"  Period {period}: "
                    f"IC={row['IC Mean']:.4f}, "
                    f"ICIR={row['Risk-Adjusted IC']:.4f}, "
                    f"t(NW)={row['t-stat(NW)']:.2f}, "
                    f"p(NW)={row['p-value(NW)']:.2e}, "
                    f"N={n}, N_eff={n_eff}"
                )

            lines.append("")

        # Skipped factors section
        if skipped_factors:
            lines.append("Skipped Factors (analysis failed)")
            lines.append("=" * 60)
            for factor_name, reason in skipped_factors.items():
                lines.append(f"  {factor_name}: {reason}")
            lines.append("")

        # Factor correlation matrix
        if factor_series and len(factor_series) > 1:
            corr = pd.concat(factor_series, axis=1).corr(method="spearman")
            lines.append("Factor Correlation Matrix (Spearman)")
            lines.append("=" * 60)
            lines.append(corr.to_string(float_format=lambda x: f"{x:.3f}"))
            lines.append("")

        summary_path = output_dir / "summary.txt"
        summary_path.write_text("\n".join(lines))
        return summary_path

    def print_summary_table(
        self,
        ic_results: dict[str, pd.DataFrame],
        factor_series: dict[str, pd.Series] | None = None,
    ) -> None:
        """Print IC/ICIR summary table to console.

        Uses ``compute_ic_summary`` for consistent metrics including
        Newey-West adjusted t-statistics.

        Args:
            ic_results: {factor_name: IC DataFrame}
            factor_series: {factor_name: Series(MultiIndex[date, asset])} for correlation
        """
        if not ic_results:
            return

        # Get all period columns
        all_periods = set()
        for ic_df in ic_results.values():
            all_periods.update(ic_df.columns)
        periods = sorted(all_periods, key=lambda x: pd.Timedelta(x))

        # Header
        header = f"  {'Factor':<20}"
        for p in periods:
            header += f"  IC({p})"
        header += "   ICIR"
        print(header)

        # Rows
        for factor_name, ic_df in ic_results.items():
            ic_summary = compute_ic_summary(ic_df)
            row = f"  {factor_name:<20}"
            icir_values = []
            for p in periods:
                if p in ic_summary.index:
                    row += f"  {ic_summary.loc[p, 'IC Mean']:>6.3f}"
                    icir_values.append(ic_summary.loc[p, "Risk-Adjusted IC"])
                else:
                    row += f"  {'N/A':>6}"

            avg_icir = np.mean(icir_values) if icir_values else 0.0
            row += f"  {avg_icir:>6.2f}"
            print(row)

        # Factor correlation matrix
        if factor_series and len(factor_series) > 1:
            corr = pd.concat(factor_series, axis=1).corr(method="spearman")
            print()
            print("Factor Correlation Matrix (Spearman):")
            print(corr.to_string(float_format=lambda x: f"{x:.3f}"))

    def generate_extended_summary(
        self,
        factor_metrics_results: dict[str, FactorMetricsResult] | None,
        output_dir: Path,
    ) -> Path | None:
        """Generate extended metrics summary to file.

        Appends to existing summary.txt or creates new file.
        """
        if not factor_metrics_results:
            return None

        lines = self._format_factor_metrics(factor_metrics_results)
        summary_path = output_dir / "summary.txt"
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write("\n" + "\n".join(lines) + "\n")
        return summary_path

    def print_extended_summary(
        self,
        factor_metrics_results: dict[str, FactorMetricsResult] | None,
    ) -> None:
        """Print extended metrics to console."""
        if factor_metrics_results:
            for line in self._format_factor_metrics(factor_metrics_results):
                print(line)

    @staticmethod
    def _format_factor_metrics(
        results: dict[str, FactorMetricsResult],
    ) -> list[str]:
        """Format factor signal quality metrics as text lines."""
        lines = [
            "",
            "Factor Signal Quality",
            "=" * 60,
        ]
        for fname, m in results.items():
            lines.append(f"Factor: {fname}")
            lines.append("-" * 40)

            periods = list(m.win_rate.index)
            hdr = "  " + " " * 16 + "".join(f"{p:>10}" for p in periods)
            lines.append(hdr)
            lines.append(
                "  Win Rate        "
                + "".join(f"{m.win_rate[p]:>9.1%} " for p in periods)
            )
            lines.append(
                "  Monotonicity    "
                + "".join(
                    f"{m.monotonicity.get(p, np.nan):>9.2f} "
                    for p in periods
                )
            )
            lines.append(
                "  IC Half-Life    "
                + "".join(
                    f"{m.ic_half_life.get(p, np.nan):>7.0f} bars"
                    if np.isfinite(m.ic_half_life.get(p, np.nan))
                    else f"{'N/A':>10} "
                    for p in periods
                )
            )
            lines.append(
                "  IC Linearity    "
                + "".join(
                    f"{m.ic_linearity.get(p, np.nan):>10.3f}" for p in periods
                )
            )
            lines.append(
                "  IC AR(1)        "
                + "".join(
                    f"{m.ic_ar1.get(p, np.nan):>10.3f}" for p in periods
                )
            )
            lines.append(f"  Coverage         {m.coverage:.1%}")
            lines.append("")

        return lines

