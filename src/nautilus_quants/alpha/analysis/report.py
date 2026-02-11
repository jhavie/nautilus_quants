"""Analysis report generator.

Generates alphalens tearsheet charts and IC/ICIR summary.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for chart generation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

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


def compute_ic_summary(ic_df: pd.DataFrame) -> pd.DataFrame:
    """Compute IC statistics with proper NaN handling.

    Unlike alphalens ``plot_information_table`` which propagates NaN
    through ``scipy.stats.ttest_1samp``, this function drops NaN values
    before computing t-stat and p-value.

    Returns a DataFrame indexed by period with columns:
        IC Mean, IC Std., Risk-Adjusted IC, t-stat(IC), p-value(IC),
        IC Skew, IC Kurtosis, N
    """
    table = pd.DataFrame()
    ic_clean = ic_df.dropna()
    table["IC Mean"] = ic_clean.mean()
    table["IC Std."] = ic_clean.std()
    table["Risk-Adjusted IC"] = ic_clean.mean() / ic_clean.std()
    t_stat, p_value = scipy_stats.ttest_1samp(ic_clean, 0, nan_policy="omit")
    table["t-stat(IC)"] = t_stat
    table["p-value(IC)"] = p_value
    table["IC Skew"] = scipy_stats.skew(ic_clean, nan_policy="omit")
    table["IC Kurtosis"] = scipy_stats.kurtosis(ic_clean, nan_policy="omit")
    table["N"] = ic_clean.count()
    nan_count = ic_df.isna().sum()
    if nan_count.any():
        table["NaN Count"] = nan_count
    return table


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
        ac[p] = perf.factor_rank_autocorrelation(factor_data, period=1)

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
    ) -> Path:
        """Generate IC/ICIR summary text file.

        Uses alphalens plot_information_table for IC Mean, IC Std,
        Risk-Adjusted IC (ICIR), t-stat, p-value, Skew, Kurtosis.
        IC DataFrames are pre-cleaned (NaN dropped) before passing to
        alphalens to avoid NaN propagation in scipy.stats.ttest_1samp.

        Args:
            ic_results: {factor_name: IC DataFrame}
            output_dir: Run output directory
            factor_series: {factor_name: Series(MultiIndex[date, asset])} for correlation

        Returns:
            Path to summary file
        """
        import alphalens.plotting as plotting

        lines = ["Factor Analysis Summary", "=" * 60, ""]

        for factor_name, ic_df in ic_results.items():
            lines.append(f"Factor: {factor_name}")
            lines.append("-" * 40)

            ic_clean = ic_df.dropna()
            ic_table = plotting.plot_information_table(ic_clean, return_df=True)
            for period in ic_table.index:
                row = ic_table.loc[period]
                n = int(ic_clean[period].count()) if period in ic_clean.columns else 0
                lines.append(
                    f"  Period {period}: "
                    f"IC={row['IC Mean']:.4f}, "
                    f"ICIR={row['Risk-Adjusted IC']:.4f}, "
                    f"t={row['t-stat(IC)']:.2f}, "
                    f"p={row['p-value(IC)']:.4f}, "
                    f"N={n}"
                )

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

        Uses alphalens plot_information_table for consistent metrics.
        IC DataFrames are pre-cleaned (NaN dropped) before passing to
        alphalens to avoid NaN propagation in scipy.stats.ttest_1samp.

        Args:
            ic_results: {factor_name: IC DataFrame}
            factor_series: {factor_name: Series(MultiIndex[date, asset])} for correlation
        """
        import alphalens.plotting as plotting

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
            ic_clean = ic_df.dropna()
            ic_table = plotting.plot_information_table(ic_clean, return_df=True)
            row = f"  {factor_name:<20}"
            icir_values = []
            for p in periods:
                if p in ic_table.index:
                    row += f"  {ic_table.loc[p, 'IC Mean']:>6.3f}"
                    icir_values.append(ic_table.loc[p, "Risk-Adjusted IC"])
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
