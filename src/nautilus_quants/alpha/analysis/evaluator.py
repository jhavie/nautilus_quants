"""Factor evaluator - core alpha analysis logic.

Computes factor values using Evaluator (unified TS+CS evaluation),
then evaluates factor quality via alphalens-reloaded.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from nautilus_trader.persistence.catalog import ParquetDataCatalog

from nautilus_quants.alpha.analysis.config import AlphaAnalysisConfig
from nautilus_quants.alpha.data_loader import CatalogDataLoader
from nautilus_quants.factors.config import FactorConfig
from nautilus_quants.factors.engine.evaluator import Evaluator
from nautilus_quants.factors.expression import parse_expression
from nautilus_quants.factors.operators.cross_sectional import CS_OPERATOR_INSTANCES
from nautilus_quants.factors.operators.math import MATH_OPERATORS
from nautilus_quants.factors.operators.time_series import TS_OPERATOR_INSTANCES

if TYPE_CHECKING:
    from nautilus_trader.model.data import Bar

logger = logging.getLogger(__name__)


class FactorEvaluator:
    """Evaluate factor quality using alphalens.

    Uses Evaluator for unified TS+CS factor computation over
    panel DataFrames (timestamps × instruments).  All factor expressions
    — including nested rank(ts_argmax(…)) — are evaluated in a single
    recursive AST pass.
    """

    def __init__(
        self,
        factor_config: FactorConfig,
        analysis_config: AlphaAnalysisConfig | None = None,
    ) -> None:
        self._factor_config = factor_config
        self._analysis_config = analysis_config

    def evaluate(
        self,
        bars_by_instrument: dict[str, list[Bar]],
    ) -> tuple[dict[str, pd.Series], pd.DataFrame]:
        """Compute all factors and prepare alphalens-compatible data.

        Args:
            bars_by_instrument: {instrument_id: [Bar, ...]}

        Returns:
            Tuple of:
            - {factor_name: Series(MultiIndex[date, asset])}
            - pricing DataFrame(index=datetime, columns=instruments)
        """
        return self._compute_all_factors(bars_by_instrument)

    def _compute_all_factors(
        self,
        bars_by_instrument: dict[str, list[Bar]],
    ) -> tuple[dict[str, pd.Series], pd.DataFrame]:
        """Unified panel-based factor computation via Evaluator.

        1. Convert bars to per-instrument DataFrames
        2. Build OHLCV panel DataFrames (timestamps × instruments)
        3. Evaluate ALL expressions (variables + factors) via Evaluator
        4. Convert to alphalens MultiIndex format
        """
        if not bars_by_instrument:
            return {}, pd.DataFrame()

        config = self._factor_config
        instruments = sorted(bars_by_instrument.keys())

        # --- Step 1: Bars -> OHLCV DataFrames per instrument ---
        ohlcv_dfs: dict[str, pd.DataFrame] = {}
        for inst_id in instruments:
            bars = bars_by_instrument[inst_id]
            if not bars:
                continue
            ohlcv_dfs[inst_id] = CatalogDataLoader.bars_to_dataframe(bars)

        # Build pricing DataFrame
        pricing = pd.DataFrame(
            {inst_id: df["close"] for inst_id, df in ohlcv_dfs.items()}
        )

        # --- Step 2: Build OHLCV panel DataFrames ---
        fields = ("open", "high", "low", "close", "volume")
        panel_fields: dict[str, pd.DataFrame | float] = {}
        for field in fields:
            series_dict = {
                inst_id: df[field]
                for inst_id, df in ohlcv_dfs.items()
            }
            panel_fields[field] = pd.concat(series_dict, axis=1)

        # Inject config parameters
        for p_name, p_val in config.parameters.items():
            panel_fields[p_name] = p_val

        # Inject funding rate data
        acfg = self._analysis_config
        if acfg and acfg.funding_rate and acfg.catalog_path:
            fr_panel = self._load_funding_rate_panel(
                acfg.catalog_path, instruments, panel_fields["close"],
            )
            if fr_panel is not None:
                panel_fields["funding_rate"] = fr_panel
                logger.info("Injected funding_rate panel: %s", fr_panel.shape)

        # Inject open interest data
        if acfg and acfg.oi_data_path:
            oi_panel = self._load_oi_panel(
                acfg.oi_data_path, instruments, panel_fields["close"],
                timeframe=acfg.oi_timeframe,
            )
            if oi_panel is not None:
                panel_fields["open_interest"] = oi_panel
                logger.info("Injected open_interest panel: %s", oi_panel.shape)

        # --- Step 3: Evaluate via Evaluator ---
        evaluator = Evaluator(
            panel_fields=panel_fields,
            ts_ops=dict(TS_OPERATOR_INSTANCES),
            cs_ops=dict(CS_OPERATOR_INSTANCES),
            math_ops=dict(MATH_OPERATORS),
            parameters=dict(config.parameters),
        )

        # Evaluate variables first (in config order)
        for var_name, var_expr in config.variables.items():
            try:
                ast = parse_expression(var_expr)
                result = evaluator.evaluate(ast)
                panel_fields[var_name] = result
            except Exception:
                logger.warning(
                    "Variable '%s' expression '%s' failed to evaluate",
                    var_name, var_expr,
                )

        # Evaluate ALL factors (no TS/CS split needed)
        factor_panels: dict[str, pd.DataFrame] = {}
        for factor_def in config.factors:
            try:
                ast = parse_expression(factor_def.expression)
                result = evaluator.evaluate(ast)
                if isinstance(result, pd.DataFrame):
                    # Inject into panel_fields for downstream factor references
                    panel_fields[factor_def.name] = result
                    factor_panels[factor_def.name] = result
                elif isinstance(result, (int, float)):
                    # Scalar → broadcast to all instruments
                    ref_panel = panel_fields.get("close")
                    if isinstance(ref_panel, pd.DataFrame):
                        broadcast_df = pd.DataFrame(
                            float(result),
                            index=ref_panel.index,
                            columns=ref_panel.columns,
                        )
                        panel_fields[factor_def.name] = broadcast_df
                        factor_panels[factor_def.name] = broadcast_df
                else:
                    logger.warning(
                        "Factor '%s' expression '%s' evaluated to %s",
                        factor_def.name, factor_def.expression,
                        type(result).__name__,
                    )
            except Exception as e:
                logger.warning(
                    "Factor '%s' expression '%s' failed: %s",
                    factor_def.name, factor_def.expression, e,
                )

        # --- Step 4: Convert to alphalens format ---
        factor_series: dict[str, pd.Series] = {}
        for fname, panel_df in factor_panels.items():
            stacked = panel_df.stack(future_stack=True).dropna()
            stacked.index.names = ["date", "asset"]
            if len(stacked) > 0:
                factor_series[fname] = stacked.astype(float)

        return factor_series, pricing

    def compute_forward_returns(
        self,
        factor_series: pd.Series,
        pricing: pd.DataFrame,
        config: AlphaAnalysisConfig,
    ) -> pd.DataFrame:
        """Pre-compute forward returns once for reuse across all factors.

        Args:
            factor_series: Any factor Series (only its index is used)
            pricing: Price DataFrame(index=datetime, columns=instruments)
            config: Analysis configuration

        Returns:
            Forward returns DataFrame with MultiIndex(date, asset)
        """
        import alphalens.utils as al_utils

        inferred = pd.infer_freq(pricing.index) if len(pricing) > 2 else "h"
        freq_offset = pd.tseries.frequencies.to_offset(inferred or "h")
        original_infer = al_utils.infer_trading_calendar

        def _crypto_calendar(factor_idx, prices_idx):
            return freq_offset

        al_utils.infer_trading_calendar = _crypto_calendar
        try:
            fwd_returns = al_utils.compute_forward_returns(
                factor=factor_series,
                prices=pricing,
                periods=config.periods,
                filter_zscore=config.filter_zscore,
            )
        finally:
            al_utils.infer_trading_calendar = original_infer

        # Rename period columns from Timedelta format (e.g. '1D1h') to
        # bar-count-based human-readable labels (e.g. '1h', '4h', '24h').
        fwd_returns = self._rename_period_columns(fwd_returns, config)
        return fwd_returns

    @staticmethod
    def _rename_period_columns(
        df: pd.DataFrame, config: AlphaAnalysisConfig,
    ) -> pd.DataFrame:
        """Rename Timedelta-style period columns to human-readable labels.

        alphalens produces column names like '1D1h', '4D4h' via its internal
        diff_custom_calendar_timedeltas logic, which can be misleading for
        sub-daily bar frequencies.  We rename by positional correspondence:
        sorted(config.periods) maps 1:1 to the column order produced by
        compute_forward_returns (which also sorts periods).
        """
        inferred_freq = df.index.levels[0].freq
        if inferred_freq is None:
            return df

        try:
            bar_td = pd.Timedelta(inferred_freq)
        except (ValueError, TypeError):
            bar_td = pd.Timedelta(
                pd.tseries.frequencies.to_offset(inferred_freq).nanos, unit="ns",
            )

        # Columns that are forward-return periods (exclude 'factor', 'factor_quantile')
        period_cols = [c for c in df.columns if c not in ("factor", "factor_quantile")]
        sorted_periods = sorted(config.periods)

        if len(period_cols) != len(sorted_periods):
            return df

        rename_map = {}
        for col, n_bars in zip(period_cols, sorted_periods):
            rename_map[col] = _format_bar_period(n_bars, bar_td)

        return df.rename(columns=rename_map)

    @staticmethod
    def _load_funding_rate_panel(
        catalog_path: str,
        instruments: list[str],
        close_panel: pd.DataFrame,
    ) -> pd.DataFrame | None:
        """Load FundingRateUpdate from catalog and build panel aligned to close.

        Returns DataFrame(index=datetime, columns=instruments) with forward-filled
        funding rate values, or None if no data found.
        """
        try:
            catalog = ParquetDataCatalog(catalog_path)
            # Workaround: NautilusTrader batch query loses per-file
            # instrument_id metadata when PyArrow merges parquet files.
            # Query per-instrument to preserve correct instrument_id.
            fr_data: list = []
            for inst in instruments:
                result = catalog.funding_rates(instrument_ids=[inst])
                if result:
                    fr_data.extend(result)
            if not fr_data:
                logger.warning("No FundingRateUpdate data in catalog")
                return None

            # Build per-instrument Series
            fr_dict: dict[str, dict] = {}
            for fru in fr_data:
                inst_id = str(fru.instrument_id)
                ts = pd.Timestamp(fru.ts_event, unit="ns")
                if inst_id not in fr_dict:
                    fr_dict[inst_id] = {}
                fr_dict[inst_id][ts] = float(fru.rate)

            if not fr_dict:
                return None

            fr_panel = pd.DataFrame(
                {k: pd.Series(v) for k, v in fr_dict.items()}
            )
            # Reindex to close_panel timestamps, forward-fill
            fr_panel = fr_panel.reindex(close_panel.index, method="ffill")
            # Fill remaining NaN columns for instruments without FR data
            for col in close_panel.columns:
                if col not in fr_panel.columns:
                    fr_panel[col] = 0.0
            return fr_panel[close_panel.columns]
        except Exception as e:
            logger.warning("Failed to load funding rate panel: %s", e)
            return None

    @staticmethod
    def _load_oi_panel(
        oi_data_path: str,
        instruments: list[str],
        close_panel: pd.DataFrame,
        timeframe: str = "4h",
    ) -> pd.DataFrame | None:
        """Load OI parquet and build panel aligned to close.

        Returns DataFrame(index=datetime, columns=instruments) with forward-filled
        open interest values, or None if no data found.
        """
        try:
            from nautilus_quants.data.transform.open_interest import (
                load_oi_lookup,
            )

            lookup = load_oi_lookup(oi_data_path, instruments, timeframe)

            oi_series: dict[str, pd.Series] = {}
            for inst_id, ts_map in lookup.items():
                if not ts_map:
                    continue
                timestamps = [pd.Timestamp(ts, unit="ns") for ts in ts_map]
                values = [d["open_interest"] for d in ts_map.values()]
                oi_series[inst_id] = pd.Series(values, index=timestamps)

            if not oi_series:
                return None

            oi_panel = pd.DataFrame(oi_series)
            oi_panel = oi_panel.reindex(close_panel.index, method="ffill")
            for col in close_panel.columns:
                if col not in oi_panel.columns:
                    oi_panel[col] = 0.0
            return oi_panel[close_panel.columns]
        except Exception as e:
            logger.warning("Failed to load OI panel: %s", e)
            return None

    def run_alphalens(
        self,
        factor_series: pd.Series,
        pricing: pd.DataFrame,
        config: AlphaAnalysisConfig,
        forward_returns: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Run alphalens analysis on a single factor.

        Args:
            factor_series: Factor values with MultiIndex(date, asset)
            pricing: Price DataFrame(index=datetime, columns=instruments)
            config: Analysis configuration
            forward_returns: Pre-computed forward returns (if None, computes internally)

        Returns:
            Dict with 'factor_data', 'ic', 'mean_returns', etc.
        """
        import alphalens.performance as perf
        import alphalens.utils as al_utils

        if forward_returns is not None:
            return run_alphalens_with_forward_returns(
                factor_series, forward_returns, config.quantiles, config.max_loss,
            )
        else:
            # Fallback: compute everything (slower, for backward compatibility)
            inferred = pd.infer_freq(pricing.index) if len(pricing) > 2 else "h"
            freq_offset = pd.tseries.frequencies.to_offset(inferred or "h")
            original_infer = al_utils.infer_trading_calendar

            def _crypto_calendar(factor_idx, prices_idx):
                return freq_offset

            al_utils.infer_trading_calendar = _crypto_calendar
            try:
                factor_data = al_utils.get_clean_factor_and_forward_returns(
                    factor=factor_series,
                    prices=pricing,
                    periods=config.periods,
                    quantiles=config.quantiles,
                    max_loss=config.max_loss,
                    filter_zscore=config.filter_zscore,
                )
            finally:
                al_utils.infer_trading_calendar = original_infer

            factor_data = self._rename_period_columns(factor_data, config)

        ic = perf.factor_information_coefficient(factor_data)
        mean_returns, _ = perf.mean_return_by_quantile(factor_data, by_date=False)

        return {
            "factor_data": factor_data,
            "ic": ic,
            "mean_returns": mean_returns,
        }


def run_alphalens_with_forward_returns(
    factor_series: pd.Series,
    forward_returns: pd.DataFrame,
    quantiles: int,
    max_loss: float,
) -> dict[str, Any]:
    """Run alphalens analysis with pre-computed forward returns.

    Module-level function suitable for ProcessPoolExecutor pickling.
    """
    import alphalens.performance as perf
    import alphalens.utils as al_utils

    factor_data = al_utils.get_clean_factor(
        factor=factor_series,
        forward_returns=forward_returns,
        quantiles=quantiles,
        max_loss=max_loss,
    )
    ic = perf.factor_information_coefficient(factor_data)
    mean_returns, _ = perf.mean_return_by_quantile(factor_data, by_date=False)
    return {
        "factor_data": factor_data,
        "ic": ic,
        "mean_returns": mean_returns,
    }


def _format_bar_period(n_bars: int, bar_td: pd.Timedelta) -> str:
    """Format a bar count into a human-readable period label.

    Examples: 1 bar of 1h -> '1h', 4 bars of 1h -> '4h', 24 bars -> '24h'
    """
    total = bar_td * n_bars
    total_seconds = int(total.total_seconds())

    if total_seconds % 86400 == 0:
        days = total_seconds // 86400
        return f"{days}d"
    if total_seconds % 3600 == 0:
        hours = total_seconds // 3600
        return f"{hours}h"
    if total_seconds % 60 == 0:
        minutes = total_seconds // 60
        return f"{minutes}m"
    return f"{total_seconds}s"
