"""Factor evaluator - core alpha analysis logic.

Computes factor values using FactorEngine + CsFactorEngine,
then evaluates factor quality via alphalens-reloaded.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from nautilus_quants.alpha.data_loader import CatalogDataLoader
from nautilus_quants.factors.config import FactorConfig
from nautilus_quants.factors.engine.cs_factor_engine import CsFactorEngine
from nautilus_quants.factors.engine.factor_engine import FactorEngine
from nautilus_quants.factors.expression import VectorizedEvaluator, parse_expression
from nautilus_quants.factors.operators.cross_sectional import CS_OPERATOR_INSTANCES
from nautilus_quants.factors.operators.math import MATH_OPERATORS
from nautilus_quants.factors.operators.time_series import TS_OPERATOR_INSTANCES

if TYPE_CHECKING:
    from nautilus_trader.model.data import Bar

    from nautilus_quants.alpha.analysis.config import AlphaAnalysisConfig

logger = logging.getLogger(__name__)


class FactorEvaluator:
    """Evaluate factor quality using alphalens.

    Uses the two-phase computation architecture:
    - Phase 1: FactorEngine computes time-series factors per bar
    - Phase 2: CsFactorEngine computes cross-sectional factors across instruments
    """

    def __init__(self, factor_config: FactorConfig) -> None:
        self._factor_config = factor_config
        self._ts_engine = FactorEngine(config=factor_config)
        self._cs_engine = CsFactorEngine(config=factor_config)

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
        """Vectorized two-phase factor computation.

        1. Convert bars to DataFrames per instrument
        2. Compute TS factors via VectorizedEvaluator (full Series, no Python loop)
        3. Build factor panels: {factor_name: DataFrame(timestamps x instruments)}
        4. Compute CS factors via CS operator compute_vectorized()
        5. Convert to alphalens MultiIndex format
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

        # --- Step 2: Compute TS factors via VectorizedEvaluator ---
        # Build variable expressions from config
        variable_exprs = dict(config.variables)
        ts_factor_names = self._cs_engine.ts_factor_names
        ts_factor_defs = {
            f.name: f for f in config.factors if f.name in ts_factor_names
        }

        # Collect per-instrument factor series, then build panels via concat
        # to avoid DataFrame fragmentation from column-by-column insertion.
        factor_series_by_name: dict[str, dict[str, pd.Series]] = {}

        for inst_id in instruments:
            if inst_id not in ohlcv_dfs:
                continue
            df = ohlcv_dfs[inst_id]
            # Base variables
            variables: dict[str, pd.Series] = {
                "open": df["open"],
                "high": df["high"],
                "low": df["low"],
                "close": df["close"],
                "volume": df["volume"],
            }

            evaluator = VectorizedEvaluator(
                variables=variables,
                ts_operators=TS_OPERATOR_INSTANCES,
                math_operators=MATH_OPERATORS,
                parameters=dict(config.parameters),
            )

            # Evaluate variable expressions (e.g., returns = close / delay(close, 1) - 1)
            for var_name, var_expr in variable_exprs.items():
                ast = parse_expression(var_expr)
                variables[var_name] = evaluator.evaluate(ast)

            # Evaluate TS factors
            for fname, fdef in ts_factor_defs.items():
                ast = parse_expression(fdef.expression)
                series_result = evaluator.evaluate(ast)
                if isinstance(series_result, (int, float)):
                    series_result = pd.Series(series_result, index=df.index)
                if fname not in factor_series_by_name:
                    factor_series_by_name[fname] = {}
                factor_series_by_name[fname][inst_id] = series_result

        # Build factor panels in one shot via pd.concat (handles index alignment)
        factor_panels: dict[str, pd.DataFrame] = {}
        for fname, series_dict in factor_series_by_name.items():
            factor_panels[fname] = pd.concat(series_dict, axis=1)

        # --- Step 3: Compute CS factors (vectorized over whole DataFrame) ---
        cs_factor_defs = [
            f for f in config.factors if f.name in self._cs_engine.cs_factor_names
        ]

        for cs_def in cs_factor_defs:
            result_df = self._evaluate_cs_expression_vectorized(
                cs_def.expression, factor_panels,
            )
            if result_df is not None:
                factor_panels[cs_def.name] = result_df
            else:
                logger.warning(
                    "CS factor '%s' expression '%s' evaluated to None "
                    "— check factor dependencies",
                    cs_def.name,
                    cs_def.expression,
                )

        # --- Step 4: Convert to alphalens format ---
        factor_series: dict[str, pd.Series] = {}
        for fname, panel_df in factor_panels.items():
            stacked = panel_df.stack(future_stack=True).dropna()
            stacked.index.names = ["date", "asset"]
            if len(stacked) > 0:
                factor_series[fname] = stacked.astype(float)

        return factor_series, pricing

    def _evaluate_cs_expression_vectorized(
        self,
        expression: str,
        factor_panels: dict[str, pd.DataFrame],
    ) -> pd.DataFrame | None:
        """Evaluate a cross-sectional expression using vectorized CS operators."""
        expression = expression.strip()

        # Handle weighted sum (e.g., "0.6 * a + 0.4 * b")
        if self._is_cs_weighted_sum(expression):
            result = self._eval_cs_weighted_sum_vec(expression, factor_panels)
            if result is not None:
                return result
            # Fallback: weighted-sum heuristic matched but parsing failed
            # (e.g., "-1 * normalize(x, true, 0)"), try function path.

        # Handle function call
        if "(" in expression:
            return self._eval_cs_function_vec(expression, factor_panels)

        # Simple variable reference
        return factor_panels.get(expression)

    def _is_cs_weighted_sum(self, expr: str) -> bool:
        """Check if expression is a weighted sum of factor names."""
        depth = 0
        i = 0
        while i < len(expr):
            char = expr[i]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif char in "+-" and depth == 0 and i > 0:
                # Skip scientific notation (e.g., 1e-3)
                if expr[i - 1] in "eE" and i >= 2 and expr[i - 2].isdigit():
                    i += 1
                    continue
                return True
            i += 1
        return False

    def _eval_cs_weighted_sum_vec(
        self,
        expression: str,
        factor_panels: dict[str, pd.DataFrame],
    ) -> pd.DataFrame | None:
        pattern = r'([+-]?\s*[\d.]+(?:[eE][+-]?\d+)?)\s*\*\s*(\w+)'
        matches = re.findall(pattern, expression)
        if not matches:
            return None

        result: pd.DataFrame | None = None
        for weight_str, factor_name in matches:
            weight = float(weight_str.replace(" ", ""))
            panel = factor_panels.get(factor_name)
            if panel is None:
                return None
            term = panel * weight
            result = term if result is None else result + term
        return result

    def _eval_cs_function_vec(
        self,
        expression: str,
        factor_panels: dict[str, pd.DataFrame],
    ) -> pd.DataFrame | None:
        match = re.match(r'(\w+)\s*\((.*)\)', expression, re.DOTALL)
        if not match:
            return None

        func_name = match.group(1)
        args_str = match.group(2)

        op_instance = CS_OPERATOR_INSTANCES.get(func_name)
        if op_instance is None:
            op_instance = CS_OPERATOR_INSTANCES.get(f"cs_{func_name}")
        if op_instance is None:
            return None

        args = self._parse_cs_args_vec(args_str, factor_panels)
        if not args or not isinstance(args[0], pd.DataFrame):
            return None

        kwargs: dict[str, Any] = {}
        # Pass extra args as keyword arguments based on operator type
        if len(args) > 1:
            # Map positional args to operator-specific kwargs
            sig_map: dict[str, list[str]] = {
                "normalize": ["use_std", "limit"],
                "winsorize": ["std_mult"],
                "scale_down": ["constant"],
                "clip_quantile": ["lower", "upper"],
                "quantile": ["driver", "sigma"],
            }
            param_names = sig_map.get(func_name, [])
            for i, pname in enumerate(param_names):
                if i + 1 < len(args):
                    kwargs[pname] = args[i + 1]

        return op_instance.compute_vectorized(args[0], **kwargs)

    def _parse_cs_args_vec(
        self,
        args_str: str,
        factor_panels: dict[str, pd.DataFrame],
    ) -> list[Any]:
        args: list[Any] = []
        depth = 0
        current = ""
        for char in args_str:
            if char == "(":
                depth += 1
                current += char
            elif char == ")":
                depth -= 1
                current += char
            elif char == "," and depth == 0:
                args.append(self._parse_single_cs_arg_vec(current.strip(), factor_panels))
                current = ""
            else:
                current += char
        if current.strip():
            args.append(self._parse_single_cs_arg_vec(current.strip(), factor_panels))
        return args

    def _parse_single_cs_arg_vec(
        self,
        arg: str,
        factor_panels: dict[str, pd.DataFrame],
    ) -> Any:
        if arg.lower() == "true":
            return True
        if arg.lower() == "false":
            return False
        try:
            val = float(arg)
            return int(val) if val == int(val) and "." not in arg and "e" not in arg.lower() else val
        except ValueError:
            pass
        if "(" in arg:
            return self._eval_cs_function_vec(arg, factor_panels)
        return factor_panels.get(arg)

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
