"""Report generation for backtest module."""

from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from nautilus_quants.backtest.exceptions import BacktestReportError
from nautilus_quants.utils.cache_keys import (
    EQUITY_SNAPSHOTS_CACHE_KEY,
    EXECUTION_STATES_CACHE_KEY,
    FACTOR_VALUES_CACHE_KEY,
    POSITION_MARKET_VALUES_CACHE_KEY,
    POSITION_METADATA_CACHE_KEY,
)
from nautilus_quants.utils.protocols import (
    BaseMetadataRenderer,
    MetadataRenderer,
)

if TYPE_CHECKING:
    from nautilus_trader.backtest.engine import BacktestEngine
    from nautilus_trader.model.position import Position

    from nautilus_quants.backtest.config import PositionVisualizationConfig, ReportConfig


class ReportGenerator:
    """Generates all backtest reports."""

    def __init__(
        self,
        engine: "BacktestEngine",
        output_dir: Path,
        config: "ReportConfig",
        metadata_renderer: MetadataRenderer | None = None,
    ) -> None:
        """Initialize report generator.

        Args:
            engine: BacktestEngine after execution
            output_dir: Directory for output files
            config: Report configuration
            metadata_renderer: Optional renderer for strategy-specific position metadata.
                              If None, uses BaseMetadataRenderer for basic columns.
        """
        self.engine = engine
        self.output_dir = Path(output_dir)
        self.config = config
        self.metadata_renderer = metadata_renderer or BaseMetadataRenderer()

    @cached_property
    def _mtm_equity(self) -> pd.Series | None:
        """Lazy-cached MTM equity from EquitySnapshotActor."""
        return self._build_equity_from_actor_snapshots()

    @cached_property
    def _market_values_index(self) -> dict[str, pd.Series]:
        """Lazy-cached per-instrument market value series from EquitySnapshotActor.

        Returns:
            Dict mapping inst_id to pd.Series of market values indexed by datetime.
            Empty dict if cache data not available (backward-compatible).
        """
        import pickle

        try:
            data = self.engine.cache.get(POSITION_MARKET_VALUES_CACHE_KEY)
        except Exception:
            return {}

        if not data:
            return {}

        try:
            points: list[tuple[int, dict[str, float]]] = pickle.loads(data)
        except Exception:
            return {}

        if not points:
            return {}

        # Transpose: list[(ts_ns, {inst_id: value})] → dict{inst_id: Series}
        by_inst: dict[str, list[tuple[pd.Timestamp, float]]] = {}
        for ts_ns, values in points:
            ts = pd.Timestamp(ts_ns, unit="ns", tz="UTC")
            for inst_id, val in values.items():
                by_inst.setdefault(inst_id, []).append((ts, val))

        index: dict[str, pd.Series] = {}
        for inst_id, pairs in by_inst.items():
            timestamps = [t for t, _ in pairs]
            vals = [v for _, v in pairs]
            index[inst_id] = pd.Series(vals, index=pd.DatetimeIndex(timestamps)).sort_index()

        return index

    @cached_property
    def _realized_equity(self) -> pd.Series:
        """Lazy-cached realized equity from engine account data."""
        return self._get_realized_equity_series()

    def _get_market_value(self, inst_id: str, ts: pd.Timestamp) -> float | None:
        """Get the market value for an instrument at a given timestamp.

        Uses asof() (forward-fill with most recent snapshot) from the cached
        per-instrument market value series built by EquitySnapshotActor.

        Args:
            inst_id: Instrument identifier string
            ts: Timestamp to query

        Returns:
            Market value float, or None if no data available.
        """
        series = self._market_values_index.get(inst_id)
        if series is None or series.empty:
            return None
        val = series.asof(ts)
        return float(val) if pd.notna(val) else None

    def _get_closed_positions(self) -> list["Position"]:
        """Get all closed positions from the engine.

        Combines position_snapshots() with positions_closed() to ensure we capture
        all trades, including the last trade that may not have been saved as a snapshot.
        Only returns positions with ts_closed > 0 to avoid 1970-01-01 date issues.

        Returns:
            List of closed positions with valid ts_closed timestamps
        """
        # Get historical snapshots (closed positions)
        all_snapshots = list(self.engine.cache.position_snapshots())
        closed_snapshots = [p for p in all_snapshots if p.ts_closed > 0]

        # Get currently closed positions (may include trades not yet in snapshots)
        closed_positions = list(self.engine.cache.positions_closed())

        # Merge: add closed positions not already in snapshots
        snapshot_ids = {p.id for p in closed_snapshots}
        for p in closed_positions:
            if p.id not in snapshot_ids and p.ts_closed > 0:
                closed_snapshots.append(p)

        return closed_snapshots

    def generate_all(self) -> dict[str, Path]:
        """Generate all configured reports.

        Returns:
            Dict mapping report type to file path
        """
        reports: dict[str, Path] = {}

        # Generate CSV reports if configured
        if "csv" in self.config.formats:
            csv_reports = self.generate_csv_reports()
            reports.update(csv_reports)

        # Generate tearsheet if configured
        if "html" in self.config.formats:
            tearsheet_path = self.generate_tearsheet()
            if tearsheet_path:
                reports["tearsheet"] = tearsheet_path

        # Generate QuantStats reports if configured
        quantstats_reports = self.generate_quantstats_reports()
        reports.update(quantstats_reports)

        # Generate position visualization if configured
        position_viz_path = self.generate_position_visualization()
        if position_viz_path:
            reports["position_viz"] = position_viz_path

        # Generate factor values CSV if data is available
        factor_csv_path = self.generate_factor_values_csv()
        if factor_csv_path:
            reports["factor_values"] = factor_csv_path

        # Generate execution report CSV if PostLimit data is available
        execution_csv_path = self.generate_execution_report_csv()
        if execution_csv_path:
            reports["execution_report"] = execution_csv_path

        return reports

    def generate_csv_reports(self) -> dict[str, Path]:
        """Generate CSV reports using ReportProvider.

        Returns:
            Dict mapping report name to file path
        """
        from nautilus_trader.analysis import ReportProvider

        reports: dict[str, Path] = {}

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get orders once for both orders and fills reports
        orders = list(self.engine.cache.orders())

        # Orders report
        try:
            if orders:
                orders_df = ReportProvider.generate_orders_report(orders)
                orders_path = self.output_dir / "orders_report.csv"
                orders_df.to_csv(orders_path)
                reports["orders"] = orders_path
            else:
                # Create empty report
                orders_path = self.output_dir / "orders_report.csv"
                orders_path.write_text("# No orders executed\n")
                reports["orders"] = orders_path
        except Exception as e:
            raise BacktestReportError(f"Failed to generate orders report: {e}") from e

        # Fills report
        try:
            if orders:
                fills_df = ReportProvider.generate_fills_report(orders)
                fills_path = self.output_dir / "fills_report.csv"
                fills_df.to_csv(fills_path)
                reports["fills"] = fills_path
            else:
                fills_path = self.output_dir / "fills_report.csv"
                fills_path.write_text("# No fills executed\n")
                reports["fills"] = fills_path
        except Exception as e:
            raise BacktestReportError(f"Failed to generate fills report: {e}") from e

        # Positions report
        try:
            positions = list(self.engine.cache.positions())
            if positions:
                positions_df = ReportProvider.generate_positions_report(positions)
                positions_path = self.output_dir / "positions_report.csv"
                positions_df.to_csv(positions_path)
                reports["positions"] = positions_path
            else:
                positions_path = self.output_dir / "positions_report.csv"
                positions_path.write_text("# No positions opened\n")
                reports["positions"] = positions_path
        except Exception as e:
            raise BacktestReportError(
                f"Failed to generate positions report: {e}"
            ) from e

        # Account report
        try:
            # Try to get account - may not exist if no trades
            accounts = list(self.engine.cache.accounts())
            if accounts:
                account = accounts[0]
                account_df = ReportProvider.generate_account_report(account)
                account_path = self.output_dir / "account_report.csv"
                account_df.to_csv(account_path)
                reports["account"] = account_path
            else:
                account_path = self.output_dir / "account_report.csv"
                account_path.write_text("# No account activity\n")
                reports["account"] = account_path
        except Exception as e:
            raise BacktestReportError(
                f"Failed to generate account report: {e}"
            ) from e

        return reports

    def generate_statistics(self) -> dict[str, Any]:
        """Extract statistics from PortfolioAnalyzer.

        Returns:
            Combined statistics dictionary
        """
        try:
            analyzer = self.engine.portfolio.analyzer

            # Get all stats
            stats_pnls = analyzer.get_performance_stats_pnls() or {}
            stats_returns = analyzer.get_performance_stats_returns() or {}
            stats_general = analyzer.get_performance_stats_general() or {}

            # Combine into single dict
            combined: dict[str, Any] = {}
            combined.update(stats_pnls)
            combined.update(stats_returns)
            combined.update(stats_general)

            # Normalize keys for convenience properties
            if "PnL (total)" in combined:
                combined["total_pnl"] = combined["PnL (total)"]
            if "PnL% (total)" in combined:
                combined["total_pnl_pct"] = combined["PnL% (total)"]
            # Sharpe — NT may return "Sharpe Ratio (252 days)"
            for k in list(combined):
                if k.startswith("Sharpe Ratio"):
                    combined["sharpe_ratio"] = combined[k]
                    break
            # Max Drawdown — try NT key first, fallback to manual calc
            dd_found = False
            for k in list(combined):
                kl = k.lower()
                if "drawdown" in kl and "duration" not in kl:
                    combined["max_drawdown"] = combined[k]
                    dd_found = True
                    break
            if not dd_found:
                # Compute max drawdown from portfolio equity curve
                try:
                    import numpy as _np
                    returns_series = analyzer.returns()
                    if returns_series is not None and len(returns_series) > 1:
                        cumulative = (1 + returns_series).cumprod()
                        running_max = cumulative.cummax()
                        drawdowns = (cumulative - running_max) / running_max
                        max_dd = float(drawdowns.min())
                        if not _np.isnan(max_dd):
                            combined["Max Drawdown"] = max_dd
                            combined["max_drawdown"] = max_dd
                except Exception:
                    pass
            if "Win Rate" in combined:
                combined["win_rate"] = combined["Win Rate"]

            return combined

        except Exception:
            # Return empty stats on error
            return {}

    def generate_tearsheet(self) -> Path | None:
        """Generate HTML tearsheet if enabled and plotly available.

        Uses only closed positions with valid ts_closed timestamps to avoid
        1970-01-01 date issues in charts.

        Returns:
            Path to tearsheet file, or None if not generated
        """
        tearsheet_config = self.config.tearsheet
        if not tearsheet_config or not tearsheet_config.enabled:
            return None

        # Check if plotly is available
        try:
            import plotly  # noqa: F401
        except ImportError:
            # Log warning but don't fail
            import warnings

            warnings.warn(
                "plotly not installed. Skipping tearsheet generation. "
                "Install with: pip install plotly>=6.3.1"
            )
            return None

        try:
            from nautilus_trader.analysis import (
                TearsheetConfig as NautilusTearsheetConfig,
                create_tearsheet_from_stats,
            )
            from nautilus_trader.analysis.config import (
                TearsheetChart,
                TearsheetDistributionChart,
                TearsheetDrawdownChart,
                TearsheetEquityChart,
                TearsheetMonthlyReturnsChart,
                TearsheetRollingSharpeChart,
                TearsheetRunInfoChart,
                TearsheetStatsTableChart,
                TearsheetYearlyReturnsChart,
            )

            chart_name_to_class: dict[str, type[TearsheetChart]] = {
                "run_info": TearsheetRunInfoChart,
                "stats_table": TearsheetStatsTableChart,
                "equity": TearsheetEquityChart,
                "drawdown": TearsheetDrawdownChart,
                "monthly_returns": TearsheetMonthlyReturnsChart,
                "distribution": TearsheetDistributionChart,
                "rolling_sharpe": TearsheetRollingSharpeChart,
                "yearly_returns": TearsheetYearlyReturnsChart,
            }

            tearsheet_path = self.output_dir / "tearsheet.html"

            # Get daily returns from account equity (same method used for QuantStats)
            returns = self._get_returns_series()

            # Get statistics from the engine's analyzer (has full stats)
            engine_analyzer = self.engine.portfolio.analyzer
            stats_pnls = engine_analyzer.get_performance_stats_pnls() or {}
            stats_returns = engine_analyzer.get_performance_stats_returns() or {}
            stats_general = engine_analyzer.get_performance_stats_general() or {}

            # Build run info
            run_info = {
                "Run ID": str(self.engine.run_id) if hasattr(self.engine, "run_id") else "N/A",
                "Run Started": str(self.engine.run_started) if hasattr(self.engine, "run_started") else "N/A",
                "Run Finished": str(self.engine.run_finished) if hasattr(self.engine, "run_finished") else "N/A",
                "Backtest Start": str(self.engine.backtest_start) if hasattr(self.engine, "backtest_start") else "N/A",
                "Backtest End": str(self.engine.backtest_end) if hasattr(self.engine, "backtest_end") else "N/A",
            }

            # Build account info
            account_info = {}
            accounts = list(self.engine.cache.accounts())
            if accounts:
                account = accounts[0]
                for currency, balance in account.starting_balances().items():
                    account_info[f"Starting Balance ({currency})"] = f"{balance.as_double():.2f} {currency}"
                for currency, balance in account.balances_total().items():
                    account_info[f"Ending Balance ({currency})"] = f"{balance.as_double():.2f} {currency}"

            # Convert str chart names → TearsheetChart instances
            chart_objects: list[TearsheetChart] = []
            for name in tearsheet_config.charts:
                chart_cls = chart_name_to_class.get(name)
                if chart_cls is not None:
                    chart_kwargs = tearsheet_config.chart_args.get(name, {})
                    chart_objects.append(chart_cls(**chart_kwargs))

            # Create tearsheet config
            config = NautilusTearsheetConfig(
                title=tearsheet_config.title,
                theme=tearsheet_config.theme,
                height=tearsheet_config.height,
                show_logo=tearsheet_config.show_logo,
                include_benchmark=tearsheet_config.include_benchmark,
                charts=chart_objects,
            )

            # Use create_tearsheet_from_stats with our filtered returns
            create_tearsheet_from_stats(
                stats_pnls=stats_pnls,
                stats_returns=stats_returns,
                stats_general=stats_general,
                returns=returns,
                output_path=str(tearsheet_path),
                title=tearsheet_config.title,
                config=config,
                run_info=run_info,
                account_info=account_info,
                engine=self.engine,
            )

            return tearsheet_path

        except Exception as e:
            raise BacktestReportError(f"Failed to generate tearsheet: {e}") from e

    def _get_returns_series(self) -> pd.Series:
        """Get returns as a pandas Series with datetime index for QuantStats.

        Priority 1: Use EquitySnapshotActor MTM equity from cache (includes
                     unrealized PnL, avoids end-of-backtest equity jumps).
        Priority 2: Fall back to realized account equity.

        Returns:
            pd.Series with daily returns indexed by datetime
        """
        # Priority 1: MTM equity from EquitySnapshotActor (cached)
        if self._mtm_equity is not None and not self._mtm_equity.empty:
            mtm_returns = self._build_daily_returns_from_equity(self._mtm_equity)
            if not mtm_returns.empty:
                return mtm_returns

        # Priority 2: Realized account equity (cached)
        if self._realized_equity.empty:
            return pd.Series(dtype=float)
        return self._build_daily_returns_from_equity(self._realized_equity)

    def _get_realized_equity_series(self) -> pd.Series:
        """Build realized equity series from account report data."""
        from nautilus_trader.analysis import ReportProvider

        accounts = list(self.engine.cache.accounts())
        if not accounts:
            return pd.Series(dtype=float)

        account = accounts[0]
        account_df = ReportProvider.generate_account_report(account)

        if account_df.empty:
            return pd.Series(dtype=float)

        equity_col = None
        for col in ["total", "equity"]:
            if col in account_df.columns:
                equity_col = col
                break

        if equity_col is None:
            return pd.Series(dtype=float)

        equity = account_df[equity_col].astype(float)
        if not isinstance(equity.index, pd.DatetimeIndex):
            equity.index = pd.to_datetime(equity.index)

        return equity.sort_index()

    def _build_daily_returns_from_equity(self, equity: pd.Series) -> pd.Series:
        """Resample equity to daily frequency and calculate daily returns."""
        if equity.empty:
            return pd.Series(dtype=float)

        daily_equity = equity.resample("1D").last().dropna()

        if len(daily_equity) < 2:
            return pd.Series(dtype=float)

        returns = daily_equity.pct_change().dropna()
        returns = returns.replace([float("inf"), float("-inf")], float("nan")).dropna()
        return returns

    def _build_equity_from_actor_snapshots(self) -> pd.Series | None:
        """Read EquitySnapshotActor data from cache and build equity Series.

        Returns:
            pd.Series indexed by datetime with equity values, or None if
            no actor data available.
        """
        import pickle

        try:
            data = self.engine.cache.get(EQUITY_SNAPSHOTS_CACHE_KEY)
        except Exception:
            return None

        if not data:
            return None

        try:
            equity_points: list[tuple[int, float]] = pickle.loads(data)
        except Exception:
            return None

        if not equity_points:
            return None

        # Build Series from (ts_ns, equity) pairs
        timestamps = [pd.Timestamp(ts_ns, unit="ns", tz="UTC") for ts_ns, _ in equity_points]
        values = [val for _, val in equity_points]

        return pd.Series(values, index=pd.DatetimeIndex(timestamps)).sort_index()

    def _build_position_records_from_cache(self) -> list[dict]:
        """Build position records directly from engine cache.

        Eliminates CSV reading and string parsing by using Position objects.

        Returns:
            List of dicts with inst_id, side, value, ts_opened, ts_closed
        """
        from nautilus_trader.model.enums import OrderSide

        records: list[dict] = []
        all_positions = self._get_closed_positions()

        # Include still-open positions
        open_positions = list(self.engine.cache.positions_open())
        seen_ids = {p.id for p in all_positions}
        for p in open_positions:
            if p.id not in seen_ids:
                all_positions.append(p)

        for pos in all_positions:
            inst_id = str(pos.instrument_id)
            side = "LONG" if pos.entry == OrderSide.BUY else "SHORT"
            qty = float(pos.peak_qty)
            value = qty * pos.avg_px_open

            ts_opened = pd.Timestamp(pos.ts_opened, unit="ns", tz="UTC")
            if pos.ts_closed > 0:
                ts_closed = pd.Timestamp(pos.ts_closed, unit="ns", tz="UTC")
            else:
                ts_closed = pd.Timestamp.max.tz_localize("UTC")

            records.append({
                "inst_id": inst_id,
                "side": side,
                "value": value,
                "ts_opened": ts_opened,
                "ts_closed": ts_closed,
            })

        return records

    def _load_position_metadata(self) -> dict[str, dict]:
        """Load position metadata from engine cache."""
        import pickle

        try:
            metadata_bytes = self.engine.cache.get(POSITION_METADATA_CACHE_KEY)
            if metadata_bytes:
                return pickle.loads(metadata_bytes)
        except Exception as e:
            import warnings

            warnings.warn(f"Failed to read position metadata from cache: {e}")
        return {}

    def _build_metadata_index(self, position_metadata: dict[str, dict]) -> dict[str, dict]:
        """Build O(1) lookup index for position metadata.

        Handles both direct inst_id keys and closed position keys with
        format 'INST.VENUE:hour_count'.

        Args:
            position_metadata: Raw metadata dict from cache

        Returns:
            Dict mapping inst_id to metadata for O(1) lookup
        """
        metadata_by_inst: dict[str, dict] = {}
        for key, meta in position_metadata.items():
            metadata_by_inst[key] = meta
            base_id = key.split(":")[0] if ":" in key else key
            if base_id not in metadata_by_inst:
                metadata_by_inst[base_id] = meta
        return metadata_by_inst

    def generate_quantstats_reports(self) -> dict[str, Path]:
        """Generate QuantStats HTML report and/or PNG charts.

        Returns:
            Dict mapping report type to file path
        """
        qs_config = self.config.quantstats
        if not qs_config or not qs_config.enabled:
            return {}

        # Check if quantstats is available
        try:
            import quantstats as qs
        except ImportError:
            import warnings

            warnings.warn(
                "quantstats not installed. Skipping QuantStats report generation. "
                "Install with: pip install quantstats>=0.0.64"
            )
            return {}

        reports: dict[str, Path] = {}
        self.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            returns = self._get_returns_series()
            if returns.empty:
                import warnings

                warnings.warn("No returns data available for QuantStats report.")
                return {}

            # Extend pandas with quantstats methods
            qs.extend_pandas()

            # Generate HTML report if configured
            if "html" in qs_config.output_format:
                html_path = self._generate_quantstats_html(qs, returns, qs_config)
                if html_path:
                    reports["quantstats_html"] = html_path

            # Generate PNG charts if configured
            if "png" in qs_config.output_format:
                png_reports = self._generate_quantstats_charts(qs, returns, qs_config)
                reports.update(png_reports)

            return reports

        except Exception as e:
            raise BacktestReportError(f"Failed to generate QuantStats reports: {e}") from e

    def _generate_quantstats_html(
        self,
        qs: Any,
        returns: pd.Series,
        qs_config: Any,
    ) -> Path | None:
        """Generate QuantStats HTML report.

        Args:
            qs: QuantStats module
            returns: Returns series
            qs_config: QuantStats configuration

        Returns:
            Path to HTML file or None
        """
        html_path = self.output_dir / "quantstats_report.html"

        # Generate full HTML report
        qs.reports.html(
            returns,
            benchmark=qs_config.benchmark,
            title=qs_config.title,
            output=str(html_path),
        )

        return html_path

    def _generate_quantstats_charts(
        self,
        qs: Any,
        returns: pd.Series,
        qs_config: Any,
    ) -> dict[str, Path]:
        """Generate QuantStats PNG charts.

        Args:
            qs: QuantStats module
            returns: Returns series
            qs_config: QuantStats configuration

        Returns:
            Dict mapping chart name to file path
        """
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend for PNG generation
        import matplotlib.pyplot as plt

        charts_dir = self.output_dir / "quantstats_charts"
        charts_dir.mkdir(parents=True, exist_ok=True)

        reports: dict[str, Path] = {}

        # Map chart names to QuantStats plot functions
        chart_functions = {
            "returns": qs.plots.returns,
            "log_returns": qs.plots.log_returns,
            "yearly_returns": qs.plots.yearly_returns,
            "monthly_heatmap": qs.plots.monthly_heatmap,
            "drawdown": qs.plots.drawdown,
            "drawdowns_periods": qs.plots.drawdowns_periods,
            "rolling_sharpe": qs.plots.rolling_sharpe,
            "rolling_volatility": qs.plots.rolling_volatility,
            "rolling_beta": qs.plots.rolling_beta,
            "histogram": qs.plots.histogram,
            "daily_returns": qs.plots.daily_returns,
        }

        for chart_name in qs_config.charts:
            if chart_name not in chart_functions:
                continue

            try:
                chart_path = charts_dir / f"{chart_name}.png"
                plot_func = chart_functions[chart_name]

                # Some plots require benchmark
                if chart_name in ("rolling_beta",) and not qs_config.benchmark:
                    continue

                fig = plot_func(returns, show=False)
                if fig is not None:
                    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    reports[f"quantstats_{chart_name}"] = chart_path

            except Exception:
                # Skip charts that fail (some need specific data)
                continue

        return reports

    def generate_factor_values_csv(self) -> Path | None:
        """Generate CSV of per-bar, per-instrument factor values.

        Reads factor data from engine cache (written by FactorEngineActor).
        Supports two formats:
        - Parquet path (cache replay mode): reads parquet, melts to CSV
        - Serialized bytes (first-run mode): unpickles snapshots, flattens to CSV

        Output CSV columns: timestamp_ns, instrument_id, factor_name, value.

        Returns:
            Path to CSV file, or None if no factor data available.
        """
        try:
            data = self.engine.cache.get(FACTOR_VALUES_CACHE_KEY)
        except Exception:
            return None

        if not data:
            return None

        # Try parquet path first (cache replay mode)
        csv_path = self._try_factor_csv_from_parquet(data)
        if csv_path:
            return csv_path

        # Fallback: serialized snapshots (first-run mode)
        return self._try_factor_csv_from_snapshots(data)

    def _try_factor_csv_from_parquet(self, data: bytes) -> Path | None:
        """Try reading factor data from a parquet file path."""
        try:
            path = Path(data.decode("utf-8"))
            if not (path.is_file() and path.suffix == ".parquet"):
                return None
            df = pd.read_parquet(path).reset_index()
            id_vars = ["ts_event_ns", "instrument_id"]
            if not all(c in df.columns for c in id_vars):
                return None
            df_melted = df.melt(
                id_vars=id_vars,
                var_name="factor_name",
                value_name="value",
            ).rename(columns={"ts_event_ns": "timestamp_ns"})
            df_melted = df_melted.dropna(subset=["value"])
            if df_melted.empty:
                return None
            self.output_dir.mkdir(parents=True, exist_ok=True)
            csv_path = self.output_dir / "factor_values_report.csv"
            df_melted.to_csv(csv_path, index=False)
            return csv_path
        except Exception:
            return None

    def _try_factor_csv_from_snapshots(self, data: bytes) -> Path | None:
        """Try reading factor data from serialized snapshots (first-run)."""
        import pickle

        try:
            snapshots: list[tuple[int, dict[str, dict[str, float]]]] = (
                pickle.loads(data)  # noqa: S301 — trusted internal cache data
            )
        except Exception:
            return None

        if not snapshots:
            return None

        rows: list[dict] = []
        for ts_ns, factors in snapshots:
            for factor_name, instrument_values in factors.items():
                for instrument_id, value in instrument_values.items():
                    rows.append({
                        "timestamp_ns": ts_ns,
                        "instrument_id": instrument_id,
                        "factor_name": factor_name,
                        "value": value,
                    })

        if not rows:
            return None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        csv_path = self.output_dir / "factor_values_report.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    def generate_execution_report_csv(self) -> Path | None:
        """Generate CSV of per-order execution states from PostLimitExecAlgorithm.

        Reads pickled OrderExecutionState dict from engine cache and writes a
        flat CSV with one row per execution sequence.

        Returns:
            Path to CSV file, or None if no execution state data available.
        """
        try:
            data = self.engine.cache.get(EXECUTION_STATES_CACHE_KEY)
        except Exception:
            return None

        if not data:
            return None

        try:
            from nautilus_quants.execution.post_limit.state import decode_execution_states

            states = decode_execution_states(data)
        except Exception:
            try:
                import pickle

                states = pickle.loads(data)
            except Exception:
                return None

        if not states:
            return None

        from nautilus_trader.model.enums import OrderSide

        rows: list[dict] = []
        for state in states.values():
            filled_qty = float(state.filled_quantity) if state.filled_quantity is not None else 0.0
            total_qty = float(state.total_quantity)
            fill_ratio = filled_qty / total_qty if total_qty > 0 else 0.0
            elapsed_ms = (
                (state.completed_ns - state.created_ns) / 1_000_000
                if state.completed_ns > 0 and state.created_ns > 0
                else 0.0
            )

            # Compute avg_fill_px (VWAP) and slippage (Implementation Shortfall)
            fill_cost = getattr(state, "fill_cost", 0.0)
            avg_fill_px = fill_cost / filled_qty if filled_qty > 0 else 0.0

            if filled_qty > 0 and state.anchor_px > 0:
                if state.side == OrderSide.BUY:
                    slippage = avg_fill_px - state.anchor_px
                else:  # SELL
                    slippage = state.anchor_px - avg_fill_px
                slippage_bps = (slippage / state.anchor_px) * 10000
            else:
                slippage = 0.0
                slippage_bps = 0.0

            rows.append({
                "primary_order_id": str(state.primary_order_id),
                "instrument_id": str(state.instrument_id),
                "side": state.side.name,
                "total_quantity": total_qty,
                "filled_quantity": filled_qty,
                "fill_ratio": round(fill_ratio, 6),
                "anchor_px": state.anchor_px,
                "last_limit_price": state.last_limit_price,
                "avg_fill_px": round(avg_fill_px, 8),
                "slippage": round(slippage, 8),
                "slippage_bps": round(slippage_bps, 4),
                "reduce_only": state.reduce_only,
                "final_state": state.state.value,
                "chase_count": state.chase_count,
                "limit_orders_submitted": state.limit_orders_submitted,
                "used_market_fallback": state.used_market_fallback,
                "created_ns": state.created_ns,
                "completed_ns": state.completed_ns,
                "elapsed_ms": round(elapsed_ms, 3),
                "timeout_secs": state.timeout_secs,
                "max_chase_attempts": state.max_chase_attempts,
                "chase_step_ticks": state.chase_step_ticks,
                "post_only": state.post_only,
            })

        self.output_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        csv_path = self.output_dir / "execution_report.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    def generate_position_visualization(self) -> Path | None:
        """Generate ECharts position timeline visualization.

        Creates an interactive HTML chart showing equity curve with long/short
        position counts and detailed position lists on hover.

        Returns:
            Path to HTML file or None if not configured/generated
        """
        viz_config = self.config.position_viz
        if not viz_config or not viz_config.enabled:
            return None

        try:
            # Get position timeline data
            timeline_data = self._get_position_timeline_data(viz_config.interval)
            if not timeline_data:
                return None

            # Create output directory
            output_subdir = self.output_dir / viz_config.output_subdir
            output_subdir.mkdir(parents=True, exist_ok=True)

            html_path = output_subdir / "position_timeline.html"

            # Generate HTML
            html_content = self._generate_echarts_html(timeline_data, viz_config)
            html_path.write_text(html_content, encoding="utf-8")

            return html_path

        except Exception as e:
            raise BacktestReportError(f"Failed to generate position visualization: {e}") from e

    def _get_position_timeline_data(self, interval: str) -> list[dict]:
        """Extract position timeline data from engine cache.

        Uses event-driven sweep algorithm for O(E + T) complexity.
        Eliminates CSV reading, ast.literal_eval, and nested loops.

        Args:
            interval: Resampling interval (e.g., "4h", "1d")

        Returns:
            List of dicts with timestamp, equity, positions, and totals
        """
        # 1. Equity from cached property — no CSV
        equity = self._mtm_equity
        if equity is not None and not equity.empty:
            equity = equity[~equity.index.duplicated(keep="last")]
        else:
            equity = self._realized_equity
            if equity.empty:
                return []

        resampled_equity = equity.resample(interval).last().dropna()
        if resampled_equity.empty:
            return []

        # 2. Positions from engine cache — no CSV
        position_records = self._build_position_records_from_cache()

        # 3. Metadata: pre-build O(1) index
        position_metadata = self._load_position_metadata()
        metadata_by_inst = self._build_metadata_index(position_metadata)

        # 4. Event-driven sweep: O(E + T)
        events: list[tuple[pd.Timestamp, int, dict]] = []
        for rec in position_records:
            events.append((rec["ts_opened"], 1, rec))   # 1 = open
            events.append((rec["ts_closed"], 0, rec))   # 0 = close (sorted before open)
        events.sort(key=lambda e: (e[0], e[1]))

        active: dict[str, dict] = {}
        render_cache: dict[str, dict] = {}
        event_idx = 0
        timeline_data: list[dict] = []

        for ts, equity_val in resampled_equity.items():
            # Advance event pointer
            while event_idx < len(events) and events[event_idx][0] <= ts:
                _, action, rec = events[event_idx]
                inst_id = rec["inst_id"]
                if action == 1:
                    active[inst_id] = rec
                else:
                    active.pop(inst_id, None)
                    render_cache.pop(inst_id, None)
                event_idx += 1

            # Build position snapshot for current timestamp
            positions: dict[str, dict] = {}
            long_count = short_count = 0
            long_total_value = short_total_value = 0.0

            for inst_id, rec in active.items():
                symbol = inst_id.split(".")[0] if "." in inst_id else inst_id

                if inst_id not in render_cache:
                    meta = metadata_by_inst.get(inst_id)
                    position_info = {
                        "side": rec["side"],
                        "ts_opened": rec["ts_opened"].isoformat(),
                    }
                    render_cache[inst_id] = self.metadata_renderer.render_position(
                        symbol=symbol,
                        position_info=position_info,
                        metadata=meta,
                        timestamp_ns=int(ts.value),
                    )

                # Use current market value (quantity × last price); fall back to 0.0
                market_value = self._get_market_value(inst_id, ts) or 0.0

                # Shallow-copy to avoid mutating the render cache
                rendered = dict(render_cache[inst_id])
                rendered["value"] = market_value
                positions[symbol] = rendered

                if rec["side"] == "LONG":
                    long_total_value += market_value
                    long_count += 1
                else:
                    short_total_value += market_value
                    short_count += 1

            timeline_data.append({
                "timestamp": ts.isoformat(),
                "equity": float(equity_val),
                "long_count": long_count,
                "short_count": short_count,
                "positions": positions,
                "long_total_value": long_total_value,
                "short_total_value": short_total_value,
            })

        return timeline_data

    def _generate_echarts_html(self, timeline_data: list[dict], viz_config: "PositionVisualizationConfig") -> str:
        """Generate ECharts HTML content with bar chart and pie chart.

        Args:
            timeline_data: Position timeline data with positions dict containing value and side
            viz_config: Visualization configuration

        Returns:
            Complete HTML string
        """
        import json
        from importlib.resources import files
        from string import Template

        # Get column configuration from renderer
        column_config = self.metadata_renderer.get_column_config()

        # Prepare data for ECharts
        timestamps = [d["timestamp"] for d in timeline_data]
        equity_data = [d["equity"] for d in timeline_data]
        long_values = [d["long_total_value"] for d in timeline_data]
        short_values = [d["short_total_value"] for d in timeline_data]

        # Calculate statistics
        if equity_data:
            start_equity = equity_data[0]
            end_equity = equity_data[-1]
            pnl = end_equity - start_equity
            pnl_pct = (pnl / start_equity * 100) if start_equity else 0
            max_equity = max(equity_data)
            min_equity = min(equity_data)
        else:
            start_equity = end_equity = pnl = pnl_pct = max_equity = min_equity = 0

        # Two-tier JSON: chart-level arrays + sparse detail data
        chart_data = {
            "timestamps": timestamps,
            "equityData": [round(e, 2) for e in equity_data],
            "longValues": [round(v, 2) for v in long_values],
            "shortValues": [round(v, 2) for v in short_values],
            "columnConfig": column_config,
        }
        detail_data: dict[int, dict] = {}
        for i, d in enumerate(timeline_data):
            if d.get("positions"):
                compact_positions = {}
                for symbol, info in d["positions"].items():
                    compact_positions[symbol] = {
                        k: round(v, 2) if isinstance(v, float) else v
                        for k, v in info.items()
                    }
                detail_data[i] = {
                    "positions": compact_positions,
                    "long_count": d["long_count"],
                    "short_count": d["short_count"],
                    "long_total_value": round(d["long_total_value"], 2),
                    "short_total_value": round(d["short_total_value"], 2),
                }
        data_json = json.dumps({"chart": chart_data, "detail": detail_data})

        # Load template file
        template_path = files("nautilus_quants.backtest.templates").joinpath("position_timeline.html")
        template_content = template_path.read_text(encoding="utf-8")

        # Substitute template variables
        template = Template(template_content)
        html_content = template.safe_substitute(
            title=viz_config.title,
            chart_height=viz_config.chart_height,
            start_equity=f"{start_equity:,.2f}",
            end_equity=f"{end_equity:,.2f}",
            pnl=f"{pnl:+,.2f}",
            pnl_class="positive" if pnl >= 0 else "negative",
            pnl_pct=f"{pnl_pct:+.2f}%",
            pnl_pct_class="positive" if pnl_pct >= 0 else "negative",
            max_equity=f"{max_equity:,.2f}",
            min_equity=f"{min_equity:,.2f}",
            data_json=data_json,
        )

        return html_content
