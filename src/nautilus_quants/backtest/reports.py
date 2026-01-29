"""Report generation for backtest module."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from nautilus_quants.backtest.exceptions import BacktestReportError

if TYPE_CHECKING:
    from nautilus_trader.backtest.engine import BacktestEngine
    from nautilus_trader.model.position import Position

    from nautilus_quants.backtest.config import ReportConfig


class ReportGenerator:
    """Generates all backtest reports."""

    def __init__(
        self,
        engine: "BacktestEngine",
        output_dir: Path,
        config: "ReportConfig",
    ) -> None:
        """Initialize report generator.

        Args:
            engine: BacktestEngine after execution
            output_dir: Directory for output files
            config: Report configuration
        """
        self.engine = engine
        self.output_dir = Path(output_dir)
        self.config = config

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
            if "Sharpe Ratio" in combined:
                combined["sharpe_ratio"] = combined["Sharpe Ratio"]
            if "Max Drawdown" in combined:
                combined["max_drawdown"] = combined["Max Drawdown"]
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

            # Create tearsheet config
            config = NautilusTearsheetConfig(
                title=tearsheet_config.title,
                theme=tearsheet_config.theme,
                height=tearsheet_config.height,
                show_logo=tearsheet_config.show_logo,
                include_benchmark=tearsheet_config.include_benchmark,
                charts=tearsheet_config.charts,
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

        Calculates daily returns from account equity changes instead of summing
        position returns. This is more accurate for HEDGING mode venues with
        multiple concurrent positions closing at the same timestamp.

        Returns:
            pd.Series with daily returns indexed by datetime
        """
        from nautilus_trader.analysis import ReportProvider

        # Get account equity data
        accounts = list(self.engine.cache.accounts())
        if not accounts:
            return pd.Series(dtype=float)

        account = accounts[0]
        account_df = ReportProvider.generate_account_report(account)

        if account_df.empty:
            return pd.Series(dtype=float)

        # Get equity column - try 'total' first (common in backtest reports),
        # then fall back to 'equity' if available
        equity_col = None
        for col in ["total", "equity"]:
            if col in account_df.columns:
                equity_col = col
                break

        if equity_col is None:
            return pd.Series(dtype=float)

        # Get equity series
        equity = account_df[equity_col].astype(float)

        # Ensure datetime index
        if not isinstance(equity.index, pd.DatetimeIndex):
            equity.index = pd.to_datetime(equity.index)

        # Resample to daily (take last value of each day)
        daily_equity = equity.resample("1D").last().dropna()

        if len(daily_equity) < 2:
            return pd.Series(dtype=float)

        # Calculate daily returns = (equity_t - equity_{t-1}) / equity_{t-1}
        daily_returns = daily_equity.pct_change().dropna()

        return daily_returns

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
