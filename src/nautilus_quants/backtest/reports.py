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
                PortfolioAnalyzer,
                TearsheetConfig as NautilusTearsheetConfig,
                create_tearsheet_from_stats,
            )
            from nautilus_trader.core.datetime import unix_nanos_to_dt

            tearsheet_path = self.output_dir / "tearsheet.html"

            # Get only closed positions with valid timestamps
            closed_positions = self._get_closed_positions()

            # Create a fresh analyzer with only closed positions
            analyzer = PortfolioAnalyzer()
            if closed_positions:
                analyzer.add_positions(closed_positions)

            # Get returns with proper datetime index
            returns = analyzer.returns()

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
