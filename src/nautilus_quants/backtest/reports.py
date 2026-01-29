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

        # Generate position visualization if configured
        position_viz_path = self.generate_position_visualization()
        if position_viz_path:
            reports["position_viz"] = position_viz_path

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
                import warnings
                warnings.warn("No position data available for visualization.")
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
        """Extract position timeline data from account report.

        Args:
            interval: Resampling interval (e.g., "4h", "1d")

        Returns:
            List of dicts with timestamp, equity, long_positions, short_positions
        """
        import ast
        from nautilus_trader.analysis import ReportProvider

        # Get account data
        accounts = list(self.engine.cache.accounts())
        if not accounts:
            return []

        account = accounts[0]
        account_df = ReportProvider.generate_account_report(account)

        if account_df.empty:
            return []

        # Get positions report for entry direction lookup
        positions = list(self.engine.cache.positions())
        position_directions: dict[str, str] = {}
        if positions:
            positions_df = ReportProvider.generate_positions_report(positions)
            if not positions_df.empty and "entry" in positions_df.columns:
                for _, row in positions_df.iterrows():
                    inst_id = row.get("instrument_id", "")
                    entry = row.get("entry", "")
                    if inst_id and entry:
                        position_directions[inst_id] = entry

        # Ensure datetime index
        if not isinstance(account_df.index, pd.DatetimeIndex):
            account_df.index = pd.to_datetime(account_df.index)

        # Get equity column
        equity_col = None
        for col in ["total", "equity"]:
            if col in account_df.columns:
                equity_col = col
                break

        if equity_col is None:
            return []

        # Resample to specified interval
        resampled = account_df.resample(interval).last().dropna(subset=[equity_col])

        timeline_data = []
        for timestamp, row in resampled.iterrows():
            equity = float(row[equity_col])

            # Parse margins to get current positions
            long_positions = []
            short_positions = []

            margins_str = row.get("margins", "[]")
            if isinstance(margins_str, str) and margins_str.strip():
                try:
                    margins = ast.literal_eval(margins_str)
                    for margin in margins:
                        if isinstance(margin, dict):
                            inst_id = margin.get("instrument_id", "")
                            if inst_id:
                                # Determine direction from positions report
                                direction = position_directions.get(inst_id, "")
                                # Extract symbol without venue
                                symbol = inst_id.split(".")[0] if "." in inst_id else inst_id
                                if direction == "BUY":
                                    long_positions.append(symbol)
                                elif direction == "SELL":
                                    short_positions.append(symbol)
                                else:
                                    # Default: assume long if direction unknown
                                    long_positions.append(symbol)
                except (ValueError, SyntaxError):
                    pass

            timeline_data.append({
                "timestamp": timestamp.isoformat(),
                "equity": equity,
                "long_count": len(long_positions),
                "short_count": len(short_positions),
                "long_positions": long_positions,
                "short_positions": short_positions,
            })

        return timeline_data

    def _generate_echarts_html(self, timeline_data: list[dict], viz_config: "PositionVisualizationConfig") -> str:
        """Generate ECharts HTML content.

        Args:
            timeline_data: Position timeline data
            viz_config: Visualization configuration

        Returns:
            Complete HTML string
        """
        import json
        from nautilus_quants.backtest.config import PositionVisualizationConfig

        # Prepare data for ECharts
        timestamps = [d["timestamp"] for d in timeline_data]
        equity_data = [d["equity"] for d in timeline_data]
        long_counts = [d["long_count"] for d in timeline_data]
        short_counts = [-d["short_count"] for d in timeline_data]  # Negative for visual distinction
        long_positions = [d["long_positions"] for d in timeline_data]
        short_positions = [d["short_positions"] for d in timeline_data]

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

        html_template = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{viz_config.title}</title>
    <script src="https://cdn.jsdelivr.net/npm/echarts@6.0.0/dist/echarts.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: Arial, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #ffffff;
            color: #333333;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 20px;
            color: #333333;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 6px;
            padding: 15px;
        }}
        .stat-label {{
            font-size: 12px;
            color: #666666;
            margin-bottom: 5px;
        }}
        .stat-value {{
            font-size: 18px;
            font-weight: 600;
            color: #333333;
        }}
        .stat-value.positive {{
            color: #28a745;
        }}
        .stat-value.negative {{
            color: #dc3545;
        }}
        #chart-container {{
            width: 100%;
            height: {viz_config.chart_height}px;
            background: #ffffff;
            border: 1px solid #e9ecef;
            border-radius: 6px;
        }}
        .position-table {{
            margin-top: 20px;
            width: 100%;
            border-collapse: collapse;
        }}
        .position-table th,
        .position-table td {{
            padding: 10px 15px;
            text-align: left;
            border-bottom: 1px solid #e9ecef;
        }}
        .position-table th {{
            background: #f8f9fa;
            font-weight: 600;
            font-size: 12px;
            color: #666666;
            text-transform: uppercase;
        }}
        .position-table td {{
            font-size: 14px;
        }}
        .long-tag {{
            display: inline-block;
            background: #d4edda;
            color: #155724;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
            margin: 2px;
        }}
        .short-tag {{
            display: inline-block;
            background: #f8d7da;
            color: #721c24;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
            margin: 2px;
        }}
        #selected-info {{
            margin-top: 20px;
            padding: 15px;
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 6px;
            display: none;
        }}
        #selected-info h3 {{
            font-size: 14px;
            margin-bottom: 10px;
            color: #666666;
        }}
        .footer {{
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #e9ecef;
            font-size: 12px;
            color: #999999;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{viz_config.title}</h1>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Starting Equity</div>
                <div class="stat-value">{start_equity:,.2f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Ending Equity</div>
                <div class="stat-value">{end_equity:,.2f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total PnL</div>
                <div class="stat-value {"positive" if pnl >= 0 else "negative"}">{pnl:+,.2f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Return</div>
                <div class="stat-value {"positive" if pnl_pct >= 0 else "negative"}">{pnl_pct:+.2f}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Max Equity</div>
                <div class="stat-value">{max_equity:,.2f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Min Equity</div>
                <div class="stat-value">{min_equity:,.2f}</div>
            </div>
        </div>

        <div id="chart-container"></div>

        <div id="selected-info">
            <h3>Selected Point Details</h3>
            <div id="selected-content"></div>
        </div>

        <div class="footer">
            Generated by Nautilus Quants &bull; Powered by ECharts
        </div>
    </div>

    <script>
        // Data
        const timestamps = {json.dumps(timestamps)};
        const equityData = {json.dumps(equity_data)};
        const longCounts = {json.dumps(long_counts)};
        const shortCounts = {json.dumps(short_counts)};
        const longPositions = {json.dumps(long_positions)};
        const shortPositions = {json.dumps(short_positions)};

        // Initialize chart
        const chart = echarts.init(document.getElementById('chart-container'));

        const option = {{
            tooltip: {{
                trigger: 'axis',
                axisPointer: {{
                    type: 'cross'
                }},
                formatter: function(params) {{
                    const idx = params[0].dataIndex;
                    const time = new Date(timestamps[idx]).toLocaleString();
                    let html = '<div style="font-weight:600;margin-bottom:8px;">' + time + '</div>';

                    params.forEach(function(item) {{
                        if (item.seriesName === 'Equity') {{
                            html += '<div>' + item.marker + ' Equity: <b>' + item.value.toLocaleString(undefined, {{minimumFractionDigits: 2}}) + '</b></div>';
                        }} else if (item.seriesName === 'Long') {{
                            html += '<div>' + item.marker + ' Long: <b>' + item.value + '</b></div>';
                        }} else if (item.seriesName === 'Short') {{
                            html += '<div>' + item.marker + ' Short: <b>' + Math.abs(item.value) + '</b></div>';
                        }}
                    }});

                    if (longPositions[idx] && longPositions[idx].length > 0) {{
                        html += '<div style="margin-top:8px;"><span style="color:#28a745;">Long:</span> ' + longPositions[idx].join(', ') + '</div>';
                    }}
                    if (shortPositions[idx] && shortPositions[idx].length > 0) {{
                        html += '<div><span style="color:#dc3545;">Short:</span> ' + shortPositions[idx].join(', ') + '</div>';
                    }}

                    return html;
                }}
            }},
            legend: {{
                data: ['Equity', 'Long', 'Short'],
                top: 10
            }},
            grid: {{
                left: '3%',
                right: '4%',
                bottom: '15%',
                top: '60',
                containLabel: true
            }},
            xAxis: {{
                type: 'category',
                data: timestamps.map(t => new Date(t).toLocaleDateString()),
                axisLabel: {{
                    rotate: 45
                }}
            }},
            yAxis: [
                {{
                    type: 'value',
                    name: 'Equity',
                    position: 'left',
                    axisLabel: {{
                        formatter: function(value) {{
                            return value.toLocaleString();
                        }}
                    }}
                }},
                {{
                    type: 'value',
                    name: 'Positions',
                    position: 'right',
                    axisLabel: {{
                        formatter: function(value) {{
                            return Math.abs(value);
                        }}
                    }}
                }}
            ],
            dataZoom: [
                {{
                    type: 'inside',
                    start: 0,
                    end: 100
                }},
                {{
                    type: 'slider',
                    start: 0,
                    end: 100,
                    bottom: 10
                }}
            ],
            series: [
                {{
                    name: 'Equity',
                    type: 'line',
                    data: equityData,
                    smooth: true,
                    lineStyle: {{
                        width: 2,
                        color: '#1890ff'
                    }},
                    areaStyle: {{
                        color: {{
                            type: 'linear',
                            x: 0,
                            y: 0,
                            x2: 0,
                            y2: 1,
                            colorStops: [
                                {{ offset: 0, color: 'rgba(24, 144, 255, 0.3)' }},
                                {{ offset: 1, color: 'rgba(24, 144, 255, 0.05)' }}
                            ]
                        }}
                    }},
                    itemStyle: {{
                        color: '#1890ff'
                    }}
                }},
                {{
                    name: 'Long',
                    type: 'bar',
                    yAxisIndex: 1,
                    data: longCounts,
                    itemStyle: {{
                        color: '#28a745'
                    }},
                    barMaxWidth: 20
                }},
                {{
                    name: 'Short',
                    type: 'bar',
                    yAxisIndex: 1,
                    data: shortCounts,
                    itemStyle: {{
                        color: '#dc3545'
                    }},
                    barMaxWidth: 20
                }}
            ]
        }};

        chart.setOption(option);

        // Handle click events
        chart.on('click', function(params) {{
            const idx = params.dataIndex;
            const selectedDiv = document.getElementById('selected-info');
            const contentDiv = document.getElementById('selected-content');

            const time = new Date(timestamps[idx]).toLocaleString();
            const equity = equityData[idx].toLocaleString(undefined, {{minimumFractionDigits: 2}});

            let html = '<table class="position-table">';
            html += '<tr><th>Time</th><th>Equity</th><th>Long Positions</th><th>Short Positions</th></tr>';
            html += '<tr>';
            html += '<td>' + time + '</td>';
            html += '<td>' + equity + '</td>';
            html += '<td>';
            if (longPositions[idx] && longPositions[idx].length > 0) {{
                longPositions[idx].forEach(p => {{
                    html += '<span class="long-tag">' + p + '</span>';
                }});
            }} else {{
                html += '-';
            }}
            html += '</td>';
            html += '<td>';
            if (shortPositions[idx] && shortPositions[idx].length > 0) {{
                shortPositions[idx].forEach(p => {{
                    html += '<span class="short-tag">' + p + '</span>';
                }});
            }} else {{
                html += '-';
            }}
            html += '</td>';
            html += '</tr></table>';

            contentDiv.innerHTML = html;
            selectedDiv.style.display = 'block';
        }});

        // Responsive
        window.addEventListener('resize', function() {{
            chart.resize();
        }});
    </script>
</body>
</html>'''

        return html_template
