"""Unit tests for position visualization report generation."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from nautilus_quants.backtest.config import PositionVisualizationConfig, ReportConfig
from nautilus_quants.backtest.reports import ReportGenerator


class TestPositionVisualizationConfig:
    """Tests for PositionVisualizationConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = PositionVisualizationConfig()

        assert config.enabled is True
        assert config.title == "Position Timeline"
        assert config.output_subdir == "echarts"
        assert config.chart_height == 500
        assert config.interval == "4h"

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = PositionVisualizationConfig(
            enabled=False,
            title="Custom Timeline",
            output_subdir="custom_charts",
            chart_height=600,
            interval="1d",
        )

        assert config.enabled is False
        assert config.title == "Custom Timeline"
        assert config.output_subdir == "custom_charts"
        assert config.chart_height == 600
        assert config.interval == "1d"

    def test_frozen_dataclass(self) -> None:
        """Test that config is immutable."""
        config = PositionVisualizationConfig()

        with pytest.raises(AttributeError):
            config.enabled = False  # type: ignore


class TestReportConfigWithPositionViz:
    """Tests for ReportConfig with position visualization integration."""

    def test_default_position_viz_is_none(self) -> None:
        """Test that position_viz is None by default."""
        config = ReportConfig()
        assert config.position_viz is None

    def test_with_position_viz_config(self) -> None:
        """Test ReportConfig with PositionVisualizationConfig."""
        viz_config = PositionVisualizationConfig(enabled=True)
        config = ReportConfig(position_viz=viz_config)

        assert config.position_viz is not None
        assert config.position_viz.enabled is True


class TestPositionVisualizationGeneration:
    """Tests for position visualization generation in ReportGenerator."""

    @pytest.fixture
    def mock_engine(self) -> MagicMock:
        """Create mock BacktestEngine."""
        engine = MagicMock()
        engine.cache.orders.return_value = []
        engine.cache.positions.return_value = []
        engine.cache.positions_closed.return_value = []
        engine.cache.position_snapshots.return_value = []
        engine.cache.accounts.return_value = []
        engine.portfolio.analyzer.get_performance_stats_pnls.return_value = {}
        engine.portfolio.analyzer.get_performance_stats_returns.return_value = {}
        engine.portfolio.analyzer.get_performance_stats_general.return_value = {}
        return engine

    @pytest.fixture
    def position_viz_config(self) -> PositionVisualizationConfig:
        """Create position visualization configuration."""
        return PositionVisualizationConfig(
            enabled=True,
            title="Test Position Timeline",
            output_subdir="echarts",
            chart_height=400,
            interval="4h",
        )

    @pytest.fixture
    def report_config_with_viz(
        self, position_viz_config: PositionVisualizationConfig
    ) -> ReportConfig:
        """Create report configuration with position visualization."""
        return ReportConfig(
            output_dir="logs/backtest_runs",
            formats=["csv"],
            position_viz=position_viz_config,
        )

    def test_generate_position_visualization_skipped_when_disabled(
        self, mock_engine: MagicMock, tmp_path: Path
    ) -> None:
        """Test generate_position_visualization skipped when disabled."""
        config = ReportConfig(
            position_viz=PositionVisualizationConfig(enabled=False),
        )

        generator = ReportGenerator(
            engine=mock_engine,
            output_dir=tmp_path,
            config=config,
        )

        result = generator.generate_position_visualization()

        assert result is None

    def test_generate_position_visualization_skipped_when_no_config(
        self, mock_engine: MagicMock, tmp_path: Path
    ) -> None:
        """Test generate_position_visualization skipped when no config."""
        config = ReportConfig(position_viz=None)

        generator = ReportGenerator(
            engine=mock_engine,
            output_dir=tmp_path,
            config=config,
        )

        result = generator.generate_position_visualization()

        assert result is None

    def test_generate_position_visualization_handles_empty_data(
        self,
        mock_engine: MagicMock,
        tmp_path: Path,
        report_config_with_viz: ReportConfig,
    ) -> None:
        """Test generate_position_visualization handles empty data gracefully."""
        generator = ReportGenerator(
            engine=mock_engine,
            output_dir=tmp_path,
            config=report_config_with_viz,
        )

        # Mock _get_position_timeline_data to return empty list
        with patch.object(generator, "_get_position_timeline_data") as mock_data:
            mock_data.return_value = []

            result = generator.generate_position_visualization()

        # Should return None for empty data
        assert result is None

    def test_generate_position_visualization_creates_file(
        self,
        mock_engine: MagicMock,
        tmp_path: Path,
        position_viz_config: PositionVisualizationConfig,
    ) -> None:
        """Test generate_position_visualization creates HTML file."""
        config = ReportConfig(position_viz=position_viz_config)

        generator = ReportGenerator(
            engine=mock_engine,
            output_dir=tmp_path,
            config=config,
        )

        # Mock _get_position_timeline_data to return sample data with positions dict
        mock_data = [
            {
                "timestamp": "2024-01-01T00:00:00",
                "equity": 100000.0,
                "long_count": 3,
                "short_count": 2,
                "positions": {
                    "BTCUSDT": {"value": 5000.0, "side": "LONG", "rank": 0, "ts_opened": "2024-01-01T00:00:00"},
                    "ETHUSDT": {"value": 3000.0, "side": "LONG", "rank": 1, "ts_opened": "2024-01-01T00:00:00"},
                    "SOLUSDT": {"value": 2000.0, "side": "LONG", "rank": 2, "ts_opened": "2024-01-01T00:00:00"},
                    "DOGEUSDT": {"value": 1500.0, "side": "SHORT", "rank": 33, "ts_opened": "2024-01-01T00:00:00"},
                    "SHIBUSDT": {"value": 500.0, "side": "SHORT", "rank": 34, "ts_opened": "2024-01-01T00:00:00"},
                },
                "long_total_value": 10000.0,
                "short_total_value": 2000.0,
            },
            {
                "timestamp": "2024-01-01T04:00:00",
                "equity": 100500.0,
                "long_count": 2,
                "short_count": 3,
                "positions": {
                    "BTCUSDT": {"value": 5000.0, "side": "LONG", "rank": 0, "ts_opened": "2024-01-01T00:00:00"},
                    "ETHUSDT": {"value": 3000.0, "side": "LONG", "rank": 1, "ts_opened": "2024-01-01T00:00:00"},
                    "DOGEUSDT": {"value": 1500.0, "side": "SHORT", "rank": 32, "ts_opened": "2024-01-01T00:00:00"},
                    "SHIBUSDT": {"value": 500.0, "side": "SHORT", "rank": 33, "ts_opened": "2024-01-01T00:00:00"},
                    "PEPEUSDT": {"value": 1000.0, "side": "SHORT", "rank": 34, "ts_opened": "2024-01-01T04:00:00"},
                },
                "long_total_value": 8000.0,
                "short_total_value": 3000.0,
            },
        ]

        with patch.object(generator, "_get_position_timeline_data") as mock_timeline:
            mock_timeline.return_value = mock_data

            result = generator.generate_position_visualization()

        assert result is not None
        assert result.exists()
        assert result.name == "position_timeline.html"
        assert (tmp_path / "echarts").exists()

    def test_generate_position_visualization_html_content(
        self,
        mock_engine: MagicMock,
        tmp_path: Path,
        position_viz_config: PositionVisualizationConfig,
    ) -> None:
        """Test generated HTML contains expected content."""
        config = ReportConfig(position_viz=position_viz_config)

        generator = ReportGenerator(
            engine=mock_engine,
            output_dir=tmp_path,
            config=config,
        )

        mock_data = [
            {
                "timestamp": "2024-01-01T00:00:00",
                "equity": 100000.0,
                "long_count": 3,
                "short_count": 2,
                "positions": {
                    "BTCUSDT": {"value": 5000.0, "side": "LONG", "rank": 0, "ts_opened": "2024-01-01T00:00:00"},
                    "ETHUSDT": {"value": 3000.0, "side": "LONG", "rank": 1, "ts_opened": "2024-01-01T00:00:00"},
                    "SOLUSDT": {"value": 2000.0, "side": "LONG", "rank": 2, "ts_opened": "2024-01-01T00:00:00"},
                    "DOGEUSDT": {"value": 1500.0, "side": "SHORT", "rank": 33, "ts_opened": "2024-01-01T00:00:00"},
                    "SHIBUSDT": {"value": 500.0, "side": "SHORT", "rank": 34, "ts_opened": "2024-01-01T00:00:00"},
                },
                "long_total_value": 10000.0,
                "short_total_value": 2000.0,
            },
        ]

        with patch.object(generator, "_get_position_timeline_data") as mock_timeline:
            mock_timeline.return_value = mock_data

            result = generator.generate_position_visualization()

        assert result is not None
        html_content = result.read_text()

        # Check for ECharts CDN
        assert "echarts" in html_content
        # Check for title
        assert position_viz_config.title in html_content
        # Check for chart containers
        assert "bar-chart" in html_content
        assert "pie-chart" in html_content
        # Check for data
        assert "100000.0" in html_content or "100,000" in html_content

    def test_generate_all_includes_position_viz(
        self,
        mock_engine: MagicMock,
        tmp_path: Path,
        report_config_with_viz: ReportConfig,
    ) -> None:
        """Test generate_all includes position visualization reports."""
        generator = ReportGenerator(
            engine=mock_engine,
            output_dir=tmp_path,
            config=report_config_with_viz,
        )

        with patch.object(generator, "generate_csv_reports") as mock_csv:
            mock_csv.return_value = {}
            with patch.object(generator, "generate_tearsheet") as mock_tearsheet:
                mock_tearsheet.return_value = None
                with patch.object(
                    generator, "generate_quantstats_reports"
                ) as mock_quantstats:
                    mock_quantstats.return_value = {}
                    with patch.object(
                        generator, "generate_position_visualization"
                    ) as mock_viz:
                        mock_viz.return_value = tmp_path / "echarts" / "position_timeline.html"

                        reports = generator.generate_all()

        mock_viz.assert_called_once()
        assert "position_viz" in reports


class TestPositionTimelineData:
    """Tests for position timeline data extraction."""

    @pytest.fixture
    def mock_engine(self) -> MagicMock:
        """Create mock BacktestEngine."""
        engine = MagicMock()
        engine.cache.positions.return_value = []
        return engine

    def test_get_position_timeline_data_with_valid_data(
        self, mock_engine: MagicMock, tmp_path: Path
    ) -> None:
        """Test _get_position_timeline_data with valid CSV data."""
        config = ReportConfig(
            position_viz=PositionVisualizationConfig(enabled=True)
        )

        generator = ReportGenerator(
            engine=mock_engine,
            output_dir=tmp_path,
            config=config,
        )

        # Create account_report.csv with margins data
        account_data = pd.DataFrame(
            {
                "total": [100000.0, 100100.0, 100050.0, 100200.0],
                "margins": [
                    "[]",
                    "[{'type': 'MarginBalance', 'instrument_id': 'BTCUSDT.BINANCE'}]",
                    "[{'type': 'MarginBalance', 'instrument_id': 'BTCUSDT.BINANCE'}, {'type': 'MarginBalance', 'instrument_id': 'ETHUSDT.BINANCE'}]",
                    "[]",
                ],
            },
            index=pd.date_range("2024-01-01", periods=4, freq="4h"),
        )
        account_data.to_csv(tmp_path / "account_report.csv")

        # Create positions_report.csv for direction and value lookup
        positions_data = pd.DataFrame({
            "instrument_id": ["BTCUSDT.BINANCE", "ETHUSDT.BINANCE"],
            "entry": ["BUY", "SELL"],
            "quantity": [0.0, 0.0],  # closed positions
            "peak_qty": [0.1, 1.0],  # use peak_qty for value calculation
            "avg_px_open": [50000.0, 3000.0],  # for value: 0.1*50000=5000, 1.0*3000=3000
        })
        positions_data.to_csv(tmp_path / "positions_report.csv")

        result = generator._get_position_timeline_data("4h")

        assert isinstance(result, list)
        assert len(result) == 4
        assert all("timestamp" in item for item in result)
        assert all("equity" in item for item in result)
        assert all("long_count" in item for item in result)
        assert all("short_count" in item for item in result)
        assert all("positions" in item for item in result)
        assert all("long_total_value" in item for item in result)
        assert all("short_total_value" in item for item in result)

    def test_get_position_timeline_data_empty_account(
        self, mock_engine: MagicMock, tmp_path: Path
    ) -> None:
        """Test _get_position_timeline_data with no account data."""
        config = ReportConfig(
            position_viz=PositionVisualizationConfig(enabled=True)
        )

        generator = ReportGenerator(
            engine=mock_engine,
            output_dir=tmp_path,
            config=config,
        )

        # No CSV file exists
        result = generator._get_position_timeline_data("4h")

        assert result == []
