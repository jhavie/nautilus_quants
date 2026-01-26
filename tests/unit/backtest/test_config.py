"""Unit tests for backtest configuration.

Tests for retained project-specific configuration classes:
- TearsheetConfig
- ReportConfig
- BacktestResult
"""

import pytest

from nautilus_quants.backtest.config import (
    BacktestResult,
    ReportConfig,
    TearsheetConfig,
)


class TestReportConfig:
    """Tests for ReportConfig dataclass."""

    def test_creates_with_defaults(self) -> None:
        """Test creation with default values."""
        config = ReportConfig()
        assert config.output_dir == "logs/backtest_runs"
        assert config.formats == ["csv", "html"]
        assert config.tearsheet is None

    def test_creates_with_tearsheet(self) -> None:
        """Test creation with tearsheet config."""
        tearsheet = TearsheetConfig(title="Custom Report")
        config = ReportConfig(tearsheet=tearsheet)
        assert config.tearsheet is not None
        assert config.tearsheet.title == "Custom Report"


class TestTearsheetConfig:
    """Tests for TearsheetConfig dataclass."""

    def test_creates_with_defaults(self) -> None:
        """Test creation with default values."""
        config = TearsheetConfig()
        assert config.enabled is True
        assert config.title == "Backtest Results"
        assert config.theme == "plotly_dark"
        assert "equity" in config.charts

    def test_creates_with_custom_values(self) -> None:
        """Test creation with custom values."""
        config = TearsheetConfig(
            enabled=False,
            title="My Backtest",
            theme="plotly_white",
        )
        assert config.enabled is False
        assert config.title == "My Backtest"
        assert config.theme == "plotly_white"


class TestBacktestResult:
    """Tests for BacktestResult dataclass."""

    def test_creates_with_required_fields(self, tmp_path) -> None:
        """Test creation with required fields."""
        from datetime import datetime
        from pathlib import Path

        result = BacktestResult(
            run_id="20250101_120000",
            success=True,
            output_dir=tmp_path,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=10.5,
            statistics={"total_pnl": 1000.0},
            reports={"csv": tmp_path / "report.csv"},
        )
        assert result.run_id == "20250101_120000"
        assert result.success is True
        assert result.total_pnl == 1000.0

    def test_properties_return_defaults(self, tmp_path) -> None:
        """Test property accessors return defaults for missing stats."""
        from datetime import datetime

        result = BacktestResult(
            run_id="test",
            success=True,
            output_dir=tmp_path,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=1.0,
            statistics={},
            reports={},
        )
        assert result.total_pnl == 0.0
        assert result.total_pnl_pct == 0.0
        assert result.sharpe_ratio == 0.0
        assert result.max_drawdown == 0.0
        assert result.win_rate == 0.0
