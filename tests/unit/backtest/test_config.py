"""Unit tests for backtest configuration."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from nautilus_quants.backtest.config import (
    BacktestConfig,
    BacktestDataConfig,
    FeeModelConfig,
    FillModelConfig,
    LatencyModelConfig,
    LoggingSettings,
    ReportConfig,
    StrategyConfig,
    TearsheetConfig,
    VenueConfig,
)
from nautilus_quants.backtest.exceptions import BacktestConfigError


class TestStrategyConfig:
    """Tests for StrategyConfig dataclass."""

    def test_creates_with_required_fields(self) -> None:
        """Test creation with required fields."""
        config = StrategyConfig(type="breakout", instrument_id="BTCUSDT")
        assert config.type == "breakout"
        assert config.instrument_id == "BTCUSDT"
        assert config.interval == "1h"  # Default interval
        assert config.params == {}

    def test_creates_with_interval(self) -> None:
        """Test creation with custom interval."""
        config = StrategyConfig(
            type="breakout",
            instrument_id="BTCUSDT",
            interval="4h",
        )
        assert config.interval == "4h"

    def test_creates_with_params(self) -> None:
        """Test creation with strategy params."""
        config = StrategyConfig(
            type="breakout",
            instrument_id="BTCUSDT",
            params={"period": 60, "threshold": 0.01},
        )
        assert config.params["period"] == 60
        assert config.params["threshold"] == 0.01


class TestBacktestDataConfig:
    """Tests for BacktestDataConfig dataclass."""

    def test_creates_with_required_fields(self) -> None:
        """Test creation with required fields."""
        config = BacktestDataConfig(
            start_date="2025-01-01",
            end_date="2025-12-31",
            catalog_path="/path/to/catalog",
        )
        assert config.start_date == "2025-01-01"
        assert config.end_date == "2025-12-31"
        assert config.bar_spec == "1m"  # Default bar spec
        assert config.instruments == []  # Default empty list
        assert config.warmup_days == 0

    def test_creates_with_instruments_list(self) -> None:
        """Test creation with instruments list."""
        config = BacktestDataConfig(
            start_date="2025-01-01",
            end_date="2025-12-31",
            catalog_path="/path/to/catalog",
            bar_spec="1m",
            instruments=["BTCUSDT", "ETHUSDT"],
        )
        assert config.instruments == ["BTCUSDT", "ETHUSDT"]

    def test_defaults_are_applied(self) -> None:
        """Test default values are applied."""
        config = BacktestDataConfig(
            start_date="2025-01-01",
            end_date="2025-12-31",
            catalog_path="/path/to/catalog",
        )
        assert config.bar_spec == "1m"  # Now has default
        assert config.catalog_path == "/path/to/catalog"
        assert config.warmup_days == 0


class TestVenueConfig:
    """Tests for VenueConfig dataclass."""

    def test_creates_with_defaults(self) -> None:
        """Test creation with default values."""
        config = VenueConfig()
        assert config.name == "BINANCE"
        assert config.oms_type == "NETTING"
        assert config.account_type == "MARGIN"
        assert config.base_currency == "USDT"
        assert config.starting_balance == "100000 USDT"
        assert config.default_leverage == 1

    def test_creates_with_nested_models(self) -> None:
        """Test creation with nested model configs."""
        config = VenueConfig(
            fill_model=FillModelConfig(prob_fill_on_limit=0.9),
            fee_model=FeeModelConfig(maker_fee=0.0001),
            latency_model=LatencyModelConfig(base_latency_ms=50),
        )
        assert config.fill_model is not None
        assert config.fill_model.prob_fill_on_limit == 0.9
        assert config.fee_model is not None
        assert config.fee_model.maker_fee == 0.0001
        assert config.latency_model is not None
        assert config.latency_model.base_latency_ms == 50


class TestFillModelConfig:
    """Tests for FillModelConfig dataclass."""

    def test_creates_with_defaults(self) -> None:
        """Test creation with default values."""
        config = FillModelConfig()
        assert config.prob_fill_on_limit == 1.0
        assert config.prob_fill_on_stop == 1.0
        assert config.prob_slippage == 0.0
        assert config.random_seed is None


class TestFeeModelConfig:
    """Tests for FeeModelConfig dataclass."""

    def test_creates_with_defaults(self) -> None:
        """Test creation with default values."""
        config = FeeModelConfig()
        assert config.type == "maker_taker"
        assert config.maker_fee == 0.0002
        assert config.taker_fee == 0.0004


class TestLatencyModelConfig:
    """Tests for LatencyModelConfig dataclass."""

    def test_creates_with_defaults(self) -> None:
        """Test creation with default values."""
        config = LatencyModelConfig()
        assert config.base_latency_ms == 0
        assert config.insert_latency_ms == 0


class TestReportConfig:
    """Tests for ReportConfig dataclass."""

    def test_creates_with_defaults(self) -> None:
        """Test creation with default values."""
        config = ReportConfig()
        assert config.output_dir == "logs/backtest_runs"
        assert config.formats == ["csv", "html"]
        assert config.tearsheet is None


class TestTearsheetConfig:
    """Tests for TearsheetConfig dataclass."""

    def test_creates_with_defaults(self) -> None:
        """Test creation with default values."""
        config = TearsheetConfig()
        assert config.enabled is True
        assert config.title == "Backtest Results"
        assert config.theme == "plotly_dark"
        assert "equity" in config.charts


class TestLoggingSettings:
    """Tests for LoggingSettings dataclass."""

    def test_creates_with_defaults(self) -> None:
        """Test creation with default values."""
        config = LoggingSettings()
        assert config.level == "INFO"
        assert config.log_to_file is True
        assert config.bypass_logging is False


class TestBacktestConfig:
    """Tests for BacktestConfig dataclass."""

    @pytest.fixture
    def valid_config_dict(self) -> dict[str, Any]:
        """Create valid configuration dictionary."""
        return {
            "strategy": {
                "type": "breakout",
                "instrument_id": "BTCUSDT",
                "interval": "1h",
                "params": {"period": 60},
            },
            "backtest": {
                "start_date": "2025-01-01",
                "end_date": "2025-12-31",
                "bar_spec": "1m",
                "instruments": ["BTCUSDT"],
                "catalog_path": "/path/to/catalog",
            },
            "venue": {
                "name": "BINANCE",
                "starting_balance": "100000 USDT",
            },
            "report": {
                "output_dir": "logs/backtest_runs",
            },
        }

    def test_from_dict_creates_config(self, valid_config_dict: dict[str, Any]) -> None:
        """Test from_dict creates config successfully."""
        config = BacktestConfig.from_dict(valid_config_dict)

        assert config.strategy.type == "breakout"
        assert config.strategy.instrument_id == "BTCUSDT"
        assert config.backtest.start_date == "2025-01-01"
        assert config.venue.name == "BINANCE"

    def test_from_yaml_loads_file(
        self, tmp_path: Path, valid_config_dict: dict[str, Any]
    ) -> None:
        """Test from_yaml loads configuration from file."""
        config_file = tmp_path / "backtest.yaml"
        with open(config_file, "w") as f:
            yaml.dump(valid_config_dict, f)

        config = BacktestConfig.from_yaml(config_file)

        assert config.strategy.type == "breakout"

    def test_from_yaml_file_not_found_raises(self) -> None:
        """Test from_yaml raises error for missing file."""
        with pytest.raises(BacktestConfigError, match="not found"):
            BacktestConfig.from_yaml("/nonexistent/path.yaml")

    def test_validate_returns_empty_for_valid(
        self, valid_config_dict: dict[str, Any]
    ) -> None:
        """Test validate returns empty list for valid config."""
        config = BacktestConfig.from_dict(valid_config_dict)
        errors = config.validate()
        assert errors == []

    def test_validate_detects_missing_strategy_type(
        self, valid_config_dict: dict[str, Any]
    ) -> None:
        """Test validate detects missing strategy type."""
        valid_config_dict["strategy"]["type"] = ""

        with pytest.raises(BacktestConfigError, match="strategy.type"):
            BacktestConfig.from_dict(valid_config_dict)

    def test_validate_detects_invalid_date_range(
        self, valid_config_dict: dict[str, Any]
    ) -> None:
        """Test validate detects end_date before start_date."""
        valid_config_dict["backtest"]["start_date"] = "2025-12-31"
        valid_config_dict["backtest"]["end_date"] = "2025-01-01"

        with pytest.raises(BacktestConfigError, match="end_date"):
            BacktestConfig.from_dict(valid_config_dict)

    def test_validate_detects_exchange_suffix_in_instrument(
        self, valid_config_dict: dict[str, Any]
    ) -> None:
        """Test validate detects exchange suffix in instrument_id."""
        valid_config_dict["strategy"]["instrument_id"] = "BTCUSDT.BINANCE"

        with pytest.raises(BacktestConfigError, match="exchange suffix"):
            BacktestConfig.from_dict(valid_config_dict)

    def test_validate_detects_missing_catalog_path(
        self, valid_config_dict: dict[str, Any]
    ) -> None:
        """Test validate detects missing catalog_path."""
        del valid_config_dict["backtest"]["catalog_path"]

        with pytest.raises(BacktestConfigError, match="catalog_path"):
            BacktestConfig.from_dict(valid_config_dict)

    def test_validate_detects_invalid_oms_type(
        self, valid_config_dict: dict[str, Any]
    ) -> None:
        """Test validate detects invalid oms_type."""
        valid_config_dict["venue"]["oms_type"] = "INVALID"

        with pytest.raises(BacktestConfigError, match="oms_type"):
            BacktestConfig.from_dict(valid_config_dict)

    def test_validate_detects_invalid_fill_probability(
        self, valid_config_dict: dict[str, Any]
    ) -> None:
        """Test validate detects fill probability out of range."""
        valid_config_dict["venue"]["fill_model"] = {"prob_fill_on_limit": 1.5}

        with pytest.raises(BacktestConfigError, match="prob_fill_on_limit"):
            BacktestConfig.from_dict(valid_config_dict)

    def test_to_dict_roundtrip(self, valid_config_dict: dict[str, Any]) -> None:
        """Test to_dict produces dict that can recreate config."""
        config = BacktestConfig.from_dict(valid_config_dict)
        config_dict = config.to_dict()

        # Should be able to create config from dict
        config2 = BacktestConfig.from_dict(config_dict)
        assert config2.strategy.type == config.strategy.type

    def test_from_dict_with_nested_tearsheet(
        self, valid_config_dict: dict[str, Any]
    ) -> None:
        """Test from_dict parses nested tearsheet config."""
        valid_config_dict["report"]["tearsheet"] = {
            "enabled": True,
            "title": "Custom Title",
            "theme": "plotly_white",
        }

        config = BacktestConfig.from_dict(valid_config_dict)

        assert config.report.tearsheet is not None
        assert config.report.tearsheet.title == "Custom Title"
        assert config.report.tearsheet.theme == "plotly_white"
