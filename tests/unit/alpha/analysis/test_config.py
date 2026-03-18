"""Tests for AlphaAnalysisConfig."""

import textwrap
from pathlib import Path

import pytest
import yaml

from nautilus_quants.alpha.analysis.config import AlphaAnalysisConfig, load_analysis_config


class TestAlphaAnalysisConfig:
    """Test AlphaAnalysisConfig dataclass."""

    def test_default_values(self):
        config = AlphaAnalysisConfig(
            catalog_path="/data/catalog",
            factor_config_path="config/factors.yaml",
            instrument_ids=["BTCUSDT.BINANCE"],
        )
        assert config.bar_spec == "1h"
        assert config.factors == []
        assert config.periods == (1, 4, 8, 24)
        assert config.quantiles == 5
        assert config.max_loss == 0.35
        assert config.filter_zscore == 20.0
        assert config.output_dir == "logs/alpha_analysis"
        assert config.output_format == ("png",)

    def test_default_charts(self):
        config = AlphaAnalysisConfig(
            catalog_path="/data/catalog",
            factor_config_path="config/factors.yaml",
            instrument_ids=["BTCUSDT.BINANCE"],
        )
        assert "quantile_returns_bar" in config.charts
        assert "cumulative_returns" in config.charts
        assert "ic_time_series" in config.charts
        assert "ic_histogram" in config.charts

    def test_custom_values(self):
        config = AlphaAnalysisConfig(
            catalog_path="/custom/path",
            factor_config_path="custom/factors.yaml",
            instrument_ids=["ETHUSDT.BINANCE", "BTCUSDT.BINANCE"],
            bar_spec="4h",
            factors=["volume", "momentum_3h"],
            periods=(1, 8),
            quantiles=10,
            max_loss=0.5,
            filter_zscore=None,
            charts=["ic_time_series"],
            output_dir="custom/output",
            output_format=("png", "html"),
        )
        assert config.catalog_path == "/custom/path"
        assert config.bar_spec == "4h"
        assert config.factors == ["volume", "momentum_3h"]
        assert config.periods == (1, 8)
        assert config.quantiles == 10
        assert config.filter_zscore is None
        assert config.output_format == ("png", "html")

    def test_frozen(self):
        config = AlphaAnalysisConfig(
            catalog_path="/data/catalog",
            factor_config_path="config/factors.yaml",
            instrument_ids=["BTCUSDT.BINANCE"],
        )
        with pytest.raises(AttributeError):
            config.catalog_path = "/other"  # type: ignore[misc]


class TestLoadAnalysisConfig:
    """Test YAML loading."""

    def test_load_full_config(self, tmp_path: Path):
        config_data = {
            "catalog_path": "/data/catalog",
            "factor_config_path": "config/factors.yaml",
            "instrument_ids": ["BTCUSDT.BINANCE", "ETHUSDT.BINANCE"],
            "bar_spec": "1h",
            "factors": ["volume", "momentum_3h"],
            "periods": [1, 4, 8, 24],
            "quantiles": 5,
            "max_loss": 0.35,
            "filter_zscore": 20.0,
            "charts": ["quantile_returns_bar", "ic_time_series"],
            "output_dir": "logs/alpha_analysis",
            "output_format": ["png"],
        }
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml.dump(config_data))

        config = load_analysis_config(yaml_path)

        assert config.catalog_path == "/data/catalog"
        assert config.factor_config_path == "config/factors.yaml"
        assert len(config.instrument_ids) == 2
        assert config.periods == (1, 4, 8, 24)
        assert config.output_format == ("png",)

    def test_load_minimal_config(self, tmp_path: Path):
        config_data = {
            "catalog_path": "/data/catalog",
            "factor_config_path": "config/factors.yaml",
            "instrument_ids": ["BTCUSDT.BINANCE"],
        }
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml.dump(config_data))

        config = load_analysis_config(yaml_path)

        assert config.bar_spec == "1h"
        assert config.factors == []
        assert config.periods == (1, 4, 8, 24)

    def test_load_missing_required_field(self, tmp_path: Path):
        config_data = {
            "catalog_path": "/data/catalog",
            # Missing factor_config_path and instrument_ids
        }
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml.dump(config_data))

        with pytest.raises(KeyError):
            load_analysis_config(yaml_path)

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            load_analysis_config(Path("/nonexistent/config.yaml"))

    def test_filter_zscore_null(self, tmp_path: Path):
        config_data = {
            "catalog_path": "/data/catalog",
            "factor_config_path": "config/factors.yaml",
            "instrument_ids": ["BTCUSDT.BINANCE"],
            "filter_zscore": None,
        }
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml.dump(config_data))

        config = load_analysis_config(yaml_path)
        assert config.filter_zscore is None
