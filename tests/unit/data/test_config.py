"""
Unit tests for the configuration loader.

Tests loading configuration from YAML files and merging CLI overrides.
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from nautilus_quants.data.config import (
    ConfigurationError,
    DownloadConfig,
    PathsConfig,
    PipelineConfig,
    ProcessConfig,
    TransformConfig,
    ValidateConfig,
    config_to_dict,
    load_config,
)


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_default_values(self, tmp_path: Path) -> None:
        """Test loading config with defaults when file doesn't exist."""
        config = load_config(tmp_path / "nonexistent.yaml")

        assert config.version == "1.0"
        assert config.download.exchange == "binance"
        assert config.download.market_type == "futures"
        assert "BTCUSDT" in config.download.symbols
        assert "1h" in config.download.timeframes

    def test_load_config_from_yaml(self, tmp_path: Path) -> None:
        """Test loading config from YAML file."""
        config_data = {
            "version": "2.0",
            "download": {
                "exchange": "binance",
                "market_type": "spot",
                "symbols": ["ETHUSDT"],
                "timeframes": ["4h"],
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
            },
            "validate": {
                "check_duplicates": False,
                "fail_on_warnings": True,
            },
            "process": {
                "max_gap_bars": 5,
            },
        }

        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(config_path)

        assert config.version == "2.0"
        assert config.download.market_type == "spot"
        assert config.download.symbols == ["ETHUSDT"]
        assert config.download.timeframes == ["4h"]
        assert config.download.start_date == "2023-01-01"
        assert config.validate.check_duplicates is False
        assert config.validate.fail_on_warnings is True
        assert config.process.max_gap_bars == 5

    def test_load_config_with_overrides(self, tmp_path: Path) -> None:
        """Test CLI overrides are applied correctly."""
        config_data = {
            "download": {
                "symbols": ["BTCUSDT"],
                "timeframes": ["1h"],
            }
        }

        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        overrides = {
            "download.symbols": "ETHUSDT,BNBUSDT",
            "download.timeframes": "4h",
            "download.start_date": "2024-06-01",
        }

        config = load_config(config_path, overrides)

        assert config.download.symbols == ["ETHUSDT", "BNBUSDT"]
        assert config.download.timeframes == ["4h"]
        assert config.download.start_date == "2024-06-01"

    def test_load_config_rate_limit(self, tmp_path: Path) -> None:
        """Test rate limit configuration is parsed correctly."""
        config_data = {
            "download": {
                "rate_limit": {
                    "max_retries": 10,
                    "initial_delay_seconds": 2.0,
                    "max_delay_seconds": 120.0,
                    "backoff_multiplier": 3.0,
                }
            }
        }

        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(config_path)

        assert config.download.rate_limit.max_retries == 10
        assert config.download.rate_limit.initial_delay_seconds == 2.0
        assert config.download.rate_limit.max_delay_seconds == 120.0
        assert config.download.rate_limit.backoff_multiplier == 3.0

    def test_load_config_paths(self, tmp_path: Path) -> None:
        """Test paths configuration is parsed correctly."""
        config_data = {
            "paths": {
                "raw_data": "/custom/raw",
                "processed_data": "/custom/processed",
                "catalog": "/custom/catalog",
                "logs": "/custom/logs",
            }
        }

        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(config_path)

        assert config.paths.raw_data == "/custom/raw"
        assert config.paths.processed_data == "/custom/processed"
        assert config.paths.catalog == "/custom/catalog"
        assert config.paths.logs == "/custom/logs"

    def test_load_config_invalid_yaml(self, tmp_path: Path) -> None:
        """Test error handling for invalid YAML."""
        config_path = tmp_path / "invalid.yaml"
        with open(config_path, "w") as f:
            f.write("invalid: yaml: content: [")

        with pytest.raises(ConfigurationError) as exc_info:
            load_config(config_path)

        assert "Invalid YAML" in str(exc_info.value)


class TestConfigToDict:
    """Tests for config_to_dict function."""

    def test_config_to_dict_roundtrip(self) -> None:
        """Test that config can be serialized and deserialized."""
        config = PipelineConfig()
        config_dict = config_to_dict(config)

        assert config_dict["version"] == "1.0"
        assert config_dict["download"]["exchange"] == "binance"
        assert config_dict["download"]["rate_limit"]["max_retries"] == 5
        assert config_dict["validate"]["check_duplicates"] is True
        assert config_dict["process"]["max_gap_bars"] == 3
        assert config_dict["transform"]["merge_files"] is True
        assert config_dict["paths"]["raw_data"] == "data/raw"


class TestDownloadConfig:
    """Tests for DownloadConfig dataclass."""

    def test_download_config_defaults(self) -> None:
        """Test DownloadConfig default values."""
        config = DownloadConfig()

        assert config.exchange == "binance"
        assert config.market_type == "futures"
        assert "BTCUSDT" in config.symbols
        assert "1h" in config.timeframes
        assert config.rate_limit.max_retries == 5
        assert config.checkpoint.enabled is True
        assert config.checkpoint.batch_size == 1000


class TestValidateConfig:
    """Tests for ValidateConfig dataclass."""

    def test_validate_config_defaults(self) -> None:
        """Test ValidateConfig default values."""
        config = ValidateConfig()

        assert config.check_duplicates is True
        assert config.check_gaps is True
        assert config.check_ohlc is True
        assert config.fail_on_warnings is False


class TestProcessConfig:
    """Tests for ProcessConfig dataclass."""

    def test_process_config_defaults(self) -> None:
        """Test ProcessConfig default values."""
        config = ProcessConfig()

        assert config.remove_duplicates is True
        assert config.keep_duplicate == "last"
        assert config.fill_small_gaps is True
        assert config.max_gap_bars == 3
        assert config.remove_invalid_ohlc is True


class TestTransformConfig:
    """Tests for TransformConfig dataclass."""

    def test_transform_config_defaults(self) -> None:
        """Test TransformConfig default values."""
        config = TransformConfig()

        assert config.output_format == "parquet"
        assert config.merge_files is True
        assert config.catalog_path == "data/catalog"


class TestPathsConfig:
    """Tests for PathsConfig dataclass."""

    def test_paths_config_defaults(self) -> None:
        """Test PathsConfig default values."""
        config = PathsConfig()

        assert config.raw_data == "data/raw"
        assert config.processed_data == "data/processed"
        assert config.catalog == "data/catalog"
        assert config.logs == "logs/data_pipeline"
