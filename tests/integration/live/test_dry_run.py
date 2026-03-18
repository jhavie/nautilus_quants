"""Integration tests for live trading dry run."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from nautilus_quants.live.exceptions import LiveConfigError
from nautilus_quants.live.runner import validate_config


def _write_config(config_dict: dict, tmp_dir: Path) -> Path:
    """Write config dict to a temp YAML file."""
    config_path = tmp_dir / "test_live.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f)
    return config_path


def _minimal_config() -> dict:
    """Build a minimal valid live config dict."""
    return {
        "venue": {
            "name": "OKX",
            "instrument_type": "SWAP",
            "contract_type": "LINEAR",
            "margin_mode": "CROSS",
            "is_demo": True,
        },
        "instruments": {
            "bar_spec": "1h",
            "instrument_ids": ["BTC-USDT-SWAP", "ETH-USDT-SWAP"],
        },
        "engine": {
            "trader_id": "TEST-DRY-001",
            "actors": [],
            "strategies": [],
        },
    }


class TestValidateConfig:
    """Tests for validate_config."""

    def test_valid_config(self, tmp_path: Path) -> None:
        config_dict = _minimal_config()
        config_path = _write_config(config_dict, tmp_path)
        assert validate_config(config_path) is True

    def test_invalid_missing_venue(self, tmp_path: Path) -> None:
        config_dict = {
            "instruments": {
                "bar_spec": "1h",
                "instrument_ids": ["BTC-USDT-SWAP"],
            },
            "engine": {"trader_id": "TEST"},
        }
        config_path = _write_config(config_dict, tmp_path)
        with pytest.raises(LiveConfigError, match="Missing 'venue'"):
            validate_config(config_path)

    def test_invalid_empty_instruments(self, tmp_path: Path) -> None:
        config_dict = _minimal_config()
        config_dict["instruments"]["instrument_ids"] = []
        config_path = _write_config(config_dict, tmp_path)
        with pytest.raises(LiveConfigError):
            validate_config(config_path)

    def test_binance_config(self, tmp_path: Path) -> None:
        config_dict = _minimal_config()
        config_dict["venue"]["name"] = "BINANCE"
        config_dict["instruments"]["instrument_ids"] = ["BTCUSDT", "ETHUSDT"]
        config_path = _write_config(config_dict, tmp_path)
        assert validate_config(config_path) is True

    def test_multiple_instruments(self, tmp_path: Path) -> None:
        config_dict = _minimal_config()
        config_dict["instruments"]["instrument_ids"] = [
            "BTC-USDT-SWAP",
            "ETH-USDT-SWAP",
            "SOL-USDT-SWAP",
            "DOGE-USDT-SWAP",
        ]
        config_path = _write_config(config_dict, tmp_path)
        assert validate_config(config_path) is True
