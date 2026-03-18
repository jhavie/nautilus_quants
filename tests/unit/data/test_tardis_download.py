"""
Unit tests for the TardisDownloader.

Tests download functionality with mocked tardis-dev datasets.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nautilus_quants.data.config import (
    TardisDownloadConfig,
    TardisPathsConfig,
    TardisPipelineConfig,
    load_tardis_config,
)
from nautilus_quants.data.download.tardis import TardisDownloadResult, TardisDownloader


class TestTardisDownloadConfig:
    """Tests for TardisDownloadConfig dataclass."""

    def test_default_values(self) -> None:
        config = TardisDownloadConfig()
        assert config.exchange == "binance-futures"
        assert config.api_key_env == "TARDIS_API_KEY"
        assert config.data_types == ("trades",)
        assert "BTCUSDT" in config.symbols
        assert config.concurrency == 5
        assert config.max_symbol_workers == 3

    def test_frozen(self) -> None:
        config = TardisDownloadConfig()
        with pytest.raises(AttributeError):
            config.exchange = "other"  # type: ignore[misc]


class TestTardisPathsConfig:
    """Tests for TardisPathsConfig dataclass."""

    def test_default_values(self) -> None:
        config = TardisPathsConfig()
        assert config.raw_data == "data/raw/tardis"
        assert config.catalog == "data/catalog"

    def test_frozen(self) -> None:
        config = TardisPathsConfig()
        with pytest.raises(AttributeError):
            config.raw_data = "other"  # type: ignore[misc]


class TestTardisPipelineConfig:
    """Tests for TardisPipelineConfig dataclass."""

    def test_default_values(self) -> None:
        config = TardisPipelineConfig()
        assert config.download.exchange == "binance-futures"
        assert config.paths.raw_data == "data/raw/tardis"

    def test_frozen(self) -> None:
        config = TardisPipelineConfig()
        with pytest.raises(AttributeError):
            config.download = TardisDownloadConfig()  # type: ignore[misc]


class TestLoadTardisConfig:
    """Tests for load_tardis_config function."""

    def test_load_defaults_when_file_missing(self, tmp_path: Path) -> None:
        config = load_tardis_config(tmp_path / "nonexistent.yaml")
        assert config.download.exchange == "binance-futures"
        assert config.paths.raw_data == "data/raw/tardis"

    def test_load_from_yaml(self, tmp_path: Path) -> None:
        import yaml

        config_data = {
            "download": {
                "exchange": "binance",
                "symbols": ["SOLUSDT"],
                "from_date": "2025-01-01",
                "to_date": "2025-06-01",
                "concurrency": 3,
                "max_symbol_workers": 2,
            },
            "paths": {
                "raw_data": "/tmp/tardis",
                "catalog": "/tmp/catalog",
            },
        }

        config_path = tmp_path / "test_tardis.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = load_tardis_config(config_path)
        assert config.download.exchange == "binance"
        assert config.download.symbols == ("SOLUSDT",)
        assert config.download.from_date == "2025-01-01"
        assert config.download.concurrency == 3
        assert config.paths.raw_data == "/tmp/tardis"

    def test_load_with_overrides(self, tmp_path: Path) -> None:
        import yaml

        config_data = {
            "download": {
                "symbols": ["BTCUSDT"],
            }
        }

        config_path = tmp_path / "test_tardis.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        overrides = {
            "download.symbols": "ETHUSDT,SOLUSDT",
            "download.from_date": "2025-03-01",
        }
        config = load_tardis_config(config_path, overrides)
        assert config.download.symbols == ("ETHUSDT", "SOLUSDT")
        assert config.download.from_date == "2025-03-01"


class TestTardisDownloader:
    """Tests for TardisDownloader class."""

    @pytest.fixture
    def config(self) -> TardisDownloadConfig:
        return TardisDownloadConfig(
            symbols=("BTCUSDT", "ETHUSDT"),
            from_date="2024-01-01",
            to_date="2024-01-02",
            max_symbol_workers=2,
        )

    @pytest.fixture
    def paths(self, tmp_path: Path) -> TardisPathsConfig:
        return TardisPathsConfig(
            raw_data=str(tmp_path / "raw" / "tardis"),
            catalog=str(tmp_path / "catalog"),
        )

    @patch("nautilus_quants.data.download.tardis.datasets")
    def test_download_all_success(
        self,
        mock_datasets: MagicMock,
        config: TardisDownloadConfig,
        paths: TardisPathsConfig,
    ) -> None:
        """Test successful download of all symbols."""
        mock_datasets.download = MagicMock(return_value=None)

        downloader = TardisDownloader(config=config, paths=paths)
        results = downloader.download_all()

        assert len(results) == 2
        assert all(r.success for r in results)
        assert {r.symbol for r in results} == {"BTCUSDT", "ETHUSDT"}
        assert mock_datasets.download.call_count == 2

    @patch("nautilus_quants.data.download.tardis.datasets")
    def test_download_single_symbol_failure(
        self,
        mock_datasets: MagicMock,
        config: TardisDownloadConfig,
        paths: TardisPathsConfig,
    ) -> None:
        """Test that a failed symbol doesn't block others."""

        def side_effect(**kwargs):
            if kwargs["symbols"] == ["ETHUSDT"]:
                raise RuntimeError("API error")

        mock_datasets.download = MagicMock(side_effect=side_effect)

        downloader = TardisDownloader(config=config, paths=paths)
        results = downloader.download_all()

        assert len(results) == 2
        success = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        assert len(success) == 1
        assert success[0].symbol == "BTCUSDT"
        assert len(failed) == 1
        assert failed[0].symbol == "ETHUSDT"
        assert "API error" in failed[0].error

    @patch("nautilus_quants.data.download.tardis.datasets")
    def test_download_symbol_passes_correct_args(
        self,
        mock_datasets: MagicMock,
        paths: TardisPathsConfig,
    ) -> None:
        """Test that correct arguments are passed to datasets.download."""
        mock_datasets.download = MagicMock(return_value=None)

        config = TardisDownloadConfig(
            exchange="binance-futures",
            symbols=("BTCUSDT",),
            data_types=("trades",),
            from_date="2024-06-01",
            to_date="2024-07-01",
            concurrency=3,
        )

        downloader = TardisDownloader(config=config, paths=paths)
        result = downloader.download_symbol("BTCUSDT")

        assert result.success
        call_kwargs = mock_datasets.download.call_args[1]
        assert call_kwargs["exchange"] == "binance-futures"
        assert call_kwargs["data_types"] == ["trades"]
        assert call_kwargs["symbols"] == ["BTCUSDT"]
        assert call_kwargs["from_date"] == "2024-06-01"
        assert call_kwargs["to_date"] == "2024-07-01"
        assert call_kwargs["concurrency"] == 3

    @patch("nautilus_quants.data.download.tardis.datasets")
    def test_download_reads_api_key_from_env(
        self,
        mock_datasets: MagicMock,
        paths: TardisPathsConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that API key is read from environment variable."""
        monkeypatch.setenv("TARDIS_API_KEY", "test-key-123")
        mock_datasets.download = MagicMock(return_value=None)

        config = TardisDownloadConfig(symbols=("BTCUSDT",))
        downloader = TardisDownloader(config=config, paths=paths)
        downloader.download_symbol("BTCUSDT")

        call_kwargs = mock_datasets.download.call_args[1]
        assert call_kwargs["api_key"] == "test-key-123"

    @patch("nautilus_quants.data.download.tardis.datasets")
    def test_download_empty_api_key_when_env_var_not_set(
        self,
        mock_datasets: MagicMock,
        paths: TardisPathsConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that empty API key is passed when env var name is used but not set."""
        monkeypatch.delenv("TARDIS_API_KEY", raising=False)
        mock_datasets.download = MagicMock(return_value=None)

        config = TardisDownloadConfig(symbols=("BTCUSDT",))
        downloader = TardisDownloader(config=config, paths=paths)
        downloader.download_symbol("BTCUSDT")

        call_kwargs = mock_datasets.download.call_args[1]
        assert call_kwargs["api_key"] == ""

    @patch("nautilus_quants.data.download.tardis.datasets")
    def test_download_literal_api_key(
        self,
        mock_datasets: MagicMock,
        paths: TardisPathsConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that literal TD.xxx key is used directly without env lookup."""
        monkeypatch.delenv("TARDIS_API_KEY", raising=False)
        mock_datasets.download = MagicMock(return_value=None)

        literal_key = "TD.test-key-12345"
        config = TardisDownloadConfig(symbols=("BTCUSDT",), api_key_env=literal_key)
        downloader = TardisDownloader(config=config, paths=paths)
        downloader.download_symbol("BTCUSDT")

        call_kwargs = mock_datasets.download.call_args[1]
        assert call_kwargs["api_key"] == literal_key

    def test_clean_removes_directory(self, tmp_path: Path) -> None:
        """Test that clean removes the exchange directory."""
        exchange_dir = tmp_path / "raw" / "tardis" / "binance-futures" / "trades"
        exchange_dir.mkdir(parents=True)
        (exchange_dir / "test.csv.gz").write_text("test")

        paths = TardisPathsConfig(raw_data=str(tmp_path / "raw" / "tardis"))
        config = TardisDownloadConfig()
        downloader = TardisDownloader(config=config, paths=paths)
        downloader.clean()

        assert not (tmp_path / "raw" / "tardis" / "binance-futures").exists()

    def test_download_result_frozen(self) -> None:
        result = TardisDownloadResult(symbol="BTC", success=True)
        with pytest.raises(AttributeError):
            result.symbol = "ETH"  # type: ignore[misc]
