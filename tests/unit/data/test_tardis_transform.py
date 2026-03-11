"""
Unit tests for the Tardis transform module.

Tests CSV.gz to Parquet conversion with mocked NautilusTrader components.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nautilus_quants.data.transform.tardis import (
    TardisTransformResult,
    transform_tardis_trades,
)


class TestTardisTransformResult:
    """Tests for TardisTransformResult dataclass."""

    def test_success_result(self) -> None:
        result = TardisTransformResult(
            success=True, symbol="BTCUSDT", files_processed=5, total_ticks=100000
        )
        assert result.success
        assert result.symbol == "BTCUSDT"
        assert result.files_processed == 5
        assert result.total_ticks == 100000
        assert result.errors == []

    def test_failure_result(self) -> None:
        result = TardisTransformResult(
            success=False,
            symbol="ETHUSDT",
            files_processed=0,
            total_ticks=0,
            errors=["No files found"],
        )
        assert not result.success
        assert result.errors == ["No files found"]


class TestTransformTardisTrades:
    """Tests for transform_tardis_trades function."""

    def test_no_files_found(self, tmp_path: Path) -> None:
        """Test returns error when no CSV.gz files found."""
        input_dir = tmp_path / "trades"
        input_dir.mkdir()

        result = transform_tardis_trades(
            input_dir=input_dir,
            catalog_path=tmp_path / "catalog",
            symbol="BTCUSDT",
        )

        assert not result.success
        assert result.files_processed == 0
        assert result.total_ticks == 0
        assert any("No CSV.gz files found" in e for e in result.errors)

    @patch("nautilus_quants.data.transform.tardis.ParquetDataCatalog")
    @patch("nautilus_quants.data.transform.tardis.TardisCSVDataLoader")
    @patch("nautilus_quants.data.transform.tardis.InstrumentId")
    def test_successful_transform(
        self,
        mock_instrument_id_cls: MagicMock,
        mock_loader_cls: MagicMock,
        mock_catalog_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test successful transformation of trade files."""
        # Setup input files
        input_dir = tmp_path / "trades"
        input_dir.mkdir()
        (input_dir / "2024-01-01_BTCUSDT.csv.gz").write_bytes(b"fake")
        (input_dir / "2024-01-02_BTCUSDT.csv.gz").write_bytes(b"fake")

        # Mock loader returns trade objects
        mock_trades_day1 = [MagicMock() for _ in range(500)]
        mock_trades_day2 = [MagicMock() for _ in range(300)]
        mock_loader = MagicMock()
        mock_loader.load_trades.side_effect = [mock_trades_day1, mock_trades_day2]
        mock_loader_cls.return_value = mock_loader

        mock_catalog = MagicMock()
        mock_catalog_cls.return_value = mock_catalog

        mock_instrument_id = MagicMock()
        mock_instrument_id_cls.from_str.return_value = mock_instrument_id

        result = transform_tardis_trades(
            input_dir=input_dir,
            catalog_path=tmp_path / "catalog",
            symbol="BTCUSDT",
        )

        assert result.success
        assert result.symbol == "BTCUSDT"
        assert result.files_processed == 2
        assert result.total_ticks == 800
        assert result.errors == []

        # Verify loader was called correctly
        assert mock_loader.load_trades.call_count == 2
        # Verify catalog.write_data was called for each file + instrument definition
        assert mock_catalog.write_data.call_count == 3

    @patch("nautilus_quants.data.transform.tardis.ParquetDataCatalog")
    @patch("nautilus_quants.data.transform.tardis.TardisCSVDataLoader")
    @patch("nautilus_quants.data.transform.tardis.InstrumentId")
    def test_partial_failure(
        self,
        mock_instrument_id_cls: MagicMock,
        mock_loader_cls: MagicMock,
        mock_catalog_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that errors in individual files don't block others."""
        input_dir = tmp_path / "trades"
        input_dir.mkdir()
        (input_dir / "2024-01-01_BTCUSDT.csv.gz").write_bytes(b"fake")
        (input_dir / "2024-01-02_BTCUSDT.csv.gz").write_bytes(b"fake")

        mock_loader = MagicMock()
        mock_loader.load_trades.side_effect = [
            [MagicMock() for _ in range(100)],  # First file succeeds
            RuntimeError("Corrupt file"),  # Second file fails
        ]
        mock_loader_cls.return_value = mock_loader

        mock_catalog = MagicMock()
        mock_catalog_cls.return_value = mock_catalog

        mock_instrument_id_cls.from_str.return_value = MagicMock()

        result = transform_tardis_trades(
            input_dir=input_dir,
            catalog_path=tmp_path / "catalog",
            symbol="BTCUSDT",
        )

        assert not result.success
        assert result.total_ticks == 100
        assert result.files_processed == 2
        assert len(result.errors) == 1
        assert "Corrupt file" in result.errors[0]

    @patch("nautilus_quants.data.transform.tardis.ParquetDataCatalog")
    @patch("nautilus_quants.data.transform.tardis.TardisCSVDataLoader")
    @patch("nautilus_quants.data.transform.tardis.InstrumentId")
    def test_instrument_id_format(
        self,
        mock_instrument_id_cls: MagicMock,
        mock_loader_cls: MagicMock,
        mock_catalog_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that instrument ID is created with correct format."""
        input_dir = tmp_path / "trades"
        input_dir.mkdir()
        (input_dir / "2024-01-01_SOLUSDT.csv.gz").write_bytes(b"fake")

        mock_loader = MagicMock()
        mock_loader.load_trades.return_value = []
        mock_loader_cls.return_value = mock_loader
        mock_catalog_cls.return_value = MagicMock()
        mock_instrument_id_cls.from_str.return_value = MagicMock()

        transform_tardis_trades(
            input_dir=input_dir,
            catalog_path=tmp_path / "catalog",
            symbol="SOLUSDT",
        )

        mock_instrument_id_cls.from_str.assert_called_once_with("SOLUSDT-PERP.BINANCE")

    @patch("nautilus_quants.data.transform.tardis.ParquetDataCatalog")
    @patch("nautilus_quants.data.transform.tardis.TardisCSVDataLoader")
    @patch("nautilus_quants.data.transform.tardis.InstrumentId")
    def test_only_matches_correct_symbol(
        self,
        mock_instrument_id_cls: MagicMock,
        mock_loader_cls: MagicMock,
        mock_catalog_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that glob only matches files for the target symbol."""
        input_dir = tmp_path / "trades"
        input_dir.mkdir()
        (input_dir / "2024-01-01_BTCUSDT.csv.gz").write_bytes(b"fake")
        (input_dir / "2024-01-01_ETHUSDT.csv.gz").write_bytes(b"fake")

        mock_loader = MagicMock()
        mock_loader.load_trades.return_value = [MagicMock()]
        mock_loader_cls.return_value = mock_loader
        mock_catalog_cls.return_value = MagicMock()
        mock_instrument_id_cls.from_str.return_value = MagicMock()

        result = transform_tardis_trades(
            input_dir=input_dir,
            catalog_path=tmp_path / "catalog",
            symbol="BTCUSDT",
        )

        # Only BTCUSDT files should be processed
        assert result.files_processed == 1
        assert result.total_ticks == 1
