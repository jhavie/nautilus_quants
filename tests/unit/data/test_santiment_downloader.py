# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for Santiment downloader (mocked san.get)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from nautilus_quants.data.config import SantimentDownloadConfig, SantimentPathsConfig
from nautilus_quants.data.download.santiment import SantimentDownloader


@pytest.fixture()
def mock_san():
    """Patch san module to avoid real API calls."""
    mock = MagicMock()
    mock.ApiConfig = MagicMock()

    def fake_get(query, from_date=None, to_date=None, interval=None):
        idx = pd.to_datetime(
            [
                "2025-10-01 00:00:00+00:00",
                "2025-10-01 04:00:00+00:00",
                "2025-10-01 08:00:00+00:00",
            ]
        )
        return pd.DataFrame({"value": [0.0001, 0.00015, 0.0002]}, index=idx)

    mock.get = fake_get
    return mock


@pytest.fixture()
def config():
    return SantimentDownloadConfig(
        metrics=("funding_rate",),
        start_date="2025-10-01",
        end_date="2025-10-02",
        symbols=("BTC", "ETH"),
        checkpoint_enabled=False,
    )


@pytest.fixture()
def paths(tmp_path):
    return SantimentPathsConfig(
        raw_data=str(tmp_path / "raw"),
        catalog=str(tmp_path / "catalog"),
        logs=str(tmp_path / "logs"),
    )


class TestSantimentDownloader:
    @patch.dict("os.environ", {"SAN_API_KEY": "test_key_123"})
    def test_download_all(self, config, paths, mock_san, tmp_path) -> None:
        with patch.dict("sys.modules", {"san": mock_san}):
            downloader = SantimentDownloader(config=config, paths=paths)
            downloader._san = mock_san
            results = downloader.download_all()

        assert len(results) == 2  # BTC + ETH
        assert all(r.success for r in results)
        assert all(r.rows == 3 for r in results)

        # Verify CSV output
        csv_path = tmp_path / "raw" / "funding_rate" / "BTC.csv"
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert len(df) == 3
        assert "timestamp_ms" in df.columns
        assert "value" in df.columns

    @patch.dict("os.environ", {"SAN_API_KEY": "test_key_123"})
    def test_checkpoint_skip(self, mock_san, tmp_path) -> None:
        raw_data = str(tmp_path / "raw")
        cp_paths = SantimentPathsConfig(
            raw_data=raw_data,
            catalog=str(tmp_path / "catalog"),
            logs=str(tmp_path / "logs"),
        )
        config_with_cp = SantimentDownloadConfig(
            metrics=("funding_rate",),
            start_date="2025-10-01",
            end_date="2025-10-02",
            symbols=("BTC",),
            checkpoint_enabled=True,
        )
        # Pre-create CSV at the exact path the downloader will check (>50 bytes)
        csv_dir = tmp_path / "raw" / "funding_rate"
        csv_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "timestamp_ms": [1700000000000, 1700014400000, 1700028800000],
                "value": [0.0001, 0.00015, -0.0002],
            }
        ).to_csv(csv_dir / "BTC.csv", index=False)

        with patch.dict("sys.modules", {"san": mock_san}):
            downloader = SantimentDownloader(config=config_with_cp, paths=cp_paths)
            downloader._san = mock_san
            results = downloader.download_all()

        assert len(results) == 1
        assert results[0].success
        # Checkpoint should have skipped — rows from existing CSV
        assert results[0].rows == 3

    def test_missing_api_key(self, config, paths) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="SAN_API_KEY"):
                SantimentDownloader(config=config, paths=paths)
