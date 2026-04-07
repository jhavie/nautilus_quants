# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Regression tests for _load_funding_rate_panel batch query bug.

NautilusTrader catalog.funding_rates() batch query loses per-file
instrument_id metadata when PyArrow merges parquet files — all records
get the first instrument's ID.  These tests verify our per-instrument
query workaround.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from nautilus_quants.alpha.analysis.evaluator import FactorEvaluator


def _make_funding_rate_update(
    instrument_id: str, rate: float, ts_ns: int,
) -> MagicMock:
    """Create a mock FundingRateUpdate."""
    fru = MagicMock()
    fru.instrument_id = MagicMock(__str__=lambda s: instrument_id)
    fru.rate = Decimal(str(rate))
    fru.ts_event = ts_ns
    return fru


def _make_close_panel(
    instruments: list[str], n_bars: int = 6, start: str = "2024-01-01",
) -> pd.DataFrame:
    """Create a close_panel with 4h frequency."""
    idx = pd.date_range(start, periods=n_bars, freq="4h")
    return pd.DataFrame(
        {inst: np.linspace(100, 110, n_bars) for inst in instruments},
        index=idx,
    )


class TestLoadFundingRatePanel:
    """Regression tests for _load_funding_rate_panel."""

    INSTRUMENTS = [
        "BTCUSDT.BINANCE",
        "ETHUSDT.BINANCE",
        "SOLUSDT.BINANCE",
    ]

    def _mock_catalog_per_instrument(self):
        """Build a mock catalog that returns correct data per-instrument.

        Simulates the workaround: each single-instrument query returns
        records with the correct instrument_id.
        """
        # 8h settlement → every 2nd bar in 4h grid
        ts_base = int(pd.Timestamp("2024-01-01").value)
        h8_ns = 8 * 3600 * 10**9

        fr_by_instrument = {
            "BTCUSDT.BINANCE": [
                _make_funding_rate_update("BTCUSDT.BINANCE", 0.0001, ts_base),
                _make_funding_rate_update("BTCUSDT.BINANCE", 0.0002, ts_base + h8_ns),
                _make_funding_rate_update("BTCUSDT.BINANCE", 0.0003, ts_base + 2 * h8_ns),
            ],
            "ETHUSDT.BINANCE": [
                _make_funding_rate_update("ETHUSDT.BINANCE", -0.0001, ts_base),
                _make_funding_rate_update("ETHUSDT.BINANCE", -0.0002, ts_base + h8_ns),
                _make_funding_rate_update("ETHUSDT.BINANCE", -0.0003, ts_base + 2 * h8_ns),
            ],
            "SOLUSDT.BINANCE": [
                _make_funding_rate_update("SOLUSDT.BINANCE", 0.0005, ts_base),
                _make_funding_rate_update("SOLUSDT.BINANCE", 0.0006, ts_base + h8_ns),
                _make_funding_rate_update("SOLUSDT.BINANCE", 0.0007, ts_base + 2 * h8_ns),
            ],
        }

        catalog = MagicMock()

        def _funding_rates(instrument_ids=None):
            if instrument_ids is None:
                return []
            if len(instrument_ids) == 1:
                return fr_by_instrument.get(instrument_ids[0], [])
            # Simulate batch bug: all records get first instrument's ID
            first_id = instrument_ids[0]
            merged = []
            for inst in instrument_ids:
                for fru in fr_by_instrument.get(inst, []):
                    broken = _make_funding_rate_update(
                        first_id, float(fru.rate), fru.ts_event,
                    )
                    merged.append(broken)
            return merged

        catalog.funding_rates = _funding_rates
        return catalog

    def test_per_instrument_preserves_ids(self):
        """Each instrument gets its own FR data, not all labeled as first."""
        close_panel = _make_close_panel(self.INSTRUMENTS)

        with patch(
            "nautilus_quants.alpha.analysis.evaluator.ParquetDataCatalog",
            return_value=self._mock_catalog_per_instrument(),
        ):
            result = FactorEvaluator._load_funding_rate_panel(
                "/fake/path", self.INSTRUMENTS, close_panel,
            )

        assert result is not None
        # All 3 instruments must have non-zero FR values
        for inst in self.INSTRUMENTS:
            assert inst in result.columns
            assert result[inst].abs().sum() > 0, (
                f"{inst} has all-zero FR — batch query bug not fixed"
            )

        # BTC should have positive rates, ETH negative
        assert result["BTCUSDT.BINANCE"].dropna().iloc[-1] > 0
        assert result["ETHUSDT.BINANCE"].dropna().iloc[-1] < 0

    def test_instruments_have_distinct_rates(self):
        """FR values must differ across instruments (not all copied from first)."""
        close_panel = _make_close_panel(self.INSTRUMENTS)

        with patch(
            "nautilus_quants.alpha.analysis.evaluator.ParquetDataCatalog",
            return_value=self._mock_catalog_per_instrument(),
        ):
            result = FactorEvaluator._load_funding_rate_panel(
                "/fake/path", self.INSTRUMENTS, close_panel,
            )

        assert result is not None
        # No two columns should be identical
        btc = result["BTCUSDT.BINANCE"].dropna().values
        eth = result["ETHUSDT.BINANCE"].dropna().values
        sol = result["SOLUSDT.BINANCE"].dropna().values
        assert not np.allclose(btc, eth), "BTC and ETH have identical FR"
        assert not np.allclose(btc, sol), "BTC and SOL have identical FR"

    def test_partial_data_fills_zero(self):
        """Instruments without FR data get 0.0 fill."""
        instruments = self.INSTRUMENTS + ["DOGEUSDT.BINANCE"]
        close_panel = _make_close_panel(instruments)

        with patch(
            "nautilus_quants.alpha.analysis.evaluator.ParquetDataCatalog",
            return_value=self._mock_catalog_per_instrument(),
        ):
            result = FactorEvaluator._load_funding_rate_panel(
                "/fake/path", instruments, close_panel,
            )

        assert result is not None
        # DOGE has no FR data → should be 0.0
        assert (result["DOGEUSDT.BINANCE"] == 0.0).all()
        # Others should have real data
        assert result["BTCUSDT.BINANCE"].abs().sum() > 0

    def test_empty_returns_none(self):
        """If no FR data exists for any instrument, returns None."""
        close_panel = _make_close_panel(["XYZUSDT.BINANCE"])

        catalog = MagicMock()
        catalog.funding_rates = MagicMock(return_value=[])

        with patch(
            "nautilus_quants.alpha.analysis.evaluator.ParquetDataCatalog",
            return_value=catalog,
        ):
            result = FactorEvaluator._load_funding_rate_panel(
                "/fake/path", ["XYZUSDT.BINANCE"], close_panel,
            )

        assert result is None
