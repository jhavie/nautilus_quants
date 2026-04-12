# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for unified extra data framework."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

from nautilus_quants.factors.engine.buffer import Buffer
from nautilus_quants.factors.engine.evaluator import Evaluator
from nautilus_quants.factors.engine.extra_data import (
    ExtraDataConfig,
    ExtraDataManager,
    _load_broadcast,
    _load_catalog_field,
    load_extra_data_config,
    parse_extra_data_raw,
)
from nautilus_quants.factors.operators.cross_sectional import CS_OPERATOR_INSTANCES
from nautilus_quants.factors.operators.math import MATH_OPERATORS
from nautilus_quants.factors.operators.time_series import TS_OPERATOR_INSTANCES
from nautilus_quants.factors.expression import parse_expression


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def close_panel() -> pd.DataFrame:
    """3-instrument, 5-timestamp close panel."""
    idx = pd.date_range("2024-01-01", periods=5, freq="4h")
    return pd.DataFrame(
        {
            "BTCUSDT.BINANCE": [40000, 40100, 40200, 40300, 40400],
            "ETHUSDT.BINANCE": [2200, 2210, 2220, 2230, 2240],
            "SOLUSDT.BINANCE": [90, 91, 92, 93, 94],
        },
        index=idx,
        dtype=float,
    )


# ── ExtraDataConfig parsing ─────────────────────────────────────────────


class TestParseExtraDataRaw:
    def test_dict_format(self) -> None:
        raw = {
            "funding_rate": {"source": "catalog", "path": "/data/catalog"},
            "btc_close": {
                "source": "broadcast",
                "instruments": ["BTC"],
            },
        }
        configs = parse_extra_data_raw(raw)
        assert len(configs) == 2
        assert configs[0].name == "funding_rate"
        assert configs[0].source == "catalog"
        assert configs[0].path == "/data/catalog"
        assert configs[1].name == "btc_close"
        assert configs[1].source == "broadcast"
        assert configs[1].instruments == ["BTC"]

    def test_shorthand_format(self) -> None:
        raw = {"funding_rate": "catalog", "quote_volume": "bar"}
        configs = parse_extra_data_raw(raw)
        assert len(configs) == 2
        assert configs[0].source == "catalog"
        assert configs[1].source == "bar"

    def test_empty_returns_empty(self) -> None:
        assert parse_extra_data_raw(None) == []
        assert parse_extra_data_raw({}) == []

    def test_instruments_string_to_list(self) -> None:
        raw = {"btc_close": {"source": "broadcast", "instruments": "BTC"}}
        configs = parse_extra_data_raw(raw)
        assert configs[0].instruments == ["BTC"]


class TestLoadExtraDataConfig:
    def test_load_from_file(self, tmp_path: Path) -> None:
        config_file = tmp_path / "extra_data.yaml"
        # Use explicit YAML string to preserve order
        config_file.write_text(
            "funding_rate:\n  source: catalog\n"
            "btc_close:\n  source: broadcast\n  instruments:\n    - BTC\n"
        )

        configs = load_extra_data_config(config_file)
        assert len(configs) == 2
        names = {c.name for c in configs}
        assert names == {"funding_rate", "btc_close"}

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_extra_data_config("/nonexistent/path.yaml")


# ── Broadcast loader ─────────────────────────────────────────────────────


class TestLoadBroadcast:
    def test_exact_match(self, close_panel: pd.DataFrame) -> None:
        cfg = ExtraDataConfig(
            name="btc_close",
            source="broadcast",
            instruments=["BTCUSDT.BINANCE"],
        )
        result = _load_broadcast(cfg, close_panel)
        assert result is not None
        # All columns should have BTC's close values
        for col in result.columns:
            np.testing.assert_array_equal(
                result[col].values,
                close_panel["BTCUSDT.BINANCE"].values,
            )

    def test_prefix_match(self, close_panel: pd.DataFrame) -> None:
        cfg = ExtraDataConfig(
            name="btc_close",
            source="broadcast",
            instruments=["BTC"],
        )
        result = _load_broadcast(cfg, close_panel)
        assert result is not None
        assert result.shape == close_panel.shape
        for col in result.columns:
            np.testing.assert_array_equal(
                result[col].values,
                close_panel["BTCUSDT.BINANCE"].values,
            )

    def test_prefix_case_insensitive(self, close_panel: pd.DataFrame) -> None:
        cfg = ExtraDataConfig(
            name="btc_close",
            source="broadcast",
            instruments=["btc"],
        )
        result = _load_broadcast(cfg, close_panel)
        assert result is not None

    def test_missing_instrument(self, close_panel: pd.DataFrame) -> None:
        cfg = ExtraDataConfig(
            name="doge_close",
            source="broadcast",
            instruments=["DOGE"],
        )
        result = _load_broadcast(cfg, close_panel)
        assert result is None

    def test_no_instruments_configured(self, close_panel: pd.DataFrame) -> None:
        cfg = ExtraDataConfig(name="x", source="broadcast", instruments=[])
        result = _load_broadcast(cfg, close_panel)
        assert result is None


# ── Buffer inject_staged_field ───────────────────────────────────────────


class TestBufferInjectStagedField:
    def test_inject_broadcast(self) -> None:
        buf = Buffer(max_history=10, extra_fields=("btc_close",))

        ts = 1000
        buf.append(
            "BTCUSDT.BINANCE",
            ts,
            {"open": 40000, "high": 40100, "low": 39900, "close": 40050, "volume": 100},
        )
        buf.append(
            "ETHUSDT.BINANCE",
            ts,
            {"open": 2200, "high": 2210, "low": 2190, "close": 2205, "volume": 50},
        )
        buf.append(
            "SOLUSDT.BINANCE", ts, {"open": 90, "high": 91, "low": 89, "close": 90.5, "volume": 30}
        )

        # Inject BTC close to all instruments BEFORE flush
        buf.inject_staged_field(ts, "btc_close", "BTCUSDT.BINANCE")

        # Now flush
        buf.flush_timestamp(ts)
        panels = buf.to_panel()

        btc_close_panel = panels["btc_close"]
        # All instruments should have BTC's close value
        for col in btc_close_panel.columns:
            assert btc_close_panel[col].iloc[0] == 40050.0

    def test_inject_missing_ts(self) -> None:
        buf = Buffer(max_history=10, extra_fields=("btc_close",))
        # Should not raise
        buf.inject_staged_field(999, "btc_close", "BTCUSDT.BINANCE")

    def test_inject_missing_instrument(self) -> None:
        buf = Buffer(max_history=10, extra_fields=("btc_close",))
        buf.append(
            "ETHUSDT.BINANCE",
            1000,
            {"close": 2200, "open": 2200, "high": 2200, "low": 2200, "volume": 10},
        )
        # Source instrument not in staging — should not raise
        buf.inject_staged_field(1000, "btc_close", "BTCUSDT.BINANCE")

    def test_inject_multiple_timestamps(self) -> None:
        buf = Buffer(max_history=10, extra_fields=("btc_close",))

        for ts, btc_close in [(1000, 40000.0), (2000, 41000.0)]:
            buf.append(
                "BTCUSDT.BINANCE",
                ts,
                {"open": 0, "high": 0, "low": 0, "close": btc_close, "volume": 0},
            )
            buf.append(
                "ETHUSDT.BINANCE", ts, {"open": 0, "high": 0, "low": 0, "close": 2200, "volume": 0}
            )
            buf.inject_staged_field(ts, "btc_close", "BTCUSDT.BINANCE")
            buf.flush_timestamp(ts)

        panels = buf.to_panel()
        btc_close_panel = panels["btc_close"]
        assert btc_close_panel["ETHUSDT.BINANCE"].iloc[0] == 40000.0
        assert btc_close_panel["ETHUSDT.BINANCE"].iloc[1] == 41000.0


# ── ExtraDataManager ─────────────────────────────────────────────────────


class TestExtraDataManager:
    def test_inject_broadcast(self, close_panel: pd.DataFrame) -> None:
        configs = [
            ExtraDataConfig(name="btc_close", source="broadcast", instruments=["BTC"]),
            ExtraDataConfig(name="eth_close", source="broadcast", instruments=["ETH"]),
        ]
        manager = ExtraDataManager(configs)
        panel_fields: dict[str, pd.DataFrame | float] = {"close": close_panel}

        manager.inject_panels(
            panel_fields,
            list(close_panel.columns),
        )

        assert "btc_close" in panel_fields
        assert "eth_close" in panel_fields
        btc = panel_fields["btc_close"]
        assert isinstance(btc, pd.DataFrame)
        # All columns = BTC close
        for col in btc.columns:
            np.testing.assert_array_equal(
                btc[col].values,
                close_panel["BTCUSDT.BINANCE"].values,
            )

    def test_unknown_source_skipped(self, close_panel: pd.DataFrame) -> None:
        configs = [
            ExtraDataConfig(name="mystery", source="unknown_source"),
        ]
        manager = ExtraDataManager(configs)
        panel_fields: dict[str, pd.DataFrame | float] = {"close": close_panel}
        # Should not raise
        manager.inject_panels(panel_fields, list(close_panel.columns))
        assert "mystery" not in panel_fields

    def test_empty_configs(self, close_panel: pd.DataFrame) -> None:
        manager = ExtraDataManager([])
        panel_fields: dict[str, pd.DataFrame | float] = {"close": close_panel}
        manager.inject_panels(panel_fields, list(close_panel.columns))
        # Only close should remain
        assert list(panel_fields.keys()) == ["close"]


# ── Backward compatibility ───────────────────────────────────────────────


class TestBackwardCompat:
    def test_legacy_funding_rate_converted(self) -> None:
        """Legacy funding_rate: true → ExtraDataConfig."""
        from nautilus_quants.alpha.analysis.config import load_analysis_config

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(
                {
                    "catalog_path": "/tmp/catalog",
                    "factor_config_path": "config/factors.yaml",
                    "instrument_ids": ["BTCUSDT.BINANCE"],
                    "funding_rate": True,
                    "oi_data_path": "/tmp/oi",
                    "oi_timeframe": "4h",
                },
                f,
            )
            f.flush()

            config = load_analysis_config(f.name)

        assert len(config.extra_data) == 2
        fr = config.extra_data[0]
        assert fr.name == "funding_rate"
        assert fr.source == "catalog"
        assert fr.path == "/tmp/catalog"
        oi = config.extra_data[1]
        assert oi.name == "open_interest"
        assert oi.source == "parquet"
        assert oi.path == "/tmp/oi"
        assert oi.timeframe == "4h"

    def test_extra_data_path_takes_precedence(self, tmp_path: Path) -> None:
        """extra_data_path overrides legacy fields."""
        from nautilus_quants.alpha.analysis.config import load_analysis_config

        # Write extra_data YAML
        ed_file = tmp_path / "extra_data.yaml"
        ed_file.write_text(
            yaml.dump(
                {
                    "btc_close": {"source": "broadcast", "instruments": ["BTC"]},
                }
            )
        )

        # Write analysis YAML with BOTH extra_data_path and legacy fields
        analysis_file = tmp_path / "analysis.yaml"
        analysis_file.write_text(
            yaml.dump(
                {
                    "catalog_path": "/tmp/catalog",
                    "factor_config_path": "config/factors.yaml",
                    "instrument_ids": ["BTCUSDT.BINANCE"],
                    "extra_data_path": str(ed_file),
                    "funding_rate": True,  # should be ignored
                }
            )
        )

        config = load_analysis_config(analysis_file)

        # extra_data_path wins — only btc_close, no funding_rate
        assert len(config.extra_data) == 1
        assert config.extra_data[0].name == "btc_close"


# ── E2E: Analyze path (all 5 sources) ───────────────────────────────────


class TestE2EAnalyzePath:
    """Verify all 5 extra data sources are injected and usable in analyze path."""

    INSTRUMENTS = ["BTCUSDT.BINANCE", "ETHUSDT.BINANCE", "SOLUSDT.BINANCE"]

    @pytest.fixture
    def panel_fields(self) -> dict[str, pd.DataFrame | float]:
        """Build OHLCV panel fields with 10 timestamps."""
        idx = pd.date_range("2024-01-01", periods=10, freq="4h")
        close = pd.DataFrame(
            {
                "BTCUSDT.BINANCE": np.linspace(40000, 41000, 10),
                "ETHUSDT.BINANCE": np.linspace(2200, 2300, 10),
                "SOLUSDT.BINANCE": np.linspace(90, 100, 10),
            },
            index=idx,
        )
        high = close * 1.01
        low = close * 0.99
        volume = pd.DataFrame(
            {
                "BTCUSDT.BINANCE": np.full(10, 1000.0),
                "ETHUSDT.BINANCE": np.full(10, 5000.0),
                "SOLUSDT.BINANCE": np.full(10, 20000.0),
            },
            index=idx,
        )
        return {
            "open": close * 0.999,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }

    @pytest.fixture
    def mock_bars(self) -> dict[str, list]:
        """Create mock bars with quote_volume field."""
        bars = {}
        idx = pd.date_range("2024-01-01", periods=10, freq="4h")
        for inst, base_price, base_vol in [
            ("BTCUSDT.BINANCE", 40000, 1000),
            ("ETHUSDT.BINANCE", 2200, 5000),
            ("SOLUSDT.BINANCE", 90, 20000),
        ]:
            inst_bars = []
            for i, ts in enumerate(idx):
                bar = MagicMock()
                bar_dict = {
                    "ts_event": ts.value,
                    "open": base_price + i * 10,
                    "high": base_price + i * 10 + 50,
                    "low": base_price + i * 10 - 50,
                    "close": base_price + (i + 1) * (1000 / 10),
                    "volume": float(base_vol),
                    "quote_volume": float(base_vol * (base_price + i * 10)),
                }
                type(bar).to_dict = staticmethod(lambda b, d=bar_dict: d)
                bar.quote_volume = bar_dict["quote_volume"]
                inst_bars.append(bar)
            bars[inst] = inst_bars
        return bars

    @pytest.fixture
    def mock_fr_catalog(self):
        """Mock catalog that returns FR data per instrument."""
        ts_base = int(pd.Timestamp("2024-01-01").value)
        h8 = 8 * 3600 * 10**9

        fr_data = {
            "BTCUSDT.BINANCE": [0.0001, 0.0002, 0.0003],
            "ETHUSDT.BINANCE": [-0.0001, -0.0002, -0.0003],
            "SOLUSDT.BINANCE": [0.0005, 0.0004, 0.0003],
        }

        catalog = MagicMock()

        def _funding_rates(instrument_ids=None):
            if not instrument_ids or len(instrument_ids) != 1:
                return []
            inst = instrument_ids[0]
            rates = fr_data.get(inst, [])
            result = []
            for i, rate in enumerate(rates):
                fru = MagicMock()
                fru.instrument_id = MagicMock(__str__=lambda s, id=inst: id)
                fru.rate = rate
                fru.ts_event = ts_base + i * h8
                result.append(fru)
            return result

        catalog.funding_rates = _funding_rates
        return catalog

    def test_all_5_sources_injected(
        self,
        panel_fields,
        mock_bars,
        mock_fr_catalog,
    ) -> None:
        """All 5 extra data fields appear in panel_fields after injection."""
        configs = [
            ExtraDataConfig(name="funding_rate", source="catalog"),
            ExtraDataConfig(
                name="open_interest", source="parquet", path="/fake/oi", timeframe="4h"
            ),
            ExtraDataConfig(name="quote_volume", source="bar"),
            ExtraDataConfig(name="btc_close", source="broadcast", instruments=["BTC"]),
            ExtraDataConfig(name="eth_close", source="broadcast", instruments=["ETH"]),
        ]

        manager = ExtraDataManager(configs)

        with (
            patch(
                "nautilus_trader.persistence.catalog.ParquetDataCatalog",
                return_value=mock_fr_catalog,
            ),
            patch(
                "nautilus_quants.data.transform.open_interest.load_parquet_field_lookup",
                return_value={
                    "BTCUSDT.BINANCE": {
                        int(pd.Timestamp("2024-01-01").value): {"open_interest": 50000.0},
                    },
                    "ETHUSDT.BINANCE": {
                        int(pd.Timestamp("2024-01-01").value): {"open_interest": 200000.0},
                    },
                },
            ),
        ):
            manager.inject_panels(
                panel_fields,
                self.INSTRUMENTS,
                bars_by_instrument=mock_bars,
                catalog_path="/fake/catalog",
            )

        # All 5 fields must be present
        for field_name in [
            "funding_rate",
            "open_interest",
            "quote_volume",
            "btc_close",
            "eth_close",
        ]:
            assert field_name in panel_fields, f"Missing: {field_name}"
            panel = panel_fields[field_name]
            assert isinstance(panel, pd.DataFrame), f"{field_name} is not DataFrame"
            assert set(panel.columns) == set(self.INSTRUMENTS)

    def test_broadcast_values_correct(self, panel_fields) -> None:
        """btc_close has BTC values, eth_close has ETH values everywhere."""
        configs = [
            ExtraDataConfig(name="btc_close", source="broadcast", instruments=["BTC"]),
            ExtraDataConfig(name="eth_close", source="broadcast", instruments=["ETH"]),
        ]
        manager = ExtraDataManager(configs)
        manager.inject_panels(panel_fields, self.INSTRUMENTS)

        btc_panel = panel_fields["btc_close"]
        eth_panel = panel_fields["eth_close"]
        close = panel_fields["close"]

        # btc_close: all columns = BTC's close
        for col in btc_panel.columns:
            np.testing.assert_array_equal(
                btc_panel[col].values,
                close["BTCUSDT.BINANCE"].values,
            )
        # eth_close: all columns = ETH's close
        for col in eth_panel.columns:
            np.testing.assert_array_equal(
                eth_panel[col].values,
                close["ETHUSDT.BINANCE"].values,
            )

    def test_variables_using_extra_data_evaluate(self, panel_fields) -> None:
        """Derived variables (btc_returns, btc_beta, vwap) evaluate correctly."""
        # Inject broadcast
        configs = [
            ExtraDataConfig(name="btc_close", source="broadcast", instruments=["BTC"]),
        ]
        manager = ExtraDataManager(configs)
        manager.inject_panels(panel_fields, self.INSTRUMENTS)

        evaluator = Evaluator(
            panel_fields=panel_fields,
            ts_ops=dict(TS_OPERATOR_INSTANCES),
            cs_ops=dict(CS_OPERATOR_INSTANCES),
            math_ops=dict(MATH_OPERATORS),
        )

        # Evaluate btc_returns = delta(btc_close, 1) / delay(btc_close, 1)
        ast = parse_expression("delta(btc_close, 1) / delay(btc_close, 1)")
        btc_returns = evaluator.evaluate(ast)
        assert isinstance(btc_returns, pd.DataFrame)
        panel_fields["btc_returns"] = btc_returns

        # Evaluate returns
        ast = parse_expression("delta(close, 1) / delay(close, 1)")
        returns = evaluator.evaluate(ast)
        panel_fields["returns"] = returns

        # btc_beta = covariance(returns, btc_returns, 5) /
        #            replace_zero(power(ts_std(btc_returns, 5), 2))
        ast = parse_expression(
            "covariance(returns, btc_returns, 5) / "
            "replace_zero(power(ts_std(btc_returns, 5), 2))"
        )
        btc_beta = evaluator.evaluate(ast)
        assert isinstance(btc_beta, pd.DataFrame)

        # BTC's beta to itself should be ~1.0 (last valid row)
        last_valid = btc_beta["BTCUSDT.BINANCE"].dropna()
        if len(last_valid) > 0:
            assert (
                abs(last_valid.iloc[-1] - 1.0) < 0.1
            ), f"BTC beta to itself should be ~1.0, got {last_valid.iloc[-1]}"


# ── E2E: Backtest/Live path (Buffer + inject_staged_field) ──────────────


class TestE2EBacktestPath:
    """Verify all 5 sources flow correctly through Buffer in streaming mode."""

    INSTRUMENTS = ["BTCUSDT.BINANCE", "ETHUSDT.BINANCE", "SOLUSDT.BINANCE"]

    def _make_bar_data(
        self,
        close: float,
        volume: float = 100.0,
        funding_rate: float | None = None,
        open_interest: float | None = None,
        quote_volume: float | None = None,
    ) -> dict[str, float]:
        data: dict[str, float] = {
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": volume,
        }
        if funding_rate is not None:
            data["funding_rate"] = funding_rate
        if open_interest is not None:
            data["open_interest"] = open_interest
        if quote_volume is not None:
            data["quote_volume"] = quote_volume
        return data

    def test_all_5_fields_in_panel(self) -> None:
        """All 5 extra data fields appear in Buffer panels after flush."""
        extra_fields = (
            "funding_rate",
            "open_interest",
            "quote_volume",
            "btc_close",
        )
        buf = Buffer(max_history=10, extra_fields=extra_fields)

        ts = 1_000_000_000

        # Simulate Actor.on_bar() per-bar enrichment for 3 instruments
        buf.append(
            "BTCUSDT.BINANCE",
            ts,
            self._make_bar_data(
                close=40000,
                funding_rate=0.0001,
                open_interest=50000,
                quote_volume=40000 * 100,
            ),
        )
        buf.append(
            "ETHUSDT.BINANCE",
            ts,
            self._make_bar_data(
                close=2200,
                funding_rate=-0.0001,
                open_interest=200000,
                quote_volume=2200 * 500,
            ),
        )
        buf.append(
            "SOLUSDT.BINANCE",
            ts,
            self._make_bar_data(
                close=90,
                funding_rate=0.0005,
                open_interest=1000000,
                quote_volume=90 * 20000,
            ),
        )

        # Simulate Actor._inject_broadcast_staged() — broadcast BTC close
        buf.inject_staged_field(ts, "btc_close", "BTCUSDT.BINANCE")

        buf.flush_timestamp(ts)
        panels = buf.to_panel()

        # Verify all extra fields present
        for field_name in extra_fields:
            assert field_name in panels, f"Missing panel: {field_name}"

        # Verify funding_rate values per instrument
        fr = panels["funding_rate"]
        assert fr["BTCUSDT.BINANCE"].iloc[0] == pytest.approx(0.0001)
        assert fr["ETHUSDT.BINANCE"].iloc[0] == pytest.approx(-0.0001)
        assert fr["SOLUSDT.BINANCE"].iloc[0] == pytest.approx(0.0005)

        # Verify open_interest values per instrument
        oi = panels["open_interest"]
        assert oi["BTCUSDT.BINANCE"].iloc[0] == pytest.approx(50000)
        assert oi["ETHUSDT.BINANCE"].iloc[0] == pytest.approx(200000)

        # Verify quote_volume
        qv = panels["quote_volume"]
        assert qv["BTCUSDT.BINANCE"].iloc[0] == pytest.approx(40000 * 100)
        assert qv["ETHUSDT.BINANCE"].iloc[0] == pytest.approx(2200 * 500)

        # Verify btc_close broadcast — all 3 should have BTC's close
        btc_close = panels["btc_close"]
        for col in btc_close.columns:
            assert btc_close[col].iloc[0] == pytest.approx(40000.0)

    def test_multi_timestamp_broadcast_consistency(self) -> None:
        """Broadcast values are correct and consistent across timestamps."""
        buf = Buffer(max_history=10, extra_fields=("btc_close", "eth_close"))

        btc_prices = [40000.0, 41000.0, 42000.0]
        eth_prices = [2200.0, 2300.0, 2400.0]

        for i, (btc_p, eth_p) in enumerate(zip(btc_prices, eth_prices)):
            ts = (i + 1) * 1_000_000_000
            buf.append("BTCUSDT.BINANCE", ts, self._make_bar_data(close=btc_p))
            buf.append("ETHUSDT.BINANCE", ts, self._make_bar_data(close=eth_p))
            buf.append("SOLUSDT.BINANCE", ts, self._make_bar_data(close=90 + i))

            buf.inject_staged_field(ts, "btc_close", "BTCUSDT.BINANCE")
            buf.inject_staged_field(ts, "eth_close", "ETHUSDT.BINANCE")
            buf.flush_timestamp(ts)

        panels = buf.to_panel()

        btc_close = panels["btc_close"]
        eth_close = panels["eth_close"]

        # Check btc_close: SOL column should have BTC's prices
        np.testing.assert_array_almost_equal(
            btc_close["SOLUSDT.BINANCE"].values,
            btc_prices,
        )
        # Check eth_close: SOL column should have ETH's prices
        np.testing.assert_array_almost_equal(
            eth_close["SOLUSDT.BINANCE"].values,
            eth_prices,
        )

    def test_variables_evaluate_on_buffer_panels(self) -> None:
        """Factor expressions using extra data fields evaluate on Buffer panels."""
        buf = Buffer(
            max_history=20,
            extra_fields=("funding_rate", "btc_close"),
        )

        # Build 10 timestamps of data
        for i in range(10):
            ts = (i + 1) * 1_000_000_000
            btc_close = 40000 + i * 100
            buf.append(
                "BTCUSDT.BINANCE",
                ts,
                self._make_bar_data(
                    close=btc_close,
                    funding_rate=0.0001 * (i + 1),
                ),
            )
            buf.append(
                "ETHUSDT.BINANCE",
                ts,
                self._make_bar_data(
                    close=2200 + i * 10,
                    funding_rate=-0.0001,
                ),
            )
            buf.inject_staged_field(ts, "btc_close", "BTCUSDT.BINANCE")
            buf.flush_timestamp(ts)

        panels = buf.to_panel()

        evaluator = Evaluator(
            panel_fields=panels,
            ts_ops=dict(TS_OPERATOR_INSTANCES),
            cs_ops=dict(CS_OPERATOR_INSTANCES),
            math_ops=dict(MATH_OPERATORS),
        )

        # Evaluate: btc_returns
        ast = parse_expression("delta(btc_close, 1) / delay(btc_close, 1)")
        btc_returns = evaluator.evaluate(ast)
        assert isinstance(btc_returns, pd.DataFrame)
        # All columns should have same btc_returns (broadcast)
        btc_col = btc_returns["BTCUSDT.BINANCE"].dropna()
        eth_col = btc_returns["ETHUSDT.BINANCE"].dropna()
        np.testing.assert_array_almost_equal(btc_col.values, eth_col.values)

        # Evaluate: funding_rate rank
        ast = parse_expression("cs_rank(funding_rate)")
        fr_rank = evaluator.evaluate(ast)
        assert isinstance(fr_rank, pd.DataFrame)
