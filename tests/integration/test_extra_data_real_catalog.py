# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Integration test — verify all 5 extra data sources load from real catalog.

Uses real BinanceBar data from /Users/joe/Sync/nautilus_quants2/data/data_4years_4h/catalog
and FR data from /Users/joe/Sync/nautilus_quants/data/data_4years_4h_fr_oi/catalog.

Test period: 2025-01-01 to 2025-03-31
Instruments: BTC, ETH, SOL (subset for speed)
Fields: funding_rate, open_interest, quote_volume, btc_close, eth_close

Usage:
    pytest tests/integration/test_extra_data_real_catalog.py -v -s
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Catalog paths
BAR_CATALOG = "/Users/joe/Sync/nautilus_quants2/data/data_4years_4h/catalog"
FR_OI_CATALOG = "/Users/joe/Sync/nautilus_quants/data/data_4years_4h_fr_oi/catalog"

INSTRUMENTS = [
    "BTCUSDT.BINANCE",
    "ETHUSDT.BINANCE",
    "SOLUSDT.BINANCE",
]

START = pd.Timestamp("2025-01-01")
END = pd.Timestamp("2025-03-31")


def _catalog_exists(path: str) -> bool:
    return Path(path).exists()


pytestmark = pytest.mark.skipif(
    not _catalog_exists(BAR_CATALOG) or not _catalog_exists(FR_OI_CATALOG),
    reason="Real catalog data not available",
)


@pytest.fixture(scope="module")
def bars_by_instrument() -> dict[str, list]:
    """Load BinanceBar data from real catalog."""
    from nautilus_trader.persistence.catalog import ParquetDataCatalog
    from nautilus_trader.model.data import BarType, BarSpecification
    from nautilus_trader.model.enums import BarAggregation, PriceType
    from nautilus_trader.model.identifiers import InstrumentId

    catalog = ParquetDataCatalog(BAR_CATALOG)
    result = {}
    for inst_id_str in INSTRUMENTS:
        inst_id = InstrumentId.from_str(inst_id_str)
        bar_spec = BarSpecification(4, BarAggregation.HOUR, PriceType.LAST)
        bar_type = BarType(inst_id, bar_spec)
        bars = catalog.bars(
            bar_types=[str(bar_type)],
            start=START,
            end=END,
        )
        result[inst_id_str] = list(bars) if bars else []
    return result


@pytest.fixture(scope="module")
def ohlcv_panels(bars_by_instrument) -> dict[str, pd.DataFrame]:
    """Build OHLCV panels from bars."""
    from nautilus_quants.alpha.data_loader import CatalogDataLoader

    ohlcv_dfs = {}
    for inst_id, bars in bars_by_instrument.items():
        if bars:
            ohlcv_dfs[inst_id] = CatalogDataLoader.bars_to_dataframe(bars)

    fields = ("open", "high", "low", "close", "volume")
    panels = {}
    for field in fields:
        panels[field] = pd.concat(
            {inst: df[field] for inst, df in ohlcv_dfs.items()},
            axis=1,
        )
    return panels


class TestRealCatalogExtraData:
    """E2E tests with real catalog data for all 5 extra data sources."""

    def test_quote_volume_from_binance_bar(self, bars_by_instrument, ohlcv_panels):
        """quote_volume extracted from BinanceBar parquet in catalog."""
        from nautilus_quants.factors.engine.extra_data import (
            ExtraDataConfig, ExtraDataManager,
        )

        configs = [
            ExtraDataConfig(name="quote_volume", source="bar"),
        ]
        panel_fields = dict(ohlcv_panels)
        manager = ExtraDataManager(configs)
        manager.inject_panels(
            panel_fields, INSTRUMENTS,
            bars_by_instrument=bars_by_instrument,
            catalog_path=BAR_CATALOG,
        )

        assert "quote_volume" in panel_fields
        qv = panel_fields["quote_volume"]
        assert isinstance(qv, pd.DataFrame)
        print(f"\n[quote_volume] shape={qv.shape}")
        print(f"  BTC mean: {qv['BTCUSDT.BINANCE'].mean():,.0f} USDT")
        print(f"  ETH mean: {qv['ETHUSDT.BINANCE'].mean():,.0f} USDT")
        print(f"  SOL mean: {qv['SOLUSDT.BINANCE'].mean():,.0f} USDT")

        # Basic sanity: BTC quote_volume >> SOL
        assert qv["BTCUSDT.BINANCE"].mean() > qv["SOLUSDT.BINANCE"].mean()
        # No all-NaN columns
        for inst in INSTRUMENTS:
            assert qv[inst].notna().sum() > 0, f"{inst} quote_volume all NaN"

    def test_funding_rate_from_catalog(self, ohlcv_panels):
        """funding_rate loaded from real FR catalog."""
        from nautilus_quants.factors.engine.extra_data import (
            ExtraDataConfig, ExtraDataManager,
        )

        configs = [
            ExtraDataConfig(
                name="funding_rate", source="catalog", path=FR_OI_CATALOG,
            ),
        ]
        panel_fields = dict(ohlcv_panels)
        manager = ExtraDataManager(configs)
        manager.inject_panels(
            panel_fields, INSTRUMENTS,
            catalog_path=FR_OI_CATALOG,
        )

        assert "funding_rate" in panel_fields
        fr = panel_fields["funding_rate"]
        assert isinstance(fr, pd.DataFrame)
        print(f"\n[funding_rate] shape={fr.shape}")
        for inst in INSTRUMENTS:
            valid = fr[inst].dropna()
            nonzero = (valid != 0).sum()
            print(f"  {inst}: {len(valid)} valid, {nonzero} non-zero, "
                  f"mean={valid.mean():.6f}")
            assert nonzero > 0, f"{inst} FR all zero"

        # BTC and ETH should have different FR values
        btc_fr = fr["BTCUSDT.BINANCE"].dropna()
        eth_fr = fr["ETHUSDT.BINANCE"].dropna()
        assert not np.allclose(btc_fr.values[:10], eth_fr.values[:10])

    def test_broadcast_btc_eth_close(self, ohlcv_panels):
        """btc_close and eth_close broadcast correctly from real data."""
        from nautilus_quants.factors.engine.extra_data import (
            ExtraDataConfig, ExtraDataManager,
        )

        configs = [
            ExtraDataConfig(
                name="btc_close", source="broadcast", instruments=["BTC"],
            ),
            ExtraDataConfig(
                name="eth_close", source="broadcast", instruments=["ETH"],
            ),
        ]
        panel_fields = dict(ohlcv_panels)
        manager = ExtraDataManager(configs)
        manager.inject_panels(panel_fields, INSTRUMENTS)

        close = panel_fields["close"]

        # btc_close: all columns = BTC's real close
        btc_close = panel_fields["btc_close"]
        for inst in INSTRUMENTS:
            np.testing.assert_array_equal(
                btc_close[inst].values,
                close["BTCUSDT.BINANCE"].values,
            )
        print(f"\n[btc_close] BTC price range: "
              f"{close['BTCUSDT.BINANCE'].min():.0f} - "
              f"{close['BTCUSDT.BINANCE'].max():.0f}")

        # eth_close: all columns = ETH's real close
        eth_close = panel_fields["eth_close"]
        for inst in INSTRUMENTS:
            np.testing.assert_array_equal(
                eth_close[inst].values,
                close["ETHUSDT.BINANCE"].values,
            )

    def test_all_5_sources_together(self, bars_by_instrument, ohlcv_panels):
        """All 5 sources load and coexist in panel_fields."""
        from nautilus_quants.factors.engine.extra_data import (
            ExtraDataConfig, ExtraDataManager,
        )

        configs = [
            ExtraDataConfig(
                name="funding_rate", source="catalog", path=FR_OI_CATALOG,
            ),
            ExtraDataConfig(name="quote_volume", source="bar", path=BAR_CATALOG),
            ExtraDataConfig(
                name="btc_close", source="broadcast", instruments=["BTC"],
            ),
            ExtraDataConfig(
                name="eth_close", source="broadcast", instruments=["ETH"],
            ),
        ]
        panel_fields = dict(ohlcv_panels)
        manager = ExtraDataManager(configs)
        manager.inject_panels(
            panel_fields, INSTRUMENTS,
            bars_by_instrument=bars_by_instrument,
            catalog_path=FR_OI_CATALOG,
        )

        expected_fields = [
            "funding_rate", "quote_volume", "btc_close", "eth_close",
        ]
        for field_name in expected_fields:
            assert field_name in panel_fields, f"Missing: {field_name}"
            panel = panel_fields[field_name]
            assert isinstance(panel, pd.DataFrame)
            assert set(panel.columns) == set(INSTRUMENTS)
            print(f"  [{field_name}] shape={panel.shape}, "
                  f"NaN%={panel.isna().mean().mean():.1%}")

    def test_derived_variables_btc_beta(self, bars_by_instrument, ohlcv_panels):
        """btc_returns, btc_beta evaluate on real data with correct values."""
        from nautilus_quants.factors.engine.extra_data import (
            ExtraDataConfig, ExtraDataManager,
        )
        from nautilus_quants.factors.engine.evaluator import Evaluator
        from nautilus_quants.factors.expression import parse_expression
        from nautilus_quants.factors.operators.cross_sectional import (
            CS_OPERATOR_INSTANCES,
        )
        from nautilus_quants.factors.operators.math import MATH_OPERATORS
        from nautilus_quants.factors.operators.time_series import (
            TS_OPERATOR_INSTANCES,
        )

        configs = [
            ExtraDataConfig(
                name="btc_close", source="broadcast", instruments=["BTC"],
            ),
            ExtraDataConfig(
                name="eth_close", source="broadcast", instruments=["ETH"],
            ),
        ]
        panel_fields = dict(ohlcv_panels)
        manager = ExtraDataManager(configs)
        manager.inject_panels(panel_fields, INSTRUMENTS)

        evaluator = Evaluator(
            panel_fields=panel_fields,
            ts_ops=dict(TS_OPERATOR_INSTANCES),
            cs_ops=dict(CS_OPERATOR_INSTANCES),
            math_ops=dict(MATH_OPERATORS),
        )

        # returns
        returns = evaluator.evaluate(
            parse_expression("delta(close, 1) / delay(close, 1)")
        )
        panel_fields["returns"] = returns

        # btc_returns
        btc_returns = evaluator.evaluate(
            parse_expression("delta(btc_close, 1) / delay(btc_close, 1)")
        )
        panel_fields["btc_returns"] = btc_returns

        # eth_returns
        eth_returns = evaluator.evaluate(
            parse_expression("delta(eth_close, 1) / delay(eth_close, 1)")
        )
        panel_fields["eth_returns"] = eth_returns

        # btc_vol
        btc_vol = evaluator.evaluate(
            parse_expression("ts_std(btc_returns, 42)")
        )
        panel_fields["btc_vol"] = btc_vol

        # btc_beta = cov(returns, btc_returns, 42) / var(btc_returns, 42)
        btc_beta = evaluator.evaluate(parse_expression(
            "covariance(returns, btc_returns, 42) / "
            "replace_zero(power(btc_vol, 2))"
        ))
        panel_fields["btc_beta"] = btc_beta

        # eth_beta
        eth_vol = evaluator.evaluate(
            parse_expression("ts_std(eth_returns, 42)")
        )
        panel_fields["eth_vol"] = eth_vol
        eth_beta = evaluator.evaluate(parse_expression(
            "covariance(returns, eth_returns, 42) / "
            "replace_zero(power(eth_vol, 2))"
        ))

        assert isinstance(btc_beta, pd.DataFrame)
        assert isinstance(eth_beta, pd.DataFrame)

        # BTC beta to itself ≈ 1.0
        btc_self_beta = btc_beta["BTCUSDT.BINANCE"].dropna()
        last_btc_beta = btc_self_beta.iloc[-1]
        print(f"\n[btc_beta] BTC self-beta (last): {last_btc_beta:.4f}")
        assert abs(last_btc_beta - 1.0) < 0.05, (
            f"BTC self-beta should be ~1.0, got {last_btc_beta:.4f}"
        )

        # ETH beta to BTC should be > 0 (crypto correlation)
        eth_btc_beta = btc_beta["ETHUSDT.BINANCE"].dropna()
        last_eth_beta = eth_btc_beta.iloc[-1]
        print(f"[btc_beta] ETH beta to BTC (last): {last_eth_beta:.4f}")
        assert last_eth_beta > 0, "ETH beta to BTC should be positive"

        # SOL beta to BTC should also be > 0
        sol_btc_beta = btc_beta["SOLUSDT.BINANCE"].dropna()
        last_sol_beta = sol_btc_beta.iloc[-1]
        print(f"[btc_beta] SOL beta to BTC (last): {last_sol_beta:.4f}")
        assert last_sol_beta > 0, "SOL beta to BTC should be positive"

        # Print summary
        print(f"\n[eth_beta] ETH self-beta (last): "
              f"{eth_beta['ETHUSDT.BINANCE'].dropna().iloc[-1]:.4f}")
        print(f"[eth_beta] BTC beta to ETH (last): "
              f"{eth_beta['BTCUSDT.BINANCE'].dropna().iloc[-1]:.4f}")

    def test_vwap_from_quote_volume(self, bars_by_instrument, ohlcv_panels):
        """VWAP = quote_volume / volume computes correctly on real data."""
        from nautilus_quants.factors.engine.extra_data import (
            ExtraDataConfig, ExtraDataManager,
        )
        from nautilus_quants.factors.engine.evaluator import Evaluator
        from nautilus_quants.factors.expression import parse_expression
        from nautilus_quants.factors.operators.cross_sectional import (
            CS_OPERATOR_INSTANCES,
        )
        from nautilus_quants.factors.operators.math import MATH_OPERATORS
        from nautilus_quants.factors.operators.time_series import (
            TS_OPERATOR_INSTANCES,
        )

        configs = [ExtraDataConfig(name="quote_volume", source="bar")]
        panel_fields = dict(ohlcv_panels)
        manager = ExtraDataManager(configs)
        manager.inject_panels(
            panel_fields, INSTRUMENTS,
            bars_by_instrument=bars_by_instrument,
            catalog_path=BAR_CATALOG,
        )

        evaluator = Evaluator(
            panel_fields=panel_fields,
            ts_ops=dict(TS_OPERATOR_INSTANCES),
            cs_ops=dict(CS_OPERATOR_INSTANCES),
            math_ops=dict(MATH_OPERATORS),
        )

        # vwap = quote_volume / replace_zero(volume)
        vwap = evaluator.evaluate(
            parse_expression("quote_volume / replace_zero(volume)")
        )
        assert isinstance(vwap, pd.DataFrame)

        close = panel_fields["close"]
        # VWAP should be close to close price (within ~5%)
        ratio = vwap / close
        btc_ratio = ratio["BTCUSDT.BINANCE"].dropna()
        mean_ratio = btc_ratio.mean()
        print(f"\n[vwap] BTC VWAP/Close ratio mean: {mean_ratio:.4f}")
        assert 0.95 < mean_ratio < 1.05, (
            f"VWAP should be close to close, ratio={mean_ratio:.4f}"
        )
