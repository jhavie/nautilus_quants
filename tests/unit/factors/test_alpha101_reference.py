# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Alpha101 reference verification tests.

Three-layer verification:
  Layer 1:  wq_ BRAIN operators standalone (using BRAIN doc examples)
  Layer 1b: Academic operators vs popbo alignment
  Layer 2:  Alpha101 expressions — full parametric run on random panel
  Layer 3:  Real BTC/ETH/SOL 4h data end-to-end
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata

from nautilus_quants.factors.builtin.alpha101 import ALPHA101_FACTORS
from nautilus_quants.factors.engine.panel_evaluator import PanelEvaluator
from nautilus_quants.factors.expression import parse_expression
from nautilus_quants.factors.operators.cross_sectional import (
    CS_OPERATOR_INSTANCES,
    CsRank,
    CsScale,
)
from nautilus_quants.factors.operators.math import MATH_OPERATORS
from nautilus_quants.factors.operators.time_series import (
    TS_OPERATOR_INSTANCES,
    TsArgmax,
    TsArgmin,
    TsRank,
    WqTsArgmax,
    WqTsArgmin,
    WqTsRank,
    ts_argmax,
    ts_argmin,
    ts_rank,
    wq_ts_argmax,
    wq_ts_argmin,
    wq_ts_rank,
)

# IndNeutralize alphas — skip these (no cross-industry data)
SKIP_INDNEUTRALIZE = {
    "alpha048", "alpha056", "alpha058", "alpha059", "alpha063", "alpha067",
    "alpha069", "alpha070", "alpha076", "alpha079", "alpha080",
    "alpha082", "alpha087", "alpha089", "alpha090", "alpha091",
    "alpha093", "alpha097", "alpha100",
}

# 12-coin data has 276 rows (4h candles) — sufficient for all Alpha101 windows (max 250)


# =========================================================================
# Layer 1: wq_ BRAIN operators standalone verification
# =========================================================================


class TestWqTsRank:
    """wq_ts_rank: (scipy_rank - 1) / (d - 1), value range [0, 1]."""

    def test_brain_example(self):
        """wq_ts_rank([200, 0, 100], d=3) → 0.5"""
        data = np.array([200.0, 0.0, 100.0])
        result = wq_ts_rank(data, 3)
        assert result == pytest.approx(0.5)

    def test_highest_is_one(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = wq_ts_rank(data, 5)
        assert result == pytest.approx(1.0)

    def test_lowest_is_zero(self):
        data = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = wq_ts_rank(data, 5)
        assert result == pytest.approx(0.0)

    def test_window_one(self):
        data = np.array([42.0])
        result = wq_ts_rank(data, 1)
        assert result == pytest.approx(0.5)

    def test_insufficient_data(self):
        data = np.array([1.0, 2.0])
        result = wq_ts_rank(data, 5)
        assert np.isnan(result)

    def test_compute_panel(self):
        """Panel output matches scalar row-by-row."""
        op = WqTsRank()
        df = pd.DataFrame({"A": [1.0, 3.0, 2.0, 5.0, 4.0],
                           "B": [5.0, 4.0, 3.0, 2.0, 1.0]})
        panel = op.compute_panel(df, 3)
        # Last row: A=[2,5,4], wq_rank=0.5; B=[3,2,1], wq_rank=0.0
        assert panel["A"].iloc[-1] == pytest.approx(0.5)
        assert panel["B"].iloc[-1] == pytest.approx(0.0)


class TestWqTsArgmax:
    """wq_ts_argmax: 0-indexed from today. Today=0, yesterday=1."""

    def test_brain_example(self):
        """wq_ts_argmax: max=9 at index 1 (oldest→today), d=6 → d-1-1=4."""
        data = np.array([4.0, 9.0, 5.0, 8.0, 2.0, 6.0])
        result = wq_ts_argmax(data, 6)
        # max is 9 at position index=1 (from oldest), offset from today = 6-1-1=4
        assert result == pytest.approx(4.0)

    def test_max_today(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = wq_ts_argmax(data, 5)
        assert result == pytest.approx(0.0)

    def test_max_oldest(self):
        data = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = wq_ts_argmax(data, 5)
        assert result == pytest.approx(4.0)

    def test_insufficient_data(self):
        data = np.array([1.0])
        result = wq_ts_argmax(data, 5)
        assert np.isnan(result)

    def test_compute_panel(self):
        op = WqTsArgmax()
        df = pd.DataFrame({"A": [4.0, 9.0, 5.0, 8.0, 2.0, 6.0]})
        panel = op.compute_panel(df, 6)
        assert panel["A"].iloc[-1] == pytest.approx(4.0)


class TestWqTsArgmin:
    """wq_ts_argmin: 0-indexed from today. Today=0, yesterday=1."""

    def test_brain_example(self):
        """wq_ts_argmin: min=2 at index 4 (oldest→today), d=6 → d-1-4=1."""
        data = np.array([4.0, 9.0, 5.0, 8.0, 2.0, 6.0])
        result = wq_ts_argmin(data, 6)
        assert result == pytest.approx(1.0)

    def test_min_today(self):
        data = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = wq_ts_argmin(data, 5)
        assert result == pytest.approx(0.0)

    def test_min_oldest(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = wq_ts_argmin(data, 5)
        assert result == pytest.approx(4.0)

    def test_insufficient_data(self):
        data = np.array([1.0])
        result = wq_ts_argmin(data, 5)
        assert np.isnan(result)

    def test_compute_panel(self):
        op = WqTsArgmin()
        df = pd.DataFrame({"A": [4.0, 9.0, 5.0, 8.0, 2.0, 6.0]})
        panel = op.compute_panel(df, 6)
        assert panel["A"].iloc[-1] == pytest.approx(1.0)


class TestWqVsAcademicRelationship:
    """Verify mathematical relationship between wq_ and academic operators."""

    def test_wq_ts_rank_from_academic(self):
        """wq_ts_rank = (ts_rank_popbo - 1) / (d - 1)."""
        data = np.array([10.0, 3.0, 7.0, 5.0, 8.0])
        d = 5
        # Academic (popbo): scipy.rankdata(method='min')[-1]
        academic = float(rankdata(data[-d:], method="min")[-1])
        wq = wq_ts_rank(data, d)
        assert wq == pytest.approx((academic - 1) / (d - 1))

    def test_wq_ts_argmax_vs_academic(self):
        """wq_ts_argmax = d - 1 - (academic_ts_argmax - 1)."""
        data = np.array([4.0, 9.0, 5.0, 8.0, 2.0, 6.0])
        d = 6
        acad = ts_argmax(data, d)  # 1-indexed from oldest → 2
        wq = wq_ts_argmax(data, d)  # 0-indexed from today → 4
        assert wq == pytest.approx(d - acad)

    def test_wq_ts_argmin_vs_academic(self):
        """wq_ts_argmin = d - 1 - (academic_ts_argmin - 1)."""
        data = np.array([4.0, 9.0, 5.0, 8.0, 2.0, 6.0])
        d = 6
        acad = ts_argmin(data, d)  # 1-indexed from oldest → 5
        wq = wq_ts_argmin(data, d)  # 0-indexed from today → 1
        assert wq == pytest.approx(d - acad)

    def test_panel_consistency(self):
        """Panel output consistent between wq_ and academic for ts_rank."""
        df = pd.DataFrame(
            np.random.RandomState(42).randn(50, 5),
            columns=[f"I{i}" for i in range(5)],
        )
        d = 10
        acad = TsRank().compute_panel(df, d)
        wq = WqTsRank().compute_panel(df, d)
        # wq = (academic - 1) / (d - 1)
        expected = (acad - 1) / (d - 1)
        pd.testing.assert_frame_equal(wq, expected)


# =========================================================================
# Layer 1b: Academic operators vs popbo alignment
# =========================================================================


class TestPopboAlignedTsRank:
    """ts_rank: compute_panel uses scipy.rankdata(method='min')[-1]."""

    def test_rank_last_value(self):
        """ts_rank([200, 0, 100], d=3) → rankdata([200,0,100])[-1] = 2."""
        data = np.array([200.0, 0.0, 100.0])
        op = TsRank()
        df = pd.DataFrame({"A": data})
        result = op.compute_panel(df, 3)
        # scipy.rankdata([200, 0, 100], method='min') = [3, 1, 2]
        assert result["A"].iloc[-1] == pytest.approx(2.0)

    def test_ascending(self):
        """Ascending data → last is highest rank."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        df = pd.DataFrame({"A": data})
        result = TsRank().compute_panel(df, 5)
        assert result["A"].iloc[-1] == pytest.approx(5.0)

    def test_descending(self):
        """Descending data → last is rank 1."""
        data = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        df = pd.DataFrame({"A": data})
        result = TsRank().compute_panel(df, 5)
        assert result["A"].iloc[-1] == pytest.approx(1.0)

    def test_ties(self):
        """Tied values → method='min' rank."""
        data = np.array([3.0, 3.0, 3.0])
        df = pd.DataFrame({"A": data})
        result = TsRank().compute_panel(df, 3)
        # rankdata([3,3,3], method='min') = [1,1,1] → last = 1
        assert result["A"].iloc[-1] == pytest.approx(1.0)


class TestPopboAlignedTsArgmax:
    """ts_argmax: np.argmax + 1 (1-indexed from oldest)."""

    def test_max_at_end(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        df = pd.DataFrame({"A": data})
        result = TsArgmax().compute_panel(df, 5)
        assert result["A"].iloc[-1] == pytest.approx(5.0)

    def test_max_at_start(self):
        data = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        df = pd.DataFrame({"A": data})
        result = TsArgmax().compute_panel(df, 5)
        assert result["A"].iloc[-1] == pytest.approx(1.0)

    def test_breakout_detection(self):
        """ts_argmax(close, 31) == 31 means today is the 31-day high."""
        data = np.array([100.0] * 30 + [105.0])
        df = pd.DataFrame({"A": data})
        result = TsArgmax().compute_panel(df, 31)
        assert result["A"].iloc[-1] == pytest.approx(31.0)


class TestPopboAlignedTsArgmin:
    """ts_argmin: np.argmin + 1 (1-indexed from oldest)."""

    def test_min_at_end(self):
        data = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        df = pd.DataFrame({"A": data})
        result = TsArgmin().compute_panel(df, 5)
        assert result["A"].iloc[-1] == pytest.approx(5.0)

    def test_min_at_start(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        df = pd.DataFrame({"A": data})
        result = TsArgmin().compute_panel(df, 5)
        assert result["A"].iloc[-1] == pytest.approx(1.0)


class TestCsRankAndScale:
    """Verify cross-sectional rank and scale operators."""

    def test_rank_percentile(self):
        """rank([4, 3, 6, 10, 2]) → popbo-aligned: rank(method='min', pct=True)."""
        df = pd.DataFrame({"A": [4], "B": [3], "C": [6], "D": [10], "E": [2]})
        result = CsRank().compute_vectorized(df)
        # Popbo: rank(method='min', pct=True) → (count_less + 1) / n
        # E=2→1/5=0.2, B=3→2/5=0.4, A=4→3/5=0.6, C=6→4/5=0.8, D=10→5/5=1.0
        assert result["E"].iloc[0] == pytest.approx(0.2)
        assert result["B"].iloc[0] == pytest.approx(0.4)
        assert result["A"].iloc[0] == pytest.approx(0.6)
        assert result["C"].iloc[0] == pytest.approx(0.8)
        assert result["D"].iloc[0] == pytest.approx(1.0)

    def test_scale_sum_abs_one(self):
        """scale(x) → each row sum(abs(x)) == 1."""
        df = pd.DataFrame(
            np.random.RandomState(42).randn(10, 5),
            columns=[f"I{i}" for i in range(5)],
        )
        result = CsScale().compute_vectorized(df)
        abs_sums = result.abs().sum(axis=1)
        for s in abs_sums:
            assert s == pytest.approx(1.0, abs=1e-10)


# =========================================================================
# Layer 2: Alpha101 expressions — full parametric run on 300x10 panel
# =========================================================================

TESTABLE_ALPHAS = sorted(set(ALPHA101_FACTORS.keys()) - SKIP_INDNEUTRALIZE)


@pytest.fixture(scope="module")
def big_panel() -> dict[str, pd.DataFrame]:
    """300x10 random panel with enough history for all alphas."""
    rng = np.random.RandomState(42)
    n_ts, n_inst = 300, 10
    instruments = [f"INST_{i}" for i in range(n_inst)]

    close = pd.DataFrame(
        rng.randn(n_ts, n_inst).cumsum(axis=0) + 100, columns=instruments,
    )
    open_ = close.shift(1).bfill() + rng.randn(n_ts, n_inst) * 0.5
    high = pd.DataFrame(
        np.maximum(open_.values, close.values) + np.abs(rng.randn(n_ts, n_inst)),
        columns=instruments,
    )
    low = pd.DataFrame(
        np.minimum(open_.values, close.values) - np.abs(rng.randn(n_ts, n_inst)),
        columns=instruments,
    )
    volume = pd.DataFrame(
        np.abs(rng.randn(n_ts, n_inst)) * 1000 + 500, columns=instruments,
    )
    returns = close / close.shift(1) - 1
    vwap = (high + low + close) / 3

    return {
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume, "returns": returns, "vwap": vwap,
    }


@pytest.fixture(scope="module")
def big_evaluator(big_panel: dict[str, pd.DataFrame]) -> PanelEvaluator:
    return PanelEvaluator(
        panel_fields=big_panel,
        ts_ops=TS_OPERATOR_INSTANCES,
        cs_ops=CS_OPERATOR_INSTANCES,
        math_ops=MATH_OPERATORS,
    )


@pytest.mark.parametrize("alpha_name", TESTABLE_ALPHAS, ids=TESTABLE_ALPHAS)
def test_alpha101_evaluates(
    alpha_name: str,
    big_panel: dict[str, pd.DataFrame],
    big_evaluator: PanelEvaluator,
) -> None:
    """Every non-IndNeutralize alpha evaluates without error on 300x10 panel."""
    expr = ALPHA101_FACTORS[alpha_name]["expression"]
    ast = parse_expression(expr)
    result = big_evaluator.evaluate(ast)

    if isinstance(result, pd.DataFrame):
        assert result.shape[1] == 10, f"{alpha_name}: wrong instrument count"
        assert not result.isna().all().all(), f"{alpha_name}: entirely NaN"
    elif isinstance(result, (int, float)):
        pass  # scalar is acceptable
    else:
        pytest.fail(f"{alpha_name}: unexpected result type {type(result)}")


# =========================================================================
# Layer 3: Real BTC/ETH/SOL 4h data end-to-end
# =========================================================================

DATA_DIR = pathlib.Path(
    "/Users/joe/Sync/nautilus_quants2/data/12coin_raw/binance"
)
SYMBOLS = [
    "AAVEUSDT", "ADAUSDT", "ARBUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT",
    "ETHUSDT", "LINKUSDT", "OPUSDT", "SOLUSDT", "SUIUSDT", "UNIUSDT",
]
N_INSTRUMENTS = len(SYMBOLS)  # 12

_REAL_DATA_AVAILABLE = DATA_DIR.exists() and all(
    list((DATA_DIR / sym / "4h").glob("*.csv")) for sym in SYMBOLS
    if (DATA_DIR / sym / "4h").exists()
)


@pytest.fixture(scope="module")
def real_panel() -> dict[str, pd.DataFrame]:
    """Load BTC/ETH/SOL 4h panel from Binance CSVs."""
    if not _REAL_DATA_AVAILABLE:
        pytest.skip("Real BTC/ETH/SOL data not available")

    frames: dict[str, pd.DataFrame] = {}
    for sym in SYMBOLS:
        csv_path = list((DATA_DIR / sym / "4h").glob("*.csv"))[0]
        df = pd.read_csv(csv_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp")
        frames[sym] = df

    fields = ["open", "high", "low", "close", "volume"]
    panel: dict[str, pd.DataFrame] = {}
    for field in fields:
        panel[field] = pd.DataFrame({sym: frames[sym][field] for sym in SYMBOLS})

    panel["returns"] = panel["close"] / panel["close"].shift(1) - 1
    panel["vwap"] = (panel["high"] + panel["low"] + panel["close"]) / 3
    return panel


@pytest.fixture(scope="module")
def real_evaluator(real_panel: dict[str, pd.DataFrame]) -> PanelEvaluator:
    return PanelEvaluator(
        panel_fields=real_panel,
        ts_ops=TS_OPERATOR_INSTANCES,
        cs_ops=CS_OPERATOR_INSTANCES,
        math_ops=MATH_OPERATORS,
    )


@pytest.mark.skipif(not _REAL_DATA_AVAILABLE, reason="Real data not found")
class TestRealDataPanel:
    """Verify panel construction from real CSVs."""

    def test_panel_shape(self, real_panel: dict[str, pd.DataFrame]) -> None:
        assert set(real_panel.keys()) >= {
            "open", "high", "low", "close", "volume", "returns", "vwap",
        }
        for field, df in real_panel.items():
            assert isinstance(df, pd.DataFrame)
            assert list(df.columns) == SYMBOLS
            assert len(df) >= 250  # Enough for all Alpha101 windows
            assert df.shape[1] == N_INSTRUMENTS
            if field != "returns":
                assert not df.isna().all().all()

    def test_data_ranges(self, real_panel: dict[str, pd.DataFrame]) -> None:
        """Sanity check: BTC close should be 5-digit range, 12 instruments."""
        btc_close = real_panel["close"]["BTCUSDT"]
        assert btc_close.min() > 10000
        assert btc_close.max() < 200000
        assert real_panel["close"].shape[1] == 12


# All non-IndNeutralize alphas (276 rows covers all windows)
_REAL_TESTABLE = sorted(
    set(ALPHA101_FACTORS.keys())
    - SKIP_INDNEUTRALIZE
)


@pytest.mark.skipif(not _REAL_DATA_AVAILABLE, reason="Real data not found")
@pytest.mark.parametrize("alpha_name", _REAL_TESTABLE, ids=_REAL_TESTABLE)
def test_alpha101_real_data(
    alpha_name: str,
    real_panel: dict[str, pd.DataFrame],
    real_evaluator: PanelEvaluator,
) -> None:
    """All non-IndNeutralize alphas evaluate on real 12-coin 4h data."""
    expr = ALPHA101_FACTORS[alpha_name]["expression"]
    ast = parse_expression(expr)
    result = real_evaluator.evaluate(ast)

    assert isinstance(result, pd.DataFrame)
    assert result.shape[1] == N_INSTRUMENTS
    assert not result.isna().all().all(), f"{alpha_name}: entirely NaN"


@pytest.mark.skipif(not _REAL_DATA_AVAILABLE, reason="Real data not found")
class TestPanelFactorEngineE2E:
    """E2E: CSV → PanelFactorEngine → factor values."""

    def test_engine_e2e(self, real_panel: dict[str, pd.DataFrame]) -> None:
        from nautilus_quants.factors.engine.panel_factor_engine import PanelFactorEngine

        engine = PanelFactorEngine(max_history=300)

        # Register a few representative alphas
        sample_alphas = ["alpha001", "alpha003", "alpha006", "alpha012", "alpha101"]
        for name in sample_alphas:
            info = ALPHA101_FACTORS[name]
            engine.register_expression_factor(name, info["expression"])

        n_rows = len(real_panel["close"])
        for row_idx in range(n_rows):
            for sym in SYMBOLS:
                bar = {
                    field: float(real_panel[field][sym].iloc[row_idx])
                    for field in ["open", "high", "low", "close", "volume"]
                }
                bar["vwap"] = float(real_panel["vwap"][sym].iloc[row_idx])
                bar["returns"] = float(real_panel["returns"][sym].iloc[row_idx])
                engine.on_bar(sym, bar, row_idx)
            engine.flush_and_compute(row_idx)

        results = engine.flush_and_compute(n_rows - 1)
        for name in sample_alphas:
            assert name in results, f"{name} missing"
            assert isinstance(results[name], dict)
            assert set(results[name].keys()) <= set(SYMBOLS)
