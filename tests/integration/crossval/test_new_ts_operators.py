# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Cross-validation: new time-series operators vs qlib / pandas reference.

Compares compute_panel() output of TsSlope, TsRsquare, TsResidual,
TsPercentile, and Ema against qlib's Slope, Rsquare, Resi, Quantile
expressions and pandas ewm respectively, using real 4h crypto data.

Qlib stores data in float32 and its regression operators (Slope, Rsquare,
Resi) use an incremental (online) Cython implementation that accumulates
running sums. Our operators use a batch (per-window) approach. Both are
mathematically equivalent, but the different arithmetic order produces
small floating-point divergences -- especially for TsResidual / Resi where
cancellation of large numbers amplifies the rounding difference.

For TsResidual we therefore use a relaxed tolerance (atol) and verify
correlation > 0.9999 rather than requiring bitwise-exact match.

Usage:
    pytest tests/integration/crossval/test_new_ts_operators.py -v
"""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Skip entire module if qlib is not installed
# ---------------------------------------------------------------------------
qlib = pytest.importorskip("qlib", reason="qlib not installed -- skipping crossval tests")

from qlib.data import D  # noqa: E402

from nautilus_quants.factors.operators.time_series import (  # noqa: E402
    Ema,
    TsPercentile,
    TsResidual,
    TsRsquare,
    TsSlope,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROVIDER_URI = "~/.qlib/qlib_data/crypto_data_perp_4h"
INSTRUMENTS = [
    "BINANCE_UM.BTCUSDT",
    "BINANCE_UM.ETHUSDT",
    "BINANCE_UM.SOLUSDT",
    "BINANCE_UM.BNBUSDT",
    "BINANCE_UM.ADAUSDT",
]
FREQ = "240min"
WINDOW = 20
QUANTILE_Q = 0.8
EMA_SPAN = 12
RTOL = 1e-6


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module", autouse=True)
def init_qlib():
    """Initialize qlib once per module."""
    qlib.init(provider_uri=PROVIDER_URI, region="cn")


@pytest.fixture(scope="module")
def close_panel() -> pd.DataFrame:
    """Load $close for all instruments as a panel DataFrame[T x N].

    Returns float32 data (qlib's native storage dtype).
    """
    df = D.features(
        instruments=INSTRUMENTS,
        fields=["$close"],
        start_time="2022-01-01",
        end_time="2025-06-30",
        freq=FREQ,
    )
    # D.features returns MultiIndex (instrument, datetime); pivot to panel
    close = df["$close"].unstack(level=0)
    return close[INSTRUMENTS]


# ---------------------------------------------------------------------------
# Qlib reference loaders
# ---------------------------------------------------------------------------
def _qlib_feature(expression: str) -> pd.DataFrame:
    """Fetch a single qlib expression for all instruments, return panel."""
    df = D.features(
        instruments=INSTRUMENTS,
        fields=[expression],
        start_time="2022-01-01",
        end_time="2025-06-30",
        freq=FREQ,
    )
    series = df.iloc[:, 0]
    panel = series.unstack(level=0)
    return panel[INSTRUMENTS]


@pytest.fixture(scope="module")
def qlib_slope() -> pd.DataFrame:
    return _qlib_feature(f"Slope($close, {WINDOW})")


@pytest.fixture(scope="module")
def qlib_rsquare() -> pd.DataFrame:
    return _qlib_feature(f"Rsquare($close, {WINDOW})")


@pytest.fixture(scope="module")
def qlib_resi() -> pd.DataFrame:
    return _qlib_feature(f"Resi($close, {WINDOW})")


@pytest.fixture(scope="module")
def qlib_quantile() -> pd.DataFrame:
    return _qlib_feature(f"Quantile($close, {WINDOW}, {QUANTILE_Q})")


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------
def _align_panels(
    naut: pd.DataFrame,
    ref: pd.DataFrame,
    label: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Align two panels on common valid entries, return flat float64 arrays.

    Returns (naut_vals, ref_vals, n_valid).
    """
    common_idx = naut.index.intersection(ref.index)
    common_cols = naut.columns.intersection(ref.columns)
    assert len(common_idx) > 0, f"{label}: no overlapping datetime index"
    assert len(common_cols) > 0, f"{label}: no overlapping columns"

    n = naut.loc[common_idx, common_cols].astype(np.float64)
    r = ref.loc[common_idx, common_cols].astype(np.float64)

    both_valid = n.notna() & r.notna()
    n_valid = int(both_valid.sum().sum())
    assert n_valid > 100, (
        f"{label}: only {n_valid} jointly valid values -- too few for meaningful comparison"
    )

    return n.values[both_valid.values], r.values[both_valid.values], n_valid


def _assert_allclose_aligned(
    naut: pd.DataFrame,
    ref: pd.DataFrame,
    rtol: float = RTOL,
    label: str = "",
) -> None:
    """Align on shared valid indices and assert allclose (rtol)."""
    naut_vals, ref_vals, n_valid = _align_panels(naut, ref, label)
    np.testing.assert_allclose(
        naut_vals,
        ref_vals,
        rtol=rtol,
        err_msg=f"{label}: nautilus vs reference mismatch ({n_valid} values compared)",
    )


def _assert_close_with_correlation(
    naut: pd.DataFrame,
    ref: pd.DataFrame,
    min_corr: float = 0.9999,
    atol: float = 0.05,
    label: str = "",
) -> None:
    """Assert two panels are close using correlation + absolute tolerance.

    Used for operators where batch vs incremental algorithms cause
    floating-point divergence that inflates relative error (e.g.
    residuals near zero).
    """
    naut_vals, ref_vals, n_valid = _align_panels(naut, ref, label)

    # 1. Correlation must be near-perfect
    corr = float(np.corrcoef(naut_vals, ref_vals)[0, 1])
    assert corr >= min_corr, (
        f"{label}: correlation {corr:.10f} < {min_corr} "
        f"({n_valid} values compared)"
    )

    # 2. Absolute differences must be small
    max_abs_diff = float(np.max(np.abs(naut_vals - ref_vals)))
    assert max_abs_diff <= atol, (
        f"{label}: max absolute difference {max_abs_diff:.6e} > {atol} "
        f"({n_valid} values compared)"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestTsSlopeVsQlib:
    """ts_slope(close, 20) vs qlib Slope($close, 20)."""

    def test_slope_matches(self, close_panel: pd.DataFrame, qlib_slope: pd.DataFrame) -> None:
        naut = TsSlope().compute_panel(close_panel, WINDOW)
        _assert_allclose_aligned(naut, qlib_slope, label="TsSlope")


class TestTsRsquareVsQlib:
    """ts_rsquare(close, 20) vs qlib Rsquare($close, 20)."""

    def test_rsquare_matches(self, close_panel: pd.DataFrame, qlib_rsquare: pd.DataFrame) -> None:
        naut = TsRsquare().compute_panel(close_panel, WINDOW)
        _assert_allclose_aligned(naut, qlib_rsquare, label="TsRsquare")


class TestTsResidualVsQlib:
    """ts_residual(close, 20) vs qlib Resi($close, 20).

    KNOWN DIFF: Qlib Resi uses an incremental (online) Cython algorithm
    with running sums (see qlib/data/_libs/rolling.pyx Resi class).
    Our TsResidual uses a batch (per-window) computation.  Both are
    mathematically equivalent OLS residuals, but the different arithmetic
    order causes floating-point divergence.  The residual = y - y_hat
    involves cancellation of large numbers, which amplifies the rounding
    difference.

    Measured on real data:
      - Correlation:     > 0.999999999
      - Max abs diff:    ~ 0.02  (for BTC prices ~40-100k)
      - P99 rel diff:    ~ 3e-4
    """

    def test_residual_matches(self, close_panel: pd.DataFrame, qlib_resi: pd.DataFrame) -> None:
        naut = TsResidual().compute_panel(close_panel, WINDOW)
        _assert_close_with_correlation(
            naut,
            qlib_resi,
            min_corr=0.999999,
            atol=0.05,
            label="TsResidual",
        )


class TestTsPercentileVsQlib:
    """ts_percentile(close, 20, 0.8) vs qlib Quantile($close, 20, 0.8)."""

    def test_percentile_matches(
        self, close_panel: pd.DataFrame, qlib_quantile: pd.DataFrame
    ) -> None:
        naut = TsPercentile().compute_panel(close_panel, WINDOW, extra_0=QUANTILE_Q)
        _assert_allclose_aligned(naut, qlib_quantile, label="TsPercentile")


class TestEmaVsPandas:
    """ema(close, 12) vs pandas ewm(span=12, adjust=False).mean().

    No qlib equivalent; pandas ewm is the canonical reference.
    Both operate on the same DataFrame so precision matches exactly.
    """

    def test_ema_matches(self, close_panel: pd.DataFrame) -> None:
        naut = Ema().compute_panel(close_panel, EMA_SPAN)
        ref = close_panel.ewm(span=EMA_SPAN, adjust=False).mean()
        _assert_allclose_aligned(naut, ref, label="Ema")


# ---------------------------------------------------------------------------
# Parametrized summary test (all 5 operators in one parametrize)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "operator_cls, window, kwargs, qlib_expr, use_qlib_ref",
    [
        (TsSlope, WINDOW, {}, f"Slope($close, {WINDOW})", True),
        (TsRsquare, WINDOW, {}, f"Rsquare($close, {WINDOW})", True),
        (TsResidual, WINDOW, {}, f"Resi($close, {WINDOW})", True),
        (TsPercentile, WINDOW, {"extra_0": QUANTILE_Q}, f"Quantile($close, {WINDOW}, {QUANTILE_Q})", True),
        (Ema, EMA_SPAN, {}, None, False),
    ],
    ids=["ts_slope", "ts_rsquare", "ts_residual", "ts_percentile", "ema"],
)
def test_operator_vs_reference(
    close_panel: pd.DataFrame,
    operator_cls,
    window: int,
    kwargs: dict,
    qlib_expr: str | None,
    use_qlib_ref: bool,
) -> None:
    """Parametrized cross-validation of all 5 new operators."""
    naut = operator_cls().compute_panel(close_panel, window, **kwargs)

    if use_qlib_ref:
        ref = _qlib_feature(qlib_expr)
    else:
        # EMA: pandas reference
        ref = close_panel.ewm(span=window, adjust=False).mean()

    if operator_cls is TsResidual:
        # Known batch-vs-incremental precision diff (see TestTsResidualVsQlib)
        _assert_close_with_correlation(
            naut, ref, min_corr=0.999999, atol=0.05, label=operator_cls.__name__,
        )
    else:
        _assert_allclose_aligned(naut, ref, label=operator_cls.__name__)
