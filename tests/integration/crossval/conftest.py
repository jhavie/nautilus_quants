# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Shared fixtures for factor engine cross-validation tests.

Provides qlib initialization, OHLCV panel loading, and comparison helpers
used across all crossval test modules.

Usage:
    pytest tests/integration/crossval/ -v
    XVAL_FULL=1 pytest tests/integration/crossval/ -v  # all instruments
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Skip entire package if qlib is not installed
# ---------------------------------------------------------------------------
qlib = pytest.importorskip("qlib", reason="qlib not installed -- skipping crossval tests")

from qlib.data import D  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROVIDER_URI = "~/.qlib/qlib_data/crypto_data_perp_4h"
FREQ = "240min"
START = "2022-01-01"
END = "2025-06-30"
RTOL = 1e-6

INSTRUMENTS_SMALL = [
    "BINANCE_UM.BTCUSDT",
    "BINANCE_UM.ETHUSDT",
    "BINANCE_UM.SOLUSDT",
    "BINANCE_UM.BNBUSDT",
    "BINANCE_UM.ADAUSDT",
    "BINANCE_UM.XRPUSDT",
    "BINANCE_UM.DOGEUSDT",
    "BINANCE_UM.AVAXUSDT",
    "BINANCE_UM.LINKUSDT",
    "BINANCE_UM.DOTUSDT",
]

INSTRUMENTS_FULL = None  # populated at runtime if XVAL_FULL=1


def get_instruments() -> list[str]:
    """Return instrument list based on XVAL_FULL env var."""
    if os.environ.get("XVAL_FULL"):
        return D.instruments(market="all")
    return INSTRUMENTS_SMALL


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def init_qlib():
    """Initialize qlib once per session."""
    qlib.init(provider_uri=PROVIDER_URI, region="cn")


@pytest.fixture(scope="session")
def instruments() -> list[str]:
    return get_instruments()


@pytest.fixture(scope="session")
def ohlcv_panel(instruments: list[str]) -> dict[str, pd.DataFrame]:
    """Load OHLCV panels for all instruments.

    Returns:
        {"open": DataFrame[T×N], "high": ..., "low": ..., "close": ..., "volume": ...}
    """
    fields = ["$open", "$high", "$low", "$close", "$volume"]
    df = D.features(
        instruments=instruments,
        fields=fields,
        start_time=START,
        end_time=END,
        freq=FREQ,
    )
    panels = {}
    for field in fields:
        name = field.replace("$", "")
        panel = df[field].unstack(level=0)
        panels[name] = panel[instruments]
    return panels


@pytest.fixture(scope="session")
def close_panel(ohlcv_panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return ohlcv_panel["close"]


@pytest.fixture(scope="session")
def high_panel(ohlcv_panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return ohlcv_panel["high"]


@pytest.fixture(scope="session")
def volume_panel(ohlcv_panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return ohlcv_panel["volume"]


# ---------------------------------------------------------------------------
# Qlib feature helper
# ---------------------------------------------------------------------------
def qlib_feature(expression: str, instruments: list[str]) -> pd.DataFrame:
    """Fetch a single qlib expression for all instruments, return panel."""
    df = D.features(
        instruments=instruments,
        fields=[expression],
        start_time=START,
        end_time=END,
        freq=FREQ,
    )
    panel = df.iloc[:, 0].unstack(level=0)
    return panel[instruments]


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------
def align_panels(
    naut: pd.DataFrame,
    ref: pd.DataFrame,
    label: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Align two panels on common valid entries, return flat float64 arrays."""
    common_idx = naut.index.intersection(ref.index)
    common_cols = naut.columns.intersection(ref.columns)
    assert len(common_idx) > 0, f"{label}: no overlapping datetime index"
    assert len(common_cols) > 0, f"{label}: no overlapping columns"

    n = naut.loc[common_idx, common_cols].astype(np.float64)
    r = ref.loc[common_idx, common_cols].astype(np.float64)

    both_valid = n.notna() & r.notna()
    n_valid = int(both_valid.sum().sum())
    assert n_valid > 100, (
        f"{label}: only {n_valid} jointly valid values -- too few"
    )
    return n.values[both_valid.values], r.values[both_valid.values], n_valid


def assert_allclose(
    naut: pd.DataFrame,
    ref: pd.DataFrame,
    rtol: float = RTOL,
    label: str = "",
) -> None:
    """Align on shared valid indices and assert allclose (rtol)."""
    naut_vals, ref_vals, n_valid = align_panels(naut, ref, label)
    np.testing.assert_allclose(
        naut_vals, ref_vals, rtol=rtol,
        err_msg=f"{label}: mismatch ({n_valid} values compared)",
    )


def assert_correlation(
    naut: pd.DataFrame,
    ref: pd.DataFrame,
    min_corr: float = 0.9999,
    atol: float = 0.05,
    label: str = "",
) -> None:
    """Assert two panels match via correlation + absolute tolerance."""
    naut_vals, ref_vals, n_valid = align_panels(naut, ref, label)
    corr = float(np.corrcoef(naut_vals, ref_vals)[0, 1])
    assert corr >= min_corr, (
        f"{label}: correlation {corr:.10f} < {min_corr} ({n_valid} values)"
    )
    max_diff = float(np.max(np.abs(naut_vals - ref_vals)))
    assert max_diff <= atol, (
        f"{label}: max abs diff {max_diff:.6e} > {atol} ({n_valid} values)"
    )


def assert_spearman(
    naut: pd.DataFrame,
    ref: pd.DataFrame,
    min_corr: float = 0.99,
    label: str = "",
) -> None:
    """Assert monotonic relationship via Spearman rank correlation."""
    from scipy.stats import spearmanr

    naut_vals, ref_vals, n_valid = align_panels(naut, ref, label)
    corr, _ = spearmanr(naut_vals, ref_vals)
    assert corr >= min_corr, (
        f"{label}: Spearman {corr:.6f} < {min_corr} ({n_valid} values)"
    )
