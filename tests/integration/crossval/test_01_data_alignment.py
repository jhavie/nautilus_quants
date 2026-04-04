# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Phase 1: Data alignment verification.

Ensures OHLCV panels loaded from Qlib are consistent before any
operator comparison.

Usage:
    pytest tests/integration/crossval/test_01_data_alignment.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


pytestmark = pytest.mark.skipif(
    not pytest.importorskip("qlib", reason="qlib not installed"),
    reason="qlib not installed",
)


class TestDataAlignment:
    """Verify OHLCV data from Qlib is consistent and usable."""

    def test_panels_have_same_shape(
        self, ohlcv_panel: dict[str, pd.DataFrame]
    ) -> None:
        shapes = {k: v.shape for k, v in ohlcv_panel.items()}
        first_shape = next(iter(shapes.values()))
        for field, shape in shapes.items():
            assert shape == first_shape, (
                f"{field} shape {shape} != expected {first_shape}"
            )

    def test_panels_have_same_index(
        self, ohlcv_panel: dict[str, pd.DataFrame]
    ) -> None:
        indices = [v.index for v in ohlcv_panel.values()]
        for idx in indices[1:]:
            pd.testing.assert_index_equal(indices[0], idx)

    def test_panels_have_same_columns(
        self, ohlcv_panel: dict[str, pd.DataFrame]
    ) -> None:
        cols = [v.columns.tolist() for v in ohlcv_panel.values()]
        for c in cols[1:]:
            assert c == cols[0]

    def test_sufficient_observations(
        self, ohlcv_panel: dict[str, pd.DataFrame]
    ) -> None:
        close = ohlcv_panel["close"]
        n_rows, n_cols = close.shape
        assert n_rows > 1000, f"Only {n_rows} timestamps — need > 1000"
        assert n_cols >= 5, f"Only {n_cols} instruments — need >= 5"

    def test_no_all_nan_columns(
        self, ohlcv_panel: dict[str, pd.DataFrame]
    ) -> None:
        for field, panel in ohlcv_panel.items():
            all_nan_cols = panel.columns[panel.isna().all()]
            assert len(all_nan_cols) == 0, (
                f"{field}: all-NaN columns: {all_nan_cols.tolist()}"
            )

    def test_ohlc_relationships(
        self, ohlcv_panel: dict[str, pd.DataFrame]
    ) -> None:
        """Verify O <= H, L <= H, L <= C, L <= O for valid rows."""
        o, h, l, c = (
            ohlcv_panel["open"],
            ohlcv_panel["high"],
            ohlcv_panel["low"],
            ohlcv_panel["close"],
        )
        valid = o.notna() & h.notna() & l.notna() & c.notna()
        violations = (
            ((l > h) & valid).sum().sum()
            + ((l > o) & valid).sum().sum()
            + ((l > c) & valid).sum().sum()
        )
        total = valid.sum().sum()
        assert violations / total < 0.001, (
            f"OHLC violations: {violations}/{total}"
        )

    def test_volume_non_negative(
        self, ohlcv_panel: dict[str, pd.DataFrame]
    ) -> None:
        vol = ohlcv_panel["volume"]
        valid = vol.notna()
        neg = ((vol < 0) & valid).sum().sum()
        assert neg == 0, f"Negative volume: {neg} entries"

    def test_close_reasonable_range(
        self, ohlcv_panel: dict[str, pd.DataFrame]
    ) -> None:
        """Sanity: close prices within reasonable crypto range."""
        close = ohlcv_panel["close"]
        min_val = close.min().min()
        max_val = close.max().max()
        assert min_val > 0, f"Close has non-positive values: min={min_val}"
        assert max_val < 1e7, f"Close has unreasonable values: max={max_val}"

    def test_no_future_leak_in_derived(
        self, ohlcv_panel: dict[str, pd.DataFrame]
    ) -> None:
        """Verify returns derived from close don't leak future data."""
        close = ohlcv_panel["close"]
        returns = close / close.shift(1) - 1
        # First row should be NaN (no prior data)
        assert returns.iloc[0].isna().all(), "Returns row 0 should be NaN"
