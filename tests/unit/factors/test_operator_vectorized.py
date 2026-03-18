"""Consistency tests: compute_vectorized vs scalar compute for all operators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nautilus_quants.factors.operators.time_series import (
    Correlation,
    Covariance,
    Delta,
    Delay,
    TsArgmax,
    TsArgmin,
    TsMax,
    TsMean,
    TsMin,
    TsRank,
    TsStd,
    TsSum,
)
from nautilus_quants.factors.operators.cross_sectional import (
    CsClipQuantile,
    CsDemean,
    CsNormalize,
    CsRank,
    CsScale,
    CsScaleDown,
    CsWinsorize,
    CsZscore,
)


# ---------------------------------------------------------------------------
# Time-series operator consistency
# ---------------------------------------------------------------------------

def _make_ts_data(n: int = 100, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.standard_normal(n).cumsum() + 100)


class TestTsOperatorVectorizedConsistency:
    """For each TS operator, compute_vectorized() must match point-by-point compute()."""

    data = _make_ts_data()
    window = 10

    def _check_consistency(self, op, data=None, window=None):
        data = data if data is not None else self.data
        window = window or self.window
        vec = op.compute_vectorized(data, window)
        for i in range(len(data)):
            arr = data.iloc[: i + 1].values
            scalar = op.compute(arr, window)
            if np.isnan(scalar):
                assert np.isnan(vec.iloc[i]), f"Mismatch at index {i}: expected NaN"
            else:
                assert vec.iloc[i] == pytest.approx(
                    scalar, nan_ok=True, abs=1e-10
                ), f"Mismatch at index {i}: vec={vec.iloc[i]}, scalar={scalar}"

    def test_ts_mean(self):
        self._check_consistency(TsMean())

    def test_ts_sum(self):
        self._check_consistency(TsSum())

    def test_ts_std(self):
        self._check_consistency(TsStd())

    def test_ts_min(self):
        self._check_consistency(TsMin())

    def test_ts_max(self):
        self._check_consistency(TsMax())

    def test_ts_rank(self):
        self._check_consistency(TsRank())

    def test_ts_argmax(self):
        self._check_consistency(TsArgmax())

    def test_ts_argmin(self):
        self._check_consistency(TsArgmin())

    def test_delta(self):
        self._check_consistency(Delta())

    def test_delay(self):
        self._check_consistency(Delay())

    def test_correlation(self):
        data2 = _make_ts_data(seed=99)
        op = Correlation()
        vec = op.compute_vectorized(self.data, self.window, data2=data2)
        for i in range(len(self.data)):
            arr1 = self.data.iloc[: i + 1].values
            arr2 = data2.iloc[: i + 1].values
            scalar = op.compute(arr1, self.window, data2=arr2)
            if np.isnan(scalar):
                # compute_vectorized uses .fillna(0) (popbo-aligned),
                # so insufficient-window NaN becomes 0.0
                assert np.isnan(vec.iloc[i]) or vec.iloc[i] == pytest.approx(0.0)
            else:
                assert vec.iloc[i] == pytest.approx(scalar, abs=1e-8)

    def test_covariance(self):
        data2 = _make_ts_data(seed=99)
        op = Covariance()
        vec = op.compute_vectorized(self.data, self.window, data2=data2)
        for i in range(len(self.data)):
            arr1 = self.data.iloc[: i + 1].values
            arr2 = data2.iloc[: i + 1].values
            scalar = op.compute(arr1, self.window, data2=arr2)
            if np.isnan(scalar):
                assert np.isnan(vec.iloc[i])
            else:
                assert vec.iloc[i] == pytest.approx(scalar, abs=1e-8)


# ---------------------------------------------------------------------------
# Cross-sectional operator consistency
# ---------------------------------------------------------------------------

def _make_cs_data(n_rows: int = 20, n_cols: int = 5, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = [f"INST_{i}" for i in range(n_cols)]
    return pd.DataFrame(rng.standard_normal((n_rows, n_cols)), columns=cols)


class TestCsOperatorVectorizedConsistency:
    """For each CS operator, compute_vectorized(df) must match row-by-row compute(dict)."""

    df = _make_cs_data()

    def _check_cs_consistency(self, op, df=None, **kwargs):
        df = df if df is not None else self.df
        vec_result = op.compute_vectorized(df, **kwargs)
        for i in range(len(df)):
            row = df.iloc[i]
            values = row.to_dict()
            scalar_result = op.compute(values, **kwargs)
            for col in df.columns:
                v = vec_result.iloc[i][col]
                s = scalar_result.get(col, float("nan"))
                if np.isnan(s):
                    assert np.isnan(v), f"Row {i}, col {col}: expected NaN"
                else:
                    assert v == pytest.approx(
                        s, nan_ok=True, abs=1e-10
                    ), f"Row {i}, col {col}: vec={v}, scalar={s}"

    def test_cs_rank(self):
        self._check_cs_consistency(CsRank())

    def test_cs_zscore(self):
        self._check_cs_consistency(CsZscore())

    def test_cs_scale(self):
        self._check_cs_consistency(CsScale())

    def test_cs_demean(self):
        self._check_cs_consistency(CsDemean())

    def test_cs_normalize_no_std(self):
        self._check_cs_consistency(CsNormalize(), use_std=False, limit=0.0)

    def test_cs_normalize_with_std(self):
        self._check_cs_consistency(CsNormalize(), use_std=True, limit=0.0)

    def test_cs_normalize_with_limit(self):
        self._check_cs_consistency(CsNormalize(), use_std=True, limit=1.0)

    def test_cs_winsorize(self):
        self._check_cs_consistency(CsWinsorize(), std_mult=1.0)

    def test_cs_scale_down(self):
        self._check_cs_consistency(CsScaleDown(), constant=0.0)

    def test_cs_clip_quantile(self):
        self._check_cs_consistency(CsClipQuantile(), lower=0.2, upper=0.8)

    def test_cs_scale_zero_total_matches_scalar(self):
        op = CsScale()
        df = pd.DataFrame([{"A": 0.0, "B": 0.0, "C": np.nan}])
        vec = op.compute_vectorized(df).iloc[0].to_dict()
        scalar = op.compute(df.iloc[0].to_dict())
        assert vec["A"] == pytest.approx(scalar["A"])
        assert vec["B"] == pytest.approx(scalar["B"])
        assert np.isnan(vec["C"])

    def test_cs_scale_down_zero_range_matches_scalar(self):
        op = CsScaleDown()
        df = pd.DataFrame([{"A": 5.0, "B": 5.0, "C": np.nan}])
        vec = op.compute_vectorized(df, constant=0.0).iloc[0].to_dict()
        scalar = op.compute(df.iloc[0].to_dict(), constant=0.0)
        assert vec["A"] == pytest.approx(scalar["A"])
        assert vec["B"] == pytest.approx(scalar["B"])
        assert np.isnan(vec["C"])

    def test_cs_normalize_zero_std_matches_scalar(self):
        op = CsNormalize()
        df = pd.DataFrame([{"A": 1.0, "B": 1.0, "C": np.nan}])
        vec = op.compute_vectorized(df, use_std=True, limit=0.0).iloc[0].to_dict()
        scalar = op.compute(df.iloc[0].to_dict(), use_std=True, limit=0.0)
        assert vec["A"] == pytest.approx(scalar["A"])
        assert vec["B"] == pytest.approx(scalar["B"])
        assert np.isnan(vec["C"])

    def test_cs_rank_ties_match_scalar(self):
        op = CsRank()
        df = pd.DataFrame([{"A": 1.0, "B": 1.0, "C": 2.0, "D": np.nan}])
        vec = op.compute_vectorized(df).iloc[0].to_dict()
        scalar = op.compute(df.iloc[0].to_dict())
        assert vec["A"] == pytest.approx(scalar["A"])
        assert vec["B"] == pytest.approx(scalar["B"])
        assert vec["C"] == pytest.approx(scalar["C"])
        assert np.isnan(vec["D"])

    def test_cs_rank_multirow_ties_and_nans_match_scalar(self):
        op = CsRank()
        df = pd.DataFrame(
            [
                {"A": 1.0, "B": 1.0, "C": 2.0, "D": np.nan},   # ties + NaN
                {"A": np.nan, "B": 3.0, "C": 3.0, "D": 3.0},  # all ties among valids
                {"A": np.nan, "B": np.nan, "C": 5.0, "D": np.nan},  # single valid
                {"A": np.nan, "B": np.nan, "C": np.nan, "D": np.nan},  # no valid
            ]
        )

        vec_df = op.compute_vectorized(df)
        for i in range(len(df)):
            scalar = op.compute(df.iloc[i].to_dict())
            for col in df.columns:
                v = vec_df.iloc[i][col]
                s = scalar.get(col, float("nan"))
                if np.isnan(s):
                    assert np.isnan(v), f"Row {i}, col {col}: expected NaN"
                else:
                    assert v == pytest.approx(s), f"Row {i}, col {col}: vec={v}, scalar={s}"
