# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Phase 2: Operator-level cross-validation — all TS operators.

Parametrized tests comparing each time-series operator against
Qlib or pandas reference, using the operator_mapping spec.

Usage:
    pytest tests/integration/crossval/test_02_ts_operators.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(
    not pytest.importorskip("qlib", reason="qlib not installed"),
    reason="qlib not installed",
)

from tests.integration.crossval.conftest import (  # noqa: E402
    assert_allclose,
    assert_correlation,
    assert_spearman,
    qlib_feature,
)
from tests.integration.crossval.operator_mapping import (  # noqa: E402
    MatchType,
    TS_OPERATORS,
)


def _get_operator_instance(cls_name: str):
    """Import and instantiate a TS operator by class name."""
    from nautilus_quants.factors.operators.time_series import TS_OPERATOR_INSTANCES

    # Find by class name
    for name, op in TS_OPERATOR_INSTANCES.items():
        if type(op).__name__ == cls_name:
            return op
    raise ValueError(f"Operator class {cls_name} not found")


# ---------------------------------------------------------------------------
# Single-data TS operators (close only)
# ---------------------------------------------------------------------------
_SINGLE_DATA_OPS = [
    op for op in TS_OPERATORS
    if op.name not in ("correlation", "covariance")
]


@pytest.mark.parametrize(
    "spec",
    _SINGLE_DATA_OPS,
    ids=[op.name for op in _SINGLE_DATA_OPS],
)
def test_ts_operator_single(
    spec,
    close_panel: pd.DataFrame,
    instruments: list[str],
) -> None:
    """Cross-validate a single-data TS operator."""
    op = _get_operator_instance(spec.operator_cls_name)
    kwargs = spec.extra_kwargs or {}
    naut = op.compute_panel(close_panel, spec.window, **kwargs)

    # Build reference: Qlib if available, else pandas
    if spec.qlib_expr is not None:
        ref = qlib_feature(
            spec.qlib_expr.format(field="close", window=spec.window),
            instruments,
        )
    elif spec.name == "ema":
        ref = close_panel.ewm(span=spec.window, adjust=False).mean()
    elif spec.name == "ts_product":
        ref = close_panel.rolling(spec.window).apply(np.prod, raw=True)
    elif spec.name == "ts_skew":
        ref = close_panel.rolling(spec.window).skew()
    else:
        pytest.skip(f"No reference for {spec.name}")

    # Compare using match_type
    if spec.match_type == MatchType.EXACT:
        assert_allclose(naut, ref, label=spec.name)

    elif spec.match_type == MatchType.CORR:
        assert_correlation(
            naut, ref, min_corr=0.9999, atol=0.05, label=spec.name,
        )

    elif spec.match_type == MatchType.SPEAR:
        assert_spearman(naut, ref, min_corr=0.99, label=spec.name)

    elif spec.match_type == MatchType.RATIO_N:
        from tests.integration.crossval.conftest import align_panels

        naut_vals, ref_vals, n_valid = align_panels(naut, ref, spec.name)
        ratio = naut_vals / np.where(ref_vals == 0, np.nan, ref_vals)
        ratio = ratio[~np.isnan(ratio)]
        np.testing.assert_allclose(
            np.median(ratio), spec.window, rtol=0.01,
            err_msg=f"{spec.name}: ratio should be ~{spec.window} (Qlib WMA bug)",
        )

    elif spec.match_type == MatchType.PANDAS:
        assert_allclose(naut, ref, label=spec.name)


# ---------------------------------------------------------------------------
# Two-data TS operators (close + volume)
# ---------------------------------------------------------------------------
_TWO_DATA_OPS = [
    op for op in TS_OPERATORS
    if op.name in ("correlation", "covariance")
]


@pytest.mark.parametrize(
    "spec",
    _TWO_DATA_OPS,
    ids=[op.name for op in _TWO_DATA_OPS],
)
def test_ts_operator_two_data(
    spec,
    close_panel: pd.DataFrame,
    volume_panel: pd.DataFrame,
    instruments: list[str],
) -> None:
    """Cross-validate a two-data TS operator (correlation, covariance)."""
    op = _get_operator_instance(spec.operator_cls_name)
    naut = op.compute_panel(close_panel, spec.window, data2=volume_panel)

    ref = qlib_feature(
        spec.qlib_expr.format(window=spec.window),
        instruments,
    )
    assert_allclose(naut, ref, label=spec.name)


# ---------------------------------------------------------------------------
# CS operators
# ---------------------------------------------------------------------------
from tests.integration.crossval.operator_mapping import CS_OPERATORS  # noqa: E402


@pytest.mark.parametrize(
    "spec",
    CS_OPERATORS,
    ids=[op.name for op in CS_OPERATORS],
)
def test_cs_operator(
    spec,
    close_panel: pd.DataFrame,
) -> None:
    """Cross-validate a CS operator against pandas row-wise reference."""
    from nautilus_quants.factors.operators.cross_sectional import CS_OPERATOR_INSTANCES

    op = None
    for name, candidate in CS_OPERATOR_INSTANCES.items():
        if type(candidate).__name__ == spec.operator_cls_name:
            op = candidate
            break

    if op is None:
        pytest.skip(f"CS operator {spec.operator_cls_name} not found")

    # Compute via nautilus — CS operators work on dict per row
    extra = spec.extra_kwargs or {}
    naut_rows = {}
    for ts in close_panel.index:
        row = close_panel.loc[ts].dropna().to_dict()
        if len(row) < 3:
            continue
        result = op.compute(row, **extra)
        if isinstance(result, dict):
            naut_rows[ts] = result
    naut = pd.DataFrame(naut_rows).T

    # Pandas reference
    if spec.name == "cs_rank":
        ref = close_panel.rank(axis=1, method="min", pct=True)
    elif spec.name == "cs_zscore":
        mu = close_panel.mean(axis=1)
        sd = close_panel.std(axis=1)
        ref = close_panel.sub(mu, axis=0).div(sd, axis=0)
    elif spec.name == "cs_demean":
        mu = close_panel.mean(axis=1)
        ref = close_panel.sub(mu, axis=0)
    elif spec.name == "cs_scale":
        abs_sum = close_panel.abs().sum(axis=1)
        ref = close_panel.div(abs_sum, axis=0)
    elif spec.name == "cs_max":
        row_max = close_panel.max(axis=1)
        ref = pd.DataFrame(
            np.tile(row_max.values[:, None], (1, close_panel.shape[1])),
            index=close_panel.index, columns=close_panel.columns,
        )
    elif spec.name == "cs_min":
        row_min = close_panel.min(axis=1)
        ref = pd.DataFrame(
            np.tile(row_min.values[:, None], (1, close_panel.shape[1])),
            index=close_panel.index, columns=close_panel.columns,
        )
    elif spec.name == "normalize":
        mu = close_panel.mean(axis=1)
        sd = close_panel.std(axis=1).replace(0, np.nan)
        ref = close_panel.sub(mu, axis=0).div(sd, axis=0)
    elif spec.name == "clip_quantile":
        lower_q = close_panel.quantile(0.2, axis=1)
        upper_q = close_panel.quantile(0.8, axis=1)
        ref = close_panel.clip(lower_q, upper_q, axis=0)
    elif spec.name == "winsorize":
        mu = close_panel.mean(axis=1)
        sd = close_panel.std(axis=1)
        lo = mu - 3.0 * sd
        hi = mu + 3.0 * sd
        ref = close_panel.clip(lo, hi, axis=0)
    else:
        pytest.skip(f"No pandas reference for {spec.name}")

    # Align columns
    common_cols = naut.columns.intersection(ref.columns)
    common_idx = naut.index.intersection(ref.index)
    naut = naut.loc[common_idx, common_cols]
    ref = ref.loc[common_idx, common_cols]

    assert_allclose(naut, ref, label=spec.name)
