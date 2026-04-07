# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Phase 3: Expression-level E2E cross-validation.

Tests complete alpha expressions against Qlib or decomposed pandas
references. Also runs a coverage check on all builtin alphas.

Usage:
    pytest tests/integration/crossval/test_03_alpha_expressions.py -v
"""

from __future__ import annotations

import sys

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
    qlib_feature,
)
from tests.integration.crossval.operator_mapping import (  # noqa: E402
    ALPHA_EXPRESSIONS,
    PANDAS_REFS,
    MatchType,
)


# ---------------------------------------------------------------------------
# Nautilus evaluator fixture
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def nautilus_evaluator():
    """Create a minimal Evaluator for computing alpha expressions."""
    sys.path.insert(0, "src")
    from nautilus_quants.factors.engine.evaluator import Evaluator
    from nautilus_quants.factors.operators.cross_sectional import CS_OPERATOR_INSTANCES
    from nautilus_quants.factors.operators.math import MATH_OPERATORS
    from nautilus_quants.factors.operators.time_series import TS_OPERATOR_INSTANCES

    operators = {}
    operators.update(TS_OPERATOR_INSTANCES)
    operators.update(CS_OPERATOR_INSTANCES)
    operators.update(MATH_OPERATORS)

    return Evaluator(operators=operators)


def _evaluate_expression(
    evaluator,
    expression: str,
    panels: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Evaluate an alpha expression on OHLCV panels."""
    from nautilus_quants.factors.expression import parse_expression

    ast = parse_expression(expression)
    result = evaluator.evaluate_panel(ast, panels)
    return result


# ---------------------------------------------------------------------------
# Parametrized alpha expression tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "spec",
    ALPHA_EXPRESSIONS,
    ids=[a.name for a in ALPHA_EXPRESSIONS],
)
def test_alpha_expression(
    spec,
    ohlcv_panel: dict[str, pd.DataFrame],
    instruments: list[str],
    nautilus_evaluator,
) -> None:
    """Cross-validate a complete alpha expression."""
    # Compute via nautilus
    naut = _evaluate_expression(nautilus_evaluator, spec.expression, ohlcv_panel)

    # Check coverage
    valid_frac = naut.notna().mean().mean()
    assert valid_frac >= spec.min_coverage, (
        f"{spec.name}: coverage {valid_frac:.2%} < {spec.min_coverage:.0%}"
    )

    # Get reference
    if spec.qlib_expr is not None:
        ref = qlib_feature(spec.qlib_expr, instruments)
    elif spec.pandas_ref is not None:
        ref_func = PANDAS_REFS[spec.pandas_ref]
        ref = ref_func(ohlcv_panel)
    else:
        pytest.skip(f"{spec.name}: no reference available")

    # Compare
    if spec.match_type == MatchType.EXACT:
        assert_allclose(naut, ref, label=spec.name)
    elif spec.match_type == MatchType.CORR:
        assert_correlation(
            naut, ref, min_corr=0.85, atol=1.0, label=spec.name,
        )
    else:
        assert_allclose(naut, ref, label=spec.name)


# ---------------------------------------------------------------------------
# Coverage test: all builtin alphas evaluate without error
# ---------------------------------------------------------------------------
class TestAlphaCoverage:
    """Verify all builtin alpha expressions produce valid output."""

    @pytest.fixture(scope="class")
    def alpha101_expressions(self) -> dict[str, str]:
        sys.path.insert(0, "src")
        from nautilus_quants.factors.builtin.alpha101 import ALPHA101_FACTORS

        return {k: v["expression"] for k, v in ALPHA101_FACTORS.items()}

    def test_alpha101_no_eval_errors(
        self,
        alpha101_expressions: dict[str, str],
        ohlcv_panel: dict[str, pd.DataFrame],
        nautilus_evaluator,
    ) -> None:
        """All Alpha101 expressions should evaluate without exceptions."""
        errors = {}
        all_nan = []
        for name, expr in alpha101_expressions.items():
            try:
                result = _evaluate_expression(nautilus_evaluator, expr, ohlcv_panel)
                if result.notna().sum().sum() == 0:
                    all_nan.append(name)
            except Exception as e:
                errors[name] = str(e)

        if errors:
            msg = "\n".join(f"  {k}: {v}" for k, v in errors.items())
            pytest.fail(f"Alpha101 evaluation errors:\n{msg}")

        # Report all-NaN (expected for some deep-nested alphas)
        if all_nan:
            print(
                f"\n  INFO: {len(all_nan)} Alpha101 factors are all-NaN "
                f"(likely deep nesting warmup): {all_nan[:5]}..."
            )
