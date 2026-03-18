# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""TDD tests for Alpha101 new operators: signed_power, if_else, decay_linear, ts_product, and aliases."""

import math

import numpy as np
import pytest

from nautilus_quants.factors.operators.math import (
    IfElse,
    SignedPower,
    if_else,
    signed_power,
)
from nautilus_quants.factors.operators.time_series import (
    DecayLinear,
    TsProduct,
    decay_linear,
    ts_product,
    ts_std,
    ts_mean,
    delta,
    delay,
    correlation,
    covariance,
    TIME_SERIES_OPERATORS,
    TS_OPERATOR_INSTANCES,
)


class TestSignedPower:
    """Tests for signed_power operator."""

    def test_positive_value(self):
        result = signed_power(3.0, 2.0)
        assert result == pytest.approx(9.0)

    def test_negative_value(self):
        result = signed_power(-9.0, 0.5)
        assert result == pytest.approx(-3.0)

    def test_negative_odd_power(self):
        # sign(-2) * |(-2)|^3 = -1 * 8 = -8
        result = signed_power(-2.0, 3.0)
        assert result == pytest.approx(-8.0)

    def test_zero(self):
        result = signed_power(0.0, 2.0)
        assert result == pytest.approx(0.0)

    def test_array(self):
        arr = np.array([-4.0, 0.0, 9.0])
        result = signed_power(arr, 0.5)
        np.testing.assert_allclose(result, [-2.0, 0.0, 3.0], rtol=1e-10)

    def test_operator_class_name(self):
        op = SignedPower()
        assert op.name == "signed_power"

    def test_operator_args(self):
        op = SignedPower()
        assert op.min_args == 2
        assert op.max_args == 2


class TestIfElse:
    """Tests for if_else operator."""

    def test_true_condition(self):
        assert if_else(1.0, 2.0, 3.0) == 2.0

    def test_false_condition(self):
        assert if_else(0.0, 2.0, 3.0) == 3.0

    def test_array_condition(self):
        cond = np.array([1.0, 0.0, 1.0, 0.0])
        result = if_else(cond, 10.0, -10.0)
        np.testing.assert_array_equal(result, [10.0, -10.0, 10.0, -10.0])

    def test_array_values(self):
        cond = np.array([1.0, 0.0])
        true_arr = np.array([5.0, 6.0])
        false_arr = np.array([7.0, 8.0])
        result = if_else(cond, true_arr, false_arr)
        np.testing.assert_array_equal(result, [5.0, 8.0])

    def test_operator_class_name(self):
        op = IfElse()
        assert op.name == "if_else"

    def test_operator_args(self):
        op = IfElse()
        assert op.min_args == 3
        assert op.max_args == 3

    def test_negative_truthy(self):
        # Non-zero values are truthy
        result = if_else(-5.0, 1.0, 2.0)
        assert result == 1.0


class TestDecayLinear:
    """Tests for decay_linear (LWMA) operator."""

    def test_basic_weights(self):
        # window=3, weights=[1/6, 2/6, 3/6]
        data = np.array([1.0, 2.0, 3.0])
        expected = (1 * 1 + 2 * 2 + 3 * 3) / 6  # = 14/6
        result = decay_linear(data, 3)
        assert result == pytest.approx(expected)

    def test_window_one(self):
        data = np.array([5.0])
        result = decay_linear(data, 1)
        assert result == pytest.approx(5.0)

    def test_insufficient_data(self):
        data = np.array([1.0, 2.0])
        result = decay_linear(data, 5)
        assert math.isnan(result)

    def test_most_recent_highest_weight(self):
        # window=2, weights=[1/3, 2/3]
        # data=[10.0, 1.0]: result = 10*(1/3) + 1*(2/3)
        data = np.array([10.0, 1.0])
        expected = 10.0 * (1 / 3) + 1.0 * (2 / 3)
        result = decay_linear(data, 2)
        assert result == pytest.approx(expected)

    def test_operator_class_name(self):
        op = DecayLinear()
        assert op.name == "decay_linear"

    def test_operator_args(self):
        op = DecayLinear()
        assert op.min_args == 2
        assert op.max_args == 2


class TestTsProduct:
    """Tests for ts_product (rolling product) operator."""

    def test_basic(self):
        data = np.array([2.0, 3.0, 4.0])
        result = ts_product(data, 3)
        assert result == pytest.approx(24.0)

    def test_window_subset(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ts_product(data, 3)
        assert result == pytest.approx(60.0)  # 3*4*5

    def test_insufficient_data(self):
        data = np.array([1.0, 2.0])
        result = ts_product(data, 5)
        assert math.isnan(result)

    def test_operator_class_name(self):
        op = TsProduct()
        assert op.name == "ts_product"


class TestAliases:
    """Test that aliases are registered and work correctly."""

    def test_all_aliases_in_time_series_operators(self):
        required = [
            "stddev", "sma", "ts_delta", "ts_delay",
            "ts_corr", "ts_covariance", "ts_std_dev",
            "ts_decay_linear", "product",
        ]
        for alias in required:
            assert alias in TIME_SERIES_OPERATORS, f"Missing alias: {alias}"

    def test_all_aliases_in_ts_operator_instances(self):
        required = [
            "stddev", "sma", "ts_delta", "ts_delay",
            "ts_corr", "ts_covariance", "ts_std_dev",
            "ts_decay_linear", "product",
        ]
        for alias in required:
            assert alias in TS_OPERATOR_INSTANCES, f"Missing instance: {alias}"

    def test_stddev_values_match_ts_std(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stddev_fn = TIME_SERIES_OPERATORS["stddev"]
        assert stddev_fn(data, 5) == pytest.approx(ts_std(data, 5))

    def test_sma_values_match_ts_mean(self):
        data = np.array([2.0, 4.0, 6.0])
        sma_fn = TIME_SERIES_OPERATORS["sma"]
        assert sma_fn(data, 3) == pytest.approx(ts_mean(data, 3))

    def test_ts_delta_values_match_delta(self):
        data = np.array([1.0, 3.0, 6.0])
        ts_delta_fn = TIME_SERIES_OPERATORS["ts_delta"]
        assert ts_delta_fn(data, 1) == pytest.approx(delta(data, 1))

    def test_ts_delay_values_match_delay(self):
        data = np.array([10.0, 20.0, 30.0])
        ts_delay_fn = TIME_SERIES_OPERATORS["ts_delay"]
        assert ts_delay_fn(data, 1) == pytest.approx(delay(data, 1))

    def test_product_alias_matches_ts_product(self):
        data = np.array([2.0, 3.0, 4.0])
        product_fn = TIME_SERIES_OPERATORS["product"]
        assert product_fn(data, 3) == pytest.approx(ts_product(data, 3))
