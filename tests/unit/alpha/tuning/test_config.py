# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for the tuning dataclass validation surface."""

from __future__ import annotations

import pytest

from nautilus_quants.alpha.tuning.config import (
    ALGORITHM_GRID,
    ALGORITHM_TPE,
    CORRECTION_BONFERRONI,
    CORRECTION_FDR_BH,
    CORRECTION_NONE,
    CV_METHOD_EXPANDING,
    CV_METHOD_ROLLING,
    PARAM_TYPE_COEFFICIENT,
    PARAM_TYPE_FIXED,
    PARAM_TYPE_SIGN,
    PARAM_TYPE_THRESHOLD,
    PARAM_TYPE_WINDOW,
    VAR_SCOPE_BROADCAST,
    VAR_SCOPE_PER_INSTRUMENT,
    CVConfig,
    ParamSpec,
    TuneConfig,
    VariableGroup,
)


class TestParamSpec:
    def test_sign_type_skips_value_range(self) -> None:
        spec = ParamSpec("p0", PARAM_TYPE_SIGN, original_value=-1.0)
        assert not spec.is_tunable
        assert spec.values is None and spec.low is None

    def test_fixed_type_skips_value_range(self) -> None:
        spec = ParamSpec("p0", PARAM_TYPE_FIXED, original_value=3.14)
        assert not spec.is_tunable

    def test_categorical_requires_values(self) -> None:
        with pytest.raises(ValueError, match="must provide either"):
            ParamSpec("p0", PARAM_TYPE_WINDOW, original_value=10.0)

    def test_continuous_requires_low_high(self) -> None:
        with pytest.raises(ValueError, match="must provide either"):
            ParamSpec("p0", PARAM_TYPE_COEFFICIENT, original_value=1.0)

    def test_values_and_range_are_mutually_exclusive(self) -> None:
        with pytest.raises(ValueError, match="cannot set both"):
            ParamSpec(
                "p0",
                PARAM_TYPE_WINDOW,
                original_value=10.0,
                values=(5.0, 10.0),
                low=0.0,
                high=20.0,
            )

    def test_window_is_tunable(self) -> None:
        spec = ParamSpec(
            "p0",
            PARAM_TYPE_WINDOW,
            original_value=10.0,
            values=(5.0, 10.0, 20.0),
        )
        assert spec.is_tunable

    def test_invalid_param_type_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid param_type"):
            ParamSpec("p0", "bogus", original_value=1.0, values=(1.0,))


class TestCVConfig:
    def test_defaults_are_valid(self) -> None:
        cfg = CVConfig()
        assert cfg.method == CV_METHOD_EXPANDING

    @pytest.mark.parametrize("method", [CV_METHOD_EXPANDING, CV_METHOD_ROLLING])
    def test_accepts_valid_methods(self, method: str) -> None:
        CVConfig(method=method)

    def test_rejects_invalid_method(self) -> None:
        with pytest.raises(ValueError, match="Invalid cv.method"):
            CVConfig(method="bogus")

    def test_rejects_negative_folds(self) -> None:
        with pytest.raises(ValueError, match="n_folds must be >= 1"):
            CVConfig(n_folds=0)

    def test_rejects_out_of_range_test_ratio(self) -> None:
        with pytest.raises(ValueError, match="cv.test_ratio must be in"):
            CVConfig(test_ratio=0.0)
        with pytest.raises(ValueError, match="cv.test_ratio must be in"):
            CVConfig(test_ratio=1.0)

    def test_rejects_when_no_training_room(self) -> None:
        with pytest.raises(ValueError, match="must leave room"):
            CVConfig(n_folds=5, test_ratio=0.2, holdout_ratio=0.1)


class TestTuneConfig:
    def test_defaults_are_valid(self) -> None:
        cfg = TuneConfig()
        assert cfg.algorithm == ALGORITHM_TPE
        assert cfg.correction_method == CORRECTION_BONFERRONI
        assert cfg.dimensions.numeric is True

    @pytest.mark.parametrize("algo", [ALGORITHM_TPE, ALGORITHM_GRID])
    def test_accepts_valid_algorithms(self, algo: str) -> None:
        TuneConfig(algorithm=algo)

    def test_rejects_invalid_algorithm(self) -> None:
        with pytest.raises(ValueError, match="Invalid algorithm"):
            TuneConfig(algorithm="bayes")

    @pytest.mark.parametrize(
        "method",
        [CORRECTION_BONFERRONI, CORRECTION_FDR_BH, CORRECTION_NONE],
    )
    def test_accepts_valid_corrections(self, method: str) -> None:
        TuneConfig(correction_method=method)

    def test_rejects_trials_below_one(self) -> None:
        with pytest.raises(ValueError, match="trials must be >= 1"):
            TuneConfig(trials=0)

    def test_rejects_register_top_k_below_one(self) -> None:
        with pytest.raises(ValueError, match="register_top_k must be >= 1"):
            TuneConfig(register_top_k=0)

    def test_rejects_stability_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="stability_min must be in"):
            TuneConfig(stability_min=1.5)

    def test_rejects_alpha_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="significance_alpha must be in"):
            TuneConfig(significance_alpha=0.0)


class TestVariableGroup:
    def test_rejects_unknown_scope(self) -> None:
        with pytest.raises(ValueError, match="Invalid scope"):
            VariableGroup(members=("a",), scope="weird")

    @pytest.mark.parametrize("scope", [VAR_SCOPE_PER_INSTRUMENT, VAR_SCOPE_BROADCAST])
    def test_accepts_valid_scopes(self, scope: str) -> None:
        VariableGroup(members=("a", "b"), scope=scope)

    def test_threshold_param_accepts_values(self) -> None:
        ParamSpec(
            "t",
            PARAM_TYPE_THRESHOLD,
            original_value=0.1,
            values=(0.05, 0.1, 0.2),
        )
