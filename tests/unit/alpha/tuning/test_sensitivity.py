# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for sensitivity.py — stability analysis around the optimum."""

from __future__ import annotations

import math

import pytest

from nautilus_quants.alpha.tuning.config import PARAM_TYPE_SIGN, PARAM_TYPE_WINDOW, ParamSpec
from nautilus_quants.alpha.tuning.sensitivity import analyze_parameter_stability


def _make_window_spec(name: str = "p1") -> ParamSpec:
    return ParamSpec(
        name=name,
        param_type=PARAM_TYPE_WINDOW,
        original_value=20.0,
        values=(10.0, 15.0, 20.0, 30.0, 40.0),
    )


class TestAnalyzeParameterStability:
    def test_flat_landscape_returns_high_stability(self) -> None:
        """If perturbations don't degrade ICIR much, stability is near 1."""
        spec = _make_window_spec()
        best = {"p1": 20.0}
        report = analyze_parameter_stability(
            best_params=best,
            numeric_specs=(spec,),
            evaluate_fn=lambda params: 0.1,  # constant
        )
        assert report.stability_score is not None
        assert report.stability_score == pytest.approx(1.0)
        assert report.is_stable(threshold=0.9)

    def test_spike_landscape_yields_low_stability(self) -> None:
        """If best is a narrow spike, stability should drop below the threshold."""
        spec = _make_window_spec()
        best = {"p1": 20.0}

        def evaluate(params):
            # |ICIR| high only at the optimum.
            return 0.3 if params["p1"] == 20.0 else 0.02

        report = analyze_parameter_stability(
            best_params=best,
            numeric_specs=(spec,),
            evaluate_fn=evaluate,
        )
        assert report.stability_score is not None
        assert report.stability_score < 0.2
        assert not report.is_stable(threshold=0.5)

    def test_sign_parameter_is_ignored(self) -> None:
        sign_spec = ParamSpec("p0", PARAM_TYPE_SIGN, original_value=-1.0)
        best = {"p0": -1.0}
        report = analyze_parameter_stability(
            best_params=best,
            numeric_specs=(sign_spec,),
            evaluate_fn=lambda _: 0.2,
        )
        # Non-tunable param should not appear in the per-parameter list.
        assert report.parameters == ()

    def test_multiple_parameters_use_worst_ratio(self) -> None:
        stable = _make_window_spec(name="stable")
        fragile = _make_window_spec(name="fragile")
        best = {"stable": 20.0, "fragile": 20.0}

        def evaluate(params):
            base = 0.2
            if params["fragile"] != 20.0:
                return 0.02  # fragile drops 10x when perturbed
            return base

        report = analyze_parameter_stability(
            best_params=best,
            numeric_specs=(stable, fragile),
            evaluate_fn=evaluate,
        )
        assert report.stability_score is not None
        assert report.stability_score < 0.2  # limited by the fragile param

    def test_evaluate_fn_error_returns_unstable(self) -> None:
        spec = _make_window_spec()
        best = {"p1": 20.0}

        def evaluate(_params):
            raise RuntimeError("boom")

        report = analyze_parameter_stability(
            best_params=best,
            numeric_specs=(spec,),
            evaluate_fn=evaluate,
        )
        # Best itself failed → stability_score is None.
        assert report.stability_score is None
