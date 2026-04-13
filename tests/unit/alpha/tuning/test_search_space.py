# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for search-space construction.

Covers:
- ``classify_parameters`` — correct param type assignment
- ``expand_config_params`` — config var → numeric literal inlining
- ``detect_operator_slots`` — outer vs inner classification, aliases
- ``detect_variable_slots`` — scope + availability filtering
- ``reconstruct_expression`` — round-trip correctness across all three dims
"""

from __future__ import annotations

import pytest

from nautilus_quants.alpha.tuning.config import (
    PARAM_TYPE_COEFFICIENT,
    PARAM_TYPE_SIGN,
    PARAM_TYPE_THRESHOLD,
    PARAM_TYPE_WINDOW,
    VAR_SCOPE_BROADCAST,
    VAR_SCOPE_PER_INSTRUMENT,
)
from nautilus_quants.alpha.tuning.search_space import (
    OPERATOR_GROUPS,
    VARIABLE_GROUPS,
    build_search_space,
    classify_parameters,
    detect_operator_slots,
    detect_variable_slots,
    expand_config_params,
    reconstruct_expression,
)
from nautilus_quants.factors.expression.normalize import expression_template

# ── Parameter classification ────────────────────────────────────────────────


class TestClassifyParameters:
    def test_negation_becomes_sign_parameter(self) -> None:
        expr = "-correlation(high, rank(volume), 10)"
        template, values = expression_template(expr)
        specs = classify_parameters(template, values, expr)
        assert len(specs) == 2
        assert specs[0].param_type == PARAM_TYPE_SIGN
        assert specs[0].original_value == -1.0
        assert not specs[0].is_tunable

    def test_ts_window_operator_classified_as_window(self) -> None:
        expr = "-correlation(high, rank(volume), 10)"
        template, values = expression_template(expr)
        specs = classify_parameters(template, values, expr)
        window_spec = specs[1]
        assert window_spec.param_type == PARAM_TYPE_WINDOW
        assert window_spec.original_value == 10.0
        assert window_spec.values is not None
        assert 10.0 in window_spec.values

    def test_fractional_value_classified_as_threshold(self) -> None:
        expr = "clip_quantile(close, 0.1, 0.9)"
        template, values = expression_template(expr)
        specs = classify_parameters(template, values, expr)
        # 0.1 and 0.9 are not windows (clip_quantile isn't a TS operator);
        # both should be thresholds with categorical 0.1-step search space.
        assert specs[0].param_type == PARAM_TYPE_THRESHOLD
        assert specs[1].param_type == PARAM_TYPE_THRESHOLD
        assert specs[0].values is not None
        assert specs[1].values is not None
        assert 0.1 in specs[0].values
        assert 0.9 in specs[1].values
        # All values are 1-decimal rounded (or 0.05 / 0.95 — the only
        # 2-decimal exceptions on the standard threshold grid).
        for v in (*specs[0].values, *specs[1].values):
            assert v in (0.05, 0.95) or abs(round(v, 1) - v) < 1e-9

    def test_threshold_snaps_off_grid_value_to_nearest(self) -> None:
        # 0.476 (a "magic number" Optuna shouldn't chase) snaps to 0.5; same
        # for 0.95805 → 0.95. Avoids p-hacking-style overfitting.
        expr = "clip_quantile(close, 0.476, 0.95805)"
        template, values = expression_template(expr)
        specs = classify_parameters(template, values, expr)
        assert 0.476 not in specs[0].values
        assert 0.95805 not in specs[1].values
        assert 0.5 in specs[0].values
        assert 0.95 in specs[1].values

    def test_large_coefficient_classified_as_coefficient(self) -> None:
        # 2.5 snaps to nearest grid point (3.0) and gets a categorical
        # ladder instead of a continuous log-scale range — eliminates
        # magic-number p-hacking risk.
        expr = "close * 2.5"
        template, values = expression_template(expr)
        specs = classify_parameters(template, values, expr)
        assert len(specs) == 1
        assert specs[0].param_type == PARAM_TYPE_COEFFICIENT
        # Now categorical, not log-scale.
        assert specs[0].values is not None
        assert specs[0].log_scale is False
        # Magic 2.5 → snaps to 3.0 grid point.
        assert 3.0 in specs[0].values
        # All candidates positive (sign of original was positive).
        assert all(v > 0 for v in specs[0].values)

    def test_negative_coefficient_preserves_sign(self) -> None:
        # -2 should NOT flip to +2 — sign carries structural meaning.
        expr = "btc_beta * -2"
        template, values = expression_template(expr)
        specs = classify_parameters(template, values, expr)
        coef = next(s for s in specs if s.param_type == PARAM_TYPE_COEFFICIENT)
        assert all(v < 0 for v in coef.values), (
            f"Negative coefficient sign should be preserved, got {coef.values}"
        )

    def test_zero_coefficient_is_fixed(self) -> None:
        # A literal 0 in the expression is structural — never tune.
        # Construct via templates manually since `close * 0` would be a
        # weird factor to start with.
        from nautilus_quants.alpha.tuning.search_space import _coefficient_search_space
        assert _coefficient_search_space(0.0) == (0.0,)

    def test_window_search_space_honours_original_magnitude(self) -> None:
        expr = "ts_mean(close, 60)"
        template, values = expression_template(expr)
        specs = classify_parameters(template, values, expr)
        win_spec = specs[0]
        assert win_spec.values is not None
        # Should cover ~ 60 / 8 .. 60 * 8 = 7.5 .. 480 (clipped to ladder).
        assert 60.0 in win_spec.values
        assert min(win_spec.values) <= 15.0
        assert max(win_spec.values) >= 120.0

    def test_context_count_mismatch_raises(self) -> None:
        expr = "close + 5"
        template, values = expression_template(expr)
        # Inject a spurious extra value to trigger the sanity check.
        with pytest.raises(ValueError, match="Parameter context count"):
            classify_parameters(template, list(values) + [1.0], expr)


# ── Config param expansion ──────────────────────────────────────────────────


class TestExpandConfigParams:
    def test_numeric_param_is_inlined(self) -> None:
        resolved = expand_config_params("ts_mean(close, w4h)", {"w4h": 5})
        assert resolved == "ts_mean(close, 5)"

    def test_only_numeric_params_are_inlined(self) -> None:
        resolved = expand_config_params("ts_mean(close, w4h)", {"w4h": 5, "label": "foo"})
        assert resolved == "ts_mean(close, 5)"

    def test_missing_param_keeps_variable_reference(self) -> None:
        # Variables not in the parameters dict are untouched; they fall back
        # to runtime resolution by the Evaluator.
        resolved = expand_config_params("ts_mean(close, w4h)", {})
        assert "w4h" in resolved

    def test_empty_parameters_still_normalises(self) -> None:
        resolved = expand_config_params(" ts_mean ( close , 5 ) ", {})
        assert resolved == "ts_mean(close, 5)"

    def test_boolean_param_is_coerced_to_float(self) -> None:
        resolved = expand_config_params("normalize(x, use_std, 0)", {"use_std": True, "x": 0.0})
        # `x` → 0 and `use_std` → 1; both are inlined as numbers.
        assert "1" in resolved


# ── Operator slot detection ────────────────────────────────────────────────


class TestDetectOperatorSlots:
    def test_outer_cs_rank_is_cs_normalize(self) -> None:
        slots = detect_operator_slots("cs_rank(-correlation(high, rank(volume), 10))")
        outer = next(s for s in slots if s.position == "outer")
        assert outer.current_op == "cs_rank"
        assert outer.group == "cs_normalize"
        # cs_normalize offers more alternatives than cs_ranking.
        assert len(outer.alternatives) == len(OPERATOR_GROUPS["cs_normalize"])

    def test_inner_rank_is_cs_ranking(self) -> None:
        slots = detect_operator_slots("cs_rank(-correlation(high, rank(volume), 10))")
        inner_slots = [s for s in slots if s.position == "inner"]
        assert len(inner_slots) == 1
        assert inner_slots[0].current_op == "cs_rank"  # alias resolved
        assert inner_slots[0].group == "cs_ranking"

    def test_rank_alias_is_recognised(self) -> None:
        slots = detect_operator_slots("rank(close)")
        assert len(slots) == 1
        assert slots[0].current_op == "cs_rank"  # canonicalised

    def test_top_level_sign_is_treated_as_outer(self) -> None:
        # ``-cs_rank(x)`` normalises to ``-1 * cs_rank(x)`` — the outer
        # wrapper should still be classified as "outer".
        slots = detect_operator_slots("-cs_rank(close)")
        outer = next(s for s in slots if s.position == "outer")
        assert outer.current_op == "cs_rank"

    def test_no_cs_wrapper_yields_no_slots(self) -> None:
        # A pure arithmetic expression has no substitutable wrappers.
        slots = detect_operator_slots("close - open")
        assert slots == ()


# ── Variable slot detection ────────────────────────────────────────────────


class TestDetectVariableSlots:
    def test_price_and_volume_slots_detected(self) -> None:
        slots = detect_variable_slots(
            "-correlation(high, rank(volume), 10)",
            available_vars={"high", "low", "open", "close", "volume", "quote_volume"},
        )
        by_group = {s.group_name for s in slots}
        assert {"price_ohlc", "volume_like"} <= by_group

    def test_scope_filtering_excludes_unavailable(self) -> None:
        slots = detect_variable_slots(
            "-correlation(high, rank(volume), 10)",
            available_vars={"high", "low", "close"},  # no volume / quote_volume
        )
        volume_slots = [s for s in slots if s.group_name == "volume_like"]
        # ``volume`` has no available peer, slot is suppressed entirely.
        assert volume_slots == []

    def test_broadcast_scope_is_reported(self) -> None:
        slots = detect_variable_slots(
            "btc_close * close",
            available_vars={"btc_close", "eth_close", "close"},
        )
        bench = next(s for s in slots if s.group_name == "benchmark_price")
        assert bench.scope == VAR_SCOPE_BROADCAST

    def test_derived_variable_is_skipped(self) -> None:
        # User-defined vars (e.g. from YAML ``variables:``) should not be
        # enumerated as swap candidates.
        slots = detect_variable_slots(
            "-correlation(vwap, rank(volume), 10)",
            available_vars={"volume", "quote_volume", "vwap"},
            derived_vars={"vwap"},
        )
        assert all(s.current_var != "vwap" for s in slots)


# ── Reconstruction ─────────────────────────────────────────────────────────


class TestReconstructExpression:
    def test_numeric_only_round_trip(self) -> None:
        template, values = expression_template("-correlation(high, rank(volume), 10)")
        rebuilt = reconstruct_expression(template, {"p0": -1.0, "p1": 10.0})
        assert rebuilt == "-1 * correlation(high, rank(volume), 10)"

    def test_numeric_overrides_window(self) -> None:
        template, values = expression_template("-correlation(high, rank(volume), 10)")
        rebuilt = reconstruct_expression(template, {"p0": -1.0, "p1": 30.0})
        assert "30" in rebuilt and "10" not in rebuilt

    def test_variable_swap_preserves_structure(self) -> None:
        expr = "-correlation(high, rank(volume), 10)"
        template, nparams, ops, vars_ = build_search_space(
            expr,
            available_vars={"high", "low", "open", "close", "volume", "quote_volume"},
            tune_variables=True,
        )
        vol_slot = next(s for s in vars_ if s.group_name == "volume_like")
        rebuilt = reconstruct_expression(
            template,
            {"p0": -1.0, "p1": 10.0},
            var_choices={vol_slot.slot_id: "quote_volume"},
            variable_slots=vars_,
        )
        assert "quote_volume" in rebuilt
        assert "volume" not in rebuilt.replace("quote_volume", "")

    def test_outer_operator_swap_with_extras(self) -> None:
        expr = "cs_rank(-correlation(high, rank(volume), 5))"
        template, nparams, ops, vars_ = build_search_space(
            expr,
            tune_operators=True,
        )
        outer = next(s for s in ops if s.position == "outer")
        rebuilt = reconstruct_expression(
            template,
            {"p0": -1.0, "p1": 5.0},
            op_choices={outer.slot_id: "winsorize"},
            operator_slots=ops,
            op_extra_params={outer.slot_id: {"std_mult": 3.0}},
        )
        assert rebuilt.startswith("winsorize(")
        assert rebuilt.rstrip(")").endswith("3")

    def test_identity_drops_wrapper(self) -> None:
        expr = "cs_rank(close)"
        template, _, ops, _ = build_search_space(
            expr,
            tune_operators=True,
        )
        rebuilt = reconstruct_expression(
            template,
            {},
            op_choices={ops[0].slot_id: "identity"},
            operator_slots=ops,
        )
        assert rebuilt == "close"

    def test_three_dim_combination(self) -> None:
        expr = "cs_rank(-correlation(high, rank(volume), 5))"
        template, nparams, ops, vars_ = build_search_space(
            expr,
            available_vars={"high", "low", "volume", "quote_volume"},
            tune_numeric=True,
            tune_operators=True,
            tune_variables=True,
        )
        outer = next(s for s in ops if s.position == "outer")
        inner = next(s for s in ops if s.position == "inner")
        vol = next(s for s in vars_ if s.group_name == "volume_like")
        rebuilt = reconstruct_expression(
            template,
            {"p0": -1.0, "p1": 20.0},
            op_choices={outer.slot_id: "cs_zscore", inner.slot_id: "identity"},
            operator_slots=ops,
            var_choices={vol.slot_id: "quote_volume"},
            variable_slots=vars_,
        )
        # Outer became cs_zscore, inner rank dropped, volume → quote_volume,
        # window → 20.
        assert rebuilt.startswith("cs_zscore(")
        assert "quote_volume" in rebuilt
        assert "rank(" not in rebuilt
        assert "20" in rebuilt


class TestBuildSearchSpace:
    def test_dimensions_off_return_empty_slots(self) -> None:
        expr = "cs_rank(-correlation(high, rank(volume), 5))"
        _, n, ops, vars_ = build_search_space(expr, tune_numeric=False, tune_operators=False, tune_variables=False)
        assert n == () and ops == () and vars_ == ()

    def test_variable_slots_respect_availability(self) -> None:
        _, _, _, vars_ = build_search_space(
            "-correlation(high, rank(volume), 10)",
            available_vars={"high", "close"},  # no volume peer
            tune_variables=True,
        )
        group_names = {s.group_name for s in vars_}
        assert "volume_like" not in group_names

    def test_config_params_are_expanded_first(self) -> None:
        template, n, _, _ = build_search_space(
            "ts_mean(close, w4h)",
            parameters={"w4h": 30},
            tune_numeric=True,
        )
        assert len(n) == 1
        assert n[0].original_value == 30.0


class TestOperatorGroupsMetadata:
    def test_cs_normalize_has_identity(self) -> None:
        names = {alt.name for alt in OPERATOR_GROUPS["cs_normalize"]}
        assert "identity" in names

    def test_cs_ranking_is_minimal(self) -> None:
        names = {alt.name for alt in OPERATOR_GROUPS["cs_ranking"]}
        assert names == {"cs_rank", "cs_zscore", "identity"}

    def test_variable_groups_catalogue_is_nonempty(self) -> None:
        assert "price_ohlc" in VARIABLE_GROUPS
        assert VARIABLE_GROUPS["price_ohlc"].scope == VAR_SCOPE_PER_INSTRUMENT
        assert "btc_close" in VARIABLE_GROUPS["benchmark_price"].members
