# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for prompt construction with theme, subsets, and constraints."""

from __future__ import annotations

from nautilus_quants.alpha.mining.agent.prompts import (
    _filter_operator_reference,
    build_generation_prompt,
)
from nautilus_quants.factors.expression.complexity import ComplexityConstraints


# ── _filter_operator_reference ───────────────────────────────────


class TestFilterOperatorReference:

    def test_filters_to_subset(self):
        result = _filter_operator_reference(("ts_mean", "cs_rank"))
        assert "ts_mean" in result
        assert "cs_rank" in result
        assert "ts_std" not in result
        assert "covariance" not in result

    def test_always_includes_ternary_and_arithmetic(self):
        result = _filter_operator_reference(("ts_mean",))
        assert "Ternary" in result or "condition ? true_expr" in result
        assert "Arithmetic" in result or "+, -, *, /" in result

    def test_empty_subset_returns_ternary_arithmetic_only(self):
        result = _filter_operator_reference(())
        assert "ts_mean" not in result
        assert "cs_rank" not in result

    def test_multiple_categories(self):
        result = _filter_operator_reference(("ts_mean", "cs_rank", "abs"))
        assert "ts_mean" in result
        assert "cs_rank" in result
        assert "abs" in result


# ── build_generation_prompt with theme ───────────────────────────


class TestPromptTheme:

    def test_theme_appears_in_prompt(self):
        prompt = build_generation_prompt(
            round_num=1, num_factors=8, bar_spec="4h",
            previous_factors=[], top_factors=[],
            theme="量价因子",
        )
        assert "Theme: 量价因子" in prompt
        assert "diverse hypotheses" in prompt

    def test_theme_takes_priority_over_hypothesis(self):
        prompt = build_generation_prompt(
            round_num=1, num_factors=8, bar_spec="4h",
            previous_factors=[], top_factors=[],
            hypothesis="should be ignored",
            theme="波动率因子",
        )
        assert "Theme: 波动率因子" in prompt
        assert "should be ignored" not in prompt

    def test_no_theme_no_hypothesis_round1_default(self):
        prompt = build_generation_prompt(
            round_num=1, num_factors=8, bar_spec="4h",
            previous_factors=[], top_factors=[],
        )
        assert "diverse alpha sources" in prompt


# ── build_generation_prompt with operator_subset ─────────────────


class TestPromptOperatorSubset:

    def test_subset_filters_operators(self):
        prompt = build_generation_prompt(
            round_num=1, num_factors=8, bar_spec="4h",
            previous_factors=[], top_factors=[],
            operator_subset=("ts_mean", "cs_rank"),
        )
        assert "ts_mean" in prompt
        assert "cs_rank" in prompt
        # ts_std should not appear in the operator reference section
        # (it may appear in construction rules examples, but not in the
        # Available Operators section heading)
        lines = prompt.split("## Available Operators")[1].split("##")[0]
        assert "ts_std" not in lines

    def test_no_subset_includes_all(self):
        prompt = build_generation_prompt(
            round_num=1, num_factors=8, bar_spec="4h",
            previous_factors=[], top_factors=[],
        )
        assert "ts_mean" in prompt
        assert "ts_std" in prompt
        assert "cs_rank" in prompt
        assert "correlation" in prompt


# ── build_generation_prompt with variable_subset ─────────────────


class TestPromptVariableSubset:

    def test_subset_limits_variables(self):
        prompt = build_generation_prompt(
            round_num=1, num_factors=8, bar_spec="4h",
            previous_factors=[], top_factors=[],
            variable_subset=("close", "volume"),
        )
        vars_section = prompt.split("## Available Variables")[1].split("##")[0]
        assert "close" in vars_section
        assert "volume" in vars_section
        assert "funding_rate" not in vars_section

    def test_subset_with_returns_adds_doc(self):
        prompt = build_generation_prompt(
            round_num=1, num_factors=8, bar_spec="4h",
            previous_factors=[], top_factors=[],
            variable_subset=("close", "returns"),
        )
        vars_section = prompt.split("## Available Variables")[1].split("##")[0]
        assert "returns" in vars_section
        assert "pre-computed" in vars_section

    def test_no_subset_includes_all(self):
        prompt = build_generation_prompt(
            round_num=1, num_factors=8, bar_spec="4h",
            previous_factors=[], top_factors=[],
        )
        assert "funding_rate" in prompt
        assert "open_interest" in prompt


# ── build_generation_prompt with constraints ─────────────────────


class TestPromptConstraints:

    def test_constraints_section_appears(self):
        prompt = build_generation_prompt(
            round_num=1, num_factors=8, bar_spec="4h",
            previous_factors=[], top_factors=[],
            constraints=ComplexityConstraints(max_depth=4, max_window=500),
        )
        assert "Hard Constraints" in prompt
        assert "4 levels" in prompt
        assert "500 bars" in prompt

    def test_no_constraints_no_section(self):
        prompt = build_generation_prompt(
            round_num=1, num_factors=8, bar_spec="4h",
            previous_factors=[], top_factors=[],
        )
        assert "Hard Constraints" not in prompt


# ── regression: backward compat ──────────────────────────────────


class TestPromptBackwardCompat:

    def test_no_new_params_same_behavior(self):
        prompt = build_generation_prompt(
            round_num=2, num_factors=8, bar_spec="4h",
            previous_factors=[
                {"expression": "cs_rank(delta(close, 6))"},
            ],
            top_factors=[
                {"expression": "cs_rank(delta(close, 6))",
                 "ic_mean": {"4h": 0.03}, "icir": {"4h": 1.5}},
            ],
            hypothesis="momentum test",
        )
        assert "momentum test" in prompt
        assert "cs_rank(delta(close, 6))" in prompt
        assert "Hard Constraints" not in prompt
        assert "Theme:" not in prompt
