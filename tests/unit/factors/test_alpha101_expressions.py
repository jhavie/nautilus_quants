# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests that all Alpha101 expressions parse correctly."""

import pytest

from nautilus_quants.factors.builtin.alpha101 import ALPHA101_FACTORS
from nautilus_quants.factors.expression.parser import parse_expression


class TestAlpha101Parse:
    """All Alpha101 expressions must parse without error."""

    @pytest.mark.parametrize("name,info", list(ALPHA101_FACTORS.items()))
    def test_all_parse(self, name, info):
        """Every alpha expression should parse without raising."""
        expr = info["expression"]
        ast = parse_expression(expr)
        assert ast is not None, f"{name} returned None AST"

    def test_count_gte_70(self):
        """We should have at least 70 alphas implemented."""
        assert len(ALPHA101_FACTORS) >= 70, f"Only {len(ALPHA101_FACTORS)} alphas registered"

    def test_all_have_description(self):
        for name, info in ALPHA101_FACTORS.items():
            assert "description" in info, f"{name} missing description"
            assert info["description"], f"{name} has empty description"

    def test_all_have_category(self):
        for name, info in ALPHA101_FACTORS.items():
            assert "category" in info, f"{name} missing category"

    def test_alpha001_uses_new_syntax(self):
        expr = ALPHA101_FACTORS["alpha001"]["expression"]
        assert "if_else" in expr or "signed_power" in expr, \
            "alpha001 should use if_else or signed_power"

    def test_alpha041_uses_power(self):
        expr = ALPHA101_FACTORS["alpha041"]["expression"]
        assert "power" in expr

    def test_alpha039_uses_decay_linear(self):
        expr = ALPHA101_FACTORS["alpha039"]["expression"]
        assert "decay_linear" in expr

    def test_alpha081_uses_ts_product(self):
        expr = ALPHA101_FACTORS["alpha081"]["expression"]
        assert "ts_product" in expr
