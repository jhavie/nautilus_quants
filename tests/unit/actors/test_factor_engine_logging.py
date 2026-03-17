# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for FactorEngineActor log preview formatting."""

from __future__ import annotations

from nautilus_quants.actors.factor_engine import _format_factor_values_preview


class TestFormatFactorValuesPreview:
    def test_formats_all_values_when_within_limit(self) -> None:
        values = {"B": 2.0, "A": 1.0, "C": 3.0}
        preview = _format_factor_values_preview(values, max_symbols=5)
        assert preview == "A=1.000000, B=2.000000, C=3.000000"

    def test_truncates_values_when_exceeding_limit(self) -> None:
        values = {
            "A": 1.0,
            "B": 2.0,
            "C": 3.0,
            "D": 4.0,
            "E": 5.0,
            "F": 6.0,
        }
        preview = _format_factor_values_preview(values, max_symbols=4)
        assert preview == "A=1.000000, B=2.000000, E=5.000000, F=6.000000, ... (+2 more)"
