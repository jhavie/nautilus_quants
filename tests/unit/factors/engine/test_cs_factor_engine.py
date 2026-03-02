# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for CsFactorEngine - Cross-sectional factor computation engine."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

from nautilus_quants.factors.engine.cs_factor_engine import CsFactorEngine


class TestCsFactorEngineDetection:
    """Tests for automatic factor type detection."""

    def test_detect_time_series_simple_variable(self):
        """Simple variable should be detected as time-series."""
        engine = CsFactorEngine()
        assert not engine._has_cs_operator("volume")
        assert not engine._has_cs_operator("close")

    def test_detect_time_series_with_ts_operators(self):
        """Expression with time-series operators should be time-series."""
        engine = CsFactorEngine()
        assert not engine._has_cs_operator("(close - delay(close, 3)) / delay(close, 3)")
        assert not engine._has_cs_operator("ts_std(close / open, 24)")
        assert not engine._has_cs_operator("correlation(close, volume, 96)")

    def test_detect_cross_sectional_with_cs_operators(self):
        """Expression with cross-sectional operators should be cross-sectional."""
        engine = CsFactorEngine()
        assert engine._has_cs_operator("normalize(volume, true, 0)")
        assert engine._has_cs_operator("winsorize(momentum, 2)")
        assert engine._has_cs_operator("cs_rank(volume)")
        assert engine._has_cs_operator("cs_zscore(momentum)")

    def test_detect_cross_sectional_nested(self):
        """Nested cross-sectional operators should be detected."""
        engine = CsFactorEngine()
        assert engine._has_cs_operator("normalize(winsorize(volume, 2), true, 0)")
        assert engine._has_cs_operator("scale(cs_rank(momentum))")


class TestCsFactorEngineReferenceExtraction:
    """Tests for variable reference extraction."""

    def test_extract_simple_references(self):
        """Should extract simple variable references."""
        engine = CsFactorEngine()
        refs = engine._extract_references("0.6 * volume_norm + 0.4 * momentum_norm")
        assert "volume_norm" in refs
        assert "momentum_norm" in refs

    def test_exclude_keywords(self):
        """Should exclude Python/expression keywords."""
        engine = CsFactorEngine()
        refs = engine._extract_references("normalize(volume, true, 0)")
        assert "true" not in refs
        assert "volume" not in refs  # Base variables excluded
        assert "normalize" not in refs  # Function name excluded

    def test_exclude_ohlcv(self):
        """Should exclude OHLCV base variables."""
        engine = CsFactorEngine()
        refs = engine._extract_references("close + open + high + low + volume")
        assert "close" not in refs
        assert "open" not in refs
        assert "high" not in refs
        assert "low" not in refs
        assert "volume" not in refs


class TestCsFactorEngineClassification:
    """Tests for factor classification with config."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock factor config."""
        from dataclasses import dataclass

        @dataclass
        class MockFactorDef:
            name: str
            expression: str

        @dataclass
        class MockConfig:
            factors: list

        # Create test factors
        factors = [
            MockFactorDef("volume", "volume"),
            MockFactorDef("momentum", "(close - delay(close, 3)) / delay(close, 3)"),
            MockFactorDef("volume_norm", "normalize(winsorize(volume, 2), true, 0)"),
            MockFactorDef("momentum_norm", "normalize(winsorize(momentum, 2), true, 0)"),
            MockFactorDef("composite", "0.6 * volume_norm + 0.4 * momentum_norm"),
        ]

        return MockConfig(factors=factors)

    def test_classify_time_series_factors(self, mock_config):
        """Should classify time-series factors correctly."""
        engine = CsFactorEngine(mock_config)
        ts_names = engine.ts_factor_names

        # volume and momentum are time-series
        assert "volume" in ts_names
        assert "momentum" in ts_names

        # Normalized versions and composite are NOT time-series
        assert "volume_norm" not in ts_names
        assert "momentum_norm" not in ts_names
        assert "composite" not in ts_names

    def test_classify_cross_sectional_factors(self, mock_config):
        """Should classify cross-sectional factors correctly."""
        engine = CsFactorEngine(mock_config)
        cs_names = engine.cs_factor_names

        # Normalized factors are cross-sectional
        assert "volume_norm" in cs_names
        assert "momentum_norm" in cs_names

        # Composite is also cross-sectional (references CS factors)
        assert "composite" in cs_names

        # Time-series factors are NOT cross-sectional
        assert "volume" not in cs_names
        assert "momentum" not in cs_names

    def test_propagate_cross_sectional_property(self, mock_config):
        """Composite factor should be CS because it references CS factors."""
        engine = CsFactorEngine(mock_config)

        # composite has no CS operators but references volume_norm (CS)
        # So it should be classified as CS
        assert "composite" in engine.cs_factor_names


class TestCsFactorEngineCompute:
    """Tests for cross-sectional factor computation."""

    def test_compute_simple_normalize(self):
        """Should compute normalize correctly."""
        engine = CsFactorEngine()

        ts_values = {
            "volume": {"A": 3.0, "B": 5.0, "C": 6.0, "D": 2.0},
        }

        # Manually set up cs_factors for testing
        from dataclasses import dataclass

        @dataclass
        class MockFactorDef:
            name: str
            expression: str

        engine._cs_factors = [
            MockFactorDef("volume_norm", "normalize(volume, false, 0)"),
        ]

        result = engine.compute(ts_values)

        # Mean = 4, so values should be [-1, 1, 2, -2]
        assert "volume_norm" in result
        assert result["volume_norm"]["A"] == pytest.approx(-1.0)
        assert result["volume_norm"]["B"] == pytest.approx(1.0)
        assert result["volume_norm"]["C"] == pytest.approx(2.0)
        assert result["volume_norm"]["D"] == pytest.approx(-2.0)

    def test_compute_weighted_sum(self):
        """Should compute weighted sum correctly."""
        engine = CsFactorEngine()

        ts_values = {
            "a": {"X": 10.0, "Y": 20.0},
            "b": {"X": 5.0, "Y": 10.0},
        }

        from dataclasses import dataclass

        @dataclass
        class MockFactorDef:
            name: str
            expression: str

        engine._cs_factors = [
            MockFactorDef("composite", "0.6 * a + 0.4 * b"),
        ]

        result = engine.compute(ts_values)

        # X: 0.6 * 10 + 0.4 * 5 = 8
        # Y: 0.6 * 20 + 0.4 * 10 = 16
        assert "composite" in result
        assert result["composite"]["X"] == pytest.approx(8.0)
        assert result["composite"]["Y"] == pytest.approx(16.0)

    def test_compute_nested_functions(self):
        """Should compute nested function calls correctly."""
        engine = CsFactorEngine()

        ts_values = {
            "volume": {"A": 3.0, "B": 5.0, "C": 100.0, "D": 2.0},  # C is outlier
        }

        from dataclasses import dataclass

        @dataclass
        class MockFactorDef:
            name: str
            expression: str

        engine._cs_factors = [
            MockFactorDef("volume_norm", "normalize(winsorize(volume, 2), true, 0)"),
        ]

        result = engine.compute(ts_values)

        # Winsorize should clip the outlier, then normalize
        assert "volume_norm" in result
        # Values should be z-scores after winsorization
        assert all(not math.isnan(v) for v in result["volume_norm"].values())

    def test_compute_with_nan(self):
        """Should handle NaN values correctly."""
        engine = CsFactorEngine()

        ts_values = {
            "a": {"X": 10.0, "Y": float("nan"), "Z": 20.0},
            "b": {"X": 5.0, "Y": 10.0, "Z": 15.0},
        }

        from dataclasses import dataclass

        @dataclass
        class MockFactorDef:
            name: str
            expression: str

        engine._cs_factors = [
            MockFactorDef("composite", "0.6 * a + 0.4 * b"),
        ]

        result = engine.compute(ts_values)

        assert result["composite"]["X"] == pytest.approx(8.0)
        assert "Y" not in result["composite"]  # NaN inputs are filtered, instrument absent
        assert result["composite"]["Z"] == pytest.approx(18.0)

    def test_compute_dependency_order(self):
        """Should compute factors in dependency order."""
        engine = CsFactorEngine()

        ts_values = {
            "raw": {"A": 10.0, "B": 20.0, "C": 30.0},
        }

        from dataclasses import dataclass

        @dataclass
        class MockFactorDef:
            name: str
            expression: str

        # intermediate depends on raw, final depends on intermediate
        engine._cs_factors = [
            MockFactorDef("intermediate", "normalize(raw, false, 0)"),
            MockFactorDef("final", "cs_scale(intermediate)"),
        ]

        result = engine.compute(ts_values)

        # Both should be computed
        assert "intermediate" in result
        assert "final" in result

        # final should sum to ~1 (scaled)
        total = sum(abs(v) for v in result["final"].values())
        assert total == pytest.approx(1.0)


class TestCsFactorEngineTopologicalSort:
    """Tests for topological sort of factors by dependency."""

    def test_sort_independent_factors(self):
        """Independent factors can be in any order."""
        from dataclasses import dataclass

        @dataclass
        class MockFactorDef:
            name: str
            expression: str

        factors = [
            MockFactorDef("a", "normalize(x, false, 0)"),
            MockFactorDef("b", "normalize(y, false, 0)"),
        ]

        engine = CsFactorEngine()
        sorted_factors = engine._topological_sort(factors, {"a", "b"})

        # Both should be included
        names = [f.name for f in sorted_factors]
        assert "a" in names
        assert "b" in names

    def test_sort_dependent_factors(self):
        """Dependent factors should be in correct order."""
        from dataclasses import dataclass

        @dataclass
        class MockFactorDef:
            name: str
            expression: str

        factors = [
            MockFactorDef("c", "0.5 * a + 0.5 * b"),  # depends on a, b
            MockFactorDef("a", "normalize(x, false, 0)"),
            MockFactorDef("b", "normalize(y, false, 0)"),
        ]

        engine = CsFactorEngine()
        sorted_factors = engine._topological_sort(factors, {"a", "b", "c"})

        names = [f.name for f in sorted_factors]
        # a and b should come before c
        assert names.index("a") < names.index("c")
        assert names.index("b") < names.index("c")
