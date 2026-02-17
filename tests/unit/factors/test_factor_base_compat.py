"""Tests for Factor base class backward compatibility (Issue 3).

Verifies that external subclasses using the old compute(self, data) signature
(without var_cache) still work through the try/except TypeError fallback
in Factor.update().
"""

from unittest.mock import MagicMock

from nautilus_quants.factors.base.factor import Factor


class _OldStyleFactor(Factor):
    """Simulates an external subclass with the old compute() signature."""

    def compute(self, data):  # noqa: D102 — intentionally no var_cache param
        return float(data.close) * 2.0


class _NewStyleFactor(Factor):
    """Subclass using the current compute() signature with var_cache."""

    def compute(self, data, var_cache=None):  # noqa: D102
        base = float(data.close)
        if var_cache and "multiplier" in var_cache:
            return base * var_cache["multiplier"]
        return base


class TestFactorBaseCompat:
    """Backward compatibility for Factor.update() with old subclass signatures."""

    def _make_data(self, close: float = 100.0):
        data = MagicMock()
        data.close = close
        return data

    def test_old_style_subclass_update_works(self):
        factor = _OldStyleFactor(name="old_factor", warmup_period=0)
        data = self._make_data(50.0)
        result = factor.update(data, var_cache={"x": 1})
        assert result == 100.0

    def test_old_style_subclass_warmup(self):
        factor = _OldStyleFactor(name="old_factor", warmup_period=2)
        data = self._make_data(50.0)

        # First call: not warmed up yet
        import math
        assert math.isnan(factor.update(data))

        # Second call: warmup_period reached, should compute
        result = factor.update(data)
        assert result == 100.0

    def test_new_style_subclass_receives_var_cache(self):
        factor = _NewStyleFactor(name="new_factor", warmup_period=0)
        data = self._make_data(10.0)
        result = factor.update(data, var_cache={"multiplier": 3.0})
        assert result == 30.0

    def test_new_style_subclass_without_var_cache(self):
        factor = _NewStyleFactor(name="new_factor", warmup_period=0)
        data = self._make_data(10.0)
        result = factor.update(data)
        assert result == 10.0
