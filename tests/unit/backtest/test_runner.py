"""Unit tests for BacktestRunner.

NOTE: BacktestRunner is DEPRECATED. Use Nautilus native BacktestRunConfig directly.
These tests verify the deprecated module still functions for legacy usage.
"""

import pytest


class TestBacktestRunnerDeprecation:
    """Tests to verify deprecation notice."""

    def test_runner_module_has_deprecation_notice(self) -> None:
        """Test that runner module contains deprecation notice."""
        from nautilus_quants.backtest import runner
        
        docstring = runner.__doc__ or ""
        assert "DEPRECATED" in docstring
