"""Integration tests for backtest edge cases and error handling.

NOTE: These tests use the legacy BacktestConfig approach which has been removed.
The tests are preserved for reference but marked as skipped.
For Actor-decoupled backtesting, use Nautilus native BacktestRunConfig directly.
"""

import pytest

# Skip all tests in this module - BacktestConfig has been removed
pytestmark = pytest.mark.skip(reason="BacktestConfig removed - use Nautilus native BacktestRunConfig")
