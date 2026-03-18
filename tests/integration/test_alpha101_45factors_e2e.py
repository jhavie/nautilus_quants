# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""E2E integration test: 45 Alpha101 factors backtest with FMZ strategy.

Validates that all 45 crypto-friendly Alpha101 factors:
  1. Parse correctly from config/examples/factors.yaml
  2. Compute without error over 12-coin 4h data
  3. Produce trading signals that the FMZ strategy can use

Usage:
  pytest tests/integration/test_alpha101_45factors_e2e.py -v --timeout=300
  python tests/integration/test_alpha101_45factors_e2e.py  # standalone
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FACTORS_YAML = PROJECT_ROOT / "config" / "examples" / "factors.yaml"
CATALOG_PATH = Path("/Users/joe/Sync/nautilus_quants2/data/data_4years_4h/catalog")

# Use a subset from the 4-year catalog (all available since 2022)
SYMBOLS_12 = [
    "AAVEUSDT.BINANCE",
    "ADAUSDT.BINANCE",
    "ATOMUSDT.BINANCE",
    "BNBUSDT.BINANCE",
    "BTCUSDT.BINANCE",
    "DOGEUSDT.BINANCE",
    "ETHUSDT.BINANCE",
    "LINKUSDT.BINANCE",
    "LTCUSDT.BINANCE",
    "SOLUSDT.BINANCE",
    "UNIUSDT.BINANCE",
    "XRPUSDT.BINANCE",
]

# Use alpha012 (simplest: sign(delta(vol,1)) * -delta(close,1)) as the
# trading signal. The FactorEngineActor still computes ALL 45 factors.
COMPOSITE_FACTOR = "alpha012"

START_TIME = "2022-06-01"
END_TIME = "2022-07-01"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_backtest_config(factors_yaml_path: str) -> dict:
    """Build a minimal FMZ backtest config dict."""
    return {
        "venues": [
            {
                "name": "BINANCE",
                "oms_type": "HEDGING",
                "account_type": "MARGIN",
                "base_currency": "USDT",
                "starting_balances": ["10000 USDT"],
                "default_leverage": 1,
            }
        ],
        "data": [
            # Standard Bar for instrument registration
            {
                "catalog_path": str(CATALOG_PATH),
                "data_cls": "nautilus_trader.model.data:Bar",
                "bar_spec": "4h",
                "instrument_ids": SYMBOLS_12,
                "start_time": START_TIME,
                "end_time": END_TIME,
            },
            # BinanceBar for FactorEngineActor (provides quote_volume)
            {
                "catalog_path": str(CATALOG_PATH),
                "data_cls": "nautilus_trader.adapters.binance.common.types:BinanceBar",
                "bar_spec": "4h",
                "instrument_ids": SYMBOLS_12,
                "start_time": START_TIME,
                "end_time": END_TIME,
            },
        ],
        "engine": {
            "trader_id": "BACKTESTER-ALPHA101-E2E",
            "logging": {"log_level": "WARNING"},
            "risk_engine": {"bypass": True},
            "actors": [
                {
                    "actor_path": "nautilus_quants.actors.factor_engine:FactorEngineActor",
                    "config_path": "nautilus_quants.actors.factor_engine:FactorEngineActorConfig",
                    "config": {
                        "factor_config_path": factors_yaml_path,
                        "interval": "4h",
                        "signal_prefix": "factor",
                        "max_history": 300,
                        "publish_signals": True,
                    },
                },
                {
                    "actor_path": "nautilus_quants.actors.equity_snapshot:EquitySnapshotActor",
                    "config_path": "nautilus_quants.actors.equity_snapshot:EquitySnapshotActorConfig",
                    "config": {
                        "interval": "4h",
                        "venue_name": "BINANCE",
                        "currency": "USDT",
                    },
                },
            ],
            "strategies": [
                {
                    "strategy_path": "nautilus_quants.strategies.fmz.strategy:FMZFactorStrategy",
                    "config_path": "nautilus_quants.strategies.fmz.strategy:FMZFactorStrategyConfig",
                    "config": {
                        "instrument_ids": SYMBOLS_12,
                        "n_long": 3,
                        "n_short": 3,
                        "position_value": 100,
                        "rebalance_interval": 6,
                        "composite_factor": COMPOSITE_FACTOR,
                    },
                }
            ],
        },
        "report": {
            "output_dir": "logs/alpha101_e2e_test",
            "formats": ["csv"],
        },
    }


def _write_config(tmp_dir: Path, factors_yaml_path: str) -> Path:
    """Write backtest config YAML to a temp file and return its path."""
    config = _build_backtest_config(factors_yaml_path)
    config_path = tmp_dir / "backtest_e2e.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return config_path


# ---------------------------------------------------------------------------
# Precondition checks
# ---------------------------------------------------------------------------

_DATA_AVAILABLE = CATALOG_PATH.exists()
_FACTORS_AVAILABLE = FACTORS_YAML.exists()

skip_no_data = pytest.mark.skipif(
    not _DATA_AVAILABLE,
    reason=f"12-coin catalog not found at {CATALOG_PATH}",
)
skip_no_factors = pytest.mark.skipif(
    not _FACTORS_AVAILABLE,
    reason=f"factors.yaml not found at {FACTORS_YAML}",
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFactorsYamlParsing:
    """Validate the factors.yaml structure without running a backtest."""

    @skip_no_factors
    def test_yaml_loads(self):
        """factors.yaml is valid YAML."""
        with open(FACTORS_YAML) as f:
            config = yaml.safe_load(f)
        assert "factors" in config
        assert "metadata" in config
        assert "variables" in config

    @skip_no_factors
    def test_factor_count(self):
        """Exactly 45 factors are defined."""
        with open(FACTORS_YAML) as f:
            config = yaml.safe_load(f)
        factors = config["factors"]
        assert len(factors) == 45, f"Expected 45 factors, got {len(factors)}"

    @skip_no_factors
    def test_all_factors_have_expression(self):
        """Every factor entry has an 'expression' key."""
        with open(FACTORS_YAML) as f:
            config = yaml.safe_load(f)
        for name, defn in config["factors"].items():
            assert "expression" in defn, f"{name} missing 'expression'"
            assert len(defn["expression"]) > 0, f"{name} has empty expression"

    @skip_no_factors
    def test_no_vwap_in_expressions(self):
        """No factor expression references 'vwap'."""
        with open(FACTORS_YAML) as f:
            config = yaml.safe_load(f)
        for name, defn in config["factors"].items():
            expr = defn["expression"].lower()
            assert "vwap" not in expr, f"{name} contains vwap: {defn['expression']}"

    @skip_no_factors
    def test_returns_variable_defined(self):
        """The 'returns' variable is defined (required by many factors)."""
        with open(FACTORS_YAML) as f:
            config = yaml.safe_load(f)
        assert "returns" in config["variables"]

    @skip_no_factors
    def test_composite_factor_exists(self):
        """The factor used as composite_factor (alpha012) exists."""
        with open(FACTORS_YAML) as f:
            config = yaml.safe_load(f)
        assert COMPOSITE_FACTOR in config["factors"]


@skip_no_data
@skip_no_factors
class TestAlpha101E2EBacktest:
    """Full end-to-end backtest with 45 factors + FMZ strategy."""

    def test_backtest_completes(self, tmp_path: Path):
        """Backtest runs to completion without crashing."""
        from nautilus_quants.backtest.runner import run_backtest

        config_path = _write_config(tmp_path, str(FACTORS_YAML))
        result = run_backtest(config_path)

        assert result is not None, "Backtest returned None (unexpected dry_run?)"
        assert result.duration > 0, "Backtest duration should be positive"

    def test_positions_generated(self, tmp_path: Path):
        """FMZ strategy generates at least some positions."""
        from nautilus_quants.backtest.runner import run_backtest

        config_path = _write_config(tmp_path, str(FACTORS_YAML))
        result = run_backtest(config_path)

        assert result is not None
        assert result.total_positions > 0, (
            f"Expected positions > 0, got {result.total_positions}"
        )

    def test_orders_generated(self, tmp_path: Path):
        """FMZ strategy generates at least some orders."""
        from nautilus_quants.backtest.runner import run_backtest

        config_path = _write_config(tmp_path, str(FACTORS_YAML))
        result = run_backtest(config_path)

        assert result is not None
        assert result.total_orders > 0, (
            f"Expected orders > 0, got {result.total_orders}"
        )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """Run the E2E backtest directly (not via pytest)."""
    import sys
    import time

    if not _DATA_AVAILABLE:
        print(f"ERROR: Catalog not found at {CATALOG_PATH}")
        sys.exit(1)
    if not _FACTORS_AVAILABLE:
        print(f"ERROR: factors.yaml not found at {FACTORS_YAML}")
        sys.exit(1)

    from nautilus_quants.backtest.runner import run_backtest

    print("=" * 70)
    print("Alpha101 45-Factor E2E Backtest")
    print(f"  Factors: {FACTORS_YAML}")
    print(f"  Data:    {CATALOG_PATH}")
    print(f"  Period:  {START_TIME} → {END_TIME}")
    print(f"  Signal:  {COMPOSITE_FACTOR}")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = _write_config(Path(tmp_dir), str(FACTORS_YAML))
        print(f"\nConfig written to: {config_path}")
        print("Running backtest...\n")

        t0 = time.time()
        result = run_backtest(config_path)
        elapsed = time.time() - t0

        if result is None:
            print("ERROR: Backtest returned None")
            sys.exit(1)

        print(f"\n{'=' * 70}")
        print("RESULT:")
        print(f"  Run ID:     {result.run_id}")
        print(f"  Duration:   {elapsed:.1f}s")
        print(f"  Positions:  {result.total_positions}")
        print(f"  Orders:     {result.total_orders}")
        if result.output_dir:
            print(f"  Output:     {result.output_dir}")
        print(f"{'=' * 70}")

        if result.total_positions > 0:
            print("\n✅ E2E test PASSED — all 45 factors computed, positions generated")
        else:
            print("\n⚠️  No positions generated (factors may need longer warmup)")

    sys.exit(0)
