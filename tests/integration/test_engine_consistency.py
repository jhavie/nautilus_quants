# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Engine Consistency Test — FactorEngine+CsFactorEngine vs PanelFactorEngine.

Feeds IDENTICAL synthetic bar data to both engines and compares the output
factor values. Covers all FMZ 4-factor composite factors:
  - TS factors: quote_volume, momentum_3h, volatility, corr
  - CS factors: volume_norm, momentum_3h_norm, volatility_norm, corr_norm
  - Composite: composite, composite_neg

If these diverge, one engine has a bug.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from nautilus_quants.factors.config import load_factor_config
from nautilus_quants.factors.engine.cs_factor_engine import CsFactorEngine
from nautilus_quants.factors.engine.factor_engine import FactorEngine
from nautilus_quants.factors.engine.panel_factor_engine import PanelFactorEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FACTOR_CONFIG_PATH = "config/fmz/factors.yaml"
N_INSTRUMENTS = 10
N_TIMESTAMPS = 150  # > 96 (correlation window) to ensure warmup is done
SEED = 42


def _generate_bars(
    n_instruments: int, n_timestamps: int, seed: int
) -> list[dict]:
    """Generate deterministic synthetic bar data.

    Returns list of dicts:
        [{"instrument_id": str, "ts": int,
          "open": float, "high": float, "low": float, "close": float,
          "volume": float, "quote_volume": float}, ...]

    Bars are ordered: all instruments for ts=0, then all for ts=1, etc.
    This mirrors Nautilus backtest replay order.
    """
    rng = np.random.RandomState(seed)
    instruments = [f"INST{i:03d}" for i in range(n_instruments)]
    bars: list[dict] = []

    # Per-instrument price state
    prices = {inst: 100.0 + rng.uniform(-20, 20) for inst in instruments}

    for t in range(n_timestamps):
        ts = (t + 1) * 3_600_000_000_000  # 1h intervals in nanoseconds
        for inst in instruments:
            prev = prices[inst]
            ret = rng.normal(0.0, 0.02)  # 2% daily vol
            close = prev * (1 + ret)
            high = close * (1 + abs(rng.normal(0, 0.005)))
            low = close * (1 - abs(rng.normal(0, 0.005)))
            open_ = prev * (1 + rng.normal(0, 0.005))
            volume = abs(rng.normal(1000, 300))
            quote_volume = volume * close

            bars.append({
                "instrument_id": inst,
                "ts": ts,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "quote_volume": quote_volume,
            })
            prices[inst] = close

    return bars


class MockBar:
    """Minimal mock bar for FactorEngine.on_bar()."""

    def __init__(self, instrument_id: str, open_: float, high: float,
                 low: float, close: float, volume: float,
                 quote_volume: float, ts_event: int):
        self.bar_type = type("BarType", (), {"instrument_id": instrument_id})()
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.quote_volume = quote_volume
        self.ts_event = ts_event
        # FactorEngine checks __dict__ for extra fields
        self.__dict__["quote_volume"] = quote_volume


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEngineConsistency:
    """Compare FactorEngine+CsFactorEngine vs PanelFactorEngine outputs."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up both engines with the same config and data."""
        self.config = load_factor_config(FACTOR_CONFIG_PATH)
        self.bars = _generate_bars(N_INSTRUMENTS, N_TIMESTAMPS, SEED)

        # --- Incremental engine (feature-019 style) ---
        self.inc_engine = FactorEngine(config=self.config, max_history=500)
        self.inc_engine.set_extra_fields(["quote_volume"])
        self.cs_engine = CsFactorEngine(config=self.config)

        # --- Panel engine (feature-018 style) ---
        self.panel_engine = PanelFactorEngine(config=self.config, max_history=500)
        self.panel_engine.set_extra_fields(["quote_volume"])

    def _run_both_engines(self) -> tuple[
        dict[str, dict[str, float]],  # incremental results at last ts
        dict[str, dict[str, float]],  # panel results at last ts
    ]:
        """Feed all bars to both engines and return last-timestamp results."""

        # --- Incremental engine ---
        # Tracks per-timestamp accumulated TS values for CS phase
        last_inc_ts_values: dict[str, dict[str, float]] = {}
        last_inc_cs_values: dict[str, dict[str, float]] = {}
        prev_ts = None

        for bar_data in self.bars:
            ts = bar_data["ts"]
            inst_id = bar_data["instrument_id"]

            # When timestamp advances, run CS phase on accumulated TS values
            if prev_ts is not None and ts != prev_ts:
                last_inc_cs_values = self.cs_engine.compute(last_inc_ts_values)
                # Merge CS results into TS results for full picture
                last_inc_full = dict(last_inc_ts_values)
                last_inc_full.update(last_inc_cs_values)

            mock_bar = MockBar(
                instrument_id=inst_id,
                open_=bar_data["open"],
                high=bar_data["high"],
                low=bar_data["low"],
                close=bar_data["close"],
                volume=bar_data["volume"],
                quote_volume=bar_data["quote_volume"],
                ts_event=ts,
            )

            ts_result = self.inc_engine.on_bar(mock_bar)

            # Accumulate TS results: merge into latest snapshot
            if ts != prev_ts:
                last_inc_ts_values = {}
            for fname, inst_vals in ts_result.factors.items():
                if fname not in last_inc_ts_values:
                    last_inc_ts_values[fname] = {}
                last_inc_ts_values[fname].update(inst_vals)

            prev_ts = ts

        # Final CS pass for last timestamp
        last_inc_cs_values = self.cs_engine.compute(last_inc_ts_values)
        inc_results = dict(last_inc_ts_values)
        inc_results.update(last_inc_cs_values)

        # --- Panel engine ---
        last_panel_results: dict[str, dict[str, float]] = {}
        prev_ts = None

        for bar_data in self.bars:
            ts = bar_data["ts"]
            inst_id = bar_data["instrument_id"]

            if prev_ts is not None and ts != prev_ts:
                last_panel_results = self.panel_engine.flush_and_compute(prev_ts)

            bar_dict = {
                "open": bar_data["open"],
                "high": bar_data["high"],
                "low": bar_data["low"],
                "close": bar_data["close"],
                "volume": bar_data["volume"],
                "quote_volume": bar_data["quote_volume"],
            }
            self.panel_engine.on_bar(inst_id, bar_dict, ts)
            prev_ts = ts

        # Final flush for last timestamp
        last_panel_results = self.panel_engine.flush_and_compute(prev_ts)

        return inc_results, last_panel_results

    # -- TS Factor Tests --

    def test_ts_factor_quote_volume(self):
        """quote_volume should be identical (raw field, no computation)."""
        inc, panel = self._run_both_engines()
        self._assert_factor_close(inc, panel, "quote_volume")

    def test_ts_factor_momentum_3h(self):
        """momentum_3h = (close - delay(close,3)) / delay(close,3)."""
        inc, panel = self._run_both_engines()
        self._assert_factor_close(inc, panel, "momentum_3h")

    def test_ts_factor_volatility(self):
        """volatility = ts_std(close/open, 24)."""
        inc, panel = self._run_both_engines()
        self._assert_factor_close(inc, panel, "volatility")

    def test_ts_factor_corr(self):
        """corr = correlation(close, quote_volume, 96)."""
        inc, panel = self._run_both_engines()
        self._assert_factor_close(inc, panel, "corr")

    # -- CS Factor Tests --

    def test_cs_factor_volume_norm(self):
        """volume_norm = normalize(clip_quantile(quote_volume, 0.2, 0.8), true, 0)."""
        inc, panel = self._run_both_engines()
        self._assert_factor_close(inc, panel, "volume_norm")

    def test_cs_factor_momentum_3h_norm(self):
        """momentum_3h_norm = normalize(clip_quantile(momentum_3h, 0.2, 0.8), true, 0)."""
        inc, panel = self._run_both_engines()
        self._assert_factor_close(inc, panel, "momentum_3h_norm")

    def test_cs_factor_volatility_norm(self):
        """volatility_norm = normalize(clip_quantile(volatility, 0.2, 0.8), true, 0)."""
        inc, panel = self._run_both_engines()
        self._assert_factor_close(inc, panel, "volatility_norm")

    def test_cs_factor_corr_norm(self):
        """corr_norm = normalize(clip_quantile(corr, 0.2, 0.8), true, 0)."""
        inc, panel = self._run_both_engines()
        self._assert_factor_close(inc, panel, "corr_norm")

    # -- Composite Factor Tests --

    def test_composite(self):
        """composite = weighted sum of normalized factors."""
        inc, panel = self._run_both_engines()
        self._assert_factor_close(inc, panel, "composite")

    def test_composite_neg(self):
        """composite_neg = negated composite."""
        inc, panel = self._run_both_engines()
        self._assert_factor_close(inc, panel, "composite_neg")

    # -- Cross-engine full snapshot comparison --

    def test_all_factors_present(self):
        """Both engines should produce the same set of factor names."""
        inc, panel = self._run_both_engines()
        assert set(inc.keys()) == set(panel.keys()), (
            f"Factor name mismatch.\n"
            f"  Incremental only: {set(inc) - set(panel)}\n"
            f"  Panel only:       {set(panel) - set(inc)}"
        )

    def test_all_instruments_present(self):
        """Both engines should produce values for the same instruments."""
        inc, panel = self._run_both_engines()
        for fname in inc:
            if fname not in panel:
                continue
            inc_insts = set(inc[fname].keys())
            pan_insts = set(panel[fname].keys())
            assert inc_insts == pan_insts, (
                f"Instrument mismatch for factor '{fname}'.\n"
                f"  Incremental only: {inc_insts - pan_insts}\n"
                f"  Panel only:       {pan_insts - inc_insts}"
            )

    # -- Helper --

    def _assert_factor_close(
        self,
        inc_results: dict[str, dict[str, float]],
        panel_results: dict[str, dict[str, float]],
        factor_name: str,
        rtol: float = 1e-6,
        atol: float = 1e-8,
    ) -> None:
        """Assert that a factor's values are close between both engines."""
        assert factor_name in inc_results, (
            f"Factor '{factor_name}' missing from incremental engine results. "
            f"Available: {list(inc_results.keys())}"
        )
        assert factor_name in panel_results, (
            f"Factor '{factor_name}' missing from panel engine results. "
            f"Available: {list(panel_results.keys())}"
        )

        inc_vals = inc_results[factor_name]
        pan_vals = panel_results[factor_name]

        # Check same instrument set
        inc_insts = set(inc_vals.keys())
        pan_insts = set(pan_vals.keys())

        if inc_insts != pan_insts:
            # Report differences but don't fail here—might be NaN handling
            only_inc = inc_insts - pan_insts
            only_pan = pan_insts - inc_insts
            msg = f"Factor '{factor_name}': instrument set differs.\n"
            if only_inc:
                msg += f"  Incremental only: {sorted(only_inc)} (values: {[inc_vals[i] for i in sorted(only_inc)]})\n"
            if only_pan:
                msg += f"  Panel only:       {sorted(only_pan)} (values: {[pan_vals[i] for i in sorted(only_pan)]})\n"
            pytest.fail(msg)

        # Compare values
        mismatches = []
        for inst in sorted(inc_insts):
            iv = inc_vals[inst]
            pv = pan_vals[inst]

            if math.isnan(iv) and math.isnan(pv):
                continue
            if math.isnan(iv) or math.isnan(pv):
                mismatches.append(
                    f"  {inst}: inc={iv}, panel={pv} (NaN mismatch)"
                )
                continue

            if not np.isclose(iv, pv, rtol=rtol, atol=atol):
                mismatches.append(
                    f"  {inst}: inc={iv:.10f}, panel={pv:.10f}, "
                    f"diff={abs(iv-pv):.2e}"
                )

        if mismatches:
            pytest.fail(
                f"Factor '{factor_name}' values differ between engines:\n"
                + "\n".join(mismatches)
            )


class TestEngineConsistencyMultiTimestamp:
    """Test consistency across MULTIPLE timestamps, not just the last one."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.config = load_factor_config(FACTOR_CONFIG_PATH)
        self.bars = _generate_bars(N_INSTRUMENTS, N_TIMESTAMPS, SEED)

    def test_composite_consistency_over_time(self):
        """Compare composite factor at EVERY timestamp after warmup."""
        inc_engine = FactorEngine(config=self.config, max_history=500)
        inc_engine.set_extra_fields(["quote_volume"])
        cs_engine = CsFactorEngine(config=self.config)
        panel_engine = PanelFactorEngine(config=self.config, max_history=500)
        panel_engine.set_extra_fields(["quote_volume"])

        # Group bars by timestamp
        bars_by_ts: dict[int, list[dict]] = {}
        for b in self.bars:
            bars_by_ts.setdefault(b["ts"], []).append(b)

        sorted_ts = sorted(bars_by_ts.keys())
        warmup = 100  # first 100 timestamps are warmup

        mismatches_per_ts: list[str] = []
        prev_ts = None
        inc_ts_values: dict[str, dict[str, float]] = {}

        for t_idx, ts in enumerate(sorted_ts):
            # -- Incremental: process all bars for this ts --
            inc_ts_values = {}
            for bar_data in bars_by_ts[ts]:
                mock_bar = MockBar(
                    instrument_id=bar_data["instrument_id"],
                    open_=bar_data["open"],
                    high=bar_data["high"],
                    low=bar_data["low"],
                    close=bar_data["close"],
                    volume=bar_data["volume"],
                    quote_volume=bar_data["quote_volume"],
                    ts_event=ts,
                )
                ts_result = inc_engine.on_bar(mock_bar)
                for fname, inst_vals in ts_result.factors.items():
                    if fname not in inc_ts_values:
                        inc_ts_values[fname] = {}
                    inc_ts_values[fname].update(inst_vals)

            inc_cs_values = cs_engine.compute(inc_ts_values)
            inc_full = dict(inc_ts_values)
            inc_full.update(inc_cs_values)

            # -- Panel: process all bars for this ts, then flush --
            for bar_data in bars_by_ts[ts]:
                bar_dict = {
                    "open": bar_data["open"],
                    "high": bar_data["high"],
                    "low": bar_data["low"],
                    "close": bar_data["close"],
                    "volume": bar_data["volume"],
                    "quote_volume": bar_data["quote_volume"],
                }
                panel_engine.on_bar(bar_data["instrument_id"], bar_dict, ts)

            panel_results = panel_engine.flush_and_compute(ts)

            # Skip warmup
            if t_idx < warmup:
                continue

            # Compare composite factor
            for fname in ("composite", "composite_neg"):
                inc_vals = inc_full.get(fname, {})
                pan_vals = panel_results.get(fname, {})

                if not inc_vals and not pan_vals:
                    continue

                common = set(inc_vals.keys()) & set(pan_vals.keys())
                for inst in sorted(common):
                    iv, pv = inc_vals[inst], pan_vals[inst]
                    if math.isnan(iv) and math.isnan(pv):
                        continue
                    if not np.isclose(iv, pv, rtol=1e-4, atol=1e-6):
                        mismatches_per_ts.append(
                            f"ts_idx={t_idx}, {fname}, {inst}: "
                            f"inc={iv:.8f}, panel={pv:.8f}, "
                            f"diff={abs(iv-pv):.2e}"
                        )

                # Check instrument set differences
                only_inc = set(inc_vals.keys()) - set(pan_vals.keys())
                only_pan = set(pan_vals.keys()) - set(inc_vals.keys())
                if only_inc or only_pan:
                    mismatches_per_ts.append(
                        f"ts_idx={t_idx}, {fname}: instrument mismatch — "
                        f"inc_only={sorted(only_inc)}, pan_only={sorted(only_pan)}"
                    )

        if mismatches_per_ts:
            # Show first 30 mismatches for readability
            shown = mismatches_per_ts[:30]
            total = len(mismatches_per_ts)
            detail = "\n".join(shown)
            pytest.fail(
                f"Engine consistency failures across timestamps "
                f"({total} total, showing first {len(shown)}):\n{detail}"
            )
