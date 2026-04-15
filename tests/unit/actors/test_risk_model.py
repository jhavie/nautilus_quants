# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for RiskModelActor core pipeline (config + embedded FactorEngine).

These tests exercise the pure/stateful helpers without spinning up a Nautilus
Actor (matching the test pattern used in test_snapshot_aggregator.py).
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest

from nautilus_quants.actors.risk_model import RiskModelActor, RiskModelActorConfig
from nautilus_quants.factors.engine.factor_engine import FactorEngine
from nautilus_quants.portfolio.config import load_portfolio_config


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_config_defaults() -> None:
    cfg = RiskModelActorConfig(portfolio_config_path="config/portfolio/portfolio.yaml")
    assert cfg.update_interval_bars == 6
    assert cfg.bar_types == []
    assert cfg.extra_data_path == ""
    assert cfg.max_history == 500
    assert cfg.flush_timeout_secs == 5


# ---------------------------------------------------------------------------
# portfolio.yaml variables registration round-trip
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_portfolio_variables_register_as_expression_factors() -> None:
    """Portfolio.yaml risk variables must register on an embedded FactorEngine
    and expose their latest values through flush_and_compute."""
    cfg = load_portfolio_config("config/portfolio/portfolio.yaml")
    risk_vars = cfg.risk_model.fundamental.variables
    assert len(risk_vars) > 0, "portfolio.yaml must declare risk variables"

    engine = FactorEngine(max_history=64)
    for var in risk_vars:
        engine.register_expression_factor(
            name=var.name,
            expression=var.expression,
            description=var.description,
        )
    registered = set(engine.factor_names)
    expected = {v.name for v in risk_vars}
    assert expected.issubset(registered)


# ---------------------------------------------------------------------------
# Buffer helpers — bind methods to a lightweight fake without Actor.__init__
# ---------------------------------------------------------------------------


class _FakeActor:
    """Minimal host exposing RiskModelActor's stateful helpers for unit tests."""

    def __init__(self, portfolio_cfg) -> None:
        self._portfolio_config = portfolio_cfg
        self._returns_buffer: OrderedDict[int, dict[str, float]] = OrderedDict()
        self._exposures_buffer: OrderedDict[int, dict[str, dict[str, float]]] = OrderedDict()
        self._market_cap_latest: dict[str, float] = {}

    # Bind real methods for direct invocation
    _buffer_returns = RiskModelActor._buffer_returns
    _buffer_exposures = RiskModelActor._buffer_exposures
    _update_market_cap_latest = RiskModelActor._update_market_cap_latest
    _trim_buffers = RiskModelActor._trim_buffers
    _assemble_returns_frame = RiskModelActor._assemble_returns_frame
    _assemble_exposures_frame = RiskModelActor._assemble_exposures_frame
    _build_wls_weights = RiskModelActor._build_wls_weights


def _load_cfg():
    return load_portfolio_config("config/portfolio/portfolio.yaml")


@pytest.mark.unit
def test_buffer_returns_populates_from_risk_values() -> None:
    """_buffer_returns extracts the 'returns' key when produced by the engine."""
    actor = _FakeActor(_load_cfg())
    risk_values = {
        "returns": {"BTC.BINANCE": 0.01, "ETH.BINANCE": -0.02},
        "btc_beta": {"BTC.BINANCE": 1.0, "ETH.BINANCE": 0.8},
    }
    actor._buffer_returns(1_000, risk_values)
    assert actor._returns_buffer[1_000] == {"BTC.BINANCE": 0.01, "ETH.BINANCE": -0.02}


@pytest.mark.unit
def test_buffer_returns_skips_when_empty() -> None:
    actor = _FakeActor(_load_cfg())
    actor._buffer_returns(1_000, {"btc_beta": {"BTC.BINANCE": 1.0}})
    assert 1_000 not in actor._returns_buffer


@pytest.mark.unit
def test_buffer_exposures_maps_factor_names_to_variable_values() -> None:
    actor = _FakeActor(_load_cfg())
    # Fundamental factors: size → log_market_cap, btc_beta → btc_beta, ...
    risk_values = {
        "log_market_cap": {"BTC.BINANCE": 11.0, "ETH.BINANCE": 10.5},
        "btc_beta": {"BTC.BINANCE": 1.0, "ETH.BINANCE": 0.9},
    }
    actor._buffer_exposures(2_000, risk_values)
    bucket = actor._exposures_buffer[2_000]
    assert bucket["size"]["BTC.BINANCE"] == pytest.approx(11.0)
    assert bucket["btc_beta"]["ETH.BINANCE"] == pytest.approx(0.9)


@pytest.mark.unit
def test_update_market_cap_latest_stores_positive_finite_values() -> None:
    actor = _FakeActor(_load_cfg())
    actor._update_market_cap_latest(
        {"market_cap": {"BTC.BINANCE": 5e9, "ETH.BINANCE": 2e9, "BAD.BINANCE": float("nan")}}
    )
    assert actor._market_cap_latest["BTC.BINANCE"] == pytest.approx(5e9)
    assert actor._market_cap_latest["ETH.BINANCE"] == pytest.approx(2e9)
    assert "BAD.BINANCE" not in actor._market_cap_latest


@pytest.mark.unit
def test_trim_buffers_caps_at_lookback_bars() -> None:
    actor = _FakeActor(_load_cfg())
    lookback = actor._portfolio_config.risk_model.statistical.lookback_bars
    for i in range(lookback + 10):
        actor._returns_buffer[i] = {"BTC.BINANCE": 0.0}
        actor._exposures_buffer[i] = {"size": {"BTC.BINANCE": 1.0}}
    actor._trim_buffers()
    assert len(actor._returns_buffer) == lookback
    assert len(actor._exposures_buffer) == lookback
    # Oldest entries dropped; newest retained
    assert (lookback + 9) in actor._returns_buffer
    assert 0 not in actor._returns_buffer


@pytest.mark.unit
def test_assemble_returns_frame_drops_sparse_columns() -> None:
    actor = _FakeActor(_load_cfg())
    # BTC fully observed, ETH only sporadic — should be dropped (0.5 threshold)
    for i in range(10):
        actor._returns_buffer[i] = {"BTC.BINANCE": 0.01}
        if i < 3:
            actor._returns_buffer[i]["ETH.BINANCE"] = 0.02
    df = actor._assemble_returns_frame()
    assert df is not None
    assert "BTC.BINANCE" in df.columns
    assert "ETH.BINANCE" not in df.columns


@pytest.mark.unit
def test_build_wls_weights_none_when_market_cap_empty() -> None:
    actor = _FakeActor(_load_cfg())
    assert actor._build_wls_weights(["BTC.BINANCE"]) is None


@pytest.mark.unit
def test_build_wls_weights_returns_sqrt_mcap() -> None:
    actor = _FakeActor(_load_cfg())
    actor._market_cap_latest = {"BTC.BINANCE": 100.0, "ETH.BINANCE": 25.0}
    w = actor._build_wls_weights(["BTC.BINANCE", "ETH.BINANCE", "MISSING.BINANCE"])
    assert w is not None
    np.testing.assert_allclose(w, np.array([10.0, 5.0, 0.0]))


@pytest.mark.unit
def test_assemble_exposures_frame_multiindex_shape() -> None:
    actor = _FakeActor(_load_cfg())
    # Populate returns (needed for column alignment) and exposures
    timestamps = [10, 20, 30]
    instruments = ["BTC.BINANCE", "ETH.BINANCE"]
    for ts in timestamps:
        actor._returns_buffer[ts] = {inst: 0.01 for inst in instruments}
        actor._exposures_buffer[ts] = {
            f.name: {inst: 0.5 for inst in instruments}
            for f in actor._portfolio_config.risk_model.fundamental.factors
        }
    returns_df = actor._assemble_returns_frame()
    assert returns_df is not None
    df = actor._assemble_exposures_frame(returns_df)
    assert df is not None
    # Rows = T × N; cols = K named factors
    n_factors = len(actor._portfolio_config.risk_model.fundamental.factors)
    assert df.shape == (len(timestamps) * len(instruments), n_factors)
    assert df.index.names == ["ts", "inst"]
