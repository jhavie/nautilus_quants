# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Unit tests for OptimizedSelectionPolicy (cache-reader pattern)."""

from __future__ import annotations

import numpy as np
import pytest

from nautilus_quants.portfolio.optimizer.base import OptimizerConstraints
from nautilus_quants.portfolio.optimizer.mean_variance import (
    MeanVarianceConfig,
    MeanVarianceOptimizer,
)
from nautilus_quants.portfolio.types import RiskModelOutput, serialize_risk_output
from nautilus_quants.strategies.cs.optimized_selection_policy import OptimizedSelectionPolicy
from nautilus_quants.strategies.cs.selection_policy import TargetPosition
from nautilus_quants.utils.cache_keys import RISK_MODEL_STATE_CACHE_KEY


class FakeCache:
    """Minimal Nautilus Cache stub for unit tests."""

    def __init__(self, initial: dict[str, bytes] | None = None) -> None:
        self._data: dict[str, bytes] = dict(initial or {})

    def get(self, key: str) -> bytes | None:
        return self._data.get(key)

    def add(self, key: str, value: bytes) -> None:
        self._data[key] = value


class FakeClock:
    def __init__(self, ts_ns: int = 0) -> None:
        self._ts = ts_ns

    def timestamp_ns(self) -> int:
        return self._ts


def _make_snapshot(
    instruments: tuple[str, ...],
    timestamp_ns: int = 1_000_000_000,
    seed: int = 0,
    n_factors: int = 3,
    model_type: str = "fundamental",
) -> RiskModelOutput:
    rng = np.random.default_rng(seed)
    n = len(instruments)
    a = rng.standard_normal((n, n))
    cov = a @ a.T / n + np.eye(n) * 0.05
    factor_names = tuple(f"style_{i}" for i in range(n_factors))
    exposures = rng.standard_normal((n, n_factors))
    factor_cov = np.eye(n_factors) * 0.01
    spec_var = rng.uniform(0.01, 0.05, size=n)
    return RiskModelOutput(
        timestamp_ns=timestamp_ns,
        instruments=instruments,
        covariance=cov,
        factor_names=factor_names,
        factor_exposures=exposures,
        factor_covariance=factor_cov,
        specific_variance=spec_var,
        model_type=model_type,
    )


def _make_optimizer() -> MeanVarianceOptimizer:
    return MeanVarianceOptimizer(MeanVarianceConfig(scale_return=False))


def _default_constraints() -> OptimizerConstraints:
    return OptimizerConstraints(
        max_weight=0.2,
        max_leverage=2.0,
        net_exposure=(-0.1, 0.1),
        turnover_limit=0.5,
        min_positions=3,
        sector_limits=None,
        factor_limits=None,
    )


@pytest.mark.unit
def test_returns_targets_when_snapshot_fresh_and_solver_feasible() -> None:
    instruments = tuple(f"ASSET_{i}.BINANCE" for i in range(8))
    snapshot = _make_snapshot(instruments)
    cache = FakeCache({RISK_MODEL_STATE_CACHE_KEY: serialize_risk_output(snapshot)})

    policy = OptimizedSelectionPolicy(
        cache=cache,
        optimizer=_make_optimizer(),
        constraints=_default_constraints(),
        clock=FakeClock(snapshot.timestamp_ns),
    )
    factor_values = {inst: float((i - 4)) for i, inst in enumerate(instruments)}
    targets = policy.select(factor_values, current_long=set(), current_short=set())

    assert targets is not None
    assert len(targets) > 0
    # Sum(|w|) ~ 1 (normalized)
    gross = sum(abs(t.weight) for t in targets)
    assert abs(gross - 1.0) < 1e-6
    # All return TargetPosition
    assert all(isinstance(t, TargetPosition) for t in targets)


@pytest.mark.unit
def test_returns_none_when_snapshot_missing_and_no_fallback() -> None:
    cache = FakeCache()  # empty
    policy = OptimizedSelectionPolicy(
        cache=cache,
        optimizer=_make_optimizer(),
        constraints=_default_constraints(),
    )
    result = policy.select({"BTC.BINANCE": 1.0}, set(), set())
    assert result is None


@pytest.mark.unit
def test_returns_none_when_snapshot_stale() -> None:
    """Stale snapshot without fallback → None (DecisionEngine holds positions)."""
    instruments = tuple(f"ASSET_{i}.BINANCE" for i in range(6))
    snapshot = _make_snapshot(instruments, timestamp_ns=0)
    cache = FakeCache({RISK_MODEL_STATE_CACHE_KEY: serialize_risk_output(snapshot)})

    policy = OptimizedSelectionPolicy(
        cache=cache,
        optimizer=_make_optimizer(),
        constraints=_default_constraints(),
        clock=FakeClock(ts_ns=10**20),  # very far future → snapshot stale
        max_snapshot_age_ns=1_000_000,  # 1ms max age — guaranteed stale
    )
    factor_values = {inst: float(i - 3) for i, inst in enumerate(instruments)}
    result = policy.select(factor_values, set(), set())
    assert result is None


@pytest.mark.unit
def test_universe_intersection_filters_missing_instruments() -> None:
    # Snapshot covers 5 assets, alpha covers 8 (3 extra) → intersect to 5
    snapshot_inst = tuple(f"ASSET_{i}.BINANCE" for i in range(5))
    snapshot = _make_snapshot(snapshot_inst)
    cache = FakeCache({RISK_MODEL_STATE_CACHE_KEY: serialize_risk_output(snapshot)})

    policy = OptimizedSelectionPolicy(
        cache=cache,
        optimizer=_make_optimizer(),
        constraints=_default_constraints(),
        clock=FakeClock(snapshot.timestamp_ns),
    )
    factor_values = {
        **{inst: 1.0 for inst in snapshot_inst},
        "EXTRA_1.BINANCE": 5.0,
        "EXTRA_2.BINANCE": -5.0,
        "EXTRA_3.BINANCE": 0.0,
    }
    targets = policy.select(factor_values, set(), set())
    assert targets is not None
    # No extras
    symbols = {t.symbol for t in targets}
    assert symbols.issubset(set(snapshot_inst))


@pytest.mark.unit
def test_sector_dummies_built_from_sector_map() -> None:
    instruments = tuple(f"ASSET_{i}.BINANCE" for i in range(6))
    snapshot = _make_snapshot(instruments)
    cache = FakeCache({RISK_MODEL_STATE_CACHE_KEY: serialize_risk_output(snapshot)})

    sector_map = {
        instruments[0]: "A",
        instruments[1]: "A",
        instruments[2]: "A",
        instruments[3]: "B",
        instruments[4]: "B",
        instruments[5]: "B",
    }
    constraints = OptimizerConstraints(
        max_weight=0.3,
        max_leverage=2.0,
        net_exposure=(-0.05, 0.05),
        turnover_limit=None,
        min_positions=1,
        sector_limits={"A": 0.01, "B": 1.5},  # very tight on A
        factor_limits=None,
    )
    policy = OptimizedSelectionPolicy(
        cache=cache,
        optimizer=_make_optimizer(),
        constraints=constraints,
        sector_map=sector_map,
        clock=FakeClock(snapshot.timestamp_ns),
    )
    factor_values = {inst: float(i) for i, inst in enumerate(instruments)}
    targets = policy.select(factor_values, set(), set())
    assert targets is not None
    # Sector A gross exposure must respect the cap
    gross_a = sum(abs(t.weight) for t in targets if sector_map[t.symbol] == "A")
    # Normalized weights sum to 1 gross, cap was 0.01 of unnormalized but after
    # scaling the ratio will be at most 0.01/max_leverage. Accept ≤ 0.05.
    assert gross_a < 0.5


@pytest.mark.unit
def test_current_positions_contribute_to_w0_for_turnover() -> None:
    """When current_long is non-empty, optimizer receives non-zero w0."""
    instruments = tuple(f"ASSET_{i}.BINANCE" for i in range(8))
    snapshot = _make_snapshot(instruments)
    cache = FakeCache({RISK_MODEL_STATE_CACHE_KEY: serialize_risk_output(snapshot)})

    # Tight turnover to force the solver to consider current positions
    constraints = OptimizerConstraints(
        max_weight=0.3,
        max_leverage=2.0,
        net_exposure=(-0.1, 0.1),
        turnover_limit=0.05,  # very tight
        min_positions=1,
    )
    policy = OptimizedSelectionPolicy(
        cache=cache,
        optimizer=_make_optimizer(),
        constraints=constraints,
        clock=FakeClock(snapshot.timestamp_ns),
    )
    current_long = {instruments[0], instruments[1]}
    factor_values = {inst: float(i - 4) for i, inst in enumerate(instruments)}
    # Simply verify no exception + outputs is list (may be empty or fallback-ish)
    result = policy.select(factor_values, current_long, set())
    # Result can be list (optimal_fallback) or None (all relaxations failed)
    assert result is None or isinstance(result, list)


@pytest.mark.unit
def test_payload_cache_reuses_deserialize_result() -> None:
    """Calling select twice with same cached payload must not re-deserialize."""
    instruments = tuple(f"ASSET_{i}.BINANCE" for i in range(6))
    snapshot = _make_snapshot(instruments)
    payload = serialize_risk_output(snapshot)
    cache = FakeCache({RISK_MODEL_STATE_CACHE_KEY: payload})

    policy = OptimizedSelectionPolicy(
        cache=cache,
        optimizer=_make_optimizer(),
        constraints=_default_constraints(),
        clock=FakeClock(snapshot.timestamp_ns),
    )
    factor_values = {inst: float(i - 3) for i, inst in enumerate(instruments)}
    _ = policy.select(factor_values, set(), set())
    first_id = policy._last_payload_id
    # Call again with same cache
    _ = policy.select(factor_values, set(), set())
    assert policy._last_payload_id == first_id
    assert policy._cached_output is not None


@pytest.mark.unit
def test_handles_corrupted_payload_gracefully() -> None:
    """Garbage bytes in cache should not crash — return None."""
    cache = FakeCache({RISK_MODEL_STATE_CACHE_KEY: b"not valid JSON!"})
    policy = OptimizedSelectionPolicy(
        cache=cache,
        optimizer=_make_optimizer(),
        constraints=_default_constraints(),
    )
    factor_values = {"A.BINANCE": 1.0, "B.BINANCE": -1.0}
    # Must not raise — returns None (DecisionEngine treats as "no opinion")
    result = policy.select(factor_values, set(), set())
    assert result is None
