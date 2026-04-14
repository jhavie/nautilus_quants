# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
OptimizedSelectionPolicy — risk-model-driven portfolio optimization.

Architecture (Cache mode, keeps DecisionEngineActor zero-aware of risk data):

    RiskModelActor ──writes──> Nautilus Cache ──reads──> OptimizedSelectionPolicy
                    (serialized RiskModelOutput)

Unlike TopKDropout / FMZ / WorldQuant policies, this policy delegates weight
determination to an Optimizer + Risk Model pair. It reads the latest
RiskModelOutput from the shared cache and solves a long-short MVO problem.

Fallback chain when risk data is unavailable or solve fails:
    1. risk snapshot missing or stale → fallback_policy
    2. solver infeasible after all relaxations → fallback_policy
    3. fallback_policy is None → return None (warmup)

Placed in strategies/cs/ alongside worldquant_selection_policy.py — this is
a CS-strategy selection plug-in, not a portfolio module concern.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

import numpy as np

from nautilus_quants.portfolio.optimizer.base import Optimizer, OptimizerConstraints
from nautilus_quants.portfolio.types import RiskModelOutput, deserialize_risk_output
from nautilus_quants.strategies.cs.selection_policy import SelectionPolicy, TargetPosition
from nautilus_quants.utils.cache_keys import RISK_MODEL_STATE_CACHE_KEY

logger = logging.getLogger(__name__)


class _CacheReader(Protocol):
    """Minimal cache interface — duck-types Nautilus CacheFacade for testing."""

    def get(self, key: str) -> bytes | None: ...


class OptimizedSelectionPolicy:
    """SelectionPolicy using a risk model (via Cache) + optimizer.

    Parameters
    ----------
    cache : CacheFacade-like
        Nautilus Cache facade. Must support ``cache.get(key: str) -> bytes | None``.
        Passed in by DecisionEngineActor when constructing the policy.
    optimizer : Optimizer
        Long-short optimizer (typically MeanVarianceOptimizer).
    constraints : OptimizerConstraints
        Portfolio construction constraints.
    sector_map : dict[str, str]
        Instrument → sector mapping. Used to build sector_dummies when
        constraints.sector_limits is non-empty. Empty dict disables sector
        constraint regardless of constraints config.
    clock : object with ``timestamp_ns()`` method | None
        Used for staleness check. If None, staleness check is skipped.
    fallback_policy : SelectionPolicy | None
        Policy invoked when risk data is missing/stale or solver infeasible.
        None means return None (signal no-opinion upstream).
    max_snapshot_age_ns : int
        Maximum age (nanoseconds) of cached risk snapshot to accept.
    """

    def __init__(
        self,
        cache: _CacheReader,
        optimizer: Optimizer,
        constraints: OptimizerConstraints,
        sector_map: dict[str, str] | None = None,
        clock: Any = None,
        fallback_policy: SelectionPolicy | None = None,
        max_snapshot_age_ns: int = 24 * 4 * 3_600_000_000_000,  # 24 × 4h
    ) -> None:
        self._cache = cache
        self._optimizer = optimizer
        self._constraints = constraints
        self._sector_map = sector_map or {}
        self._clock = clock
        self._fallback = fallback_policy
        self._max_age_ns = max_snapshot_age_ns

        # Reuse deserialize result when the underlying snapshot's timestamp_ns
        # hasn't advanced. Content-stable (works for both in-memory and
        # DB-backed caches that may return fresh bytes objects each call).
        self._cached_output: RiskModelOutput | None = None

    def select(
        self,
        factor_values: dict[str, float],
        current_long: set[str],
        current_short: set[str],
    ) -> list[TargetPosition] | None:
        now_ns = self._current_ts_ns()
        snapshot = self._read_risk_snapshot(now_ns)
        if snapshot is None:
            logger.debug("No fresh risk snapshot; using fallback policy")
            return self._call_fallback(factor_values, current_long, current_short)

        # Align universe: intersection of alpha keys and snapshot instruments
        universe = sorted(set(factor_values.keys()) & set(snapshot.instruments))
        if len(universe) < max(self._constraints.min_positions, 1):
            logger.debug(
                "Universe %d < min_positions %d; fallback",
                len(universe),
                self._constraints.min_positions,
            )
            return self._call_fallback(factor_values, current_long, current_short)

        alpha_vec, cov_mat, w0 = self._align_inputs(
            factor_values=factor_values,
            snapshot=snapshot,
            current_long=current_long,
            current_short=current_short,
            universe=universe,
        )

        factor_exposures = self._align_factor_exposures(snapshot, universe)
        sector_dummies = self._build_sector_dummies(universe)

        result = self._optimizer.solve(
            alpha=alpha_vec,
            covariance=cov_mat,
            current_weights=w0,
            constraints=self._constraints,
            factor_exposures=factor_exposures,
            sector_dummies=sector_dummies,
        )

        if result.status == "infeasible":
            logger.warning("Optimizer infeasible across all relaxations; fallback")
            return self._call_fallback(factor_values, current_long, current_short)

        targets = self._to_targets(result.weights, factor_values, universe)
        return targets

    def _read_risk_snapshot(self, now_ns: int) -> RiskModelOutput | None:
        """Read and deserialize latest risk snapshot, memoized by timestamp_ns."""
        payload = self._cache.get(RISK_MODEL_STATE_CACHE_KEY)
        if payload is None:
            return None
        try:
            output = deserialize_risk_output(payload)
        except Exception as exc:
            logger.error("Failed to deserialize risk snapshot: %s", exc)
            return None
        # Reuse the prior decoded instance if the model's timestamp is unchanged
        # (keeps downstream structures keyed on `id(output)` stable).
        cached = self._cached_output
        if cached is not None and cached.timestamp_ns == output.timestamp_ns:
            output = cached
        else:
            self._cached_output = output
        if self._max_age_ns > 0 and now_ns > 0:
            age = now_ns - output.timestamp_ns
            if age > self._max_age_ns:
                logger.debug("Risk snapshot stale: age=%dns > max=%dns", age, self._max_age_ns)
                return None
        return output

    def _current_ts_ns(self) -> int:
        if self._clock is None:
            return 0
        try:
            return int(self._clock.timestamp_ns())
        except AttributeError:
            return 0

    def _align_inputs(
        self,
        factor_values: dict[str, float],
        snapshot: RiskModelOutput,
        current_long: set[str],
        current_short: set[str],
        universe: list[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build aligned (alpha, covariance, w0) numpy arrays.

        Covariance is sliced from snapshot.covariance using index positions of
        the intersected universe.
        """
        alpha = np.array([float(factor_values[inst]) for inst in universe], dtype=np.float64)

        # Build index of snapshot instruments for O(1) lookup
        idx_map = {inst: i for i, inst in enumerate(snapshot.instruments)}
        rows = np.array([idx_map[inst] for inst in universe], dtype=np.intp)
        cov_mat = snapshot.covariance[np.ix_(rows, rows)].copy()

        # Current weights: equal-weight assumption for current positions,
        # sign by leg. Rationale: SelectionPolicy does not receive actual
        # weights, only membership sets. For turnover constraint this gives
        # a reasonable proxy.
        n_long = max(len(current_long), 1)
        n_short = max(len(current_short), 1)
        w0 = np.zeros(len(universe), dtype=np.float64)
        for i, inst in enumerate(universe):
            if inst in current_long:
                w0[i] = 1.0 / n_long
            elif inst in current_short:
                w0[i] = -1.0 / n_short
        return alpha, cov_mat, w0

    def _align_factor_exposures(
        self,
        snapshot: RiskModelOutput,
        universe: list[str],
    ) -> np.ndarray | None:
        """Slice snapshot.factor_exposures to universe rows, or None."""
        if snapshot.factor_exposures is None:
            return None
        idx_map = {inst: i for i, inst in enumerate(snapshot.instruments)}
        rows = np.array([idx_map[inst] for inst in universe], dtype=np.intp)
        return snapshot.factor_exposures[rows, :].copy()

    def _build_sector_dummies(self, universe: list[str]) -> np.ndarray | None:
        """Build (N, S) one-hot sector matrix aligned to constraints.sector_limits."""
        limits = self._constraints.sector_limits
        if not limits or not self._sector_map:
            return None
        sector_names = list(limits.keys())
        s_to_idx = {s: i for i, s in enumerate(sector_names)}
        n = len(universe)
        dummies = np.zeros((n, len(sector_names)), dtype=np.float64)
        for i, inst in enumerate(universe):
            sector = self._sector_map.get(inst)
            if sector is None or sector not in s_to_idx:
                continue
            dummies[i, s_to_idx[sector]] = 1.0
        return dummies

    def _to_targets(
        self,
        weights: np.ndarray,
        factor_values: dict[str, float],
        universe: list[str],
    ) -> list[TargetPosition]:
        """Normalize to sum(|w|)=1 and emit TargetPosition list.

        Matches WorldQuantSelectionPolicy output convention.
        """
        gross = float(np.sum(np.abs(weights)))
        if gross < 1e-12:
            return []
        normalized = weights / gross
        targets: list[TargetPosition] = []
        for inst, w in zip(universe, normalized):
            if abs(w) < 1e-10:
                continue
            targets.append(
                TargetPosition(
                    symbol=inst,
                    weight=float(w),
                    factor=float(factor_values.get(inst, 0.0)),
                )
            )
        # Sort by factor (matching other policies' convention)
        targets.sort(key=lambda t: t.factor)
        return targets

    def _call_fallback(
        self,
        factor_values: dict[str, float],
        current_long: set[str],
        current_short: set[str],
    ) -> list[TargetPosition] | None:
        if self._fallback is None:
            return None
        return self._fallback.select(factor_values, current_long, current_short)
