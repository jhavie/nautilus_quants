# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Optimizer Protocol, constraints and result types.

Protocol-based pluggable slot: any class with the ``solve`` signature is a
valid Optimizer. No inheritance required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class OptimizerConstraints:
    """Portfolio construction constraints.

    Attributes
    ----------
    max_weight : float
        Per-instrument weight cap: |w_i| ≤ max_weight (e.g. 0.05 = 5%).
    max_leverage : float
        Gross leverage cap: sum(|w|) ≤ max_leverage (e.g. 2.0 = 200% gross).
    net_exposure : tuple[float, float]
        Allowed range for sum(w). (-0.05, 0.05) = near-market-neutral.
    turnover_limit : float | None
        L1 turnover cap: sum(|w - w0|) ≤ turnover_limit. None disables.
    min_positions : int
        Minimum count of non-zero positions for the solver to accept the result.
    sector_limits : dict[str, float] | None
        Per-sector gross exposure cap: sum(|w_i| for i in sector) ≤ cap.
    factor_limits : dict[str, float] | None
        Per-named-factor signed exposure cap: |X_k' w| ≤ cap. Only meaningful
        when the risk model provides factor_exposures with named factors
        (Fundamental model).
    """

    max_weight: float = 0.05
    max_leverage: float = 2.0
    net_exposure: tuple[float, float] = (-0.05, 0.05)
    turnover_limit: float | None = 0.3
    min_positions: int = 10
    sector_limits: dict[str, float] | None = None
    factor_limits: dict[str, float] | None = None


@dataclass(frozen=True)
class OptimizerResult:
    """Optimizer output.

    Attributes
    ----------
    weights : np.ndarray
        (N,) optimal weights. sum(|weights|) normalized by caller before
        publishing TargetPosition (convention: sum(|w|) = 1).
    status : str
        "optimal" | "optimal_fallback" | "infeasible" | "error".
        "optimal_fallback" means some constraints were relaxed.
    objective_value : float | None
        Final objective function value (if solved).
    risk_value : float | None
        w' Σ w at the solution (portfolio variance).
    alpha_value : float | None
        w' α at the solution (expected return).
    turnover : float | None
        L1 turnover sum(|w - w0|) if w0 provided.
    relaxed : tuple[str, ...]
        Names of constraints relaxed during fallback (empty if "optimal").
    """

    weights: np.ndarray
    status: str
    objective_value: float | None = None
    risk_value: float | None = None
    alpha_value: float | None = None
    turnover: float | None = None
    relaxed: tuple[str, ...] = field(default_factory=tuple)


class Optimizer(Protocol):
    """Protocol for portfolio weight optimizers."""

    def solve(
        self,
        alpha: np.ndarray,
        covariance: np.ndarray,
        current_weights: np.ndarray | None,
        constraints: OptimizerConstraints,
        factor_exposures: np.ndarray | None = None,
        sector_dummies: np.ndarray | None = None,
    ) -> OptimizerResult:
        """Solve for optimal portfolio weights.

        Parameters
        ----------
        alpha : np.ndarray
            (N,) expected returns.
        covariance : np.ndarray
            (N, N) covariance matrix.
        current_weights : np.ndarray | None
            (N,) current portfolio weights (for turnover constraint). None = 0.
        constraints : OptimizerConstraints
            Problem constraints.
        factor_exposures : np.ndarray | None
            (N, K) factor exposure matrix for factor-based limits.
        sector_dummies : np.ndarray | None
            (N, S) sector membership indicator matrix for sector limits.

        Returns
        -------
        OptimizerResult
            Solution with status and diagnostics.
        """
        ...
