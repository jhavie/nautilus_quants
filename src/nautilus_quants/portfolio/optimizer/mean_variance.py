# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
MeanVarianceOptimizer — long-short mean-variance optimization via cvxpy.

Problem formulation (not benchmark-tracking, not long-only):

    maximize    w' α  -  λ · w' Σ w
    subject to  net_lo ≤ sum(w) ≤ net_hi           (near-market-neutral)
                |w_i| ≤ max_weight                  (single position cap)
                sum(|w|) ≤ max_leverage             (gross leverage cap)
                sum(|w - w0|) ≤ turnover_limit      (optional turnover)
                Σ_i∈sector |w_i| ≤ sector_cap       (optional per-sector)
                |X_k' w| ≤ factor_cap               (optional per-named-factor)

Fallback strategy (if infeasible):
    1. Drop turnover constraint → retry
    2. Drop sector constraints → retry
    3. Drop factor constraints → retry
    4. Return status="infeasible" with zero weights

Solver: ECOS via cvxpy (lightweight, no extra dependencies beyond cvxpy).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from nautilus_quants.portfolio.optimizer.base import (
    Optimizer,
    OptimizerConstraints,
    OptimizerResult,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MeanVarianceConfig:
    """MeanVarianceOptimizer configuration.

    Attributes
    ----------
    risk_aversion : float
        λ coefficient on risk term. Larger = more risk-averse.
    solver : str
        cvxpy solver name. "ECOS" is fast and reliable for this problem shape.
    epsilon : float
        Weights below this magnitude are zeroed after solve (numerical cleanup).
    scale_return : bool
        Rescale alpha to match volatility of Σ (Qlib convention for stability).
    fallback_relax : tuple[str, ...]
        Order of constraint relaxation when infeasible.
    """

    risk_aversion: float = 1.0
    solver: str = "CLARABEL"  # cvxpy 1.4+ default; ECOS/SCS also supported
    epsilon: float = 5e-5
    scale_return: bool = True
    fallback_relax: tuple[str, ...] = ("turnover", "sector", "factor")


class MeanVarianceOptimizer(Optimizer):
    """Long-short MVO optimizer (cvxpy)."""

    def __init__(self, config: MeanVarianceConfig) -> None:
        self._config = config

    def solve(
        self,
        alpha: np.ndarray,
        covariance: np.ndarray,
        current_weights: np.ndarray | None,
        constraints: OptimizerConstraints,
        factor_exposures: np.ndarray | None = None,
        sector_dummies: np.ndarray | None = None,
    ) -> OptimizerResult:
        cfg = self._config
        n = len(alpha)
        if covariance.shape != (n, n):
            raise ValueError(f"covariance shape {covariance.shape} != ({n}, {n})")

        if current_weights is None:
            w0 = np.zeros(n, dtype=np.float64)
        else:
            w0 = np.asarray(current_weights, dtype=np.float64)
            if w0.shape != (n,):
                raise ValueError(f"current_weights shape {w0.shape} != ({n},)")

        alpha = np.asarray(alpha, dtype=np.float64)
        if cfg.scale_return and np.std(alpha) > 1e-12:
            # Rescale alpha to match portfolio-level volatility (Qlib convention)
            target_vol = float(np.sqrt(max(np.mean(np.diag(covariance)), 1e-12)))
            alpha = alpha / np.std(alpha) * target_vol

        # Relaxation attempts in order: full → drop N constraints progressively
        relax_stages: list[tuple[str, ...]] = [()]
        for step in cfg.fallback_relax:
            relax_stages.append(relax_stages[-1] + (step,))

        last_error: Exception | None = None
        for relax in relax_stages:
            try:
                weights, obj, risk_val, alpha_val = self._solve_cvxpy(
                    alpha=alpha,
                    covariance=covariance,
                    w0=w0,
                    constraints=constraints,
                    factor_exposures=factor_exposures,
                    sector_dummies=sector_dummies,
                    skip=relax,
                )
            except (cp.error.SolverError, ValueError) as exc:
                last_error = exc
                logger.debug("MVO solve failed at relax=%s: %s", relax, exc)
                continue
            if weights is None:
                continue

            # Numerical cleanup
            weights = np.where(np.abs(weights) < cfg.epsilon, 0.0, weights)
            status = "optimal" if not relax else "optimal_fallback"
            turnover = float(np.sum(np.abs(weights - w0)))
            return OptimizerResult(
                weights=weights,
                status=status,
                objective_value=obj,
                risk_value=risk_val,
                alpha_value=alpha_val,
                turnover=turnover,
                relaxed=relax,
            )

        logger.warning("MVO infeasible across all relaxation stages: %s", last_error)
        return OptimizerResult(
            weights=np.zeros(n, dtype=np.float64),
            status="infeasible",
        )

    # ------------------------------------------------------------------
    # Internal: single cvxpy solve
    # ------------------------------------------------------------------

    def _solve_cvxpy(
        self,
        alpha: np.ndarray,
        covariance: np.ndarray,
        w0: np.ndarray,
        constraints: OptimizerConstraints,
        factor_exposures: np.ndarray | None,
        sector_dummies: np.ndarray | None,
        skip: tuple[str, ...],
    ) -> tuple[np.ndarray | None, float | None, float | None, float | None]:
        cfg = self._config
        n = len(alpha)
        w = cp.Variable(n)

        cons: list[cp.Constraint] = []
        # Per-instrument absolute weight cap (|w_i| ≤ max_weight)
        cons.append(cp.abs(w) <= constraints.max_weight)
        # Gross leverage: sum(|w|) ≤ max_leverage (equivalent to ||w||_1)
        cons.append(cp.norm1(w) <= constraints.max_leverage)
        # Net exposure band
        net_lo, net_hi = constraints.net_exposure
        cons.append(cp.sum(w) >= net_lo)
        cons.append(cp.sum(w) <= net_hi)

        # Turnover cap
        if "turnover" not in skip and constraints.turnover_limit is not None:
            cons.append(cp.norm1(w - w0) <= constraints.turnover_limit)

        # Per-sector gross exposure cap
        if "sector" not in skip and sector_dummies is not None and constraints.sector_limits:
            # sector_dummies shape: (N, S) one-hot per sector
            sector_names = list(constraints.sector_limits.keys())
            s = sector_dummies.shape[1]
            if s != len(sector_names):
                raise ValueError(
                    f"sector_dummies cols ({s}) != len(sector_limits) ({len(sector_names)})"
                )
            for idx, name in enumerate(sector_names):
                limit = constraints.sector_limits[name]
                members = sector_dummies[:, idx]
                cons.append(cp.norm1(cp.multiply(members, w)) <= limit)

        # Per-named-factor signed exposure cap
        if "factor" not in skip and factor_exposures is not None and constraints.factor_limits:
            # factor_exposures shape: (N, K)
            factor_names = list(constraints.factor_limits.keys())
            if factor_exposures.shape[1] < len(factor_names):
                raise ValueError(
                    f"factor_exposures has {factor_exposures.shape[1]} cols, "
                    f"expected ≥ {len(factor_names)} for factor_limits"
                )
            for k, name in enumerate(factor_names):
                limit = constraints.factor_limits[name]
                exposure_vec = factor_exposures[:, k]
                cons.append(cp.abs(exposure_vec @ w) <= limit)

        # Objective: max α'w − λ w'Σw
        risk_term = cp.quad_form(w, cp.psd_wrap(covariance))
        objective = cp.Maximize(alpha @ w - cfg.risk_aversion * risk_term)

        prob = cp.Problem(objective, cons)
        prob.solve(solver=cfg.solver, warm_start=True)
        if prob.status not in ("optimal", "optimal_inaccurate"):
            return None, None, None, None

        w_val = np.asarray(w.value, dtype=np.float64)
        risk_val = float(w_val @ covariance @ w_val)
        alpha_val = float(alpha @ w_val)
        return w_val, float(prob.value), risk_val, alpha_val
