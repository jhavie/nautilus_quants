# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
StatisticalRiskModel — PCA / Factor Analysis / Shrinkage covariance.

Ported from Qlib:
- qlib/model/riskmodel/structured.py (PCA/FA decomposition)
- qlib/model/riskmodel/shrink.py (Ledoit-Wolf + OAS shrinkage)

Produces RiskModelOutput with either:
- Decomposed form (F, cov_b, var_u) when method in {"pca", "fa"}
- Full covariance only when method == "shrink"

Factor names are synthetic ("PC_0", "PC_1", ...) — not human-interpretable.
For interpretable factor exposure monitoring, use FundamentalRiskModel instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nautilus_quants.portfolio.risk_model.base import RiskModel, StatisticalModelConfig
from nautilus_quants.portfolio.risk_model.preprocessing import preprocess_returns
from nautilus_quants.portfolio.types import RiskModelOutput

METHOD_PCA = "pca"
METHOD_FA = "fa"
METHOD_SHRINK = "shrink"

SHRINK_LW = "lw"
SHRINK_OAS = "oas"

TARGET_CONST_VAR = "const_var"
TARGET_CONST_CORR = "const_corr"
TARGET_SINGLE_FACTOR = "single_factor"


class StatisticalRiskModel(RiskModel):
    """PCA / FA / Shrinkage covariance estimator (Qlib-compatible).

    Parameters
    ----------
    config : StatisticalModelConfig
        See base.StatisticalModelConfig for all fields.

    Notes
    -----
    PCA and FA use ``sklearn.decomposition``. Shrinkage uses numpy-only
    closed-form formulas (no sklearn dependency on shrink path).
    """

    def __init__(self, config: StatisticalModelConfig) -> None:
        self._config = config

    def fit(
        self,
        returns: pd.DataFrame,
        timestamp_ns: int,
        exposures: pd.DataFrame | None = None,  # noqa: ARG002 — ignored for statistical
        weights: np.ndarray | None = None,  # noqa: ARG002
    ) -> RiskModelOutput:
        cfg = self._config
        if returns.shape[0] < cfg.min_history_bars:
            raise ValueError(f"need at least {cfg.min_history_bars} rows, got {returns.shape[0]}")

        matrix, instruments = preprocess_returns(
            returns,
            winsorize_quantile=cfg.winsorize_quantile,
            nan_option=cfg.nan_option,
            scale_pct=cfg.scale_return_pct,
            assume_centered=cfg.assume_centered,
        )
        # preprocess_returns may return MaskedArray; sklearn needs ndarray.
        # For mask mode we fall back to ndarray (NaN already handled as fill).
        x = np.asarray(matrix, dtype=np.float64)

        if cfg.method in (METHOD_PCA, METHOD_FA):
            return self._fit_factor_decomposition(x, instruments, timestamp_ns)
        if cfg.method == METHOD_SHRINK:
            return self._fit_shrinkage(x, instruments, timestamp_ns)
        raise ValueError(f"unknown method: {cfg.method!r}")

    @property
    def min_history(self) -> int:
        return self._config.min_history_bars

    @property
    def model_type(self) -> str:
        return "statistical"

    # ------------------------------------------------------------------
    # PCA / FA decomposition
    # ------------------------------------------------------------------

    def _fit_factor_decomposition(
        self,
        x: np.ndarray,
        instruments: tuple[str, ...],
        timestamp_ns: int,
    ) -> RiskModelOutput:
        """PCA or FA: X = B @ F.T + U → (F, cov(B), var(U)).

        Matches Qlib StructuredCovEstimator._predict.
        """
        from sklearn.decomposition import PCA, FactorAnalysis  # local import to keep

        cfg = self._config
        k = min(cfg.num_factors, max(1, min(x.shape) - 1))

        solver_cls = PCA if cfg.method == METHOD_PCA else FactorAnalysis
        model = solver_cls(n_components=k, random_state=0).fit(x)

        factor_exposures = np.asarray(model.components_, dtype=np.float64).T  # (N, K)
        scores = np.asarray(model.transform(x), dtype=np.float64)  # (T, K)
        residuals = x - scores @ factor_exposures.T

        factor_cov = np.cov(scores, rowvar=False)
        if factor_cov.ndim == 0:
            # Single factor edge case: np.cov returns scalar
            factor_cov = np.array([[float(factor_cov)]], dtype=np.float64)
        var_u = np.var(residuals, axis=0, dtype=np.float64)

        covariance = factor_exposures @ factor_cov @ factor_exposures.T + np.diag(var_u)
        covariance = _ensure_symmetric(covariance)

        prefix = "PC" if cfg.method == METHOD_PCA else "FA"
        factor_names = tuple(f"{prefix}_{i}" for i in range(k))

        return RiskModelOutput(
            timestamp_ns=timestamp_ns,
            instruments=instruments,
            covariance=covariance,
            factor_names=factor_names,
            factor_exposures=factor_exposures,
            factor_covariance=factor_cov,
            specific_variance=var_u,
            model_type="statistical",
            factor_returns_history=scores,  # (T, K)
            specific_returns_history=residuals,  # (T, N)
            instrument_returns=x,  # (T, N)
        )

    # ------------------------------------------------------------------
    # Shrinkage
    # ------------------------------------------------------------------

    def _fit_shrinkage(
        self,
        x: np.ndarray,
        instruments: tuple[str, ...],
        timestamp_ns: int,
    ) -> RiskModelOutput:
        """Shrinkage covariance: S_hat = (1 - α) * S + α * F.

        Matches Qlib ShrinkCovEstimator._predict.
        """
        cfg = self._config
        # Empirical covariance (centered via preprocessing); normalize by T not T-1.
        s = (x.T @ x) / x.shape[0]

        target = self._build_shrink_target(x, s, cfg.shrink_target)
        alpha = self._solve_shrink_alpha(x, s, target, cfg.shrinkage, cfg.shrink_target)

        if alpha > 0.0:
            s = (1.0 - alpha) * s + alpha * target
        covariance = _ensure_symmetric(s)

        return RiskModelOutput(
            timestamp_ns=timestamp_ns,
            instruments=instruments,
            covariance=covariance,
            factor_names=None,
            factor_exposures=None,
            factor_covariance=None,
            specific_variance=np.diag(covariance).copy(),
            model_type="statistical",
            instrument_returns=x,  # (T, N) for VaR
        )

    @staticmethod
    def _build_shrink_target(x: np.ndarray, s: np.ndarray, target: str) -> np.ndarray:
        if target == TARGET_CONST_VAR:
            n = s.shape[0]
            f = np.eye(n)
            np.fill_diagonal(f, float(np.mean(np.diag(s))))
            return f
        if target == TARGET_CONST_CORR:
            var = np.diag(s)
            sqrt_var = np.sqrt(np.maximum(var, 1e-12))
            covar = np.outer(sqrt_var, sqrt_var)
            n = s.shape[0]
            if n < 2:
                return s.copy()
            r_bar = (np.sum(s / covar) - n) / (n * (n - 1))
            f = r_bar * covar
            np.fill_diagonal(f, var)
            return f
        if target == TARGET_SINGLE_FACTOR:
            mkt = np.nanmean(x, axis=1)
            t = len(x)
            cov_mkt = (x.T @ mkt) / t
            var_mkt = float(mkt @ mkt) / t
            if var_mkt < 1e-12:
                return s.copy()
            f = np.outer(cov_mkt, cov_mkt) / var_mkt
            np.fill_diagonal(f, np.diag(s))
            return f
        raise ValueError(f"unknown shrink_target: {target!r}")

    @staticmethod
    def _solve_shrink_alpha(
        x: np.ndarray,
        s: np.ndarray,
        f: np.ndarray,
        shrinkage: str | float | None,
        target: str,
    ) -> float:
        if shrinkage is None:
            return 0.0
        if isinstance(shrinkage, (int, float)) and not isinstance(shrinkage, bool):
            alpha = float(shrinkage)
            return max(0.0, min(1.0, alpha))
        if shrinkage == SHRINK_OAS:
            return _shrink_alpha_oas(x, s)
        if shrinkage == SHRINK_LW:
            if target == TARGET_CONST_VAR:
                return _shrink_alpha_lw_const_var(x, s, f)
            if target == TARGET_CONST_CORR:
                return _shrink_alpha_lw_const_corr(x, s)
            if target == TARGET_SINGLE_FACTOR:
                return _shrink_alpha_lw_single_factor(x, s)
        raise ValueError(f"unsupported shrinkage={shrinkage!r} with target={target!r}")


def _ensure_symmetric(m: np.ndarray) -> np.ndarray:
    """Symmetrize to counteract tiny numerical asymmetry."""
    return 0.5 * (m + m.T)


def _shrink_alpha_oas(x: np.ndarray, s: np.ndarray) -> float:
    """Oracle Approximating Shrinkage (OAS) parameter for const_var target only.

    Matches Qlib ShrinkCovEstimator._get_shrink_param_oas.
    """
    tr_s2 = float(np.sum(s**2))
    tr2_s = float(np.trace(s)) ** 2
    n, p = x.shape
    if p <= 2 or (n + 1 - 2 / p) == 0:
        return 0.0
    a = (1.0 - 2.0 / p) * (tr_s2 + tr2_s)
    b = (n + 1 - 2.0 / p) * (tr_s2 + tr2_s / p)
    if b == 0:
        return 0.0
    return max(0.0, min(1.0, a / b))


def _shrink_alpha_lw_const_var(x: np.ndarray, s: np.ndarray, f: np.ndarray) -> float:
    """Ledoit-Wolf α for const_var target."""
    t = x.shape[0]
    y = x**2
    phi = float(np.sum(y.T @ y / t - s**2))
    gamma = float(np.linalg.norm(s - f, "fro") ** 2)
    if gamma <= 0:
        return 0.0
    kappa = phi / gamma
    return max(0.0, min(1.0, kappa / t))


def _shrink_alpha_lw_const_corr(x: np.ndarray, s: np.ndarray) -> float:
    """Ledoit-Wolf α for const_corr target (matches Qlib implementation)."""
    t, n = x.shape
    if n < 2:
        return 0.0
    var = np.diag(s)
    sqrt_var = np.sqrt(np.maximum(var, 1e-12))
    denom = np.outer(sqrt_var, sqrt_var)
    r_bar = (np.sum(s / denom) - n) / (n * (n - 1))

    y = x**2
    phi_mat = y.T @ y / t - s**2
    phi = float(np.sum(phi_mat))

    theta_mat = (x**3).T @ x / t - var[:, None] * s
    np.fill_diagonal(theta_mat, 0.0)
    rho = float(np.sum(np.diag(phi_mat))) + r_bar * float(
        np.sum(np.outer(1.0 / sqrt_var, sqrt_var) * theta_mat)
    )

    # Re-build F to compute gamma (norm-squared distance)
    f = r_bar * denom
    np.fill_diagonal(f, var)
    gamma = float(np.linalg.norm(s - f, "fro") ** 2)
    if gamma <= 0:
        return 0.0
    kappa = (phi - rho) / gamma
    return max(0.0, min(1.0, kappa / t))


def _shrink_alpha_lw_single_factor(x: np.ndarray, s: np.ndarray) -> float:
    """Ledoit-Wolf α for single_factor target."""
    t = x.shape[0]
    mkt = np.nanmean(x, axis=1)
    cov_mkt = (x.T @ mkt) / t
    var_mkt = float(mkt @ mkt) / t
    if var_mkt < 1e-12:
        return 0.0

    y = x**2
    phi = float(np.sum(y.T @ y)) / t - float(np.sum(s**2))

    rdiag = float(np.sum(y**2)) / t - float(np.sum(np.diag(s) ** 2))
    z = x * mkt[:, None]
    v1 = y.T @ z / t - cov_mkt[:, None] * s
    roff1 = (
        float(np.sum(v1 * cov_mkt[:, None].T)) / var_mkt
        - float(np.sum(np.diag(v1) * cov_mkt)) / var_mkt
    )
    v3 = z.T @ z / t - var_mkt * s
    roff3 = (
        float(np.sum(v3 * np.outer(cov_mkt, cov_mkt))) / var_mkt**2
        - float(np.sum(np.diag(v3) * cov_mkt**2)) / var_mkt**2
    )
    roff = 2.0 * roff1 - roff3
    rho = rdiag + roff

    f = np.outer(cov_mkt, cov_mkt) / var_mkt
    np.fill_diagonal(f, np.diag(s))
    gamma = float(np.linalg.norm(s - f, "fro") ** 2)
    if gamma <= 0:
        return 0.0
    kappa = (phi - rho) / gamma
    return max(0.0, min(1.0, kappa / t))
