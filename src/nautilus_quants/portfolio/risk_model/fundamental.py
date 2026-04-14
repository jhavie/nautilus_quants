# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
FundamentalRiskModel — Barra-style weighted least-squares (WLS) cross-sectional
regression risk model.

Model:
    r_t = X_{t-1} f_t + u_t      per period t
where
    r_t : (N,) instrument returns
    X_{t-1} : (N, K) factor exposures at period t-1 (Market + Styles + Sectors)
    f_t : (K,) factor returns
    u_t : (N,) idiosyncratic residuals

Estimation (per period):
    f_t = (X' W X)^{-1} X' W r_t
WLS weights W = diag(wls_weights_i), typically sqrt(market_cap).

Outputs:
    cov_b = cov(f_t)              (K, K)  factor covariance
    var_u = var(u_{·,i})           (N,)   specific (idiosyncratic) variance
    covariance = X_T cov_b X_T' + diag(var_u)   full (N, N)

Notes
-----
- **Named factors are interpretable.** Unlike Statistical PCA, factor_names
  here are things like "btc_beta", "size", "sector_L1" — downstream exposure
  monitoring is meaningful.
- Exposures fed via ``exposures`` parameter must be panel-shaped
  (MultiIndex (timestamp, instrument), columns = factor names). For the
  *latest* F, we slice the last timestamp; for historical WLS, we run the
  regression for every available period.
- Sector dummies are appended from ``config.sector_map``. Missing instruments
  default to the ``"Other"`` sector.
- All inputs are aligned to the returns universe (same instruments/timestamps).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nautilus_quants.portfolio.risk_model.base import FundamentalModelConfig, RiskModel
from nautilus_quants.portfolio.risk_model.preprocessing import (
    center_columns,
    handle_nan,
    winsorize_returns,
)
from nautilus_quants.portfolio.types import RiskModelOutput


class FundamentalRiskModel(RiskModel):
    """Barra-style WLS cross-sectional regression risk model.

    Parameters
    ----------
    config : FundamentalModelConfig
        See base.FundamentalModelConfig for all fields.
    """

    def __init__(self, config: FundamentalModelConfig) -> None:
        self._config = config

    def fit(
        self,
        returns: pd.DataFrame,
        timestamp_ns: int,
        exposures: pd.DataFrame | None = None,
        weights: np.ndarray | None = None,
    ) -> RiskModelOutput:
        cfg = self._config
        if returns.shape[0] < cfg.min_history_bars:
            raise ValueError(f"need at least {cfg.min_history_bars} rows, got {returns.shape[0]}")
        if exposures is None:
            raise ValueError("FundamentalRiskModel requires factor exposures panel")

        instruments = tuple(str(c) for c in returns.columns)

        # Winsorize returns then align to same index (winsorize returns pandas DataFrame)
        winsorized = winsorize_returns(
            returns,
            lower_quantile=cfg.winsorize_quantile,
            upper_quantile=1.0 - cfg.winsorize_quantile,
        )
        scale = 100.0 if cfg.scale_return_pct else 1.0
        r_panel = winsorized * scale

        # Build per-period (T, N, K) exposures tensor. Style factors come from
        # `exposures` (MultiIndex (timestamp, instrument) → K style cols), sector
        # dummies are appended. We winsorize style exposures cross-sectionally.
        style_tensor, style_names = _assemble_style_tensor(
            exposures,
            index=r_panel.index,
            instruments=instruments,
            style_names=tuple(f.name for f in cfg.factors),
            winsorize_sigma=cfg.winsorize_exposures_sigma,
        )
        sector_tensor, sector_names = _assemble_sector_dummies(
            instruments=instruments,
            sector_map=cfg.sector_map,
        )  # (N, S)
        # Broadcast sector dummies across time (static)
        sector_tensor_time = np.broadcast_to(
            sector_tensor, (r_panel.shape[0], *sector_tensor.shape)
        )

        # Stack [styles | sectors] along factor axis → (T, N, K_total)
        x_tensor = np.concatenate([style_tensor, sector_tensor_time], axis=2)
        factor_names = tuple(style_names) + tuple(sector_names)
        k_total = x_tensor.shape[2]

        # WLS weights (N,) — typically sqrt(market_cap). If None, equal weights.
        if weights is None:
            wls_w = np.ones(len(instruments), dtype=np.float64)
        else:
            wls_w = np.asarray(weights, dtype=np.float64)
            if wls_w.shape != (len(instruments),):
                raise ValueError(f"weights shape {wls_w.shape} != ({len(instruments)},)")
        wls_w = np.where(np.isfinite(wls_w) & (wls_w > 0), wls_w, 0.0)

        # Run WLS per period: f_t = (X'WX)^-1 X'W r_t
        # Using lstsq for numerical stability. Skip periods where |wls_w| == 0 or
        # returns has all NaN.
        r_values = handle_nan(r_panel.to_numpy(dtype=np.float64), cfg.nan_option)
        r_values = np.asarray(r_values)  # force ndarray

        t_obs = r_values.shape[0]
        factor_returns = np.full((t_obs, k_total), np.nan, dtype=np.float64)
        residuals = np.zeros_like(r_values)

        sqrt_w = np.sqrt(wls_w)
        for t in range(t_obs):
            xt = x_tensor[t]  # (N, K)
            rt = r_values[t]  # (N,)
            mask = np.isfinite(rt) & (sqrt_w > 0)
            if mask.sum() < k_total + 1:
                residuals[t] = 0.0
                continue
            x_eff = xt[mask] * sqrt_w[mask][:, None]
            r_eff = rt[mask] * sqrt_w[mask]
            sol, *_ = np.linalg.lstsq(x_eff, r_eff, rcond=None)
            factor_returns[t] = sol
            residuals[t, mask] = rt[mask] - xt[mask] @ sol

        # cov_b: drop all-NaN rows before computing cov
        valid_rows = ~np.all(np.isnan(factor_returns), axis=1)
        clean_fr = factor_returns[valid_rows]
        # Replace any residual NaN factor returns (rare) with period mean
        col_mean = np.nanmean(clean_fr, axis=0)
        clean_fr = np.where(np.isnan(clean_fr), col_mean, clean_fr)
        clean_fr = center_columns(clean_fr, assume_centered=cfg.assume_centered)
        cov_b = np.cov(clean_fr, rowvar=False)
        if cov_b.ndim == 0:
            cov_b = np.array([[float(cov_b)]], dtype=np.float64)

        # var_u: specific variance per instrument (optionally shrunk)
        var_u = np.var(residuals, axis=0, dtype=np.float64)
        if cfg.shrink_specific:
            var_u = _bayesian_shrink_specific_variance(var_u, wls_w)

        # Latest-period exposures (used as F in Σ = F cov_b F' + D)
        f_latest = x_tensor[-1]  # (N, K_total)
        covariance = f_latest @ cov_b @ f_latest.T + np.diag(var_u)
        covariance = _ensure_symmetric(covariance)

        return RiskModelOutput(
            timestamp_ns=timestamp_ns,
            instruments=instruments,
            covariance=covariance,
            factor_names=factor_names,
            factor_exposures=f_latest,
            factor_covariance=cov_b,
            specific_variance=var_u,
            model_type="fundamental",
        )

    @property
    def min_history(self) -> int:
        return self._config.min_history_bars

    @property
    def model_type(self) -> str:
        return "fundamental"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _assemble_style_tensor(
    exposures: pd.DataFrame,
    index: pd.Index,
    instruments: tuple[str, ...],
    style_names: tuple[str, ...],
    winsorize_sigma: float,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Unstack style factor panel to (T, N, K) and z-score / winsorize each period.

    Accepts two exposures DataFrame formats:
    1. MultiIndex (timestamp, instrument), columns = factor names
    2. Columns = factor names, single index of ``timestamp`` with each cell being a
       dict of {instrument: value} — used by some callers.

    Returns
    -------
    tensor : np.ndarray of shape (T, N, K)
    names : tuple of factor column names used (subset of style_names available)
    """
    if not isinstance(exposures.index, pd.MultiIndex):
        raise ValueError("exposures must have a MultiIndex (timestamp, instrument)")

    missing = [n for n in style_names if n not in exposures.columns]
    if missing:
        raise ValueError(f"exposures missing style factors: {missing}")

    t_obs = len(index)
    n_inst = len(instruments)
    k_style = len(style_names)
    tensor = np.zeros((t_obs, n_inst, k_style), dtype=np.float64)

    # Reindex to (T × N) for each factor
    for k, name in enumerate(style_names):
        # Unstack factor k to a TxN DataFrame
        panel = exposures[name].unstack(level=-1)
        panel = panel.reindex(index=index, columns=list(instruments))
        arr = panel.to_numpy(dtype=np.float64)
        # Replace NaN with 0 before z-score (missing exposure → neutral)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        # Per-period cross-sectional z-score
        mean = arr.mean(axis=1, keepdims=True)
        std = arr.std(axis=1, keepdims=True, ddof=0)
        std = np.where(std < 1e-12, 1.0, std)
        z = (arr - mean) / std
        # Winsorize ±sigma
        z = np.clip(z, -winsorize_sigma, winsorize_sigma)
        tensor[:, :, k] = z

    return tensor, style_names


def _assemble_sector_dummies(
    instruments: tuple[str, ...],
    sector_map: dict[str, str],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build one-hot sector membership matrix.

    Unknown instruments are assigned to sector "Other".
    Sectors with zero members are dropped from the output.
    """
    n = len(instruments)
    # Collect sectors in deterministic (sorted) order
    assignments = [sector_map.get(inst, "Other") for inst in instruments]
    sectors = sorted({s for s in assignments})
    s_to_idx = {s: i for i, s in enumerate(sectors)}

    dummies = np.zeros((n, len(sectors)), dtype=np.float64)
    for i, s in enumerate(assignments):
        dummies[i, s_to_idx[s]] = 1.0

    # Prefix sector names to avoid colliding with style names
    sector_names = tuple(f"sector_{s}" for s in sectors)
    # Drop empty columns (should not happen after sorting but keeps dtype safe)
    col_sum = dummies.sum(axis=0)
    keep = col_sum > 0
    dummies = dummies[:, keep]
    sector_names = tuple(n for n, k in zip(sector_names, keep) if k)
    return dummies, sector_names


def _bayesian_shrink_specific_variance(
    var_u: np.ndarray,
    wls_w: np.ndarray,
) -> np.ndarray:
    """Shrink specific variances toward the WLS-weighted cross-sectional mean.

    Simple Barra-style Bayesian shrinkage:
        v_i_new = 0.5 * v_i + 0.5 * v_bar
    where v_bar is the weighted average of diag(D). Matches the spirit of the
    Barra USE4 decile-based shrinkage without decile segmentation (overkill for
    ~100 crypto assets).
    """
    total_w = wls_w.sum()
    if total_w <= 0:
        return var_u.copy()
    v_bar = float(np.sum(var_u * wls_w) / total_w)
    return 0.5 * var_u + 0.5 * v_bar


def _ensure_symmetric(m: np.ndarray) -> np.ndarray:
    return 0.5 * (m + m.T)
