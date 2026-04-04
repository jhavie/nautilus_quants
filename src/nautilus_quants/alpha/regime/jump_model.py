# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Statistical Jump Model wrapper for crypto regime detection.

Wraps the `jumpmodels` library with crypto-specific defaults and
end-to-end hyperparameter optimization (lambda tuned by composite ICIR,
not classification accuracy).

Based on: Yu, Mulvey, Nie (JPM 2026) + Cortese et al. (2023)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from jumpmodels.jump import JumpModel

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegimeLabels:
    """Result of regime detection."""

    labels: pd.Series  # int labels (0, 1, 2...) indexed by timestamp
    n_states: int
    jump_penalty: float
    state_stats: dict[int, dict[str, float]]  # {state: {mean_ret, vol, count}}


class CryptoRegimeDetector:
    """Statistical Jump Model regime detector for crypto markets.

    Uses BTC (+ optionally ETH) features to identify market regimes.
    States are sorted by cumulative return: state 0 = bear, last = bull.

    Args:
        n_states: Number of regime states (2 or 3).
        jump_penalty: Lambda parameter controlling regime persistence.
            Higher = fewer regime switches.
    """

    def __init__(
        self,
        n_states: int = 3,
        jump_penalty: float = 10.0,
    ) -> None:
        self._n_states = n_states
        self._jump_penalty = jump_penalty
        self._model: JumpModel | None = None

    def fit(
        self,
        features: pd.DataFrame,
        returns: pd.Series | None = None,
    ) -> RegimeLabels:
        """Fit Jump Model on feature matrix and return regime labels.

        Args:
            features: Feature DataFrame (timestamps × features).
            returns: Return series for state sorting (default: BTC returns).

        Returns:
            RegimeLabels with per-timestamp integer labels.
        """
        self._model = JumpModel(
            n_components=self._n_states,
            jump_penalty=self._jump_penalty,
            verbose=0,
        )
        self._model.fit(features, ret_ser=returns, sort_by="cumret")
        labels = self._model.predict(features)

        if isinstance(labels, np.ndarray):
            labels = pd.Series(labels, index=features.index, name="regime")

        # Compute state statistics
        state_stats: dict[int, dict[str, float]] = {}
        if returns is not None:
            aligned_ret = returns.reindex(labels.index)
            for state in range(self._n_states):
                mask = labels == state
                state_ret = aligned_ret[mask].dropna()
                state_stats[state] = {
                    "mean_ret": float(state_ret.mean()) if len(state_ret) else 0.0,
                    "vol": float(state_ret.std()) if len(state_ret) else 0.0,
                    "count": int(mask.sum()),
                    "pct": float(mask.mean()),
                }
        else:
            for state in range(self._n_states):
                mask = labels == state
                state_stats[state] = {
                    "mean_ret": 0.0,
                    "vol": 0.0,
                    "count": int(mask.sum()),
                    "pct": float(mask.mean()),
                }

        n_switches = int((labels != labels.shift()).sum())
        logger.info(
            "JumpModel fitted: %d states, lambda=%.1f, switches=%d, "
            "avg_regime_len=%.0f bars",
            self._n_states,
            self._jump_penalty,
            n_switches,
            len(labels) / max(n_switches, 1),
        )
        for state, ss in state_stats.items():
            logger.info(
                "  State %d: count=%d (%.1f%%), mean_ret=%.4f, vol=%.4f",
                state,
                ss["count"],
                ss["pct"] * 100,
                ss["mean_ret"],
                ss["vol"],
            )

        return RegimeLabels(
            labels=labels,
            n_states=self._n_states,
            jump_penalty=self._jump_penalty,
            state_stats=state_stats,
        )

    def predict_online(self, features: pd.DataFrame) -> pd.Series:
        """Online prediction using fitted model.

        Args:
            features: Feature DataFrame (may include new observations).

        Returns:
            Series of regime labels.
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        labels = self._model.predict(features)
        if isinstance(labels, np.ndarray):
            labels = pd.Series(labels, index=features.index, name="regime")
        return labels


def optimize_jump_penalty(
    features: pd.DataFrame,
    returns: pd.Series,
    factor_dfs: dict[str, pd.DataFrame],
    fwd_returns: pd.DataFrame,
    weights: dict[str, float] | None = None,
    n_states: int = 3,
    lambda_candidates: list[float] | None = None,
    min_weight: float = 0.02,
) -> tuple[float, float, dict[str, dict[str, float]]]:
    """Find optimal jump penalty by maximizing regime-aware composite ICIR.

    This implements the paper's core innovation: lambda is tuned by
    portfolio performance (ICIR), not by classification accuracy.

    Args:
        features: Regime feature DataFrame.
        returns: BTC return series for state sorting.
        factor_dfs: {factor_name: DataFrame(T×N)} of raw factor values.
        fwd_returns: Forward returns DataFrame.
        weights: Equal weights if None.
        n_states: Number of states.
        lambda_candidates: Lambda values to search.
        min_weight: Floor weight per factor.

    Returns:
        (best_lambda, best_icir, best_weight_map).
    """
    from nautilus_quants.alpha.regime.regime_ic_analysis import (
        _cs_normalize,
        compute_regime_aware_weights,
        compute_regime_ic,
    )
    from scipy import stats

    if lambda_candidates is None:
        lambda_candidates = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    if weights is None:
        weights = {n: 1.0 / len(factor_dfs) for n in factor_dfs}

    factor_names = list(factor_dfs.keys())
    norm_dfs = {n: _cs_normalize(df) for n, df in factor_dfs.items()}

    best_lambda = lambda_candidates[0]
    best_icir = -np.inf
    best_weight_map: dict[str, dict[str, float]] = {}

    for lam in lambda_candidates:
        detector = CryptoRegimeDetector(n_states=n_states, jump_penalty=lam)
        result = detector.fit(features, returns=returns)
        labels = result.labels

        # Convert int labels to regime strings (state 0 = bear, last = bull)
        label_map = {}
        if n_states == 2:
            label_map = {0: "bear", 1: "bull"}
        elif n_states == 3:
            label_map = {0: "bear", 1: "neutral", 2: "bull"}
        else:
            for i in range(n_states):
                label_map[i] = f"state_{i}"

        regime = labels.map(label_map)

        # Compute per-factor regime IC
        regime_results = compute_regime_ic(factor_dfs, fwd_returns, regime)

        # Compute per-regime weights (ICIR-proportional)
        weight_map = compute_regime_aware_weights(regime_results, min_weight)

        # Evaluate: compute regime-aware composite ICIR
        ic_values = []
        common_dates = norm_dfs[factor_names[0]].index.intersection(
            fwd_returns.index
        ).intersection(regime.index)

        for dt in common_dates:
            r = fwd_returns.loc[dt].dropna()
            current_regime = regime.loc[dt]
            if current_regime not in weight_map:
                continue
            w = weight_map[current_regime]
            f = sum(
                w.get(n, 0) * norm_dfs[n].loc[dt] for n in factor_names
            ).dropna()
            common = f.index.intersection(r.index)
            if len(common) < 20:
                continue
            corr, _ = stats.spearmanr(f[common].values, r[common].values)
            if not np.isnan(corr):
                ic_values.append(corr)

        if not ic_values:
            continue
        ic_arr = np.array(ic_values)
        icir = float(ic_arr.mean() / ic_arr.std()) if ic_arr.std() > 0 else 0.0

        logger.info("lambda=%.1f → ICIR=%.4f (n_ic=%d)", lam, icir, len(ic_arr))

        if icir > best_icir:
            best_icir = icir
            best_lambda = lam
            best_weight_map = weight_map

    logger.info("Best lambda=%.1f → ICIR=%.4f", best_lambda, best_icir)
    return best_lambda, best_icir, best_weight_map
