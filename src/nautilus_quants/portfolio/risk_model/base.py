# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
RiskModel Protocol and shared configuration.

Pluggable slot pattern: Statistical (PCA) and Fundamental (Barra WLS) both
implement this Protocol. RiskModelActor selects implementation via config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from nautilus_quants.factors.config import FactorDefinition
from nautilus_quants.portfolio.types import RiskModelOutput


@dataclass(frozen=True)
class RiskModelConfig:
    """Common risk model configuration fields.

    Attributes
    ----------
    lookback_bars : int
        Historical observations used for each estimation. Typical: 240 bars
        (≈40 days at 4h bars).
    min_history_bars : int
        Minimum observations before first estimation (warmup).
    winsorize_quantile : float
        Returns winsorize at (q, 1-q) quantiles before estimation. 0.025 matches
        Qlib prepare_riskdata.py default (2.5%/97.5%).
    scale_return_pct : bool
        Multiply returns by 100 before estimation (Qlib convention).
    assume_centered : bool
        If False, subtract cross-sectional mean before covariance.
    nan_option : str
        How to handle NaN in returns: "ignore" | "fill" | "mask".
        "fill" replaces NaN with 0. "mask" uses np.ma.MaskedArray.
    """

    lookback_bars: int = 240
    min_history_bars: int = 60
    winsorize_quantile: float = 0.025
    scale_return_pct: bool = True
    assume_centered: bool = False
    nan_option: str = "fill"


@dataclass(frozen=True)
class StatisticalModelConfig(RiskModelConfig):
    """Statistical (PCA/Shrinkage) model configuration.

    Attributes
    ----------
    method : str
        "pca" | "fa" | "shrink". PCA and FA produce decomposed (F, cov_b, var_u);
        shrink produces full covariance only.
    num_factors : int
        Number of latent factors for PCA/FA decomposition.
    shrinkage : str | float | None
        Ledoit-Wolf shrinkage: "lw" | "oas" | float ∈ [0, 1] | None.
        Only used when method == "shrink".
    shrink_target : str
        "const_var" | "const_corr" | "single_factor".
    """

    method: str = "pca"
    num_factors: int = 10
    shrinkage: str | float | None = "lw"
    shrink_target: str = "const_corr"


@dataclass(frozen=True)
class FundamentalFactorSpec:
    """Single named risk factor specification (Fundamental model).

    Exposures are always cross-sectionally z-scored and winsorized to
    ``winsorize_exposures_sigma`` inside FundamentalRiskModel; no per-factor
    transform override is supported today.

    Attributes
    ----------
    name : str
        Risk factor display name (e.g., "btc_beta", "size").
    variable : str
        Source variable name from FactorValues (computed by FactorEngine).
    """

    name: str
    variable: str


@dataclass(frozen=True)
class FundamentalModelConfig(RiskModelConfig):
    """Fundamental (Barra-style WLS) model configuration.

    Attributes
    ----------
    variables : tuple[FactorDefinition, ...]
        Portfolio-defined risk variable expressions. Computed by RiskModelActor's
        embedded FactorEngine (not by the alpha FactorEngineActor). Registration
        order = YAML insertion order = dependency order.
    factors : tuple[FundamentalFactorSpec, ...]
        Named risk factors. Each ``factor.variable`` must name one of the
        ``variables`` above (or be an extra_data field name like
        ``funding_rate``).
    sector_map : dict[str, str]
        Instrument → sector mapping (e.g., "BTCUSDT.BINANCE" → "L1").
    wls_weight_source : str
        "market_cap" | "equal". When "market_cap", RiskModelActor reads
        ``variables["market_cap"]`` for WLS weights.
    winsorize_exposures_sigma : float
        Cross-sectional winsorize threshold for factor exposures (±σ). 3.0 matches Barra.
    sector_constraint : bool
        Enforce cap-weighted sum of sector factor returns = 0 (Barra constraint).
    shrink_specific : bool
        Apply Bayesian shrinkage to specific variances toward cap-decile means.
    """

    variables: tuple[FactorDefinition, ...] = field(default_factory=tuple)
    factors: tuple[FundamentalFactorSpec, ...] = field(default_factory=tuple)
    sector_map: dict[str, str] = field(default_factory=dict)
    wls_weight_source: str = "market_cap"
    winsorize_exposures_sigma: float = 3.0
    sector_constraint: bool = True
    shrink_specific: bool = True


class RiskModel(Protocol):
    """Protocol for risk model estimators.

    Implementations: StatisticalRiskModel, FundamentalRiskModel.
    """

    def fit(
        self,
        returns: pd.DataFrame,
        timestamp_ns: int,
        exposures: pd.DataFrame | None = None,
        weights: np.ndarray | None = None,
    ) -> RiskModelOutput:
        """Estimate covariance and (optionally) factor decomposition.

        Parameters
        ----------
        returns : pd.DataFrame
            Returns matrix of shape (T, N). Index = timestamps, columns = instruments.
            Contains up to ``lookback_bars`` observations (may include NaN per config).
        timestamp_ns : int
            Event timestamp (ns) to stamp on the RiskModelOutput.
        exposures : pd.DataFrame | None
            Fundamental-only. Latest-period factor exposures of shape (N, K).
            Index = instruments, columns = risk factor names. Ignored for statistical.
        weights : np.ndarray | None
            Fundamental-only. WLS weights of shape (N,) (typically sqrt(market_cap)).

        Returns
        -------
        RiskModelOutput
            Immutable output with covariance (and optional decomposition).
        """
        ...

    @property
    def min_history(self) -> int:
        """Minimum observations required for a valid fit."""
        ...

    @property
    def model_type(self) -> str:
        """Returns "statistical" or "fundamental"."""
        ...
