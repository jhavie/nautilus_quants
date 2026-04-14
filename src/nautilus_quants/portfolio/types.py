# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Portfolio types — RiskModelOutput dataclass and cache serialization.

Data flow (Cache mode, not msgbus):
    RiskModelActor → cache.add(RISK_MODEL_STATE_CACHE_KEY, serialize_risk_output(output))
    OptimizedSelectionPolicy.select → deserialize_risk_output(cache.get(KEY)) → use

Serialization uses JSON (with numpy.tolist()) to match FactorValues pattern
(factors/factor_values.py). Human-readable, no extra deps, adequate perf for
100x100 covariance matrices (~200KB JSON payload).

No @customdataclass decorator here — RiskModelOutput does NOT travel via
Nautilus msgbus. DecisionEngineActor has zero knowledge of it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RiskModelOutput:
    """Immutable output of a risk model estimation.

    Both Statistical (PCA) and Fundamental (Barra) risk models produce this
    same output shape. The ``is_interpretable`` flag distinguishes them:
    - Statistical: factor_names are synthetic ("PC_0", "PC_1", ...)
    - Fundamental: factor_names are human-meaningful ("size", "btc_beta", ...)

    Attributes
    ----------
    timestamp_ns : int
        Event timestamp of the underlying data (nanoseconds).
    instruments : tuple[str, ...]
        N instrument IDs (sorted deterministically).
    covariance : np.ndarray
        (N, N) full covariance matrix. Primary optimizer input.
    factor_names : tuple[str, ...] | None
        K factor names (None for pure statistical with no decomposition exposed).
    factor_exposures : np.ndarray | None
        (N, K) factor exposure matrix X. None if no decomposition.
    factor_covariance : np.ndarray | None
        (K, K) factor covariance matrix cov_b.
    specific_variance : np.ndarray | None
        (N,) idiosyncratic variance vector var_u (diagonal of D).
    model_type : str
        "statistical" or "fundamental". Used by monitor to decide whether
        to publish named exposure alerts.
    """

    timestamp_ns: int
    instruments: tuple[str, ...]
    covariance: np.ndarray
    factor_names: tuple[str, ...] | None = None
    factor_exposures: np.ndarray | None = None
    factor_covariance: np.ndarray | None = None
    specific_variance: np.ndarray | None = None
    model_type: str = "statistical"
    # Optional instrument → sector mapping, set by Fundamental model for
    # downstream consumers (SnapshotAggregator) that want to compute
    # sector-level exposures without re-reading portfolio.yaml.
    sector_map: dict[str, str] | None = None

    @property
    def is_decomposed(self) -> bool:
        """Whether factor decomposition components are available."""
        return self.factor_exposures is not None

    @property
    def is_interpretable(self) -> bool:
        """Whether factor_names are human-meaningful (Barra yes, PCA no)."""
        if not self.factor_names:
            return False
        return self.model_type == "fundamental"

    @property
    def n_instruments(self) -> int:
        return len(self.instruments)

    @property
    def n_factors(self) -> int:
        return len(self.factor_names) if self.factor_names else 0


def serialize_risk_output(output: RiskModelOutput) -> bytes:
    """Serialize RiskModelOutput to bytes for Nautilus Cache storage.

    Format: JSON with numpy arrays as nested lists (``.tolist()``). Matches
    FactorValues serialization pattern (json.dumps(...).encode("utf-8")).

    Parameters
    ----------
    output : RiskModelOutput
        The model output to serialize.

    Returns
    -------
    bytes
        UTF-8 JSON-encoded payload suitable for ``cache.add(key, bytes)``.
    """
    payload: dict[str, Any] = {
        "timestamp_ns": output.timestamp_ns,
        "instruments": list(output.instruments),
        "covariance": output.covariance.tolist(),
        "factor_names": list(output.factor_names) if output.factor_names else None,
        "factor_exposures": (
            output.factor_exposures.tolist() if output.factor_exposures is not None else None
        ),
        "factor_covariance": (
            output.factor_covariance.tolist() if output.factor_covariance is not None else None
        ),
        "specific_variance": (
            output.specific_variance.tolist() if output.specific_variance is not None else None
        ),
        "model_type": output.model_type,
        "sector_map": dict(output.sector_map) if output.sector_map else None,
    }
    return json.dumps(payload).encode("utf-8")


def deserialize_risk_output(payload: bytes) -> RiskModelOutput:
    """Deserialize bytes from Cache into a RiskModelOutput.

    Parameters
    ----------
    payload : bytes
        UTF-8 JSON-encoded payload from ``cache.get(key)``.

    Returns
    -------
    RiskModelOutput
        Reconstructed immutable output.
    """
    data = json.loads(payload.decode("utf-8"))
    return RiskModelOutput(
        timestamp_ns=int(data["timestamp_ns"]),
        instruments=tuple(data["instruments"]),
        covariance=np.asarray(data["covariance"], dtype=np.float64),
        factor_names=tuple(data["factor_names"]) if data.get("factor_names") else None,
        factor_exposures=(
            np.asarray(data["factor_exposures"], dtype=np.float64)
            if data.get("factor_exposures") is not None
            else None
        ),
        factor_covariance=(
            np.asarray(data["factor_covariance"], dtype=np.float64)
            if data.get("factor_covariance") is not None
            else None
        ),
        specific_variance=(
            np.asarray(data["specific_variance"], dtype=np.float64)
            if data.get("specific_variance") is not None
            else None
        ),
        model_type=str(data.get("model_type", "statistical")),
        sector_map=dict(data["sector_map"]) if data.get("sector_map") else None,
    )
