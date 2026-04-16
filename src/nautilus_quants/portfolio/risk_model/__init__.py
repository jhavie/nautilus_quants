# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Risk model pluggable slot.

Two implementations share the RiskModel Protocol:
- StatisticalRiskModel: PCA + Shrinkage (ported from Qlib)
- FundamentalRiskModel: Barra-style WLS cross-sectional regression

Both produce RiskModelOutput. Used by RiskModelActor for computation.
"""

from nautilus_quants.portfolio.risk_model.base import RiskModel, RiskModelConfig

__all__ = [
    "RiskModel",
    "RiskModelConfig",
]
