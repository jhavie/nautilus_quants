# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Portfolio monitoring — pure functions for Grafana reporting.

No Actor here. Called by SnapshotAggregatorActor (health, exposure) and
FactorEngineActor (IC). Grafana handles alerting via its own rule engine.
"""

from nautilus_quants.portfolio.monitor.attribution import compute_pnl_attribution
from nautilus_quants.portfolio.monitor.concentration import compute_concentration
from nautilus_quants.portfolio.monitor.exposure import (
    check_factor_limits,
    check_sector_limits,
    compute_portfolio_exposure,
)
from nautilus_quants.portfolio.monitor.factor import (
    compute_factor_health,
    compute_factor_ic,
)
from nautilus_quants.portfolio.monitor.risk_decomposition import compute_risk_budget
from nautilus_quants.portfolio.monitor.var import compute_historical_var

__all__ = [
    "check_factor_limits",
    "check_sector_limits",
    "compute_concentration",
    "compute_factor_health",
    "compute_factor_ic",
    "compute_historical_var",
    "compute_pnl_attribution",
    "compute_portfolio_exposure",
    "compute_risk_budget",
]
