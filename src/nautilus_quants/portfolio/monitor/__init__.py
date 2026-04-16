# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Portfolio exposure monitor — pure function for risk reporting.

No Actor here. Called by SnapshotAggregatorActor's periodic timer, which
reads the latest risk snapshot from Cache, computes raw exposures, and
writes ``snapshot:risk`` JSON for Grafana. Grafana handles alerting via
its own rule engine — no limit-breach detection lives in this module.
"""

from nautilus_quants.portfolio.monitor.attribution import compute_pnl_attribution
from nautilus_quants.portfolio.monitor.concentration import compute_concentration
from nautilus_quants.portfolio.monitor.exposure import (
    check_factor_limits,
    check_sector_limits,
    compute_portfolio_exposure,
)
from nautilus_quants.portfolio.monitor.risk_decomposition import compute_risk_budget
from nautilus_quants.portfolio.monitor.var import compute_historical_var

__all__ = [
    "check_factor_limits",
    "check_sector_limits",
    "compute_concentration",
    "compute_historical_var",
    "compute_pnl_attribution",
    "compute_portfolio_exposure",
    "compute_risk_budget",
]
