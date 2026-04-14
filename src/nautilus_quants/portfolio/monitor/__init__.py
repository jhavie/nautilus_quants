# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Portfolio exposure monitor — pure functions for risk reporting.

No Actor here. Called by SnapshotAggregatorActor (the existing monitoring
aggregator) in its periodic timer, which reads the latest risk snapshot from
Cache and writes ``snapshot:risk`` JSON for Grafana.
"""

from nautilus_quants.portfolio.monitor.exposure import (
    Breach,
    check_factor_limits,
    check_sector_limits,
    compute_portfolio_exposure,
)

__all__ = [
    "Breach",
    "compute_portfolio_exposure",
    "check_factor_limits",
    "check_sector_limits",
]
