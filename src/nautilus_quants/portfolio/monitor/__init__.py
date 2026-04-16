# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Portfolio exposure monitor — pure function for risk reporting.

No Actor here. Called by SnapshotAggregatorActor's periodic timer, which
reads the latest risk snapshot from Cache, computes raw exposures, and
writes ``snapshot:risk`` JSON for Grafana. Grafana handles alerting via
its own rule engine — no limit-breach detection lives in this module.
"""

from nautilus_quants.portfolio.monitor.exposure import compute_portfolio_exposure

__all__ = [
    "compute_portfolio_exposure",
]
