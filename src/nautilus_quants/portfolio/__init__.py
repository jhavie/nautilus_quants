# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Portfolio module: risk modeling, portfolio optimization, and exposure monitoring.

Pure computation layer with no NautilusTrader dependency. Consumed by:
- actors/risk_model.py (RiskModelActor writes RiskModelOutput to Cache)
- strategies/cs/optimized_selection_policy.py (reads Cache, calls optimizer)
- actors/snapshot_aggregator.py (reads Cache, calls monitor functions)

See: /Users/joe/.claude/plans/resilient-painting-star.md (feature/047 plan).
"""

from nautilus_quants.portfolio.types import (
    RiskModelOutput,
    deserialize_risk_output,
    serialize_risk_output,
)

__all__ = [
    "RiskModelOutput",
    "serialize_risk_output",
    "deserialize_risk_output",
]
