# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Portfolio optimizer pluggable slot.

Implementations share the Optimizer Protocol. Currently only MeanVarianceOptimizer
(long-short cvxpy) is provided, but new optimizers (risk parity, Kelly, etc.)
can be added by implementing Optimizer.
"""

from nautilus_quants.portfolio.optimizer.base import (
    Optimizer,
    OptimizerConstraints,
    OptimizerResult,
)

__all__ = [
    "Optimizer",
    "OptimizerConstraints",
    "OptimizerResult",
]
