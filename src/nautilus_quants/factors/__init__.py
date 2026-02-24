# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Factor Framework for Nautilus Trader.

This module provides a configuration-driven factor computation framework
supporting Alpha101-style expressions with time-series and cross-sectional operators.

Core Components:
    - FactorEngine: Actor that computes factors and publishes FactorValues
    - Factor: Base class for all factors (time-series, cross-sectional, expression-based)
    - Operators: ts_mean, ts_max, delay, cs_rank, cs_zscore, etc.
    - Expression Engine: Parses and evaluates Alpha101-style expressions

Example:
    ```python
    from nautilus_quants.factors import FactorEngine, load_factor_config

    config = load_factor_config("config/factors.yaml")
    engine = FactorEngine(config)
    ```
"""

from nautilus_quants.factors.base import (
    CrossSectionalFactor,
    ExpressionFactor,
    Factor,
    TimeSeriesFactor,
)
from nautilus_quants.factors.config import (
    ConfigValidationError,
    FactorConfig,
    FactorDefinition,
    PerformanceConfig,
    load_factor_config,
)
from nautilus_quants.factors.engine import (
    DependencyResolver,
    FactorEngine,
    FactorEngineActor,
    FactorEngineActorConfig,
)
from nautilus_quants.factors.types import FactorInput, FactorValues

__all__ = [
    # Data types
    "FactorInput",
    "FactorValues",
    # Configuration
    "FactorConfig",
    "FactorDefinition",
    "PerformanceConfig",
    "ConfigValidationError",
    "load_factor_config",
    # Factor classes
    "Factor",
    "ExpressionFactor",
    "TimeSeriesFactor",
    "CrossSectionalFactor",
    # Engine (Nautilus-native Actor)
    "FactorEngineActor",
    "FactorEngineActorConfig",
    # Engine (standalone)
    "FactorEngine",
    "DependencyResolver",
]
