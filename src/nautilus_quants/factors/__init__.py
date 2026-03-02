# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Factor Framework for Nautilus Trader.

This module provides a configuration-driven factor computation framework
supporting Alpha101-style expressions with time-series and cross-sectional operators.

Core Components:
    - PanelFactorEngine: Evaluates factor expressions on panel DataFrames [T x N]
    - FactorEngineActor: Nautilus Actor wrapper that publishes FactorValues
    - Operators: ts_mean, ts_max, delay, rank, normalize, etc.
    - Expression Engine: Parses and evaluates Alpha101-style expressions via AST

Example:
    ```python
    from nautilus_quants.factors import PanelFactorEngine, load_factor_config

    config = load_factor_config("config/factors.yaml")
    engine = PanelFactorEngine(config)
    ```
"""

from nautilus_quants.factors.config import (
    ConfigValidationError,
    FactorConfig,
    FactorDefinition,
    PerformanceConfig,
    load_factor_config,
)
from nautilus_quants.factors.engine import (
    FactorEngineActor,
    FactorEngineActorConfig,
    PanelFactorEngine,
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
    # Engine (Nautilus-native Actor)
    "FactorEngineActor",
    "FactorEngineActorConfig",
    # Engine (standalone)
    "PanelFactorEngine",
]
