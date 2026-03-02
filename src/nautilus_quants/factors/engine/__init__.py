# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Factor Engine module - Panel-based computation engine.

Provides the PanelFactorEngine and supporting components for
multi-instrument factor computation using DataFrame[T x N] architecture.

Constitution Compliance:
    - FactorEngineActor extends Nautilus Actor (Principle I)
    - FactorEngineActorConfig extends ActorConfig (Principle II)
"""

from nautilus_quants.actors.factor_engine import (
    FactorEngineActor,
    FactorEngineActorConfig,
)
from nautilus_quants.factors.engine.panel_buffer import PanelBuffer
from nautilus_quants.factors.engine.panel_evaluator import PanelEvaluator
from nautilus_quants.factors.engine.panel_factor_engine import PanelFactorEngine

__all__ = [
    # Nautilus-native Actor (Constitution compliant)
    "FactorEngineActor",
    "FactorEngineActorConfig",
    # Panel engine
    "PanelFactorEngine",
    "PanelBuffer",
    "PanelEvaluator",
]
