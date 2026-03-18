# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Factor Engine module - Panel-based computation engine.

Provides the FactorEngine and supporting components for
multi-instrument factor computation using DataFrame[T x N] architecture.

Constitution Compliance:
    - FactorEngineActor extends Nautilus Actor (Principle I)
    - FactorEngineActorConfig extends ActorConfig (Principle II)
"""

from nautilus_quants.actors.factor_engine import (
    FactorEngineActor,
    FactorEngineActorConfig,
)
from nautilus_quants.factors.engine.buffer import Buffer
from nautilus_quants.factors.engine.evaluator import Evaluator
from nautilus_quants.factors.engine.factor_engine import FactorEngine

__all__ = [
    # Nautilus-native Actor (Constitution compliant)
    "FactorEngineActor",
    "FactorEngineActorConfig",
    # Panel engine
    "FactorEngine",
    "Buffer",
    "Evaluator",
]
