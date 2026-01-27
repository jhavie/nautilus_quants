# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Factor Engine module - Core computation engine.

Provides the FactorEngine class and supporting components for
multi-instrument factor computation.

Constitution Compliance:
    - FactorEngineActor extends Nautilus Actor (Principle I)
    - FactorEngineConfig extends ActorConfig (Principle II)
"""

from nautilus_quants.factors.engine.actor import (
    FactorEngineActor,
    FactorEngineActorConfig,
)
from nautilus_quants.factors.engine.data_synchronizer import (
    DataSynchronizer,
    InstrumentData,
)
from nautilus_quants.factors.engine.dependency_resolver import (
    CircularDependencyError,
    DependencyError,
    DependencyResolver,
)
from nautilus_quants.factors.engine.factor_engine import FactorEngine

__all__ = [
    # Nautilus-native Actor (Constitution compliant)
    "FactorEngineActor",
    "FactorEngineActorConfig",
    # Core engine (standalone usage)
    "FactorEngine",
    "DataSynchronizer",
    "InstrumentData",
    "DependencyResolver",
    "DependencyError",
    "CircularDependencyError",
]
