"""Alpha module - factor research tools.

Provides factor quality evaluation (IC/ICIR/quantile analysis) via alphalens.
"""

from nautilus_quants.alpha.analysis.config import (
    AlphaAnalysisConfig,
    MetricsConfig,
    load_analysis_config,
)
from nautilus_quants.alpha.analysis.evaluator import FactorEvaluator
from nautilus_quants.alpha.data_loader import CatalogDataLoader

__all__ = [
    "AlphaAnalysisConfig",
    "CatalogDataLoader",
    "FactorEvaluator",
    "MetricsConfig",
    "load_analysis_config",
]
