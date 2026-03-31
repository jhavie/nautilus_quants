"""Alpha analysis configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class AlphaAnalysisConfig:
    """Configuration for alpha factor analysis.

    Attributes:
        catalog_path: Path to parquet data catalog
        factor_config_path: Path to factors YAML definition
        instrument_ids: List of instrument IDs to analyze
        bar_spec: Bar specification (e.g., "1h", "4h")
        factors: Factor names to analyze (empty = all)
        periods: Forward return periods in bar counts
        quantiles: Number of quantiles for grouping
        max_loss: Maximum allowed loss ratio for alphalens
        filter_zscore: Z-score threshold for outlier filtering (None to disable)
        charts: List of chart types to generate
        output_dir: Base output directory
        output_format: Output file formats
    """

    catalog_path: str
    factor_config_path: str
    instrument_ids: list[str]
    bar_spec: str = "1h"

    factors: list[str] = field(default_factory=list)

    periods: tuple[int, ...] = (1, 4, 8, 24)
    quantiles: int = 5
    max_loss: float = 0.35
    filter_zscore: float | None = 20.0

    charts: list[str] = field(default_factory=lambda: [
        "quantile_returns_bar",
        "quantile_returns_violin",
        "cumulative_returns",
        "cumulative_returns_long_short",
        "quantile_spread",
        "returns_table",
        "ic_time_series",
        "ic_histogram",
        "ic_qq",
        "monthly_ic_heatmap",
        "turnover",
        "turnover_table",
        "factor_rank_autocorrelation",
        "event_study",
        "events_distribution",
        "quantile_statistics_table",
    ])

    output_dir: str = "logs/alpha_analysis"
    output_format: tuple[str, ...] = ("png",)
    factor_cache_path: str = ""


def load_analysis_config(path: str | Path) -> AlphaAnalysisConfig:
    """Load analysis configuration from YAML file.

    Args:
        path: Path to YAML configuration file

    Returns:
        Parsed AlphaAnalysisConfig

    Raises:
        FileNotFoundError: If config file doesn't exist
        KeyError: If required fields are missing
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    periods = raw.get("periods", (1, 4, 8, 24))
    output_format = raw.get("output_format", ("png",))

    return AlphaAnalysisConfig(
        catalog_path=raw["catalog_path"],
        factor_config_path=raw["factor_config_path"],
        instrument_ids=raw["instrument_ids"],
        bar_spec=raw.get("bar_spec", "1h"),
        factors=raw.get("factors", []) or [],
        periods=tuple(periods),
        quantiles=raw.get("quantiles", 5),
        max_loss=raw.get("max_loss", 0.35),
        filter_zscore=raw.get("filter_zscore", 20.0),
        charts=raw.get("charts", [
            "quantile_returns_bar",
            "quantile_returns_violin",
            "cumulative_returns",
            "cumulative_returns_long_short",
            "quantile_spread",
            "returns_table",
            "ic_time_series",
            "ic_histogram",
            "ic_qq",
            "monthly_ic_heatmap",
            "turnover",
            "turnover_table",
            "factor_rank_autocorrelation",
            "event_study",
            "events_distribution",
            "quantile_statistics_table",
        ]),
        output_dir=raw.get("output_dir", "logs/alpha_analysis"),
        output_format=tuple(output_format),
        factor_cache_path=raw.get("factor_cache_path", ""),
    )
