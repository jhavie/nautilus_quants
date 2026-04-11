"""Alpha analysis configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from nautilus_quants.factors.engine.extra_data import (
    ExtraDataConfig,
    load_extra_data_config,
    parse_extra_data_raw,
)


@dataclass(frozen=True)
class JumpModelConfig:
    """Jump Model regime detection parameters.

    Attributes:
        n_states: Number of regime states.
        jump_penalty: Lambda controlling regime persistence (higher=fewer switches).
        feature_set: Feature set for regime detection.
        refit_window: Rolling window size in bars (0=expanding window).
        refit_interval: How often to refit in bars.
        min_train: Minimum bars required before first fit.
    """

    n_states: int = 3
    jump_penalty: float = 5000.0
    feature_set: str = "full"  # full/paper/returns_only
    refit_window: int = 0      # 0=expanding, >0=rolling window size
    refit_interval: int = 504  # refit every N bars (~84 days at 4h)
    min_train: int = 1008      # minimum training bars (~168 days at 4h)


@dataclass(frozen=True)
class EmaConfig:
    """EMA baseline regime detection parameters."""

    span: int = 200


@dataclass(frozen=True)
class RegimeConfig:
    """Configuration for regime-conditional factor analysis.

    Attributes:
        regime_instrument: Instrument for regime detection (e.g. BTC).
        jump_model: Jump Model hyperparameters.
        ema: EMA baseline parameters.
        forward_period: Forward return period in bar counts for IC.
        min_obs: Minimum cross-sectional observations per timestamp.
        min_weight: Floor weight per factor per regime.
        export_weights: Whether to export per-regime weight_map YAML.
    """

    regime_instrument: str = "BTCUSDT.BINANCE"
    jump_model: JumpModelConfig = field(default_factory=JumpModelConfig)
    ema: EmaConfig = field(default_factory=EmaConfig)
    forward_period: int = 1
    min_obs: int = 20
    min_weight: float = 0.02
    export_weights: bool = True


@dataclass(frozen=True)
class MetricsConfig:
    """Configuration for extended factor metrics.

    Attributes:
        factor_metrics: Enable factor signal quality metrics
            (Win Rate, Coverage, IC Half-Life, Monotonicity,
             IC Linearity, IC AR(1))
    """

    factor_metrics: bool = False


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
        min_observations: Minimum factor-return observations to run analysis
        min_coverage: Minimum coverage ratio (valid obs / total possible) to run analysis
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
    min_observations: int = 100
    min_coverage: float = 0.1

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
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    regime: RegimeConfig | None = None

    # Unified extra data (replaces funding_rate/oi_data_path)
    extra_data_path: str = ""
    extra_data: list[ExtraDataConfig] = field(default_factory=list)

    # Deprecated: kept for backward compatibility
    funding_rate: bool = False
    oi_data_path: str = ""
    oi_timeframe: str = "4h"

    # Registry auto-persist configuration
    registry_env: str = "test"
    registry_db_dir: str = "logs/registry"
    registry_enabled: bool = True


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

    # Parse registry config
    reg = raw.get("registry", {}) or {}

    # Load extra_data: file path takes precedence, then inline, then legacy
    extra_data_path = raw.get("extra_data_path", "")
    if extra_data_path:
        extra_data = load_extra_data_config(extra_data_path)
    elif raw.get("extra_data"):
        extra_data = parse_extra_data_raw(raw["extra_data"])
    else:
        # Backward compat: convert legacy fields to ExtraDataConfig
        extra_data = []
        if raw.get("funding_rate"):
            extra_data.append(ExtraDataConfig(
                name="funding_rate",
                source="catalog",
                path=raw.get("catalog_path", ""),
            ))
        if raw.get("oi_data_path"):
            extra_data.append(ExtraDataConfig(
                name="open_interest",
                source="parquet",
                path=raw["oi_data_path"],
                timeframe=raw.get("oi_timeframe", "4h"),
            ))

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
        min_observations=raw.get("min_observations", 100),
        min_coverage=raw.get("min_coverage", 0.1),
        charts=raw.get("charts", []),
        output_dir=raw.get("output_dir", "logs/alpha_analysis"),
        output_format=tuple(output_format),
        factor_cache_path=raw.get("factor_cache_path", ""),
        metrics=_parse_metrics_config(raw.get("metrics", {})),
        regime=_parse_regime_config(raw.get("regime")),
        extra_data_path=extra_data_path,
        extra_data=extra_data,
        funding_rate=raw.get("funding_rate", False),
        oi_data_path=raw.get("oi_data_path", ""),
        oi_timeframe=raw.get("oi_timeframe", "4h"),
        registry_env=reg.get("env", "test"),
        registry_db_dir=reg.get("db_dir", "logs/registry"),
        registry_enabled=reg.get("enabled", True),
    )


def _parse_metrics_config(raw: dict | None) -> MetricsConfig:
    """Parse metrics configuration from YAML dict."""
    if not raw:
        return MetricsConfig()
    return MetricsConfig(
        factor_metrics=raw.get("factor_metrics", False),
    )


def _parse_regime_config(raw: dict | None) -> RegimeConfig | None:
    """Parse regime analysis configuration from YAML dict."""
    if not raw:
        return None

    jm_raw = raw.get("jump_model", {}) or {}
    ema_raw = raw.get("ema", {}) or {}

    return RegimeConfig(
        regime_instrument=raw.get("regime_instrument", "BTCUSDT.BINANCE"),
        jump_model=JumpModelConfig(
            n_states=jm_raw.get("n_states", 3),
            jump_penalty=jm_raw.get("jump_penalty", 100.0),
            feature_set=jm_raw.get("feature_set", "full"),
            refit_window=jm_raw.get("refit_window", 0),
            refit_interval=jm_raw.get("refit_interval", 126),
            min_train=jm_raw.get("min_train", 504),
        ),
        ema=EmaConfig(
            span=ema_raw.get("span", 20),
        ),
        forward_period=raw.get("forward_period", 1),
        min_obs=raw.get("min_obs", 20),
        min_weight=raw.get("min_weight", 0.02),
        export_weights=raw.get("export_weights", True),
    )
