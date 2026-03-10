"""
Configuration loader for the data pipeline.

Loads configuration from YAML file and merges with CLI overrides.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import yaml


@dataclass
class RateLimitConfig:
    """Rate limit configuration for API requests."""

    max_retries: int = 5
    initial_delay_seconds: float = 4.0
    max_delay_seconds: float = 60.0
    backoff_multiplier: float = 2.0


@dataclass
class CheckpointConfig:
    """Checkpoint configuration for resumable downloads."""

    enabled: bool = True
    batch_size: int = 1000


@dataclass
class DownloadConfig:
    """Download configuration."""

    exchange: str = "binance"
    market_type: Literal["spot", "futures"] = "futures"
    symbols: list[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    timeframes: list[str] = field(default_factory=lambda: ["1h", "4h"])
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)


@dataclass
class ValidateConfig:
    """Validation configuration."""

    check_duplicates: bool = True
    check_gaps: bool = True
    check_ohlc: bool = True
    fail_on_warnings: bool = False


@dataclass
class ProcessConfig:
    """Processing configuration."""

    remove_duplicates: bool = True
    keep_duplicate: Literal["first", "last"] = "last"
    fill_small_gaps: bool = True
    max_gap_bars: int = 3
    remove_invalid_ohlc: bool = True


@dataclass
class TransformConfig:
    """Transform configuration."""

    output_format: str = "parquet"
    merge_files: bool = True
    catalog_path: str = "data/catalog"
    maker_fee: str = "0.0002"
    taker_fee: str = "0.0004"
    margin_init: str = "0.05"
    margin_maint: str = "0.025"
    bar_class: str = "Bar"


@dataclass
class PathsConfig:
    """Paths configuration."""

    raw_data: str = "data/raw"
    processed_data: str = "data/processed"
    catalog: str = "data/catalog"
    logs: str = "logs/data_pipeline"


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""

    version: str = "1.0"
    download: DownloadConfig = field(default_factory=DownloadConfig)
    validate: ValidateConfig = field(default_factory=ValidateConfig)
    process: ProcessConfig = field(default_factory=ProcessConfig)
    transform: TransformConfig = field(default_factory=TransformConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


@dataclass(frozen=True)
class TardisDownloadConfig:
    """Download configuration for Tardis tick data."""

    exchange: str = "binance-futures"
    api_key_env: str = "TARDIS_API_KEY"
    data_types: tuple[str, ...] = ("trades",)
    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT")
    from_date: str = "2024-01-01"
    to_date: str = "2024-07-01"
    concurrency: int = 5
    max_symbol_workers: int = 3


@dataclass(frozen=True)
class TardisPathsConfig:
    """Paths configuration for Tardis data."""

    raw_data: str = "data/raw/tardis"
    catalog: str = "data/catalog"


@dataclass(frozen=True)
class TardisPipelineConfig:
    """Complete Tardis pipeline configuration.

    No transform config needed — TardisCSVDataLoader infers precision automatically.
    """

    download: TardisDownloadConfig = TardisDownloadConfig()
    paths: TardisPathsConfig = TardisPathsConfig()


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""

    pass


def _parse_rate_limit(data: dict) -> RateLimitConfig:
    """Parse rate limit configuration from dict."""
    return RateLimitConfig(
        max_retries=data.get("max_retries", 5),
        initial_delay_seconds=data.get("initial_delay_seconds", 4.0),
        max_delay_seconds=data.get("max_delay_seconds", 60.0),
        backoff_multiplier=data.get("backoff_multiplier", 2.0),
    )


def _parse_checkpoint(data: dict) -> CheckpointConfig:
    """Parse checkpoint configuration from dict."""
    return CheckpointConfig(
        enabled=data.get("enabled", True),
        batch_size=data.get("batch_size", 1000),
    )


def _parse_download(data: dict) -> DownloadConfig:
    """Parse download configuration from dict."""
    return DownloadConfig(
        exchange=data.get("exchange", "binance"),
        market_type=data.get("market_type", "futures"),
        symbols=data.get("symbols", ["BTCUSDT", "ETHUSDT"]),
        timeframes=data.get("timeframes", ["1h", "4h"]),
        start_date=data.get("start_date", "2024-01-01"),
        end_date=data.get("end_date", "2024-12-31"),
        rate_limit=_parse_rate_limit(data.get("rate_limit", {})),
        checkpoint=_parse_checkpoint(data.get("checkpoint", {})),
    )


def _parse_validate(data: dict) -> ValidateConfig:
    """Parse validation configuration from dict."""
    return ValidateConfig(
        check_duplicates=data.get("check_duplicates", True),
        check_gaps=data.get("check_gaps", True),
        check_ohlc=data.get("check_ohlc", True),
        fail_on_warnings=data.get("fail_on_warnings", False),
    )


def _parse_process(data: dict) -> ProcessConfig:
    """Parse processing configuration from dict."""
    return ProcessConfig(
        remove_duplicates=data.get("remove_duplicates", True),
        keep_duplicate=data.get("keep_duplicate", "last"),
        fill_small_gaps=data.get("fill_small_gaps", True),
        max_gap_bars=data.get("max_gap_bars", 3),
        remove_invalid_ohlc=data.get("remove_invalid_ohlc", True),
    )


def _parse_transform(data: dict) -> TransformConfig:
    """Parse transform configuration from dict."""
    return TransformConfig(
        output_format=data.get("output_format", "parquet"),
        merge_files=data.get("merge_files", True),
        catalog_path=data.get("catalog_path", "data/catalog"),
        maker_fee=data.get("maker_fee", "0.0002"),
        taker_fee=data.get("taker_fee", "0.0004"),
        margin_init=data.get("margin_init", "0.05"),
        margin_maint=data.get("margin_maint", "0.025"),
        bar_class=data.get("bar_class", "Bar"),
    )


def _parse_paths(data: dict) -> PathsConfig:
    """Parse paths configuration from dict."""
    return PathsConfig(
        raw_data=data.get("raw_data", "data/raw"),
        processed_data=data.get("processed_data", "data/processed"),
        catalog=data.get("catalog", "data/catalog"),
        logs=data.get("logs", "logs/data_pipeline"),
    )


def load_config(
    config_path: Path | str = "config/data.yaml",
    overrides: Optional[dict[str, Any]] = None,
) -> PipelineConfig:
    """Load configuration from YAML file and merge with CLI overrides.

    Args:
        config_path: Path to the YAML configuration file
        overrides: Optional dictionary of CLI overrides

    Returns:
        PipelineConfig with merged configuration

    Raises:
        ConfigurationError: If configuration file is invalid or not found
    """
    config_path = Path(config_path)

    # Load from YAML file
    if config_path.exists():
        try:
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in {config_path}: {e}")
    else:
        # Use defaults if config file doesn't exist
        data = {}

    # Apply CLI overrides
    if overrides:
        data = _merge_overrides(data, overrides)

    # Parse into typed configuration
    config = PipelineConfig(
        version=data.get("version", "1.0"),
        download=_parse_download(data.get("download", {})),
        validate=_parse_validate(data.get("validate", {})),
        process=_parse_process(data.get("process", {})),
        transform=_parse_transform(data.get("transform", {})),
        paths=_parse_paths(data.get("paths", {})),
    )

    return config


def _merge_overrides(data: dict, overrides: dict) -> dict:
    """Merge CLI overrides into configuration data.

    Supports dotted keys like 'download.symbols' for nested values.
    """
    result = data.copy()

    for key, value in overrides.items():
        if value is None:
            continue

        # Handle dotted keys
        parts = key.split(".")
        current = result

        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Handle comma-separated lists for symbols and timeframes
        if parts[-1] in ("symbols", "timeframes") and isinstance(value, str):
            current[parts[-1]] = [v.strip() for v in value.split(",")]
        else:
            current[parts[-1]] = value

    return result


def _parse_tardis_download(data: dict) -> TardisDownloadConfig:
    """Parse Tardis download configuration from dict."""
    data_types = data.get("data_types", ["trades"])
    symbols = data.get("symbols", ["BTCUSDT", "ETHUSDT"])
    return TardisDownloadConfig(
        exchange=data.get("exchange", "binance-futures"),
        api_key_env=data.get("api_key_env", "TARDIS_API_KEY"),
        data_types=tuple(data_types),
        symbols=tuple(symbols),
        from_date=data.get("from_date", "2024-01-01"),
        to_date=data.get("to_date", "2024-07-01"),
        concurrency=data.get("concurrency", 5),
        max_symbol_workers=data.get("max_symbol_workers", 3),
    )


def _parse_tardis_paths(data: dict) -> TardisPathsConfig:
    """Parse Tardis paths configuration from dict."""
    return TardisPathsConfig(
        raw_data=data.get("raw_data", "data/raw/tardis"),
        catalog=data.get("catalog", "data/catalog"),
    )


def load_tardis_config(
    config_path: Path | str = "config/examples/tardis_data.yaml",
    overrides: Optional[dict[str, Any]] = None,
) -> TardisPipelineConfig:
    """Load Tardis configuration from YAML file and merge with CLI overrides.

    Args:
        config_path: Path to the YAML configuration file
        overrides: Optional dictionary of CLI overrides

    Returns:
        TardisPipelineConfig with merged configuration

    Raises:
        ConfigurationError: If configuration file is invalid or not found
    """
    config_path = Path(config_path)

    if config_path.exists():
        try:
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in {config_path}: {e}")
    else:
        data = {}

    if overrides:
        data = _merge_overrides(data, overrides)

    return TardisPipelineConfig(
        download=_parse_tardis_download(data.get("download", {})),
        paths=_parse_tardis_paths(data.get("paths", {})),
    )


def config_to_dict(config: PipelineConfig) -> dict:
    """Convert PipelineConfig to dictionary for serialization."""
    return {
        "version": config.version,
        "download": {
            "exchange": config.download.exchange,
            "market_type": config.download.market_type,
            "symbols": config.download.symbols,
            "timeframes": config.download.timeframes,
            "start_date": config.download.start_date,
            "end_date": config.download.end_date,
            "rate_limit": {
                "max_retries": config.download.rate_limit.max_retries,
                "initial_delay_seconds": config.download.rate_limit.initial_delay_seconds,
                "max_delay_seconds": config.download.rate_limit.max_delay_seconds,
                "backoff_multiplier": config.download.rate_limit.backoff_multiplier,
            },
            "checkpoint": {
                "enabled": config.download.checkpoint.enabled,
                "batch_size": config.download.checkpoint.batch_size,
            },
        },
        "validate": {
            "check_duplicates": config.validate.check_duplicates,
            "check_gaps": config.validate.check_gaps,
            "check_ohlc": config.validate.check_ohlc,
            "fail_on_warnings": config.validate.fail_on_warnings,
        },
        "process": {
            "remove_duplicates": config.process.remove_duplicates,
            "keep_duplicate": config.process.keep_duplicate,
            "fill_small_gaps": config.process.fill_small_gaps,
            "max_gap_bars": config.process.max_gap_bars,
            "remove_invalid_ohlc": config.process.remove_invalid_ohlc,
        },
        "transform": {
            "output_format": config.transform.output_format,
            "merge_files": config.transform.merge_files,
            "catalog_path": config.transform.catalog_path,
            "maker_fee": config.transform.maker_fee,
            "taker_fee": config.transform.taker_fee,
            "margin_init": config.transform.margin_init,
            "margin_maint": config.transform.margin_maint,
            "bar_class": config.transform.bar_class,
        },
        "paths": {
            "raw_data": config.paths.raw_data,
            "processed_data": config.paths.processed_data,
            "catalog": config.paths.catalog,
            "logs": config.paths.logs,
        },
    }
