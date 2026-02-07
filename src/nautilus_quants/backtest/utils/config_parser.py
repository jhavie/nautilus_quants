"""Configuration parsing utilities for backtest module."""

import copy
from pathlib import Path
from typing import Any

from nautilus_quants.backtest.config import (
    PositionVisualizationConfig,
    QuantStatsConfig,
    ReportConfig,
    TearsheetConfig,
)
from nautilus_quants.backtest.utils.bar_spec import format_bar_spec


def parse_report_config(config_dict: dict) -> ReportConfig | None:
    """Parse report configuration from YAML dict.

    Args:
        config_dict: Full YAML config dictionary

    Returns:
        ReportConfig if report section exists, None otherwise
    """
    report_section = config_dict.get("report")
    if not report_section:
        return None

    # Parse tearsheet config if present
    tearsheet_config = None
    tearsheet_section = report_section.get("tearsheet")
    if tearsheet_section:
        tearsheet_config = TearsheetConfig(
            enabled=tearsheet_section.get("enabled", True),
            title=tearsheet_section.get("title", "Backtest Results"),
            theme=tearsheet_section.get("theme", "plotly_dark"),
            height=tearsheet_section.get("height", 1500),
            show_logo=tearsheet_section.get("show_logo", True),
            include_benchmark=tearsheet_section.get("include_benchmark", False),
            benchmark_name=tearsheet_section.get("benchmark_name", "Benchmark"),
            charts=tearsheet_section.get(
                "charts",
                [
                    "run_info",
                    "stats_table",
                    "equity",
                    "drawdown",
                    "monthly_returns",
                    "distribution",
                    "rolling_sharpe",
                    "yearly_returns",
                ],
            ),
        )

    # Parse quantstats config if present
    quantstats_config = None
    quantstats_section = report_section.get("quantstats")
    if quantstats_section:
        quantstats_config = QuantStatsConfig(
            enabled=quantstats_section.get("enabled", False),
            title=quantstats_section.get("title", "QuantStats Report"),
            benchmark=quantstats_section.get("benchmark"),
            output_format=quantstats_section.get("output_format", ["html"]),
            charts=quantstats_section.get(
                "charts",
                [
                    "returns",
                    "log_returns",
                    "yearly_returns",
                    "monthly_heatmap",
                    "drawdown",
                    "rolling_sharpe",
                    "rolling_volatility",
                ],
            ),
        )

    # Parse position_viz config if present
    position_viz_config = None
    position_viz_section = report_section.get("position_viz")
    if position_viz_section:
        position_viz_config = PositionVisualizationConfig(
            enabled=position_viz_section.get("enabled", True),
            title=position_viz_section.get("title", "Position Timeline"),
            output_subdir=position_viz_section.get("output_subdir", "echarts"),
            chart_height=position_viz_section.get("chart_height", 500),
            interval=position_viz_section.get("interval", "4h"),
            metadata_renderer=position_viz_section.get("metadata_renderer"),
        )

    return ReportConfig(
        output_dir=report_section.get("output_dir", "logs/backtest_runs"),
        formats=report_section.get("formats", ["csv", "html"]),
        tearsheet=tearsheet_config,
        quantstats=quantstats_config,
        position_viz=position_viz_config,
    )


def get_nautilus_config_dict(config_dict: dict) -> dict:
    """Extract only Nautilus-compatible config keys.

    Removes project-specific keys (report, logging) that are not part of
    Nautilus BacktestRunConfig schema.

    Args:
        config_dict: Full YAML config dictionary

    Returns:
        Dict with only Nautilus-compatible keys
    """
    # Keys that Nautilus BacktestRunConfig expects
    nautilus_keys = {"venues", "data", "engine", "batch_size_bytes"}
    return {k: v for k, v in config_dict.items() if k in nautilus_keys}


def extract_data_configs(config_dict: dict) -> list[dict]:
    """Extract instrument_ids and bar_spec from data section.

    Expects catalog format with instrument_ids (plural), catalog_path, and bar_spec.

    Args:
        config_dict: Full YAML config dictionary

    Returns:
        List of dicts with instrument_id and bar_spec for each data source
    """
    data_section = config_dict.get("data", [])
    result = []

    for data_config in data_section:
        instrument_ids = data_config.get("instrument_ids", [])
        catalog_path = data_config.get("catalog_path", "")
        bar_spec = data_config.get("bar_spec", "")

        if not instrument_ids or not catalog_path or not bar_spec:
            continue

        # Convert bar_spec to native format (e.g., "1h" -> "1-HOUR-LAST")
        try:
            native_spec = format_bar_spec(bar_spec, include_source=False)
        except ValueError:
            # Assume already in native format
            native_spec = bar_spec

        for inst_id in instrument_ids:
            bar_type = f"{inst_id}-{native_spec}-EXTERNAL"
            result.append(
                {
                    "instrument_id": inst_id,
                    "bar_spec": native_spec,
                    "bar_type": bar_type,
                }
            )

    return result


def inject_data_configs(config_dict: dict, data_configs: list[dict]) -> dict:
    """Inject data configs into actor and strategy configurations.

    This allows actors and strategies to know which bar types to subscribe to
    without querying the cache (which is empty at on_start time).

    Args:
        config_dict: Full YAML config dictionary
        data_configs: List of extracted data configs

    Returns:
        Modified config dict with injected bar_type info
    """
    if not data_configs:
        return config_dict

    # Deep copy to avoid modifying original
    config_dict = copy.deepcopy(config_dict)

    engine = config_dict.get("engine", {})

    # Inject into actors
    actors = engine.get("actors", [])
    for actor in actors:
        actor_config = actor.get("config", {})
        # Inject bar_types list if not already present
        if "bar_types" not in actor_config:
            actor_config["bar_types"] = [dc["bar_type"] for dc in data_configs]
        actor["config"] = actor_config

    # Inject into strategies
    strategies = engine.get("strategies", [])
    for strategy in strategies:
        strategy_config = strategy.get("config", {})
        # Inject bar_types list if not already present (for multi-instrument strategies)
        if "bar_types" not in strategy_config:
            strategy_config["bar_types"] = [dc["bar_type"] for dc in data_configs]
        # Inject bar_type if not already present and we have data configs (for single-instrument strategies)
        if "bar_type" not in strategy_config and data_configs:
            # Find matching data config by instrument_id
            instrument_id = strategy_config.get("instrument_id")
            for dc in data_configs:
                if dc["instrument_id"] == instrument_id:
                    strategy_config["bar_type"] = dc["bar_type"]
                    break
            else:
                # Default to first data config if no match
                strategy_config["bar_type"] = data_configs[0]["bar_type"]
        strategy["config"] = strategy_config

    return config_dict


def inject_logging_config(config_dict: dict, output_dir: Path) -> dict:
    """Inject logging configuration to write log file to output directory.

    Args:
        config_dict: Full YAML config dictionary
        output_dir: Output directory for log file

    Returns:
        Modified config dict with logging config
    """
    config_dict = copy.deepcopy(config_dict)

    engine = config_dict.get("engine", {})
    logging_config = engine.get("logging", {})

    # Set log file configuration
    logging_config["log_level_file"] = logging_config.get("log_level", "INFO")
    logging_config["log_directory"] = str(output_dir)
    logging_config["log_file_name"] = "nautilus.log"

    engine["logging"] = logging_config
    config_dict["engine"] = engine

    return config_dict
