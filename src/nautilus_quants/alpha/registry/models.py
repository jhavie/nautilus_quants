# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Data models for the Factor Registry v2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── Registry configuration ──


@dataclass(frozen=True)
class RegistryConfig:
    """Configuration for the factor registry persistence layer."""

    env: str = "test"
    db_dir: str = "logs/registry"
    enabled: bool = True


# ── Factor lifecycle ──

VALID_STATUSES = {"candidate", "active", "archived"}
STATUS_TRANSITIONS: dict[str, set[str]] = {
    "candidate": {"active"},
    "active": {"archived"},
    "archived": {"candidate"},
}


@dataclass(frozen=True)
class FactorRecord:
    """A single factor's metadata in the registry.

    factor_id follows the convention ``{source}_{key}``
    (e.g. ``alpha101_alpha044_8h``).
    """

    factor_id: str
    expression: str
    prototype: str = ""
    description: str = ""
    source: str = ""
    status: str = "candidate"
    tags: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    variables: dict[str, str] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""


# ── Config snapshots ──


@dataclass(frozen=True)
class ConfigSnapshot:
    """Universal config snapshot stored in configs_snapshot table.

    ``type`` distinguishes the config kind:
    ``"factors"`` / ``"analysis"`` / ``"backtest"``.
    """

    config_id: str
    type: str
    config_name: str = ""
    config_json: dict[str, Any] = field(default_factory=dict)
    file_path: str = ""
    config_hash: str = ""
    created_at: str = ""


# ── Analysis metrics ──


@dataclass(frozen=True)
class AnalysisMetrics:
    """Per-factor, per-period metrics from alpha analysis.

    Stores all output from ``compute_ic_summary()`` and
    ``compute_all_factor_metrics()``.
    """

    run_id: str
    factor_id: str
    period: str
    # IC statistics (11 columns from compute_ic_summary)
    ic_mean: float | None = None
    ic_std: float | None = None
    icir: float | None = None
    t_stat_ic: float | None = None
    p_value_ic: float | None = None
    t_stat_nw: float | None = None
    p_value_nw: float | None = None
    n_eff: int | None = None
    ic_skew: float | None = None
    ic_kurtosis: float | None = None
    n_samples: int | None = None
    # Signal quality (6 fields from FactorMetricsResult)
    win_rate: float | None = None
    monotonicity: float | None = None
    ic_half_life: float | None = None
    ic_linearity: float | None = None
    ic_ar1: float | None = None
    coverage: float | None = None
    # Portfolio metrics
    mean_return: float | None = None
    turnover: float | None = None
    # Config references (FK → configs_snapshot)
    factor_config_id: str = ""
    analysis_config_id: str = ""
    # Meta
    output_dir: str = ""
    timeframe: str = ""
    created_at: str = ""


# ── Backtest ──


@dataclass(frozen=True)
class BacktestRunRecord:
    """A single backtest run record."""

    backtest_id: str
    config_id: str = ""
    factor_config_id: str = ""
    output_dir: str = ""
    strategy_name: str = ""
    instrument_count: int = 0
    timeframe: str = ""
    started_at: str = ""
    finished_at: str = ""
    duration_seconds: float = 0.0
    total_pnl: float | None = None
    total_pnl_pct: float | None = None
    sharpe_ratio: float | None = None
    max_drawdown: float | None = None
    win_rate: float | None = None
    statistics_json: dict[str, Any] = field(default_factory=dict)
    reports_json: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BacktestFactor:
    """Link between a backtest run and a factor (M:N junction)."""

    backtest_id: str
    factor_id: str
    role: str = "component"
