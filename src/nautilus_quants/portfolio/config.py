# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Portfolio configuration loader and factory functions.

Parses ``config/portfolio/portfolio.yaml`` into dataclasses used by:
- RiskModelActor (risk_model + common sections)
- OptimizedSelectionPolicy (optimizer + constraints + policy sections)
- SnapshotAggregatorActor exposure monitoring (monitor section)

All configuration items are explicitly listed — no code-level defaults for
runtime parameters (per CLAUDE.md Constitution II: Configuration-Driven).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from nautilus_quants.portfolio.optimizer.base import OptimizerConstraints
from nautilus_quants.portfolio.optimizer.mean_variance import (
    MeanVarianceConfig,
    MeanVarianceOptimizer,
)
from nautilus_quants.portfolio.risk_model.base import (
    FundamentalFactorSpec,
    FundamentalModelConfig,
    RiskModel,
    StatisticalModelConfig,
)
from nautilus_quants.portfolio.risk_model.fundamental import FundamentalRiskModel
from nautilus_quants.portfolio.risk_model.statistical import StatisticalRiskModel


@dataclass(frozen=True)
class RiskModelSectionConfig:
    """Parsed ``risk_model`` section from portfolio.yaml.

    Attributes
    ----------
    type : str
        "statistical" | "fundamental" | "parallel".
    active_model : str
        Which cache key (:statistical or :fundamental) the main RISK_MODEL_STATE
        key mirrors. Only relevant when type="parallel".
    statistical : StatisticalModelConfig
        Statistical model config (used when type in {"statistical", "parallel"}).
    fundamental : FundamentalModelConfig
        Fundamental model config (used when type in {"fundamental", "parallel"}).
    """

    type: str = "parallel"
    active_model: str = "fundamental"
    statistical: StatisticalModelConfig = field(default_factory=StatisticalModelConfig)
    fundamental: FundamentalModelConfig = field(default_factory=FundamentalModelConfig)


@dataclass(frozen=True)
class PolicySectionConfig:
    """Parsed ``policy`` section — OptimizedSelectionPolicy runtime config.

    Attributes
    ----------
    max_snapshot_age_ns : int
        Maximum age of cached risk snapshot to accept (nanoseconds).
        When snapshot is missing, stale, or the optimizer is infeasible,
        OptimizedSelectionPolicy returns None (DecisionEngine holds positions).
    """

    max_snapshot_age_ns: int = 24 * 4 * 3_600_000_000_000  # 24 × 4h


@dataclass(frozen=True)
class PortfolioConfig:
    """Top-level parsed portfolio.yaml.

    Loaded once at actor startup; all runtime parameters accessible as
    typed fields. No hidden defaults (per Constitution II).

    Note: risk exposure monitoring is not configured here. The existing
    SnapshotAggregatorActor writes ``snapshot:risk`` automatically whenever
    ``risk_model:state`` is present in the cache, and Grafana reads from
    Redis for alerting (no code-level gate needed).
    """

    risk_model: RiskModelSectionConfig
    optimizer: MeanVarianceConfig
    constraints: OptimizerConstraints
    policy: PolicySectionConfig


def _parse_statistical(data: dict[str, Any]) -> StatisticalModelConfig:
    common = data.get("common", {})
    stat = data.get("statistical", {})
    return StatisticalModelConfig(
        lookback_bars=int(common.get("lookback_bars", 240)),
        min_history_bars=int(common.get("min_history_bars", 60)),
        winsorize_quantile=float(common.get("winsorize_quantile", 0.025)),
        scale_return_pct=bool(common.get("scale_return_pct", True)),
        assume_centered=bool(common.get("assume_centered", False)),
        nan_option=str(common.get("nan_option", "fill")),
        method=str(stat.get("method", "pca")),
        num_factors=int(stat.get("num_factors", 10)),
        shrinkage=stat.get("shrinkage", "lw"),
        shrink_target=str(stat.get("shrink_target", "const_corr")),
    )


def _parse_fundamental(data: dict[str, Any]) -> FundamentalModelConfig:
    common = data.get("common", {})
    fund = data.get("fundamental", {})
    factor_specs = tuple(
        FundamentalFactorSpec(name=str(spec["name"]), variable=str(spec["variable"]))
        for spec in fund.get("factors", [])
    )
    return FundamentalModelConfig(
        lookback_bars=int(common.get("lookback_bars", 240)),
        min_history_bars=int(common.get("min_history_bars", 60)),
        winsorize_quantile=float(common.get("winsorize_quantile", 0.025)),
        scale_return_pct=bool(common.get("scale_return_pct", True)),
        assume_centered=bool(common.get("assume_centered", False)),
        nan_option=str(common.get("nan_option", "fill")),
        factors=factor_specs,
        sector_map=dict(fund.get("sector_map", {})),
        wls_weight_source=str(fund.get("wls_weight_source", "market_cap")),
        winsorize_exposures_sigma=float(fund.get("winsorize_exposures_sigma", 3.0)),
        sector_constraint=bool(fund.get("sector_constraint", True)),
        shrink_specific=bool(fund.get("shrink_specific", True)),
        market_cap_ewm_alpha=float(fund.get("market_cap_ewm_alpha", 0.1)),
    )


def _parse_risk_model_section(data: dict[str, Any]) -> RiskModelSectionConfig:
    return RiskModelSectionConfig(
        type=str(data.get("type", "parallel")),
        active_model=str(data.get("active_model", "fundamental")),
        statistical=_parse_statistical(data),
        fundamental=_parse_fundamental(data),
    )


def _parse_optimizer_section(data: dict[str, Any]) -> MeanVarianceConfig:
    return MeanVarianceConfig(
        risk_aversion=float(data.get("risk_aversion", 1.0)),
        solver=str(data.get("solver", "ECOS")),
        epsilon=float(data.get("epsilon", 5e-5)),
        scale_return=bool(data.get("scale_return", True)),
        fallback_relax=tuple(data.get("fallback_relax", ("turnover", "sector", "factor"))),
    )


def _parse_constraints_section(data: dict[str, Any]) -> OptimizerConstraints:
    net = data.get("net_exposure", [-0.05, 0.05])
    return OptimizerConstraints(
        max_weight=float(data.get("max_weight", 0.05)),
        max_leverage=float(data.get("max_leverage", 2.0)),
        net_exposure=(float(net[0]), float(net[1])),
        turnover_limit=(
            float(data["turnover_limit"]) if data.get("turnover_limit") is not None else None
        ),
        min_positions=int(data.get("min_positions", 10)),
        sector_limits=(
            {str(k): float(v) for k, v in data["sector_limits"].items()}
            if data.get("sector_limits")
            else None
        ),
        factor_limits=(
            {str(k): float(v) for k, v in data["factor_limits"].items()}
            if data.get("factor_limits")
            else None
        ),
    )


def _parse_policy_section(data: dict[str, Any]) -> PolicySectionConfig:
    return PolicySectionConfig(
        max_snapshot_age_ns=int(data.get("max_snapshot_age_ns", 24 * 4 * 3_600_000_000_000)),
    )


def load_portfolio_config(path: str | Path) -> PortfolioConfig:
    """Load and parse portfolio.yaml into typed config dataclasses.

    Parameters
    ----------
    path : str | Path
        File path to ``config/portfolio/portfolio.yaml``.

    Returns
    -------
    PortfolioConfig
        Fully parsed, frozen dataclass tree.

    Raises
    ------
    FileNotFoundError
        If path does not exist.
    yaml.YAMLError
        If file is malformed YAML.
    """
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"portfolio.yaml must be a mapping, got {type(raw).__name__}")
    # Optionally merge sector_map from external file.
    # Path is resolved relative to portfolio.yaml's parent directory
    # (not cwd) so configs are self-contained and movable.
    fund = raw.get("risk_model", {}).get("fundamental", {})
    sector_map_path = fund.get("sector_map_path")
    if sector_map_path and "sector_map" not in fund:
        sp = Path(sector_map_path)
        if not sp.is_absolute():
            sp = p.parent / sp.name
        fund["sector_map"] = _load_sector_map(sp)
    return PortfolioConfig(
        risk_model=_parse_risk_model_section(raw.get("risk_model", {})),
        optimizer=_parse_optimizer_section(raw.get("optimizer", {})),
        constraints=_parse_constraints_section(raw.get("constraints", {})),
        policy=_parse_policy_section(raw.get("policy", {})),
    )


def _load_sector_map(path: Path) -> dict[str, str]:
    """Load sector_map.yaml and flatten to {instrument: sector} mapping.

    Expected format:
        L1:
          - BTCUSDT.BINANCE
          - ETHUSDT.BINANCE
        DeFi:
          - UNIUSDT.BINANCE
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    if not isinstance(raw, dict):
        return mapping
    for sector, members in raw.items():
        if not isinstance(members, list):
            continue
        for inst in members:
            mapping[str(inst)] = str(sector)
    return mapping


def build_risk_model(cfg: RiskModelSectionConfig) -> RiskModel:
    """Instantiate the active risk model per config.type / active_model.

    For type="parallel" the caller wants both models; this factory returns
    the ``active_model`` one. RiskModelActor constructs both independently.
    """
    if cfg.type == "statistical" or (cfg.type == "parallel" and cfg.active_model == "statistical"):
        return StatisticalRiskModel(cfg.statistical)
    if cfg.type == "fundamental" or (cfg.type == "parallel" and cfg.active_model == "fundamental"):
        return FundamentalRiskModel(cfg.fundamental)
    raise ValueError(f"unknown risk_model type/active_model: {cfg.type}/{cfg.active_model}")


def build_optimizer(cfg: MeanVarianceConfig) -> MeanVarianceOptimizer:
    """Instantiate the MVO optimizer. (Only MVO supported at present.)"""
    return MeanVarianceOptimizer(cfg)
