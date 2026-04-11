# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Factor Framework Configuration.

This module handles loading and validation of factor configurations
from YAML files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class FactorDefinition:
    """
    Definition of a single factor.

    Attributes:
        name: Factor key in YAML (e.g. "alpha044_8h")
        expression: Alpha101-style expression string
        description: Human-readable description
        tags: Labels for grouping (replaces category). E.g. ["reversal", "volume"]
        prototype: Groups parameter variants of the same base factor
    """
    name: str
    expression: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    prototype: str = ""


@dataclass(frozen=True)
class CompositeConfig:
    """Declarative composite factor configuration.

    Attributes:
        name: Output factor name (default "composite")
        transform: Transform applied before weighting (normalize/cs_rank/cs_zscore/raw)
        nan_policy: How to handle NaN in composite (strict/fill_neutral)
        weights: Factor name → weight mapping
    """
    name: str = "composite"
    transform: str = "normalize"
    nan_policy: str = "strict"
    weights: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PerformanceConfig:
    """
    Performance monitoring configuration.
    
    Attributes:
        max_compute_time_ms: Maximum allowed computation time per bar
        enable_timing: Whether to log timing information
        warning_threshold_ms: Threshold for warning logs
    """
    max_compute_time_ms: float = 1.0
    enable_timing: bool = True
    warning_threshold_ms: float = 0.5


@dataclass
class FactorConfig:
    """
    Complete factor framework configuration.
    
    Attributes:
        name: Configuration name
        version: Configuration version
        description: Optional description
        parameters: Global parameters referenced in expressions
        variables: Reusable variable definitions (expressions)
        factors: List of factor definitions
        performance: Performance monitoring settings
    """
    name: str = "default"
    version: str = "1.0"
    description: str = ""
    source: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    variables: dict[str, str] = field(default_factory=dict)
    factors: list[FactorDefinition] = field(default_factory=list)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    _pipeline: list[FactorDefinition] = field(default_factory=list)

    @property
    def all_factors(self) -> list[FactorDefinition]:
        """All factors: base (factors) + auto-generated (composite/derived)."""
        return self.factors + self._pipeline

    def get_parameter(self, name: str, default: Any = None) -> Any:
        """Get a parameter value by name."""
        return self.parameters.get(name, default)
    
    def get_variable(self, name: str) -> str | None:
        """Get a variable expression by name."""
        return self.variables.get(name)
    
    def get_factor(self, name: str) -> FactorDefinition | None:
        """Get a factor definition by name."""
        for factor in self.factors:
            if factor.name == name:
                return factor
        return None


def generate_factor_id(source: str, key: str) -> str:
    """Generate a factor_id from source prefix and YAML key.

    Examples:
        >>> generate_factor_id("alpha101", "alpha044_8h")
        'alpha101_alpha044_8h'
        >>> generate_factor_id("", "sma_60")
        'sma_60'
    """
    if source:
        return f"{source}_{key}"
    return key


_TRANSFORM_TEMPLATES: dict[str, str] = {
    "normalize": "normalize({factor}, true, 0)",
    "cs_rank": "cs_rank({factor})",
    "cs_zscore": "cs_zscore({factor})",
}

_TRANSFORM_SUFFIXES: dict[str, str] = {
    "normalize": "_norm",
    "cs_rank": "_ranked",
    "cs_zscore": "_zscored",
}

_NEUTRAL_VALUES: dict[str, float] = {
    "normalize": 0.0,
    "cs_zscore": 0.0,
    "cs_rank": 0.5,
    "raw": 0.0,
}


def _build_composite_pipeline(
    raw: dict[str, Any],
) -> list[FactorDefinition]:
    """Generate pipeline FactorDefinitions from a composite config."""
    weights: dict[str, float] = raw.get("weights", {})
    if not weights:
        return []

    transform = raw.get("transform", "normalize")
    nan_policy = raw.get("nan_policy", "strict")
    comp_name = raw.get("name", "composite")

    pipeline: list[FactorDefinition] = []
    terms: list[str] = []

    if transform == "raw":
        # No intermediate factors
        for factor_name, weight in weights.items():
            if nan_policy == "fill_neutral":
                neutral = _NEUTRAL_VALUES.get(transform, 0.0)
                filled_name = f"{factor_name}_filled"
                pipeline.append(FactorDefinition(
                    name=filled_name,
                    expression=f"fill_nan({factor_name}, {neutral})",
                ))
                terms.append(f"{weight} * {filled_name}")
            else:
                terms.append(f"{weight} * {factor_name}")
    else:
        template = _TRANSFORM_TEMPLATES.get(transform)
        suffix = _TRANSFORM_SUFFIXES.get(transform, f"_{transform}")
        if template is None:
            template = f"{transform}({{factor}})"

        for factor_name, weight in weights.items():
            derived_name = f"{factor_name}{suffix}"
            pipeline.append(FactorDefinition(
                name=derived_name,
                expression=template.format(factor=factor_name),
            ))
            if nan_policy == "fill_neutral":
                neutral = _NEUTRAL_VALUES.get(transform, 0.0)
                filled_name = f"{factor_name}_filled"
                pipeline.append(FactorDefinition(
                    name=filled_name,
                    expression=f"fill_nan({derived_name}, {neutral})",
                ))
                terms.append(f"{weight} * {filled_name}")
            else:
                terms.append(f"{weight} * {derived_name}")

    composite_expr = " + ".join(terms)
    pipeline.append(FactorDefinition(
        name=comp_name,
        expression=composite_expr,
    ))
    return pipeline


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


def validate_factor_config(config: FactorConfig) -> list[str]:
    """
    Validate a factor configuration.
    
    Args:
        config: The configuration to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors: list[str] = []
    
    # Check for empty factors
    if not config.factors:
        # This is a warning, not an error - empty config is valid
        pass
    
    # Validate each factor
    seen_names: set[str] = set()
    for factor in config.factors:
        # Check for duplicate names
        if factor.name in seen_names:
            errors.append(f"Duplicate factor name: {factor.name}")
        seen_names.add(factor.name)
        
        # Check for empty expression
        if not factor.expression.strip():
            errors.append(f"Factor '{factor.name}' has empty expression")
        
        # Check for invalid factor names (must be valid Python identifier)
        if not factor.name.isidentifier():
            errors.append(
                f"Factor name '{factor.name}' is not a valid identifier"
            )
    
    # Validate variable names
    for var_name, var_expr in config.variables.items():
        if not var_name.isidentifier():
            errors.append(
                f"Variable name '{var_name}' is not a valid identifier"
            )
        if not var_expr.strip():
            errors.append(f"Variable '{var_name}' has empty expression")
    
    # Validate performance config
    if config.performance.max_compute_time_ms <= 0:
        errors.append("max_compute_time_ms must be positive")
    if config.performance.warning_threshold_ms < 0:
        errors.append("warning_threshold_ms must be non-negative")
    
    return errors


def load_factor_config(path: str | Path) -> FactorConfig:
    """
    Load factor configuration from a YAML file.
    
    Args:
        path: Path to the YAML configuration file
        
    Returns:
        Parsed and validated FactorConfig
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        ConfigValidationError: If the config is invalid
        yaml.YAMLError: If the YAML is malformed
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)
    
    if raw_config is None:
        raw_config = {}
    
    # Parse metadata
    metadata = raw_config.get("metadata", {})
    name = metadata.get("name", "default")
    version = metadata.get("version", "1.0")
    description = metadata.get("description", "")
    source = metadata.get("source", "")

    # Parse parameters
    parameters = raw_config.get("parameters", {})

    # Parse variables
    variables = raw_config.get("variables", {})

    # Parse factors
    factors_raw = raw_config.get("factors", {})
    factors: list[FactorDefinition] = []
    for factor_name, factor_data in factors_raw.items():
        if factor_data is None:
            continue
        raw_tags = factor_data.get("tags", [])
        # Backward compat: migrate category → tags if tags empty
        if not raw_tags and factor_data.get("category"):
            raw_tags = [factor_data["category"]]
        factors.append(FactorDefinition(
            name=factor_name,
            expression=factor_data.get("expression", ""),
            description=factor_data.get("description", ""),
            tags=raw_tags,
            prototype=factor_data.get("prototype", ""),
        ))
    
    # Parse performance config
    perf_raw = raw_config.get("performance", {})
    performance = PerformanceConfig(
        max_compute_time_ms=perf_raw.get("max_compute_time_ms", 1.0),
        enable_timing=perf_raw.get("enable_timing", True),
        warning_threshold_ms=perf_raw.get("warning_threshold_ms", 0.5),
    )
    
    # Parse composite config → auto-generate pipeline factors
    pipeline: list[FactorDefinition] = []
    composite_raw = raw_config.get("composite")
    if composite_raw and isinstance(composite_raw, dict):
        pipeline = _build_composite_pipeline(composite_raw)

    config = FactorConfig(
        name=name,
        version=version,
        description=description,
        source=source,
        parameters=parameters,
        variables=variables,
        factors=factors,
        performance=performance,
        _pipeline=pipeline,
    )
    
    # Validate
    errors = validate_factor_config(config)
    if errors:
        raise ConfigValidationError(
            f"Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )
    
    return config
