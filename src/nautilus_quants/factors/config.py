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
        name: Factor identifier
        expression: Alpha101-style expression string
        description: Human-readable description
        category: Optional category for grouping (momentum, volatility, etc.)
    """
    name: str
    expression: str
    description: str = ""
    category: str = ""


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
    parameters: dict[str, Any] = field(default_factory=dict)
    variables: dict[str, str] = field(default_factory=dict)
    factors: list[FactorDefinition] = field(default_factory=list)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
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
        factors.append(FactorDefinition(
            name=factor_name,
            expression=factor_data.get("expression", ""),
            description=factor_data.get("description", ""),
            category=factor_data.get("category", ""),
        ))
    
    # Parse performance config
    perf_raw = raw_config.get("performance", {})
    performance = PerformanceConfig(
        max_compute_time_ms=perf_raw.get("max_compute_time_ms", 1.0),
        enable_timing=perf_raw.get("enable_timing", True),
        warning_threshold_ms=perf_raw.get("warning_threshold_ms", 0.5),
    )
    
    config = FactorConfig(
        name=name,
        version=version,
        description=description,
        parameters=parameters,
        variables=variables,
        factors=factors,
        performance=performance,
    )
    
    # Validate
    errors = validate_factor_config(config)
    if errors:
        raise ConfigValidationError(
            f"Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )
    
    return config
