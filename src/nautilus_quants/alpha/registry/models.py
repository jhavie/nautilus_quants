# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Data models for the Factor Registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FactorRecord:
    """A single factor's metadata in the registry.

    Attributes
    ----------
    factor_id : str
        Unique identifier (e.g. "alpha001").
    expression : str
        Alpha101-style DSL expression.
    description : str
        Human-readable description.
    category : str
        Barra-style category (momentum / volatility / volume / ...).
    source : str
        Origin of the factor (alpha101 / alpha191 / technical / mined).
    status : str
        Lifecycle state: candidate → active → archived.
    created_at : str
        ISO-8601 creation timestamp.
    updated_at : str
        ISO-8601 last-update timestamp.
    ic_mean : float | None
        Mean information coefficient (populated by Feature 035+).
    icir : float | None
        IC information ratio (populated by Feature 035+).
    score : float | None
        Composite quality score 0-100 (populated by Feature 035+).
    bar_spec : str
        Bar specification used for the latest analysis (e.g. "4h").
    """

    factor_id: str
    expression: str
    description: str = ""
    category: str = ""
    source: str = ""
    status: str = "candidate"
    context_id: str = ""
    created_at: str = ""
    updated_at: str = ""
    ic_mean: float | None = None
    icir: float | None = None
    score: float | None = None
    bar_spec: str = ""


# Valid status values and allowed transitions.
VALID_STATUSES = {"candidate", "active", "archived"}
STATUS_TRANSITIONS: dict[str, set[str]] = {
    "candidate": {"active"},
    "active": {"archived"},
    "archived": {"candidate"},
}


@dataclass(frozen=True)
class FactorVersion:
    """Immutable snapshot of a factor expression at a point in time.

    Attributes
    ----------
    factor_id : str
        Parent factor identifier.
    version : int
        Monotonically increasing version number (starts at 1).
    expression : str
        The expression text for this version.
    reason : str
        Why the expression was changed.
    created_at : str
        ISO-8601 timestamp of this version.
    """

    factor_id: str
    version: int
    expression: str
    reason: str = ""
    created_at: str = ""


@dataclass(frozen=True)
class ConfigContext:
    """Configuration-level context stored alongside factors.

    Preserves variables and parameters from the original factors.yaml
    so that ``export_factors_yaml`` can produce a round-trip-safe YAML.

    Attributes
    ----------
    context_id : str
        Identifier (typically ``metadata.name`` from factors.yaml).
    variables : dict[str, str]
        Reusable variable expressions (e.g. ``{"returns": "delta(close,1)/delay(close,1)"}``).
    parameters : dict[str, Any]
        Global numeric parameters (e.g. ``{"short_window": 24}``).
    metadata : dict[str, str]
        Original YAML metadata section (name, version, description).
    created_at : str
        ISO-8601 creation timestamp.
    """

    context_id: str
    variables: dict[str, str] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)
    created_at: str = ""


@dataclass(frozen=True)
class AnalysisResult:
    """Per-factor, per-bar_spec, per-period analysis metrics.

    Populated by Feature 035 (``alpha analyze --write-registry``).

    Attributes
    ----------
    factor_id : str
        Factor identifier.
    bar_spec : str
        Bar specification (e.g. "4h").
    period : int
        Forward-return period in bars.
    ic_mean : float | None
        Mean information coefficient.
    ic_std : float | None
        Standard deviation of IC.
    icir : float | None
        IC information ratio (ic_mean / ic_std).
    mean_return : float | None
        Mean return spread.
    turnover : float | None
        Factor portfolio turnover.
    analyzed_at : str
        ISO-8601 timestamp of the analysis run.
    """

    factor_id: str
    bar_spec: str
    period: int
    ic_mean: float | None = None
    ic_std: float | None = None
    icir: float | None = None
    mean_return: float | None = None
    turnover: float | None = None
    analyzed_at: str = ""
