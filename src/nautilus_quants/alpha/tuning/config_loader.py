# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Read the ``tune:`` section from an alpha_mining YAML.

The loader is separate from ``config.py`` to keep the dataclasses free of
YAML-parsing logic — ``config.py`` defines *what* a config looks like,
``config_loader.py`` defines *how* to read one from disk. Mirrors the split
used by ``alpha/analysis/config.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from nautilus_quants.alpha.tuning.config import (
    ALGORITHM_TPE,
    CORRECTION_BONFERRONI,
    CV_METHOD_EXPANDING,
    CandidatesConfig,
    CVConfig,
    DimensionsConfig,
    EligibilityConfig,
    TuneConfig,
)


def load_tune_config(
    path: str | Path,
    *,
    overrides: dict[str, Any] | None = None,
) -> TuneConfig:
    """Parse a tune config from ``path``.

    Accepts either:

    * A file containing a top-level ``tune:`` section (typical
      ``alpha_mining.yaml``), or
    * A standalone YAML whose root **is** the tune section (for tests).

    ``overrides`` is a flat mapping of field-name → value applied after
    parsing. The supported keys mirror the CLI flags that ``alpha tune``
    exposes: ``dimensions.numeric`` / ``dimensions.operators`` /
    ``dimensions.variables`` / ``by_prototype`` / ``trials`` /
    ``register_top_k`` / ``enabled`` etc.
    """
    raw = _read_raw(path)
    tune_section = raw.get("tune", raw)
    return build_tune_config(tune_section, overrides=overrides)


def _read_raw(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Tune config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Tune config at {p} must be a mapping, got {type(raw).__name__}")
    return raw


def build_tune_config(
    raw: dict[str, Any],
    *,
    overrides: dict[str, Any] | None = None,
) -> TuneConfig:
    """Coerce a raw YAML mapping into a ``TuneConfig``."""
    overrides = overrides or {}

    dims_raw = dict(raw.get("dimensions", {}) or {})
    for k in ("numeric", "operators", "variables"):
        override_key = f"dimensions.{k}"
        if override_key in overrides:
            dims_raw[k] = bool(overrides[override_key])
    dimensions = DimensionsConfig(
        numeric=bool(dims_raw.get("numeric", True)),
        operators=bool(dims_raw.get("operators", False)),
        variables=bool(dims_raw.get("variables", False)),
    )

    cand_raw = dict(raw.get("candidates", {}) or {})
    eligibility_raw = dict(cand_raw.get("eligibility", {}) or {})
    eligibility = EligibilityConfig(
        icir_abs_min=float(eligibility_raw.get("icir_abs_min", 0.05)),
        t_stat_nw_abs_min=float(eligibility_raw.get("t_stat_nw_abs_min", 1.5)),
        coverage_min=float(eligibility_raw.get("coverage_min", 0.30)),
        n_samples_min=int(eligibility_raw.get("n_samples_min", 1000)),
        min_valid_periods=int(eligibility_raw.get("min_valid_periods", 1)),
    )
    candidates = CandidatesConfig(
        env=str(cand_raw.get("env", "test")),
        prototype=cand_raw.get("prototype"),
        source=cand_raw.get("source"),
        status=cand_raw.get("status", "candidate"),
        tags=tuple(cand_raw.get("tags", []) or ()),
        eligibility=eligibility,
    )

    cv_raw = dict(raw.get("cv", {}) or {})
    cv = CVConfig(
        method=str(cv_raw.get("method", CV_METHOD_EXPANDING)),
        n_folds=int(cv_raw.get("n_folds", 3)),
        test_ratio=float(cv_raw.get("test_ratio", 0.167)),
        holdout_ratio=float(cv_raw.get("holdout_ratio", 0.167)),
        gap_bars=int(cv_raw.get("gap_bars", 0)),
    )

    charts = tuple(raw.get("charts", []) or ())
    search_overrides = dict(raw.get("search_space_overrides", {}) or {})

    config = TuneConfig(
        enabled=bool(overrides.get("enabled", raw.get("enabled", False))),
        source=str(raw.get("source", "tune")),
        output_dir=str(raw.get("output_dir", "logs/alpha_tune")),
        candidates=candidates,
        dimensions=dimensions,
        by_prototype=bool(overrides.get("by_prototype", raw.get("by_prototype", True))),
        algorithm=str(overrides.get("algorithm", raw.get("algorithm", ALGORITHM_TPE))),
        trials=int(overrides.get("trials", raw.get("trials", 50))),
        n_jobs=int(overrides.get("n_jobs", raw.get("n_jobs", 1))),
        early_stop_patience=int(
            overrides.get(
                "early_stop_patience", raw.get("early_stop_patience", 20)
            )
        ),
        early_stop_min_trials=int(
            overrides.get(
                "early_stop_min_trials", raw.get("early_stop_min_trials", 30)
            )
        ),
        timeout=int(raw.get("timeout", 3600)),
        seed=int(raw.get("seed", 42)),
        register_top_k=int(overrides.get("register_top_k", raw.get("register_top_k", 3))),
        search_space_overrides=search_overrides,
        cv=cv,
        stability_min=float(raw.get("stability_min", 0.5)),
        correction_method=str(raw.get("correction_method", CORRECTION_BONFERRONI)),
        significance_alpha=float(raw.get("significance_alpha", 0.05)),
        charts=charts,
        forward_horizon_bars=int(raw.get("forward_horizon_bars", 1)),
        ic_mean_weight=float(raw.get("ic_mean_weight", 0.0)),
    )
    return config
