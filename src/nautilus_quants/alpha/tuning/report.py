# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Text + chart artefacts for a tune run.

The CLI writes one ``run_summary.json``, one ``tune_result.csv`` and a
per-prototype / per-factor subdirectory with JSON + optional matplotlib
figures. Chart generation is gated on ``TuneConfig.charts``: when the list
is empty (the default) we skip matplotlib entirely, which keeps batch tune
runs fast and reproducible.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Mapping

from nautilus_quants.alpha.tuning.config import TuneConfig, TuneResult
from nautilus_quants.alpha.tuning.variant_registration import RegistrationSummary

logger = logging.getLogger(__name__)

# Chart kinds the YAML's ``charts:`` list may enable.
CHART_OPTIMIZATION_HISTORY = "optimization_history"
CHART_PARAMETER_IMPORTANCE = "parameter_importance"
CHART_PARAMETER_LANDSCAPE = "parameter_landscape"
CHART_OPERATOR_COMPARISON = "operator_comparison"
CHART_VARIABLE_COMPARISON = "variable_comparison"

KNOWN_CHARTS = frozenset(
    {
        CHART_OPTIMIZATION_HISTORY,
        CHART_PARAMETER_IMPORTANCE,
        CHART_PARAMETER_LANDSCAPE,
        CHART_OPERATOR_COMPARISON,
        CHART_VARIABLE_COMPARISON,
    }
)


# ── Path helpers ───────────────────────────────────────────────────────────


def build_run_dir(output_dir: str | Path, run_id: str) -> Path:
    root = Path(output_dir) / run_id
    root.mkdir(parents=True, exist_ok=True)
    return root


def build_factor_dir(run_dir: Path, label: str, index: int | None = None) -> Path:
    """Nested directory per prototype / factor.

    ``label`` is either a prototype name (``proto_001_alpha044``) or a
    sanitised factor id (``factor_alpha101_alpha044_8h``). When ``index`` is
    provided we left-pad it (``proto_NNN_...``) to keep directory listings
    stable.
    """
    safe = label.replace("/", "_").replace(" ", "_")
    if index is not None:
        dirname = f"proto_{index:03d}_{safe}"
    else:
        dirname = f"factor_{safe}"
    path = run_dir / dirname
    path.mkdir(parents=True, exist_ok=True)
    return path


# ── Serialisation ─────────────────────────────────────────────────────────


def _tune_result_to_dict(result: TuneResult) -> dict:
    """``asdict`` with tuples coerced to lists for JSON-safety."""

    def _fix(obj):
        if isinstance(obj, tuple):
            return [_fix(x) for x in obj]
        if isinstance(obj, list):
            return [_fix(x) for x in obj]
        if isinstance(obj, dict):
            return {k: _fix(v) for k, v in obj.items()}
        return obj

    return _fix(asdict(result))


def write_factor_artefacts(
    factor_dir: Path,
    tune_result: TuneResult,
    registration: RegistrationSummary | None = None,
) -> None:
    """Write per-factor JSON artefacts.

    Files produced (always):
      - ``trials.json`` — top_k trial list
      - ``best_params.yaml`` — lightweight YAML of the winning config
      - ``holdout_metrics.json`` — walk-forward validation figures
      - ``tune_result.json`` — full ``TuneResult`` dump
      - ``registration_summary.json`` — which variants were upserted
    """
    factor_dir.mkdir(parents=True, exist_ok=True)

    # trials.json
    trials_payload = [
        {
            "trial_number": t.trial_number,
            "expression": t.expression,
            "params": t.params,
            "cv_icir": list(t.cv_icir),
            "mean_icir": t.mean_icir,
            "objective_value": t.objective_value,
        }
        for t in tune_result.top_k
    ]
    (factor_dir / "trials.json").write_text(json.dumps(trials_payload, indent=2, ensure_ascii=False))

    # best_params.yaml (hand-rolled, keep dependency on yaml out of the hot path)
    best_lines = ["best_expression: |"]
    best_lines.append(f"  {tune_result.best_expression}")
    best_lines.append("best_params:")
    for name, value in tune_result.best_params.items():
        if isinstance(value, str):
            best_lines.append(f"  {name}: '{value}'")
        else:
            best_lines.append(f"  {name}: {value}")
    best_lines.append(f"best_icir_cv: {tune_result.best_icir_cv}")
    (factor_dir / "best_params.yaml").write_text("\n".join(best_lines) + "\n")

    # holdout_metrics.json
    (factor_dir / "holdout_metrics.json").write_text(
        json.dumps(
            {
                "holdout_icir": tune_result.holdout_icir,
                "holdout_t_stat_nw": tune_result.holdout_t_stat_nw,
                "holdout_p_value_nw": tune_result.holdout_p_value_nw,
                "stability_score": tune_result.stability_score,
                "adjusted_p_value": tune_result.adjusted_p_value,
                "n_trials": tune_result.n_trials,
                "n_pruned": tune_result.n_pruned,
            },
            indent=2,
        )
    )

    # Full dump.
    (factor_dir / "tune_result.json").write_text(json.dumps(_tune_result_to_dict(tune_result), indent=2, ensure_ascii=False))

    if registration is not None:
        payload = {
            "source_factor_id": registration.source_factor_id,
            "n_registered": registration.n_registered,
            "n_updated": registration.n_updated,
            "n_skipped": registration.n_skipped,
            "variants": [
                {
                    "factor_id": v.factor_id,
                    "expression": v.expression,
                    "outcome": v.outcome,
                    "status": v.status,
                }
                for v in registration.variants
            ],
        }
        (factor_dir / "registration_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))


# ── Charts (optional) ──────────────────────────────────────────────────────


def generate_charts(
    factor_dir: Path,
    tune_result: TuneResult,
    requested: Iterable[str],
) -> list[Path]:
    """Render the subset of charts listed in ``requested``.

    Silently skips unknown chart names. Failures are logged but don't raise —
    charts are auxiliary, never block a tune run.
    """
    requested_set = {c for c in requested if c in KNOWN_CHARTS}
    if not requested_set:
        return []

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover — optional dependency
        logger.debug("matplotlib unavailable; skipping tune charts")
        return []

    charts_dir = factor_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if CHART_OPTIMIZATION_HISTORY in requested_set:
        path = charts_dir / "optimization_history.png"
        _plot_optimization_history(tune_result, path, plt)
        if path.exists():
            written.append(path)

    if CHART_PARAMETER_IMPORTANCE in requested_set:
        path = charts_dir / "parameter_importance.png"
        _plot_parameter_importance(tune_result, path, plt)
        if path.exists():
            written.append(path)

    if CHART_OPERATOR_COMPARISON in requested_set:
        path = charts_dir / "operator_comparison.png"
        _plot_comparison(
            tune_result.operator_comparison,
            "Operator comparison",
            path,
            plt,
        )
        if path.exists():
            written.append(path)

    if CHART_VARIABLE_COMPARISON in requested_set:
        path = charts_dir / "variable_comparison.png"
        _plot_comparison(
            tune_result.variable_comparison,
            "Variable comparison",
            path,
            plt,
        )
        if path.exists():
            written.append(path)

    if CHART_PARAMETER_LANDSCAPE in requested_set:
        path = charts_dir / "parameter_landscape.png"
        _plot_parameter_landscape(tune_result, path, plt)
        if path.exists():
            written.append(path)

    return written


def _plot_optimization_history(
    tune_result: TuneResult,
    out: Path,
    plt,
) -> None:
    if not tune_result.top_k:
        return
    ordered = sorted(tune_result.top_k, key=lambda t: t.trial_number)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    xs = [t.trial_number for t in ordered]
    ys = [t.objective_value for t in ordered]
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Objective (|ICIR|)")
    ax.set_title("Optimization history (top_k)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _plot_parameter_importance(
    tune_result: TuneResult,
    out: Path,
    plt,
) -> None:
    importance = tune_result.param_importance
    if not importance:
        return
    items = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in items]
    values = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(6, max(2.5, 0.3 * len(items))))
    ax.barh(names, values, color="#4C72B0")
    ax.invert_yaxis()
    ax.set_xlabel("Relative importance")
    ax.set_title("Parameter importance")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _plot_comparison(
    comparison: Mapping[str, Mapping[str, float]] | None,
    title: str,
    out: Path,
    plt,
) -> None:
    if not comparison:
        return
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for slot_id, per_option in comparison.items():
        if not per_option:
            continue
        items = sorted(per_option.items(), key=lambda kv: kv[1], reverse=True)
        names = [k for k, _ in items]
        values = [v for _, v in items]
        ax.plot(names, values, marker="o", label=slot_id)
    ax.set_ylabel("Best |ICIR| observed")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _plot_parameter_landscape(
    tune_result: TuneResult,
    out: Path,
    plt,
) -> None:
    # Only meaningful when the top_k trials cover a 2D grid in ``params``.
    params_by_name: dict[str, list[float]] = {}
    scores: list[float] = []
    for trial in tune_result.top_k:
        numeric_params = {k: v for k, v in trial.params.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        for k, v in numeric_params.items():
            params_by_name.setdefault(k, []).append(float(v))
        scores.append(trial.objective_value)
    if len(params_by_name) < 2 or not scores:
        return
    first_two = list(params_by_name.keys())[:2]
    xs = params_by_name[first_two[0]]
    ys = params_by_name[first_two[1]]
    if len(xs) != len(ys) or len(xs) != len(scores):
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(xs, ys, c=scores, cmap="viridis", s=60)
    ax.set_xlabel(first_two[0])
    ax.set_ylabel(first_two[1])
    ax.set_title("Parameter landscape (top_k)")
    fig.colorbar(sc, ax=ax, label="|ICIR|")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


# ── Run-level artefacts ────────────────────────────────────────────────────


def write_run_summary(
    run_dir: Path,
    *,
    tune_config: TuneConfig,
    results: list[tuple[str, TuneResult, RegistrationSummary | None]],
) -> Path:
    """Write the top-level ``run_summary.json`` + CSV table.

    Each entry of ``results`` is ``(label, TuneResult, RegistrationSummary)``
    where ``label`` is the prototype or factor_id.
    """
    rows: list[dict] = []
    for label, result, registration in results:
        rows.append(
            {
                "label": label,
                "template": result.template,
                "best_expression": result.best_expression,
                "best_icir_cv": result.best_icir_cv,
                "holdout_icir": result.holdout_icir,
                "holdout_t_stat_nw": result.holdout_t_stat_nw,
                "holdout_p_value_nw": result.holdout_p_value_nw,
                "stability_score": result.stability_score,
                "adjusted_p_value": result.adjusted_p_value,
                "n_trials": result.n_trials,
                "n_pruned": result.n_pruned,
                "n_registered": registration.n_registered if registration else 0,
                "n_skipped": registration.n_skipped if registration else 0,
                "registered_factor_ids": (list(registration.factor_ids) if registration else []),
            }
        )

    summary_path = run_dir / "run_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "tune_config": _tune_config_to_dict(tune_config),
                "n_groups": len(results),
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    # Minimal CSV — columns ordered for stable diffs.
    csv_lines = [
        ",".join(
            [
                "label",
                "best_expression",
                "best_icir_cv",
                "holdout_icir",
                "stability_score",
                "n_trials",
                "n_registered",
            ]
        )
    ]
    for row in rows:
        csv_lines.append(
            ",".join(
                [
                    _csv_escape(row["label"]),
                    _csv_escape(row["best_expression"]),
                    str(row["best_icir_cv"]),
                    str(row["holdout_icir"]),
                    str(row["stability_score"]),
                    str(row["n_trials"]),
                    str(row["n_registered"]),
                ]
            )
        )
    (run_dir / "tune_result.csv").write_text("\n".join(csv_lines) + "\n")
    return summary_path


def _tune_config_to_dict(cfg: TuneConfig) -> dict:
    """Flatten TuneConfig for JSON serialisation."""
    from dataclasses import asdict

    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()}


def _csv_escape(value: str) -> str:
    if value is None:
        return ""
    text = str(value)
    if any(ch in text for ch in [",", "\n", '"']):
        text = '"' + text.replace('"', '""') + '"'
    return text
