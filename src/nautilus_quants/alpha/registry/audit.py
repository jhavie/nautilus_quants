# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Factor registry audit — duplicate detection and prototype grouping."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import json
import logging
import re

from dataclasses import dataclass, field

from nautilus_quants.alpha.registry.models import FactorRecord
from nautilus_quants.alpha.registry.repository import FactorRepository
from nautilus_quants.factors.expression.normalize import (
    expression_hash,
    expression_template,
)


# ── Repair data models ──


@dataclass(frozen=True)
class OrphanGroup:
    """A group of metrics that used a different expression than the current one."""

    expression: str
    expression_hash: str
    run_ids: list[str] = field(default_factory=list)
    metric_count: int = 0


@dataclass(frozen=True)
class ConflictReport:
    """Report for a factor with metrics from conflicting expressions."""

    factor_id: str
    current_expression: str
    current_hash: str
    source: str
    orphan_groups: list[OrphanGroup] = field(default_factory=list)


@dataclass(frozen=True)
class RepairAction:
    """A single repair action taken or planned."""

    action: str  # "split" | "delete"
    factor_id: str
    new_factor_id: str
    run_ids: list[str] = field(default_factory=list)
    detail: str = ""

logger = logging.getLogger(__name__)

# ── Prototype lookup tables (builtin sources) ─────────────────────────────

def _build_prototype_map() -> dict[str, str]:
    """Build {factor_name: prototype} from all builtin factor libraries."""
    mapping: dict[str, str] = {}
    try:
        from nautilus_quants.factors.builtin.ta_factors import TA_FACTORS
        for name, meta in TA_FACTORS.items():
            mapping[name] = meta.get("prototype", name)
    except ImportError:
        pass
    return mapping

_PROTOTYPE_MAP: dict[str, str] | None = None

def _get_prototype_map() -> dict[str, str]:
    global _PROTOTYPE_MAP
    if _PROTOTYPE_MAP is None:
        _PROTOTYPE_MAP = _build_prototype_map()
    return _PROTOTYPE_MAP


# Pattern: strip trailing digits from names like MA10, CORD60, sma_20
_TRAILING_DIGITS_RE = re.compile(r"^(.+?)_?(\d+)$")


def find_expression_duplicates(
    repo: FactorRepository,
) -> list[list[FactorRecord]]:
    """Find groups of factors with identical expressions.

    Returns a list of groups (each group has ≥2 factors with the same
    normalized expression hash).
    """
    factors = repo.list_factors()
    by_hash: dict[str, list[FactorRecord]] = defaultdict(list)
    for f in factors:
        h = expression_hash(f.expression)
        by_hash[h].append(f)
    return [group for group in by_hash.values() if len(group) >= 2]


def suggest_prototype_groups(
    repo: FactorRepository,
) -> dict[str, list[tuple[str, dict[str, float]]]]:
    """Suggest prototype groupings based on expression templates.

    Returns ``{template: [(factor_id, {p0: val, ...}), ...]}``
    for groups with ≥2 members (i.e. factors that share the same
    expression structure with different numeric parameters).
    """
    factors = repo.list_factors()
    by_template: dict[str, list[tuple[str, dict[str, float]]]] = defaultdict(
        list,
    )
    for f in factors:
        try:
            tmpl, vals = expression_template(f.expression)
        except Exception:
            continue
        params = {f"p{i}": v for i, v in enumerate(vals)}
        by_template[tmpl].append((f.factor_id, params))
    return {
        tmpl: members
        for tmpl, members in by_template.items()
        if len(members) >= 2
    }


def dedup_factors(
    repo: FactorRepository,
    keep_source: str | None = None,
    dry_run: bool = True,
) -> list[tuple[str, str]]:
    """Remove duplicate factors (by expression hash).

    Within each duplicate group, keeps one factor:
    - If *keep_source* is given, prefer that source.
    - Otherwise, keep the lexicographically smallest factor_id.

    Returns list of ``(removed_factor_id, kept_factor_id)`` pairs.

    When *dry_run* is True (default), nothing is actually deleted.
    """
    groups = find_expression_duplicates(repo)
    removed: list[tuple[str, str]] = []

    for group in groups:
        # Pick the keeper
        if keep_source:
            preferred = [f for f in group if f.source == keep_source]
            rest = [f for f in group if f.source != keep_source]
            if preferred:
                preferred.sort(key=lambda f: f.factor_id)
                keeper = preferred[0]
                to_remove = preferred[1:] + rest
            else:
                group.sort(key=lambda f: f.factor_id)
                keeper = group[0]
                to_remove = group[1:]
        else:
            group.sort(key=lambda f: f.factor_id)
            keeper = group[0]
            to_remove = group[1:]

        for f in to_remove:
            removed.append((f.factor_id, keeper.factor_id))
            if not dry_run:
                repo.delete_factor(f.factor_id)

    return removed


def _build_template_proto_map(
    factors: list[FactorRecord],
) -> dict[str, str]:
    """Build {factor_id: inferred_prototype} from expression templates.

    Factors with the same expression template (same structure, different
    numbers) share a prototype.  The prototype name is the longest common
    prefix of the bare factor names in the group.
    """
    # template → [(factor_id, bare_name), ...]
    by_template: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for f in factors:
        try:
            tmpl, _ = expression_template(f.expression)
        except Exception:
            continue
        bare = f.factor_id
        if f.source and bare.startswith(f"{f.source}_"):
            bare = bare[len(f.source) + 1:]
        by_template[tmpl].append((f.factor_id, bare))

    result: dict[str, str] = {}
    for tmpl, members in by_template.items():
        if len(members) < 2:
            continue
        # Derive prototype = longest common prefix of bare names,
        # then strip trailing _ or digits
        bare_names = [m[1] for m in members]
        prefix = _common_prefix(bare_names).rstrip("_")
        if not prefix:
            continue
        for fid, _ in members:
            result[fid] = prefix
    return result


def _common_prefix(strings: list[str]) -> str:
    """Longest common prefix of a list of strings."""
    if not strings:
        return ""
    prefix = strings[0]
    for s in strings[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix


def backfill_factors(
    repo: FactorRepository,
    dry_run: bool = True,
) -> dict[str, int]:
    """Backfill expression_hash, prototype, and parameters for all factors.

    For each factor:
    1. Compute and store ``expression_hash``.
    2. Fix prototype via expression template grouping: factors with the
       same template share a prototype (derived from common name prefix).
       Falls back to builtin map (``ta_factors.py``).
    3. Compute ``expression_template`` and store extracted numbers
       as ``{p0: val, p1: val, ...}`` in parameters.

    Returns ``{"hash": N, "prototype": N, "parameters": N}`` counts.
    """
    proto_map = _get_prototype_map()
    factors = repo.list_factors()
    counts = {"hash": 0, "prototype": 0, "parameters": 0}

    # Build prototype map from expression templates (primary method)
    tmpl_proto_map = _build_template_proto_map(factors)

    for f in factors:
        updates: dict[str, Any] = {}

        # 1. expression_hash
        try:
            h = expression_hash(f.expression)
        except Exception:
            h = ""
        row = repo._db.fetch_one(
            "SELECT expression_hash FROM factors WHERE factor_id = ?",
            [f.factor_id],
        )
        stored_hash = (row[0] or "") if row else ""
        if h and h != stored_hash:
            updates["expression_hash"] = h
            counts["hash"] += 1

        # 2. prototype — from template grouping, then builtin map
        bare_name = f.factor_id
        if f.source and bare_name.startswith(f"{f.source}_"):
            bare_name = bare_name[len(f.source) + 1:]

        correct_proto = (
            tmpl_proto_map.get(f.factor_id)
            or proto_map.get(bare_name)
        )
        if correct_proto and f.prototype != correct_proto:
            updates["prototype"] = correct_proto
            counts["prototype"] += 1

        # 3. parameters from expression_template
        try:
            _, vals = expression_template(f.expression)
            if vals:
                extracted = {
                    f"p{i}": _export_num(v) for i, v in enumerate(vals)
                }
                merged = {
                    k: v for k, v in f.parameters.items()
                    if not k.startswith("p") or not k[1:].isdigit()
                }
                merged.update(extracted)
                if merged != f.parameters:
                    updates["parameters"] = merged
                    counts["parameters"] += 1
        except Exception:
            pass

        # Apply updates
        if updates and not dry_run:
            set_clauses = []
            params_list: list[Any] = []
            for col, val in updates.items():
                set_clauses.append(f"{col} = ?")
                if col == "parameters":
                    params_list.append(json.dumps(val))
                else:
                    params_list.append(val)
            params_list.append(f.factor_id)
            repo._db.execute(
                f"UPDATE factors SET {', '.join(set_clauses)} "
                f"WHERE factor_id = ?",
                params_list,
            )

    return counts


def _export_num(v: float) -> int | float:
    if v == int(v) and abs(v) < 1e15:
        return int(v)
    return v


# ── Conflict detection and repair ────────────────────────────────────────


def find_conflicting_factors(
    repo: FactorRepository,
) -> list[ConflictReport]:
    """Find factors with metrics from conflicting expressions.

    For each factor, checks whether all metrics' ``factor_config_id``
    reference config snapshots that contain the same expression.
    Returns a report for each factor with orphaned metrics.
    """
    # Find factors that have metrics from multiple factor_config_ids
    rows = repo._db.fetch_all(
        "SELECT factor_id, COUNT(DISTINCT factor_config_id) AS n "
        "FROM alpha_analysis_metrics "
        "GROUP BY factor_id "
        "HAVING n > 1"
    )
    if not rows:
        return []

    reports: list[ConflictReport] = []
    for fid, _ in rows:
        factor = repo.get_factor(fid)
        if factor is None:
            continue

        current_hash = factor.expression_hash
        if not current_hash:
            try:
                current_hash = expression_hash(factor.expression)
            except Exception:
                continue

        # Get all (run_id, factor_config_id) pairs
        metrics_rows = repo._db.fetch_all(
            "SELECT DISTINCT run_id, factor_config_id "
            "FROM alpha_analysis_metrics WHERE factor_id = ?",
            [fid],
        )

        # Strip source prefix to get the bare name
        bare_name = fid
        if factor.source and bare_name.startswith(f"{factor.source}_"):
            bare_name = bare_name[len(factor.source) + 1:]

        # Group by expression hash
        orphan_by_hash: dict[str, list[str]] = defaultdict(list)
        for run_id, cfg_id in metrics_rows:
            cfg = repo.get_config_snapshot(cfg_id)
            if cfg is None:
                continue
            factors_dict = cfg.config_json.get("factors", {})
            if bare_name not in factors_dict:
                continue
            expr = factors_dict[bare_name].get("expression", "")
            try:
                h = expression_hash(expr)
            except Exception:
                continue
            if h != current_hash:
                orphan_by_hash[h].append(run_id)

        if not orphan_by_hash:
            continue

        orphan_groups = []
        for h, run_ids in orphan_by_hash.items():
            # Get expression from any config
            sample_run = run_ids[0]
            sample_cfg_id = next(
                cfg_id for rid, cfg_id in metrics_rows if rid == sample_run
            )
            cfg = repo.get_config_snapshot(sample_cfg_id)
            expr = cfg.config_json["factors"][bare_name]["expression"]
            n_metrics = repo._db.fetch_one(
                "SELECT COUNT(*) FROM alpha_analysis_metrics "
                "WHERE factor_id = ? AND run_id IN "
                f"({','.join('?' for _ in run_ids)})",
                [fid, *run_ids],
            )[0]
            orphan_groups.append(OrphanGroup(
                expression=expr,
                expression_hash=h,
                run_ids=run_ids,
                metric_count=n_metrics,
            ))

        reports.append(ConflictReport(
            factor_id=fid,
            current_expression=factor.expression,
            current_hash=current_hash,
            source=factor.source,
            orphan_groups=orphan_groups,
        ))

    return reports


def repair_factors(
    repo: FactorRepository,
    dry_run: bool = True,
) -> list[RepairAction]:
    """Repair factors with metrics from conflicting expressions.

    For each orphaned expression group:
    - If a factor with the same expression already exists → merge metrics
      into that factor.
    - Otherwise → create a new factor ``{name}_{hash[:8]}`` and reassign
      metrics to it.

    Returns a list of actions taken (or planned if dry_run).
    """
    conflicts = find_conflicting_factors(repo)
    actions: list[RepairAction] = []

    for report in conflicts:
        for orphan in report.orphan_groups:
            # Prefer existing factor with the same expression
            existing_match = repo.find_by_expression_hash(
                orphan.expression_hash,
            )
            if (
                existing_match is not None
                and existing_match.factor_id != report.factor_id
            ):
                target_fid = existing_match.factor_id
                action_type = "merge"
            else:
                target_fid = (
                    f"{report.factor_id}"
                    f"_{orphan.expression_hash[:8]}"
                )
                action_type = "split"

            actions.append(RepairAction(
                action=action_type,
                factor_id=report.factor_id,
                new_factor_id=target_fid,
                run_ids=orphan.run_ids,
                detail=orphan.expression[:80],
            ))
            if not dry_run:
                if action_type == "split":
                    # Create new factor for orphaned expression
                    existing = repo.get_factor(report.factor_id)
                    new_record = FactorRecord(
                        factor_id=target_fid,
                        expression=orphan.expression,
                        expression_hash=orphan.expression_hash,
                        source=existing.source if existing else "",
                        status="candidate",
                        tags=existing.tags if existing else [],
                        variables=(
                            existing.variables if existing else {}
                        ),
                    )
                    repo.upsert_factor(new_record)
                # Reassign metrics to target (new or existing)
                for run_id in orphan.run_ids:
                    # Delete conflicting rows in target to avoid PK violation
                    repo._db.execute(
                        "DELETE FROM alpha_analysis_metrics "
                        "WHERE factor_id = ? AND run_id = ? AND period IN "
                        "(SELECT period FROM alpha_analysis_metrics "
                        " WHERE factor_id = ? AND run_id = ?)",
                        [target_fid, run_id, report.factor_id, run_id],
                    )
                    repo._db.execute(
                        "UPDATE alpha_analysis_metrics "
                        "SET factor_id = ? "
                        "WHERE factor_id = ? AND run_id = ?",
                        [target_fid, report.factor_id, run_id],
                    )

    return actions
