# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Generate a quantitative IC / ICIR uplift report for tuned variants.

For every factor with ``source = 'alpha_tune_sanapi'`` in the registry, find
its source factor (by stripping the ``_tuneN_<hash>`` suffix from
``factor_id``), pair their per-period metrics, and emit a markdown report
covering:

- aggregate stats (mean / median uplift, % improved, ...)
- top winners + losers
- one row per source factor showing original vs best variant

Usage:
    python scripts/tune_uplift_report.py [output_path.md]

Defaults to writing ``logs/alpha_tune_sanapi/uplift_report.md``.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import duckdb

DB_PATH = "/Users/joe/Sync/nautilus_quants/logs/registry/test.duckdb"
TUNE_SOURCE = "alpha_tune_sanapi"
DEFAULT_OUTPUT = Path(
    "/Users/joe/Sync/worktrees/feature-046-alpha-tune/logs/alpha_tune_sanapi/uplift_report.md"
)

# factor_id format: ``{source_factor_id}_tuneN_<hash8>``
_TUNE_SUFFIX_RE = re.compile(r"_tune\d+_[0-9a-f]{8}$")


def strip_tune_suffix(tune_factor_id: str) -> str:
    return _TUNE_SUFFIX_RE.sub("", tune_factor_id)


def fetch_metrics(
    conn: duckdb.DuckDBPyConnection,
    factor_ids: Iterable[str],
) -> dict[str, dict[str, dict[str, float]]]:
    """Fetch latest metrics keyed by ``{factor_id: {period: {ic, icir, t_stat_nw}}}``."""
    ids = list(factor_ids)
    if not ids:
        return {}
    placeholders = ",".join(["?"] * len(ids))
    rows = conn.execute(
        f"""
        SELECT factor_id, period, ic_mean, icir, t_stat_nw, n_samples, run_id
        FROM alpha_analysis_metrics
        WHERE factor_id IN ({placeholders})
        ORDER BY factor_id, period, run_id DESC
        """,
        ids,
    ).fetchall()
    out: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    seen: set[tuple[str, str]] = set()
    for fid, period, ic, icir, t_stat, n, run_id in rows:
        key = (fid, period)
        if key in seen:
            continue
        seen.add(key)
        out[fid][period] = {
            "ic": ic,
            "icir": icir,
            "t_stat_nw": t_stat,
            "n_samples": n,
            "run_id": run_id,
        }
    return out


def main() -> None:
    output_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(DB_PATH, read_only=True)

    # 1. All tuned variants
    tune_factors = conn.execute(
        "SELECT factor_id, expression FROM factors WHERE source = ?", [TUNE_SOURCE]
    ).fetchall()
    print(f"Found {len(tune_factors)} tuned variants in registry")

    # 2. Group by source factor (strip _tuneN_<hash>)
    by_source: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for fid, expr in tune_factors:
        src = strip_tune_suffix(fid)
        if src == fid:
            continue  # not a tuned variant
        by_source[src].append((fid, expr))
    print(f"Grouped into {len(by_source)} source factors")

    # 3. Look up source factor expressions + metrics
    source_factors = conn.execute(
        f"SELECT factor_id, expression FROM factors "
        f"WHERE factor_id IN ({','.join(['?'] * len(by_source))})",
        list(by_source.keys()),
    ).fetchall()
    src_expr_map: dict[str, str] = {fid: expr for fid, expr in source_factors}
    print(f"Resolved {len(src_expr_map)}/{len(by_source)} source factor expressions")

    all_factor_ids = set(src_expr_map.keys())
    for variants in by_source.values():
        for fid, _ in variants:
            all_factor_ids.add(fid)
    metrics = fetch_metrics(conn, all_factor_ids)

    # 4. For each source, pick best variant by max |ICIR| at any period
    rows: list[dict] = []
    skipped_no_src_metrics = 0
    skipped_no_variant_metrics = 0
    for src_id, variants in sorted(by_source.items()):
        if src_id not in src_expr_map:
            continue
        src_metrics = metrics.get(src_id, {})
        if not src_metrics:
            skipped_no_src_metrics += 1
            continue
        # Pick best variant
        best_variant_id, best_variant_metrics = None, {}
        best_variant_score = -1.0
        for vid, _ in variants:
            vm = metrics.get(vid, {})
            if not vm:
                continue
            score = max(
                (abs(p.get("icir", 0.0)) for p in vm.values() if p.get("icir") is not None),
                default=-1.0,
            )
            if score > best_variant_score:
                best_variant_score = score
                best_variant_id = vid
                best_variant_metrics = vm
        if best_variant_id is None:
            skipped_no_variant_metrics += 1
            continue

        # Pick the source factor's STRONGEST period for fair comparison
        src_best_period = max(
            src_metrics.keys(),
            key=lambda p: abs(src_metrics[p].get("icir") or 0),
        )
        var_best_period = max(
            best_variant_metrics.keys(),
            key=lambda p: abs(best_variant_metrics[p].get("icir") or 0),
        )
        src_best = src_metrics[src_best_period]
        var_best = best_variant_metrics[var_best_period]

        rows.append(
            {
                "src_id": src_id,
                "src_expr": src_expr_map[src_id],
                "var_id": best_variant_id,
                "var_expr": next(e for f, e in variants if f == best_variant_id),
                "src_period": src_best_period,
                "var_period": var_best_period,
                "src_ic": src_best.get("ic"),
                "src_icir": src_best.get("icir"),
                "src_t_nw": src_best.get("t_stat_nw"),
                "var_ic": var_best.get("ic"),
                "var_icir": var_best.get("icir"),
                "var_t_nw": var_best.get("t_stat_nw"),
                "n_variants": len(variants),
            }
        )

    print(f"\nPaired {len(rows)} (source, best-variant) factors")
    print(f"  Skipped (no source metrics): {skipped_no_src_metrics}")
    print(f"  Skipped (no variant metrics): {skipped_no_variant_metrics}")

    # 5. Compute uplift metrics
    def _abs(x):
        return abs(x) if x is not None else 0.0

    def _pct(num, denom):
        if denom is None or denom == 0:
            return float("inf")
        return (num - denom) / abs(denom) * 100

    for r in rows:
        r["abs_icir_orig"] = _abs(r["src_icir"])
        r["abs_icir_var"] = _abs(r["var_icir"])
        r["abs_icir_uplift"] = r["abs_icir_var"] - r["abs_icir_orig"]
        r["abs_icir_pct"] = _pct(r["abs_icir_var"], r["abs_icir_orig"])
        r["abs_ic_orig"] = _abs(r["src_ic"])
        r["abs_ic_var"] = _abs(r["var_ic"])
        r["abs_ic_uplift"] = r["abs_ic_var"] - r["abs_ic_orig"]
        r["abs_ic_pct"] = _pct(r["abs_ic_var"], r["abs_ic_orig"])

    # 6. Aggregate stats
    n = len(rows)
    n_icir_better = sum(1 for r in rows if r["abs_icir_uplift"] > 0)
    n_ic_better = sum(1 for r in rows if r["abs_ic_uplift"] > 0)
    n_doubled = sum(1 for r in rows if r["abs_icir_var"] >= 2 * r["abs_icir_orig"] and r["abs_icir_orig"] > 0)
    n_tripled = sum(1 for r in rows if r["abs_icir_var"] >= 3 * r["abs_icir_orig"] and r["abs_icir_orig"] > 0)

    mean_icir_orig = sum(r["abs_icir_orig"] for r in rows) / n if n else 0
    mean_icir_var = sum(r["abs_icir_var"] for r in rows) / n if n else 0
    median_uplift = sorted(r["abs_icir_uplift"] for r in rows)[n // 2] if n else 0

    sorted_by_uplift = sorted(rows, key=lambda r: r["abs_icir_uplift"], reverse=True)
    sorted_by_pct = sorted(
        [r for r in rows if r["abs_icir_orig"] > 0.01],  # exclude near-zero baselines
        key=lambda r: r["abs_icir_pct"], reverse=True,
    )
    losers = sorted(rows, key=lambda r: r["abs_icir_uplift"])[:5]

    # 7. Render markdown
    lines: list[str] = []

    def w(s: str = "") -> None:
        lines.append(s)

    w("# Tune Uplift Report — alpha_tune_sanapi")
    w("")
    w(f"_Generated for {n} (source, best-variant) pairs._")
    w("")
    w("**Methodology**: For each source factor with at least one tuned variant, "
      "pair against the variant with the strongest |ICIR| (any period). Compare on "
      "the period that maximises |ICIR| for each side independently — this gives "
      "the fairest 'best vs best' comparison.")
    w("")

    # Section: Aggregate stats
    w("## Aggregate Statistics")
    w("")
    w(f"| Metric | Value |")
    w(f"|--------|-------|")
    w(f"| Total source factors compared | {n} |")
    w(f"| ICIR improved (\\|ICIR\\| went up) | **{n_icir_better} ({n_icir_better/n*100:.0f}%)** |")
    w(f"| IC improved | {n_ic_better} ({n_ic_better/n*100:.0f}%) |")
    w(f"| Doubled \\|ICIR\\| (≥ 2x) | **{n_doubled} ({n_doubled/n*100:.0f}%)** |")
    w(f"| Tripled \\|ICIR\\| (≥ 3x) | {n_tripled} ({n_tripled/n*100:.0f}%) |")
    w(f"| Mean original \\|ICIR\\| | {mean_icir_orig:.4f} |")
    w(f"| Mean tuned \\|ICIR\\| | **{mean_icir_var:.4f}** |")
    w(f"| Mean uplift (Δ\\|ICIR\\|) | **{mean_icir_var - mean_icir_orig:+.4f}** |")
    w(f"| Median uplift | {median_uplift:+.4f} |")
    w("")

    # Section: top uplift (absolute)
    w("## Top 15 by Absolute |ICIR| Uplift (Δ)")
    w("")
    w("| # | Source factor | Δ\\|ICIR\\| | \\|ICIR\\| orig | \\|ICIR\\| tuned | %  |")
    w("|---|---------------|-----------|---------------|----------------|----|")
    for i, r in enumerate(sorted_by_uplift[:15], 1):
        pct = "∞" if r["abs_icir_pct"] == float("inf") else f"{r['abs_icir_pct']:+.0f}%"
        w(
            f"| {i} | `{r['src_id']}` | **{r['abs_icir_uplift']:+.4f}** | "
            f"{r['abs_icir_orig']:.4f} | {r['abs_icir_var']:.4f} | {pct} |"
        )
    w("")

    # Section: top uplift (relative %)
    w("## Top 15 by Relative |ICIR| Uplift (%)")
    w("")
    w("_(excluding cases where original \\|ICIR\\| < 0.01 to avoid division noise)_")
    w("")
    w("| # | Source factor | %  | \\|ICIR\\| orig | \\|ICIR\\| tuned | Δ |")
    w("|---|---------------|----|---------------|----------------|---|")
    for i, r in enumerate(sorted_by_pct[:15], 1):
        w(
            f"| {i} | `{r['src_id']}` | **{r['abs_icir_pct']:+.0f}%** | "
            f"{r['abs_icir_orig']:.4f} | {r['abs_icir_var']:.4f} | "
            f"{r['abs_icir_uplift']:+.4f} |"
        )
    w("")

    # Section: losers
    w("## Bottom 5 (tune made it worse)")
    w("")
    w("| Source factor | Δ\\|ICIR\\| | \\|ICIR\\| orig | \\|ICIR\\| tuned |")
    w("|---------------|-----------|---------------|----------------|")
    for r in losers:
        w(
            f"| `{r['src_id']}` | {r['abs_icir_uplift']:+.4f} | "
            f"{r['abs_icir_orig']:.4f} | {r['abs_icir_var']:.4f} |"
        )
    w("")

    # Section: full per-factor table
    w("## Per-Factor Detail (sorted by absolute uplift, desc)")
    w("")
    w("Each row pairs the **source factor**'s strongest period with the **best "
      "tuned variant**'s strongest period.")
    w("")
    for r in sorted_by_uplift:
        sym = "🟢" if r["abs_icir_uplift"] > 0 else ("🔴" if r["abs_icir_uplift"] < 0 else "⚪")
        pct_str = "∞" if r["abs_icir_pct"] == float("inf") else f"{r['abs_icir_pct']:+.0f}%"
        w(f"### {sym} `{r['src_id']}`")
        w(f"- **uplift**: Δ\\|ICIR\\| = `{r['abs_icir_uplift']:+.4f}` ({pct_str}); "
          f"Δ\\|IC\\| = `{r['abs_ic_uplift']:+.4f}`")
        w(f"- **n_variants**: {r['n_variants']}")
        w(f"")
        w(f"| | Expression | Period | IC | \\|IC\\| | ICIR | \\|ICIR\\| | t(NW) |")
        w(f"|---|------------|--------|----|-------|------|-----------|-------|")
        src_t = f"{r['src_t_nw']:+.2f}" if r["src_t_nw"] is not None else "–"
        var_t = f"{r['var_t_nw']:+.2f}" if r["var_t_nw"] is not None else "–"
        w(
            f"| **orig** | `{r['src_expr']}` | {r['src_period']} | "
            f"{r['src_ic']:+.4f} | {r['abs_ic_orig']:.4f} | "
            f"{r['src_icir']:+.4f} | {r['abs_icir_orig']:.4f} | {src_t} |"
        )
        w(
            f"| **tuned** | `{r['var_expr']}` | {r['var_period']} | "
            f"{r['var_ic']:+.4f} | {r['abs_ic_var']:.4f} | "
            f"{r['var_icir']:+.4f} | {r['abs_icir_var']:.4f} | {var_t} |"
        )
        w("")
        w("---")
        w("")

    output_path.write_text("\n".join(lines))
    print(f"\n✓ Report written to: {output_path}")
    print(f"  ({len(lines)} lines)")
    conn.close()


if __name__ == "__main__":
    main()
