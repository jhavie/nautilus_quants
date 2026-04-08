# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""HDBSCAN factor clustering and Super Alpha composition.

Implements the clustering layer of the promote pipeline:
  1. HDBSCAN on Spearman correlation distance -> auto-detect clusters + noise
  2. Within-cluster PCA / equal / ICIR composition -> Super Alpha expressions
  3. Super Alpha FactorRecord generation for DuckDB storage
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClusterConfig:
    """Multi-algorithm clustering configuration."""

    enabled: bool = False
    algorithm: str = "agglomerative"  # "hdbscan" | "agglomerative"
    # --- Common ---
    distance_metric: str = "spearman"
    composition_method: str = "pca"  # "pca" | "equal" | "icir"
    single_factor_passthrough: bool = True
    noise_passthrough: bool = True
    super_alpha_prefix: str = "SA"
    component_status: str = "component"
    # --- HDBSCAN ---
    min_cluster_size: int = 3
    cluster_selection_method: str = "eom"  # "eom" | "leaf"
    min_samples: int | None = None
    # --- Agglomerative ---
    linkage: str = "average"  # "average" | "complete" | "single"
    distance_threshold: float = 0.3  # |corr| < 1-threshold → 不合并
    # --- Distance ---
    abs_correlation: bool = True  # True: 1-|corr|, False: (1-corr)/2


# ---------------------------------------------------------------------------
# Super Alpha dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SuperAlpha:
    """A Super Alpha composed from a cluster of correlated factors."""

    name: str  # "SA_000", "SA_001"
    expression: str  # DSL: "cs_rank(0.35*(expr1) + 0.40*(expr2))"
    members: list[str]  # member factor_ids
    weights: dict[str, float]  # member_id -> weight
    method: str  # "pca" | "equal" | "icir"
    cluster_id: int
    tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Step 1: Clustering (HDBSCAN / Agglomerative)
# ---------------------------------------------------------------------------


def _corr_to_distance(
    corr_matrix: pd.DataFrame,
    abs_correlation: bool = True,
) -> np.ndarray:
    """Convert correlation matrix to distance.

    abs_correlation=True:  1 - |corr|.  ±0.9 → 0.1 (same cluster)
    abs_correlation=False: (1 - corr)/2. +0.9 → 0.05, -0.9 → 0.95
    """
    if abs_correlation:
        distance = 1.0 - corr_matrix.abs().values
    else:
        distance = (1.0 - corr_matrix.values) / 2.0
    np.fill_diagonal(distance, 0.0)
    distance = (distance + distance.T) / 2.0
    return np.clip(distance, 0.0, 1.0)


def _cluster_hdbscan(
    factor_ids: list[str],
    distance: np.ndarray,
    config: ClusterConfig,
) -> tuple[dict[int, list[str]], list[str]]:
    """HDBSCAN density-based clustering."""
    try:
        import hdbscan
    except ImportError:
        raise ImportError(
            "hdbscan is required for clustering. "
            "Install: pip install hdbscan"
        )

    min_samples = config.min_samples or config.min_cluster_size
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=config.min_cluster_size,
        min_samples=min_samples,
        metric="precomputed",
        cluster_selection_method=config.cluster_selection_method,
    )
    labels = clusterer.fit_predict(distance)

    clusters: dict[int, list[str]] = {}
    noise: list[str] = []
    for fid, label in zip(factor_ids, labels):
        if label == -1:
            noise.append(fid)
        else:
            clusters.setdefault(label, []).append(fid)

    logger.info(
        "HDBSCAN: %d clusters, %d noise from %d total",
        len(clusters), len(noise), len(factor_ids),
    )
    return clusters, noise


def _cluster_agglomerative(
    factor_ids: list[str],
    distance: np.ndarray,
    config: ClusterConfig,
) -> tuple[dict[int, list[str]], list[str]]:
    """Agglomerative hierarchical clustering with distance threshold."""
    from sklearn.cluster import AgglomerativeClustering

    clusterer = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage=config.linkage,
        distance_threshold=config.distance_threshold,
    )
    labels = clusterer.fit_predict(distance)

    clusters: dict[int, list[str]] = {}
    for fid, label in zip(factor_ids, labels):
        clusters.setdefault(int(label), []).append(fid)

    logger.info(
        "Agglomerative: %d clusters from %d total "
        "(linkage=%s, threshold=%.3f)",
        len(clusters), len(factor_ids),
        config.linkage, config.distance_threshold,
    )
    return clusters, []  # No noise concept


def cluster_factors(
    corr_matrix: pd.DataFrame,
    config: ClusterConfig,
) -> tuple[dict[int, list[str]], list[str]]:
    """Cluster factors using the configured algorithm.

    Args:
        corr_matrix: Factor Spearman correlation matrix.
        config: Clustering configuration.

    Returns:
        (clusters, noise_factors):
            clusters: {cluster_id -> [factor_ids]}
            noise_factors: [factor_ids] (empty for agglomerative)
    """
    factor_ids = corr_matrix.index.tolist()
    distance = _corr_to_distance(corr_matrix, config.abs_correlation)

    if config.algorithm == "hdbscan":
        return _cluster_hdbscan(factor_ids, distance, config)
    elif config.algorithm == "agglomerative":
        return _cluster_agglomerative(factor_ids, distance, config)
    else:
        raise ValueError(f"Unknown clustering algorithm: {config.algorithm}")


# ---------------------------------------------------------------------------
# Step 2: Within-cluster composition
# ---------------------------------------------------------------------------


def compose_super_alpha(
    cluster_id: int,
    member_ids: list[str],
    member_expressions: dict[str, str],
    factor_panels: dict[str, pd.DataFrame] | None = None,
    method: str = "pca",
    icir_values: dict[str, float] | None = None,
    prefix: str = "SA",
) -> SuperAlpha:
    """Compose a Super Alpha from a cluster of factors.

    Args:
        cluster_id: HDBSCAN cluster label.
        member_ids: Factor IDs in this cluster.
        member_expressions: {factor_id -> DSL expression string}.
        factor_panels: {factor_id -> DataFrame(T x N)} -- needed for PCA.
        method: "pca" | "equal" | "icir".
        icir_values: {factor_id -> |ICIR|} -- needed for icir method.
        prefix: Super Alpha name prefix.

    Returns:
        SuperAlpha with computed weights and DSL expression.
    """
    n = len(member_ids)

    if n == 1:
        # Single factor: passthrough with weight 1.0
        fid = member_ids[0]
        return SuperAlpha(
            name=f"{prefix}_{cluster_id:03d}",
            expression=member_expressions[fid],
            members=member_ids,
            weights={fid: 1.0},
            method="passthrough",
            cluster_id=cluster_id,
        )

    if method == "equal":
        weights = _compose_equal(member_ids)

    elif method == "pca":
        weights = _compose_pca(member_ids, factor_panels, n)

    elif method == "icir":
        weights = _compose_icir(member_ids, icir_values, n)

    else:
        raise ValueError(f"Unknown composition method: {method}")

    # Build DSL expression: cs_rank(w1*(expr1) + w2*(expr2) + ...)
    expression = _build_dsl_expression(member_ids, weights, member_expressions)

    return SuperAlpha(
        name=f"{prefix}_{cluster_id:03d}",
        expression=expression,
        members=member_ids,
        weights=weights,
        method=method,
        cluster_id=cluster_id,
    )


def _compose_equal(member_ids: list[str]) -> dict[str, float]:
    """Equal-weight composition."""
    n = len(member_ids)
    return {fid: round(1.0 / n, 4) for fid in member_ids}


def _compose_pca(
    member_ids: list[str],
    factor_panels: dict[str, pd.DataFrame] | None,
    n: int,
) -> dict[str, float]:
    """PCA-based composition using first principal component loadings."""
    if factor_panels is None:
        raise ValueError("factor_panels required for PCA composition")

    try:
        from sklearn.decomposition import PCA
    except ImportError:
        raise ImportError(
            "scikit-learn is required for PCA composition. "
            "Install: pip install scikit-learn"
        )

    # Align all panels to common timestamps
    common_idx: pd.Index | None = None
    for fid in member_ids:
        if fid in factor_panels:
            if common_idx is None:
                common_idx = factor_panels[fid].index
            else:
                common_idx = common_idx.intersection(factor_panels[fid].index)

    if common_idx is None or len(common_idx) == 0:
        logger.warning(
            "No common timestamps for PCA, falling back to equal weights",
        )
        return _compose_equal(member_ids)

    # Build matrix: rows = (timestamp, instrument) pairs, cols = factors
    panels = []
    for fid in member_ids:
        panel = factor_panels[fid].loc[common_idx]
        panels.append(panel.values.flatten())  # T*N

    matrix = np.column_stack(panels)  # (T*N, n_factors)

    # Drop rows with any NaN
    valid = ~np.isnan(matrix).any(axis=1)
    matrix_clean = matrix[valid]

    if len(matrix_clean) < n:
        logger.warning("Insufficient data for PCA, falling back to equal")
        return _compose_equal(member_ids)

    pca = PCA(n_components=1)
    pca.fit(matrix_clean)
    pc1 = pca.components_[0]  # shape (n_factors,)

    # Normalize to sum of abs = 1
    abs_sum = np.abs(pc1).sum()
    if abs_sum > 0:
        pc1 = pc1 / abs_sum

    return {fid: float(round(w, 4)) for fid, w in zip(member_ids, pc1)}


def _compose_icir(
    member_ids: list[str],
    icir_values: dict[str, float] | None,
    n: int,
) -> dict[str, float]:
    """ICIR-weighted composition (absolute ICIR as weights)."""
    if icir_values is None:
        raise ValueError("icir_values required for ICIR composition")

    raw = {fid: abs(icir_values.get(fid, 0.0)) for fid in member_ids}
    total = sum(raw.values())
    if total > 0:
        return {fid: round(v / total, 4) for fid, v in raw.items()}
    return _compose_equal(member_ids)


def _build_dsl_expression(
    member_ids: list[str],
    weights: dict[str, float],
    member_expressions: dict[str, str],
) -> str:
    """Build DSL expression: cs_rank(w1*(expr1) + w2*(expr2) + ...)."""
    terms = []
    for fid in member_ids:
        w = weights[fid]
        expr = member_expressions[fid]
        if abs(w) < 1e-6:
            continue
        terms.append(f"{w} * ({expr})")
    inner = " + ".join(terms)
    return f"cs_rank({inner})"


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def build_super_alphas(
    factor_ids: list[str],
    factor_expressions: dict[str, str],
    corr_matrix: pd.DataFrame,
    config: ClusterConfig,
    factor_panels: dict[str, pd.DataFrame] | None = None,
    icir_values: dict[str, float] | None = None,
    factor_tags: dict[str, list[str]] | None = None,
) -> tuple[list[SuperAlpha], list[str]]:
    """Full pipeline: cluster -> compose -> return Super Alphas.

    Args:
        factor_ids: IDs of factors to cluster.
        factor_expressions: {factor_id -> DSL expression}.
        corr_matrix: Spearman correlation matrix.
        config: Clustering configuration.
        factor_panels: Factor value panels (needed for PCA).
        icir_values: ICIR values (needed for ICIR method).
        factor_tags: {factor_id -> tags} for inheriting tags.

    Returns:
        (super_alphas, noise_factor_ids):
            super_alphas: List of SuperAlpha objects.
            noise_factor_ids: Factor IDs discarded as noise.
    """
    # Filter corr matrix to requested factor_ids
    available = [f for f in factor_ids if f in corr_matrix.index]
    if not available:
        logger.warning("No factors found in correlation matrix")
        return [], factor_ids

    sub_corr = corr_matrix.loc[available, available]

    # Step 1: HDBSCAN clustering
    clusters, noise = cluster_factors(sub_corr, config)

    # Step 2: Compose each cluster into a Super Alpha
    super_alphas: list[SuperAlpha] = []
    for cluster_id, members in sorted(clusters.items()):
        if len(members) == 1 and not config.single_factor_passthrough:
            noise.extend(members)
            continue

        method = (
            config.composition_method if len(members) > 1 else "passthrough"
        )
        member_exprs = {
            fid: factor_expressions[fid]
            for fid in members
            if fid in factor_expressions
        }
        sa = compose_super_alpha(
            cluster_id=cluster_id,
            member_ids=members,
            member_expressions=member_exprs,
            factor_panels=factor_panels,
            method=method,
            icir_values=icir_values,
            prefix=config.super_alpha_prefix,
        )

        # Inherit tags from members
        if factor_tags:
            all_tags: set[str] = set()
            for fid in members:
                all_tags.update(factor_tags.get(fid, []))
            all_tags.add(f"cluster_{cluster_id}")
            sa = SuperAlpha(
                name=sa.name,
                expression=sa.expression,
                members=sa.members,
                weights=sa.weights,
                method=sa.method,
                cluster_id=sa.cluster_id,
                tags=sorted(all_tags),
            )

        super_alphas.append(sa)

    # Step 3: Noise factors -> standalone Super Alphas (if enabled)
    passthrough_noise: list[str] = []
    if config.noise_passthrough and noise:
        next_id = max(clusters.keys(), default=-1) + 1
        for i, fid in enumerate(noise):
            if fid not in factor_expressions:
                continue
            sa = SuperAlpha(
                name=f"{config.super_alpha_prefix}_{next_id + i:03d}",
                expression=factor_expressions[fid],
                members=[fid],
                weights={fid: 1.0},
                method="passthrough",
                cluster_id=next_id + i,
                tags=sorted(
                    {*(factor_tags.get(fid, []) if factor_tags else []), "noise_passthrough"},
                ),
            )
            super_alphas.append(sa)
            passthrough_noise.append(fid)
        remaining_noise = [f for f in noise if f not in passthrough_noise]
    else:
        remaining_noise = noise

    logger.info(
        "Built %d Super Alphas from %d clusters "
        "(%d noise passthrough, %d noise discarded)",
        len(super_alphas),
        len(clusters),
        len(passthrough_noise),
        len(remaining_noise),
    )
    return super_alphas, remaining_noise


# ──────────────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────────────


def plot_cluster_heatmap(
    corr_matrix: pd.DataFrame,
    clusters: dict[int, list[str]],
    noise: list[str],
    output_path: str | None = None,
    title: str = "Factor Correlation — HDBSCAN Clusters",
) -> None:
    """Generate a correlation heatmap with cluster annotations.

    Reorders the correlation matrix so that factors in the same cluster
    are adjacent. Cluster boundaries are drawn as rectangles. Noise
    factors are grouped at the bottom-right.

    Args:
        corr_matrix: Factor Spearman correlation matrix.
        clusters: ``{cluster_id -> [factor_ids]}`` from :func:`cluster_factors`.
        noise: Factor IDs labelled as noise.
        output_path: File path for the PNG. ``None`` → display only.
        title: Chart title.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns

    # Build ordered factor list: clusters first (sorted by id), noise last
    ordered_ids: list[str] = []
    cluster_boundaries: list[tuple[int, int, int]] = []  # (start, size, cid)
    for cid in sorted(clusters):
        start = len(ordered_ids)
        members = clusters[cid]
        ordered_ids.extend(members)
        cluster_boundaries.append((start, len(members), cid))
    noise_start = len(ordered_ids)
    ordered_ids.extend(noise)

    # Filter to factors present in corr_matrix
    available = [f for f in ordered_ids if f in corr_matrix.index]
    sub_corr = corr_matrix.loc[available, available]

    # Color palette for clusters
    n_clusters = len(clusters)
    palette = sns.color_palette("husl", max(n_clusters, 1))

    fig, ax = plt.subplots(figsize=(max(12, len(available) * 0.4), max(10, len(available) * 0.35)))
    annot_fontsize = max(5, min(8, 200 // max(len(available), 1)))
    sns.heatmap(
        sub_corr, vmin=-1, vmax=1, center=0,
        cmap="RdBu_r", ax=ax,
        xticklabels=True, yticklabels=True,
        linewidths=0.1,
        annot=True, fmt=".2f",
        annot_kws={"size": annot_fontsize},
    )

    # Draw cluster boundary rectangles
    for start, size, cid in cluster_boundaries:
        color = palette[cid % len(palette)]
        rect = mpatches.FancyBboxPatch(
            (start, start), size, size,
            boxstyle="round,pad=0.02",
            linewidth=2.5, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)

    # Draw noise boundary (if any)
    if noise:
        n_noise = len([n for n in noise if n in corr_matrix.index])
        if n_noise > 0:
            rect = mpatches.FancyBboxPatch(
                (noise_start, noise_start), n_noise, n_noise,
                boxstyle="round,pad=0.02",
                linewidth=2, edgecolor="gray", facecolor="none",
                linestyle="--",
            )
            ax.add_patch(rect)

    # Legend
    legend_handles = []
    for start, size, cid in cluster_boundaries:
        color = palette[cid % len(palette)]
        legend_handles.append(
            mpatches.Patch(
                facecolor=color, alpha=0.3, edgecolor=color,
                label=f"Cluster {cid} ({size} factors)",
            )
        )
    if noise:
        legend_handles.append(
            mpatches.Patch(
                facecolor="gray", alpha=0.2, edgecolor="gray",
                label=f"Noise ({len(noise)} factors)",
            )
        )
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper left", bbox_to_anchor=(1.02, 1),
            fontsize=8,
        )

    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.xticks(fontsize=6, rotation=90)
    plt.yticks(fontsize=6)
    plt.tight_layout()

    if output_path:
        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Cluster heatmap saved: %s", output_path)
    plt.close(fig)
