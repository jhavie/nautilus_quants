# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for HDBSCAN factor clustering and Super Alpha composition."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nautilus_quants.alpha.registry.clustering import (
    ClusterConfig,
    SuperAlpha,
    build_super_alphas,
    cluster_factors,
    compose_super_alpha,
)
from nautilus_quants.factors.expression.parser import parse_expression


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_block_corr(
    n_factors: int,
    groups: list[list[int]],
    within_corr: float = 0.85,
    between_corr: float = 0.05,
) -> pd.DataFrame:
    """Build a synthetic block-diagonal correlation matrix.

    Args:
        n_factors: Total number of factors.
        groups: List of index groups (e.g. [[0,1,2],[3,4,5]]).
        within_corr: Correlation within each group.
        between_corr: Correlation across groups.

    Returns:
        Correlation matrix as pd.DataFrame.
    """
    corr = np.full((n_factors, n_factors), between_corr)
    np.fill_diagonal(corr, 1.0)
    for group in groups:
        for i in group:
            for j in group:
                if i != j:
                    corr[i, j] = within_corr
    ids = [f"f{i}" for i in range(n_factors)]
    return pd.DataFrame(corr, index=ids, columns=ids)


def _make_factor_expressions(ids: list[str]) -> dict[str, str]:
    """Generate trivial DSL expressions for each factor ID."""
    return {fid: f"cs_rank(close)" for fid in ids}


def _make_correlated_panels(
    n_timestamps: int = 100,
    n_instruments: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create 3 correlated factor panels from a shared base signal."""
    np.random.seed(seed)
    base = np.random.randn(n_timestamps, n_instruments)
    f1 = pd.DataFrame(base + np.random.randn(n_timestamps, n_instruments) * 0.1)
    f2 = pd.DataFrame(base + np.random.randn(n_timestamps, n_instruments) * 0.2)
    f3 = pd.DataFrame(base + np.random.randn(n_timestamps, n_instruments) * 0.15)
    return f1, f2, f3


# ---------------------------------------------------------------------------
# Test: cluster_factors
# ---------------------------------------------------------------------------


class TestClusterFactors:
    """Tests for HDBSCAN clustering on correlation distance."""

    def test_cluster_high_corr_factors(self) -> None:
        """Two blocks of 3 highly-correlated factors should yield 2 clusters."""
        corr = _build_block_corr(
            n_factors=6,
            groups=[[0, 1, 2], [3, 4, 5]],
            within_corr=0.85,
            between_corr=0.05,
        )
        config = ClusterConfig(
            enabled=True,
            algorithm="hdbscan",
            min_cluster_size=3,
            min_samples=2,
        )

        clusters, noise = cluster_factors(corr, config)

        # Must produce exactly 2 clusters
        assert len(clusters) == 2, (
            f"Expected 2 clusters, got {len(clusters)}: {clusters}"
        )

        # Each cluster should have exactly 3 members
        sizes = sorted(len(v) for v in clusters.values())
        assert sizes == [3, 3], f"Expected [3, 3] cluster sizes, got {sizes}"

        # Factors 0-2 should be in one cluster, 3-5 in the other
        group_a = {"f0", "f1", "f2"}
        group_b = {"f3", "f4", "f5"}
        cluster_sets = [set(v) for v in clusters.values()]
        assert group_a in cluster_sets, (
            f"Group A {group_a} not found in clusters: {cluster_sets}"
        )
        assert group_b in cluster_sets, (
            f"Group B {group_b} not found in clusters: {cluster_sets}"
        )

    def test_noise_detection(self) -> None:
        """An uncorrelated factor appended to block-corr should be noise."""
        # 6 factors in 2 blocks + 1 noise factor (f6)
        n = 7
        corr = _build_block_corr(
            n_factors=n,
            groups=[[0, 1, 2], [3, 4, 5]],
            within_corr=0.85,
            between_corr=0.05,
        )
        # f6 row/col already has between_corr=0.05, which is near zero -- noise

        config = ClusterConfig(
            enabled=True,
            algorithm="hdbscan",
            min_cluster_size=3,
            min_samples=2,
        )

        clusters, noise = cluster_factors(corr, config)

        assert "f6" in noise, (
            f"Expected f6 in noise, got noise={noise}, clusters={clusters}"
        )

    def test_noise_passthrough(self) -> None:
        """Noise factors should become standalone Super Alphas when enabled."""
        n = 7
        corr = _build_block_corr(
            n_factors=n,
            groups=[[0, 1, 2], [3, 4, 5]],
            within_corr=0.85,
            between_corr=0.05,
        )
        factor_ids = [f"f{i}" for i in range(n)]
        factor_expressions = {fid: f"cs_rank(close_{fid})" for fid in factor_ids}

        config = ClusterConfig(
            enabled=True,
            algorithm="hdbscan",
            min_cluster_size=3,
            min_samples=2,
            composition_method="equal",
            noise_passthrough=True,
        )

        super_alphas, noise = build_super_alphas(
            factor_ids=factor_ids,
            factor_expressions=factor_expressions,
            corr_matrix=corr,
            config=config,
        )

        # f6 should be a passthrough Super Alpha, not discarded
        assert len(noise) == 0, f"Expected no discarded noise, got {noise}"
        passthrough_sas = [sa for sa in super_alphas if sa.method == "passthrough"]
        assert len(passthrough_sas) >= 1
        passthrough_members = [m for sa in passthrough_sas for m in sa.members]
        assert "f6" in passthrough_members

    def test_noise_discarded_when_disabled(self) -> None:
        """Noise factors should be discarded when noise_passthrough=False."""
        n = 7
        corr = _build_block_corr(
            n_factors=n,
            groups=[[0, 1, 2], [3, 4, 5]],
            within_corr=0.85,
            between_corr=0.05,
        )
        factor_ids = [f"f{i}" for i in range(n)]
        factor_expressions = {fid: f"cs_rank(close_{fid})" for fid in factor_ids}

        config = ClusterConfig(
            enabled=True,
            algorithm="hdbscan",
            min_cluster_size=3,
            min_samples=2,
            composition_method="equal",
            noise_passthrough=False,
        )

        super_alphas, noise = build_super_alphas(
            factor_ids=factor_ids,
            factor_expressions=factor_expressions,
            corr_matrix=corr,
            config=config,
        )

        assert "f6" in noise


class TestAgglomerativeClustering:
    """Tests for Agglomerative hierarchical clustering."""

    def test_two_blocks(self) -> None:
        """Two blocks of correlated factors → 2 clusters, no noise."""
        corr = _build_block_corr(
            n_factors=6,
            groups=[[0, 1, 2], [3, 4, 5]],
            within_corr=0.85,
            between_corr=0.05,
        )
        config = ClusterConfig(
            enabled=True,
            algorithm="agglomerative",
            linkage="average",
            distance_threshold=0.3,
        )

        clusters, noise = cluster_factors(corr, config)

        assert len(clusters) == 2
        assert len(noise) == 0  # No noise in agglomerative

    def test_no_noise(self) -> None:
        """Agglomerative assigns every factor — outlier gets its own cluster."""
        corr = _build_block_corr(
            n_factors=7,
            groups=[[0, 1, 2], [3, 4, 5]],
            within_corr=0.85,
            between_corr=0.05,
        )
        config = ClusterConfig(
            enabled=True,
            algorithm="agglomerative",
            linkage="average",
            distance_threshold=0.3,
        )

        clusters, noise = cluster_factors(corr, config)

        assert len(noise) == 0
        all_members = [fid for members in clusters.values() for fid in members]
        assert "f6" in all_members  # Not discarded

    def test_threshold_controls_granularity(self) -> None:
        """Lower threshold → more clusters."""
        corr = _build_block_corr(
            n_factors=6,
            groups=[[0, 1, 2], [3, 4, 5]],
            within_corr=0.85,
            between_corr=0.05,
        )

        # Tight threshold — should split more
        config_tight = ClusterConfig(
            enabled=True,
            algorithm="agglomerative",
            linkage="average",
            distance_threshold=0.05,
        )
        clusters_tight, _ = cluster_factors(corr, config_tight)

        # Loose threshold — should merge more
        config_loose = ClusterConfig(
            enabled=True,
            algorithm="agglomerative",
            linkage="average",
            distance_threshold=0.5,
        )
        clusters_loose, _ = cluster_factors(corr, config_loose)

        assert len(clusters_tight) >= len(clusters_loose)

    def test_unknown_algorithm_raises(self) -> None:
        """Unknown algorithm name should raise ValueError."""
        corr = _build_block_corr(n_factors=4, groups=[[0, 1, 2]], within_corr=0.9)
        config = ClusterConfig(enabled=True, algorithm="kmeans")

        with pytest.raises(ValueError, match="Unknown clustering algorithm"):
            cluster_factors(corr, config)


# ---------------------------------------------------------------------------
# Test: compose_super_alpha
# ---------------------------------------------------------------------------


class TestComposeSuperAlpha:
    """Tests for within-cluster Super Alpha composition."""

    def test_single_factor_passthrough(self) -> None:
        """A cluster with 1 factor should passthrough with weight=1.0."""
        member_ids = ["f0"]
        member_exprs = {"f0": "cs_rank(close)"}

        sa = compose_super_alpha(
            cluster_id=0,
            member_ids=member_ids,
            member_expressions=member_exprs,
            method="pca",  # method is ignored for single factor
        )

        assert sa.method == "passthrough"
        assert sa.weights == {"f0": 1.0}
        assert sa.expression == "cs_rank(close)"
        assert sa.name == "SA_000"

    def test_compose_equal_weights(self) -> None:
        """Equal-weight composition should give 1/N weights."""
        member_ids = ["f0", "f1", "f2", "f3"]
        member_exprs = {fid: f"cs_rank(close_{fid})" for fid in member_ids}

        sa = compose_super_alpha(
            cluster_id=1,
            member_ids=member_ids,
            member_expressions=member_exprs,
            method="equal",
        )

        assert sa.method == "equal"
        expected_w = round(1.0 / 4, 4)
        for fid in member_ids:
            assert sa.weights[fid] == pytest.approx(expected_w, abs=1e-4)

        # Weights should sum to ~1.0
        assert sum(sa.weights.values()) == pytest.approx(1.0, abs=1e-3)

    def test_compose_pca(self) -> None:
        """PCA composition on correlated factors gives weights that sum(|w|)=1."""
        # Use asymmetric noise to ensure PCA loadings are non-uniform.
        # f0 is almost pure signal, f1 has moderate noise, f2 has heavy noise.
        np.random.seed(42)
        n_t, n_i = 200, 5
        base = np.random.randn(n_t, n_i)
        f0 = pd.DataFrame(base + np.random.randn(n_t, n_i) * 0.01)
        f1 = pd.DataFrame(base + np.random.randn(n_t, n_i) * 0.5)
        f2 = pd.DataFrame(base + np.random.randn(n_t, n_i) * 2.0)

        member_ids = ["f0", "f1", "f2"]
        member_exprs = {fid: f"cs_rank(close_{fid})" for fid in member_ids}
        panels = {"f0": f0, "f1": f1, "f2": f2}

        sa = compose_super_alpha(
            cluster_id=0,
            member_ids=member_ids,
            member_expressions=member_exprs,
            factor_panels=panels,
            method="pca",
        )

        assert sa.method == "pca"
        weights = list(sa.weights.values())

        # sum(|w|) should be ~1.0
        assert sum(abs(w) for w in weights) == pytest.approx(1.0, abs=1e-3)

        # With asymmetric noise, PCA weights must differ from equal 1/3
        equal_w = 1.0 / 3
        deviations = [abs(abs(w) - equal_w) for w in weights]
        assert any(d > 0.02 for d in deviations), (
            f"PCA weights should not all be equal to 1/3: {weights}"
        )


# ---------------------------------------------------------------------------
# Test: DSL expression parsing
# ---------------------------------------------------------------------------


class TestDSLExpression:
    """Tests for generated DSL expression validity."""

    def test_super_alpha_expression_parseable(self) -> None:
        """Generated DSL expression must be parseable by parse_expression()."""
        member_ids = ["f0", "f1"]
        member_exprs = {
            "f0": "cs_rank(close)",
            "f1": "ts_mean(volume, 10)",
        }

        sa = compose_super_alpha(
            cluster_id=0,
            member_ids=member_ids,
            member_expressions=member_exprs,
            method="equal",
        )

        # Should not raise
        ast = parse_expression(sa.expression)
        assert ast is not None


# ---------------------------------------------------------------------------
# Test: build_super_alphas end-to-end pipeline
# ---------------------------------------------------------------------------


class TestBuildSuperAlphasE2E:
    """End-to-end test of the full clustering + composition pipeline."""

    def test_build_super_alphas_end_to_end(self) -> None:
        """Full pipeline: 6 factors in 2 blocks -> 2 Super Alphas."""
        factor_ids = [f"f{i}" for i in range(6)]
        factor_expressions = {fid: f"cs_rank(close_{fid})" for fid in factor_ids}
        corr = _build_block_corr(
            n_factors=6,
            groups=[[0, 1, 2], [3, 4, 5]],
            within_corr=0.85,
            between_corr=0.05,
        )

        # Build simple panels for equal-weight (no PCA needed)
        np.random.seed(99)
        panels = {
            fid: pd.DataFrame(np.random.randn(50, 3))
            for fid in factor_ids
        }
        tags = {
            "f0": ["momentum"],
            "f1": ["momentum"],
            "f2": ["momentum"],
            "f3": ["value"],
            "f4": ["value"],
            "f5": ["value"],
        }

        config = ClusterConfig(
            enabled=True,
            algorithm="agglomerative",
            linkage="average",
            distance_threshold=0.3,
            composition_method="equal",
            single_factor_passthrough=True,
        )

        super_alphas, noise = build_super_alphas(
            factor_ids=factor_ids,
            factor_expressions=factor_expressions,
            corr_matrix=corr,
            config=config,
            factor_panels=panels,
            factor_tags=tags,
        )

        # Should get 2 Super Alphas
        assert len(super_alphas) == 2, (
            f"Expected 2 Super Alphas, got {len(super_alphas)}"
        )

        # Each should have 3 members
        for sa in super_alphas:
            assert len(sa.members) == 3
            assert sa.method == "equal"
            assert sa.name.startswith("SA_")
            # Tags should be inherited
            assert len(sa.tags) > 0
            assert any(t.startswith("cluster_") for t in sa.tags)

        # Noise should be empty (all 6 factors clustered)
        assert len(noise) == 0, f"Expected no noise, got {noise}"

        # Each SA expression should be parseable
        for sa in super_alphas:
            ast = parse_expression(sa.expression)
            assert ast is not None
