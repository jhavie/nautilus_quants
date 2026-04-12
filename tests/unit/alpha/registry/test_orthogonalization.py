# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for Lowdin symmetric orthogonalization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nautilus_quants.alpha.registry.orthogonalization import (
    OrthConfig,
    OrthResult,
    compute_lowdin_matrix,
    orthogonalize_factor_weights,
    orthogonalize_weights,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_uncorrelated_factors(
    n_obs: int = 500,
    n_factors: int = 3,
    seed: int = 42,
) -> np.ndarray:
    """Generate uncorrelated factor columns via independent normals."""
    rng = np.random.RandomState(seed)
    return rng.randn(n_obs, n_factors)


def _make_correlated_factors(
    n_obs: int = 500,
    corr: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    """Generate 3 factors with pairwise correlation ~corr.

    Technique: X_i = sqrt(corr)*Z + sqrt(1-corr)*E_i
    where Z is the common factor and E_i are idiosyncratic.
    """
    rng = np.random.RandomState(seed)
    z = rng.randn(n_obs, 1)
    e = rng.randn(n_obs, 3)
    factors = np.sqrt(corr) * z + np.sqrt(1.0 - corr) * e
    return factors


def _make_nearly_identical_factors(
    n_obs: int = 500,
    noise: float = 1e-8,
    seed: int = 42,
) -> np.ndarray:
    """Generate 2 nearly identical factors (extreme collinearity)."""
    rng = np.random.RandomState(seed)
    base = rng.randn(n_obs, 1)
    f1 = base.copy()
    f2 = base + rng.randn(n_obs, 1) * noise
    return np.column_stack([f1, f2])


# ---------------------------------------------------------------------------
# Test: compute_lowdin_matrix
# ---------------------------------------------------------------------------


class TestComputeLowdinMatrix:
    """Tests for the Lowdin S^{-1/2} computation."""

    def test_lowdin_identity_for_orthogonal_input(self) -> None:
        """Uncorrelated factors -> S^{-1/2} should be close to identity."""
        factors = _make_uncorrelated_factors(n_obs=1000, n_factors=3, seed=0)

        S_inv_sqrt, eigenvalues, cond = compute_lowdin_matrix(factors)

        # S^{-1/2} should be approximately identity
        identity = np.eye(3)
        off_diag_max = np.max(np.abs(S_inv_sqrt - identity) * (1 - identity))
        assert off_diag_max < 0.1, (
            f"Off-diagonal elements too large: max={off_diag_max:.4f}\n"
            f"S_inv_sqrt:\n{S_inv_sqrt}"
        )

        # Condition number should be close to 1
        assert cond < 2.0, f"Condition number too high for orthogonal input: {cond}"

    def test_lowdin_overlap_above_090(self) -> None:
        """Moderately correlated factors -> all overlap scores > 0.90."""
        factors = _make_correlated_factors(n_obs=1000, corr=0.5, seed=42)

        S_inv_sqrt, eigenvalues, cond = compute_lowdin_matrix(factors)

        # Apply transform and check overlap
        # Normalize columns for fair comparison
        means = factors.mean(axis=0)
        stds = factors.std(axis=0)
        stds[stds == 0] = 1.0
        normalized = (factors - means) / stds

        orth_values = normalized @ S_inv_sqrt

        for i in range(3):
            corr_val = np.corrcoef(normalized[:, i], orth_values[:, i])[0, 1]
            assert corr_val > 0.90, (
                f"Factor {i} overlap {corr_val:.4f} < 0.90"
            )

    def test_lowdin_regularization(self) -> None:
        """Near-singular input: regularization clips small eigenvalues or raises."""
        factors = _make_nearly_identical_factors(n_obs=500, noise=1e-8)

        # With default max_condition_number=1e6, this may or may not raise
        # depending on exact condition number after regularization.
        # The key contract: either it succeeds with clipped eigenvalues,
        # or it raises ValueError for too-high condition number.
        try:
            S_inv_sqrt, eigenvalues, cond = compute_lowdin_matrix(
                factors,
                min_eigenvalue=1e-6,
                max_condition_number=1e6,
            )
            # If it succeeds, eigenvalues should be >= min_eigenvalue
            assert np.all(eigenvalues >= 1e-6), (
                f"Eigenvalues below min: {eigenvalues}"
            )
        except ValueError as e:
            # Expected: condition number too high
            assert "Condition number" in str(e) or "Insufficient" in str(e), (
                f"Unexpected ValueError: {e}"
            )


# ---------------------------------------------------------------------------
# Test: orthogonalize_weights
# ---------------------------------------------------------------------------


class TestOrthogonalizeWeights:
    """Tests for weight transformation."""

    def test_orthogonalize_weights(self) -> None:
        """Transformed weights should have sum(|w|) = 1.0."""
        factors = _make_correlated_factors(n_obs=500, corr=0.5, seed=42)
        S_inv_sqrt, _, _ = compute_lowdin_matrix(factors)

        factor_ids = ["f0", "f1", "f2"]
        original = {"f0": 0.5, "f1": 0.3, "f2": 0.2}

        result = orthogonalize_weights(original, S_inv_sqrt, factor_ids)

        # sum(|w|) must equal 1.0
        abs_sum = sum(abs(v) for v in result.values())
        assert abs_sum == pytest.approx(1.0, abs=1e-3), (
            f"sum(|w|) = {abs_sum}, expected 1.0. Weights: {result}"
        )

        # All factor_ids present
        assert set(result.keys()) == set(factor_ids)

        # Weights should differ from original (transform has an effect)
        original_vec = [original[fid] for fid in factor_ids]
        result_vec = [result[fid] for fid in factor_ids]
        assert original_vec != pytest.approx(result_vec, abs=0.01), (
            f"Weights unchanged after transform: {result}"
        )


# ---------------------------------------------------------------------------
# Test: full pipeline
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """End-to-end test of orthogonalize_factor_weights()."""

    def test_full_pipeline(self) -> None:
        """Full pipeline returns OrthResult with all fields populated."""
        np.random.seed(42)
        n_timestamps = 100
        n_instruments = 5

        # Create moderately correlated panels
        base = np.random.randn(n_timestamps, n_instruments)
        panels = {
            "f0": pd.DataFrame(
                base + np.random.randn(n_timestamps, n_instruments) * 0.3,
            ),
            "f1": pd.DataFrame(
                base + np.random.randn(n_timestamps, n_instruments) * 0.4,
            ),
            "f2": pd.DataFrame(
                base + np.random.randn(n_timestamps, n_instruments) * 0.35,
            ),
        }
        factor_ids = ["f0", "f1", "f2"]
        original_weights = {"f0": 0.5, "f1": 0.3, "f2": 0.2}
        config = OrthConfig(enabled=True, method="lowdin")

        result = orthogonalize_factor_weights(
            factor_ids=factor_ids,
            factor_panels=panels,
            original_weights=original_weights,
            config=config,
        )

        # Type check
        assert isinstance(result, OrthResult)

        # Transform matrix shape
        assert result.transform_matrix.shape == (3, 3)

        # Orthogonalized weights populated
        assert set(result.orth_weights.keys()) == set(factor_ids)
        abs_sum = sum(abs(v) for v in result.orth_weights.values())
        assert abs_sum == pytest.approx(1.0, abs=1e-3)

        # Original weights preserved
        assert result.original_weights == original_weights

        # Eigenvalues array
        assert len(result.eigenvalues) == 3
        assert np.all(result.eigenvalues > 0)

        # Condition number is positive and finite
        assert result.condition_number > 0
        assert np.isfinite(result.condition_number)

        # Overlap scores populated for each factor
        assert set(result.overlap_scores.keys()) == set(factor_ids)
        for fid, score in result.overlap_scores.items():
            assert 0.0 <= score <= 1.0 or score == pytest.approx(
                0.0, abs=0.01
            ), f"Overlap for {fid} out of range: {score}"
