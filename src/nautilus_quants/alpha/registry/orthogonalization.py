# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Löwdin symmetric orthogonalization for factor weight optimization.

Computes the Löwdin transform S^{-1/2} that maximizes overlap between
original and orthogonalized factors, then transforms composite weights
to eliminate redundant exposure between correlated factors.

References:
    - Löwdin, P.-O. (1950). On the non-orthogonality problem.
    - Klein & Chow (2013). Orthogonalized factors and systematic risk decomposition.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrthConfig:
    """Löwdin orthogonalization configuration."""

    enabled: bool = False
    method: str = "lowdin"  # "lowdin" | "none"
    min_eigenvalue: float = 1e-6  # Clip eigenvalues below this
    max_condition_number: float = 1e6  # Skip if condition number exceeds


@dataclass(frozen=True)
class OrthResult:
    """Result of Löwdin orthogonalization."""

    transform_matrix: np.ndarray  # S^{-1/2}, shape (N, N)
    orth_weights: dict[str, float]  # factor_id -> orthogonalized weight
    original_weights: dict[str, float]  # factor_id -> original weight
    eigenvalues: np.ndarray  # eigenvalues of S
    condition_number: float  # lambda_max / lambda_min
    overlap_scores: dict[str, float]  # factor_id -> corr(F_i, F_i_orth)


def compute_lowdin_matrix(
    factor_values: np.ndarray,
    min_eigenvalue: float = 1e-6,
    max_condition_number: float = 1e6,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute Löwdin symmetric orthogonalization matrix S^{-1/2}.

    Args:
        factor_values: Shape (T*N_instruments, N_factors) -- stacked
            cross-sections. Each column is one factor's values across all
            timestamps and instruments.
        min_eigenvalue: Regularization -- clip eigenvalues below this.
        max_condition_number: If condition number exceeds this, raise
            ValueError.

    Returns:
        (S_inv_sqrt, eigenvalues, condition_number):
            S_inv_sqrt: Löwdin transform matrix, shape (N_factors, N_factors).
            eigenvalues: Eigenvalues of S (after regularization).
            condition_number: lambda_max / lambda_min.

    Raises:
        ValueError: If condition number exceeds max_condition_number or
            insufficient valid rows.
    """
    # Drop rows with any NaN
    valid = ~np.isnan(factor_values).any(axis=1)
    clean = factor_values[valid]

    if len(clean) < factor_values.shape[1]:
        raise ValueError(
            f"Insufficient valid rows ({len(clean)}) "
            f"for {factor_values.shape[1]} factors"
        )

    # Normalize columns to zero mean, unit variance
    means = clean.mean(axis=0)
    stds = clean.std(axis=0)
    stds[stds == 0] = 1.0  # prevent division by zero
    normalized = (clean - means) / stds

    # Compute correlation matrix S = (1/T) * F^T F
    n_obs = len(normalized)
    S = normalized.T @ normalized / n_obs

    # Eigendecomposition: S = U diag(lambda) U^T
    eigenvalues, eigenvectors = np.linalg.eigh(S)

    # Regularize small eigenvalues
    eigenvalues = np.maximum(eigenvalues, min_eigenvalue)

    # Check condition number
    cond = eigenvalues.max() / eigenvalues.min()
    if cond > max_condition_number:
        raise ValueError(
            f"Condition number {cond:.1f} exceeds threshold "
            f"{max_condition_number:.0f}. "
            f"Factors are too collinear for orthogonalization."
        )

    # S^{-1/2} = U @ diag(1/sqrt(lambda)) @ U^T
    S_inv_sqrt = (
        eigenvectors
        @ np.diag(1.0 / np.sqrt(eigenvalues))
        @ eigenvectors.T
    )

    return S_inv_sqrt, eigenvalues, float(cond)


def orthogonalize_weights(
    original_weights: dict[str, float],
    transform_matrix: np.ndarray,
    factor_ids: list[str],
) -> dict[str, float]:
    """Apply Löwdin transform to composite weights.

    Args:
        original_weights: {factor_id -> weight} (e.g., ICIR weights).
        transform_matrix: S^{-1/2} matrix from compute_lowdin_matrix.
        factor_ids: Ordered factor IDs matching transform_matrix columns.

    Returns:
        {factor_id -> orthogonalized weight}, normalized to sum(|w|) = 1.
    """
    w = np.array([original_weights.get(fid, 0.0) for fid in factor_ids])
    w_orth = transform_matrix.T @ w

    # Normalize to sum(|w|) = 1
    abs_sum = np.abs(w_orth).sum()
    if abs_sum > 0:
        w_orth = w_orth / abs_sum

    return {
        fid: float(round(w_val, 4))
        for fid, w_val in zip(factor_ids, w_orth)
    }


def validate_overlap(
    factor_values: np.ndarray,
    transform_matrix: np.ndarray,
    factor_ids: list[str],
    min_overlap: float = 0.90,
) -> dict[str, float]:
    """Validate that orthogonalized factors preserve overlap with originals.

    Computes corr(F_i, F_i_orth) for each factor. Löwdin should yield > 0.90.

    Args:
        factor_values: Shape (T*N, N_factors).
        transform_matrix: S^{-1/2} matrix.
        factor_ids: Ordered factor IDs.
        min_overlap: Warn if any overlap falls below this.

    Returns:
        {factor_id -> overlap_score (Pearson correlation)}.
    """
    valid = ~np.isnan(factor_values).any(axis=1)
    clean = factor_values[valid]

    if len(clean) == 0:
        return {fid: 0.0 for fid in factor_ids}

    # Orthogonalized values
    orth_values = clean @ transform_matrix

    overlaps: dict[str, float] = {}
    for i, fid in enumerate(factor_ids):
        corr = np.corrcoef(clean[:, i], orth_values[:, i])[0, 1]
        if np.isnan(corr):
            corr = 0.0
        overlaps[fid] = float(round(corr, 4))
        if corr < min_overlap:
            logger.warning(
                "Low overlap for %s: %.3f < %.3f",
                fid,
                corr,
                min_overlap,
            )

    return overlaps


def orthogonalize_factor_weights(
    factor_ids: list[str],
    factor_panels: dict[str, pd.DataFrame],
    original_weights: dict[str, float],
    config: OrthConfig,
) -> OrthResult:
    """Full orthogonalization pipeline.

    Args:
        factor_ids: Ordered factor IDs.
        factor_panels: {factor_id -> DataFrame(T x N_instruments)}.
        original_weights: {factor_id -> weight} to transform.
        config: Orthogonalization configuration.

    Returns:
        OrthResult with transform matrix, weights, and diagnostics.
    """
    # Stack panels into (T*N, N_factors) matrix
    # Align to common timestamps
    common_idx = None
    for fid in factor_ids:
        if fid in factor_panels:
            idx = factor_panels[fid].index
            common_idx = (
                idx if common_idx is None
                else common_idx.intersection(idx)
            )

    if common_idx is None or len(common_idx) == 0:
        raise ValueError("No common timestamps across factor panels")

    columns = []
    for fid in factor_ids:
        panel = factor_panels[fid].loc[common_idx]
        columns.append(panel.values.flatten())

    matrix = np.column_stack(columns)

    # Compute Löwdin transform
    S_inv_sqrt, eigenvalues, cond = compute_lowdin_matrix(
        matrix,
        min_eigenvalue=config.min_eigenvalue,
        max_condition_number=config.max_condition_number,
    )

    # Transform weights
    orth_weights = orthogonalize_weights(
        original_weights, S_inv_sqrt, factor_ids,
    )

    # Validate overlap
    overlaps = validate_overlap(matrix, S_inv_sqrt, factor_ids)

    return OrthResult(
        transform_matrix=S_inv_sqrt,
        orth_weights=orth_weights,
        original_weights=original_weights,
        eigenvalues=eigenvalues,
        condition_number=cond,
        overlap_scores=overlaps,
    )
