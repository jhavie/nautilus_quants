# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
WorldQuantSelectionPolicy — WorldQuant BRAIN 7-step portfolio construction
as a pluggable SelectionPolicy.

Extracted from WorldQuantAlphaStrategy. Pure computation, zero Nautilus
dependencies. Steps:
  1. (external) Raw alpha vector from FactorValues
  2. Apply delay (use previous period's data if delay=1)
  3. Market neutralization (subtract mean → sum(alpha) = 0)
  4. Scale (divide by |sum| → sum(|alpha|) = 1)
  5. Apply linear decay to normalized weights (if decay > 0)
  6. Truncation (cap max single-asset weight, iterative convergence)
  7. (external) Capital allocation by DecisionEngineActor
"""

from __future__ import annotations

import math

from nautilus_quants.strategies.cs.selection_policy import TargetPosition


class WorldQuantSelectionPolicy:
    """WorldQuant BRAIN 7-step pipeline as SelectionPolicy.

    Parameters
    ----------
    delay : int
        Data delay. 0 = current data, 1 = previous period (avoids look-ahead).
    decay : int
        Linear decay window. 0 = no decay, N = weighted average over N periods.
    neutralization : str
        "MARKET" = subtract cross-sectional mean. "NONE" = no neutralization.
    truncation : float
        Max weight per instrument. 0.0 = no truncation.
    enable_long : bool
        Whether to include long positions (positive weights).
    enable_short : bool
        Whether to include short positions (negative weights).
    """

    def __init__(
        self,
        delay: int = 1,
        decay: int = 0,
        neutralization: str = "MARKET",
        truncation: float = 0.0,
        enable_long: bool = True,
        enable_short: bool = True,
    ) -> None:
        self._delay = delay
        self._decay = decay
        self._neutralization = neutralization
        self._truncation = truncation
        self._enable_long = enable_long
        self._enable_short = enable_short
        # Stateful buffers
        self._prev_alpha: dict[str, float] | None = None
        self._alpha_history: list[dict[str, float]] = []

    def select(
        self,
        factor_values: dict[str, float],
        current_long: set[str],
        current_short: set[str],
    ) -> list[TargetPosition] | None:
        """Run BRAIN pipeline and return target portfolio.

        Returns None during warmup (delay buffer not filled) to signal
        "no opinion" — caller should hold all positions as-is.
        """
        alpha = self._apply_delay(factor_values)
        if alpha is None:
            return None

        weights = self._process_alpha(alpha)
        if not weights:
            return []

        result = []
        for symbol, w in weights.items():
            if w > 1e-8 and self._enable_long:
                result.append(TargetPosition(symbol, w, factor_values.get(symbol, 0.0)))
            elif w < -1e-8 and self._enable_short:
                result.append(TargetPosition(symbol, w, factor_values.get(symbol, 0.0)))
        return sorted(result, key=lambda t: t.factor)

    # ------------------------------------------------------------------
    # Step 2: Delay
    # ------------------------------------------------------------------

    def _apply_delay(self, raw: dict[str, float]) -> dict[str, float] | None:
        """Apply delay buffer.

        delay=0: use current period's data directly.
        delay=1: use previous period's data (returns None during warmup).
        """
        if self._delay == 0:
            return raw
        prev = self._prev_alpha
        self._prev_alpha = raw
        return prev

    # ------------------------------------------------------------------
    # Steps 3-6: Process alpha
    # ------------------------------------------------------------------

    def _process_alpha(self, alpha: dict[str, float]) -> dict[str, float]:
        """Apply neutralize, scale, decay, truncate. Returns final weight vector."""
        alpha = {k: v for k, v in alpha.items() if not math.isnan(v)}
        if not alpha:
            return {}

        # Step 3: Market neutralization
        if self._neutralization == "MARKET":
            alpha = self._neutralize(alpha)

        # Step 4: Scale (sum(|alpha|) = 1)
        alpha = self._scale(alpha)

        # Step 5: Decay + re-scale
        alpha = self._apply_decay(alpha)
        alpha = self._scale(alpha)

        # Step 6: Iterative truncation + re-scale
        if self._truncation > 0:
            for _ in range(20):
                truncated = self._truncate(alpha)
                alpha = self._scale(truncated)
                if max(abs(v) for v in alpha.values()) <= self._truncation + 1e-10:
                    break

        return alpha

    # ------------------------------------------------------------------
    # Step 3: Neutralization
    # ------------------------------------------------------------------

    @staticmethod
    def _neutralize(alpha: dict[str, float]) -> dict[str, float]:
        """Market neutralization: subtract cross-sectional mean.

        Result: sum(alpha) = 0 → market neutral.
        """
        valid = {k: v for k, v in alpha.items() if not math.isnan(v)}
        if not valid:
            return alpha
        mean = sum(valid.values()) / len(valid)
        result = {k: v - mean for k, v in valid.items()}
        for k, v in alpha.items():
            if math.isnan(v):
                result[k] = v
        return result

    # ------------------------------------------------------------------
    # Step 4: Scale
    # ------------------------------------------------------------------

    @staticmethod
    def _scale(alpha: dict[str, float]) -> dict[str, float]:
        """Scale alpha vector so sum(|alpha|) = 1."""
        valid = {k: v for k, v in alpha.items() if not math.isnan(v)}
        total_abs = sum(abs(v) for v in valid.values())
        if total_abs == 0:
            return alpha
        return {k: v / total_abs for k, v in valid.items()}

    # ------------------------------------------------------------------
    # Step 5: Decay
    # ------------------------------------------------------------------

    def _apply_decay(self, alpha: dict[str, float]) -> dict[str, float]:
        """Apply linear decay weighted average to normalized weights.

        decay=0: return unchanged.
        decay=N: weighted average over last N periods.
        Weights: oldest=1, ..., newest=N.
        """
        if self._decay == 0:
            return alpha

        self._alpha_history.append(alpha)
        if len(self._alpha_history) > self._decay:
            self._alpha_history.pop(0)

        n = len(self._alpha_history)
        weights = list(range(1, n + 1))
        total_weight = sum(weights)

        all_keys: set[str] = set()
        for hist in self._alpha_history:
            all_keys.update(hist.keys())

        result: dict[str, float] = {}
        for key in sorted(all_keys):
            weighted_sum = sum(
                w * hist.get(key, 0.0)
                for w, hist in zip(weights, self._alpha_history)
            )
            result[key] = weighted_sum / total_weight
        return result

    # ------------------------------------------------------------------
    # Step 6: Truncation
    # ------------------------------------------------------------------

    def _truncate(self, alpha: dict[str, float]) -> dict[str, float]:
        """Cap individual instrument weights at truncation threshold."""
        t = self._truncation
        return {k: max(-t, min(t, v)) for k, v in alpha.items()}
