# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Configuration for PostLimitExecAlgorithm."""

from __future__ import annotations

from nautilus_trader.config import ExecAlgorithmConfig
from nautilus_trader.model.identifiers import ExecAlgorithmId


class PostLimitExecAlgorithmConfig(ExecAlgorithmConfig, frozen=True):
    """Configuration for the PostLimit execution algorithm.

    Parameters
    ----------
    exec_algorithm_id : ExecAlgorithmId, default "PostLimit"
        Unique algorithm identifier.
    offset_ticks : int, default 0
        Initial price offset from BBO in ticks.
        BUY:  limit = best_bid + offset_ticks * tick_size
        SELL: limit = best_ask - offset_ticks * tick_size
        0 = pure BBO pegging, 1 = improve BBO by 1 tick, -1 = retreat 1 tick.
    timeout_secs : float, default 15.0
        Seconds to wait before canceling and re-posting (or falling back).
    max_chase_attempts : int, default 3
        Maximum number of chase (cancel + re-post) cycles.
    chase_step_ticks : int, default 1
        Additional tick offset per chase iteration (cumulative with offset_ticks).
    fallback_to_market : bool, default True
        Submit a market order when max chases are exhausted.
    post_only : bool, default True
        Use post-only / maker-only flag on limit orders.
        When True the algorithm clamps the price so it never crosses the
        opposite BBO (prevents taker fills on exchanges that reject post-only
        cross orders).
    """

    exec_algorithm_id: ExecAlgorithmId | None = ExecAlgorithmId("PostLimit")
    offset_ticks: int = 0
    timeout_secs: float = 15.0
    max_chase_attempts: int = 3
    chase_step_ticks: int = 1
    fallback_to_market: bool = True
    post_only: bool = True
