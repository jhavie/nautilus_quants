# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Configuration classes for CS strategy components."""

import msgspec

from nautilus_trader.config import ActorConfig, StrategyConfig


class DecisionEngineActorConfig(ActorConfig, frozen=True):
    """
    Configuration for DecisionEngineActor.

    Parameters
    ----------
    position_value : float, default 300.0
        Fixed position value per instrument in quote currency (USDT).
    composite_factor : str, default "composite"
        Name of the composite factor to use for ranking.
    rebalance_interval : int, default 1
        Rebalance every N signals.
    selection_policy : str, default "FMZSelectionPolicy"
        Selection policy class name. Available:
        "FMZSelectionPolicy", "TopKDropoutSelectionPolicy".
    n_long : int, default 40
        FMZ: number of instruments to long (lowest composite values).
    n_short : int, default 40
        FMZ: number of instruments to short (highest composite values).
    topk_long : int | None
        TopK: number of instruments in long pool.
    topk_short : int | None
        TopK: number of instruments in short pool.
    n_drop_long : int | None
        TopK: max instruments to rotate out of long pool per rebalance.
    n_drop_short : int | None
        TopK: max instruments to rotate out of short pool per rebalance.
    rebalance_to_weights : bool, default False
        When True, rebalance positions to target weights proportional to
        factor scores (qlib-style). When False, use fixed position_value.
    venue_name : str | None
        Venue name for NAV lookup (required when rebalance_to_weights=True).
    min_rebalance_pct : float, default 0.05
        Minimum position value delta (as fraction of current value) to
        trigger a resize. Prevents excessive micro-adjustments.
    """

    position_value: float = 300.0
    composite_factor: str = "composite"
    rebalance_interval: int = 1
    selection_policy: str = "FMZSelectionPolicy"
    n_long: int = 40
    n_short: int = 40
    topk_long: int | None = None
    topk_short: int | None = None
    n_drop_long: int | None = None
    n_drop_short: int | None = None
    rebalance_to_weights: bool = False
    venue_name: str | None = None
    min_rebalance_pct: float = 0.05


class BracketConfig(msgspec.Struct, frozen=True):
    """
    Configuration for bracket order TP/SL attachment.

    Parameters
    ----------
    take_profit_pct : float | None
        Take-profit percentage (e.g. 0.03 = 3%). None to disable TP.
    stop_loss_pct : float | None
        Stop-loss percentage (e.g. 0.02 = 2%). None to disable SL.
    tp_order_type : str
        TP order type: "LIMIT" or "MARKET_IF_TOUCHED".
    sl_order_type : str
        SL order type: "STOP_MARKET" or "STOP_LIMIT".
    entry_exec_algorithm_id : str | None
        Exec algorithm for the entry order (e.g. "PostLimit").
        Auto-set by CSStrategy when execution_policy is PostLimitExecutionPolicy.
    """

    take_profit_pct: float | None = None
    stop_loss_pct: float | None = None
    tp_order_type: str = "LIMIT"
    sl_order_type: str = "STOP_MARKET"
    entry_exec_algorithm_id: str | None = None


class CSStrategyConfig(StrategyConfig, frozen=True):
    """
    Configuration for CSStrategy.

    Parameters
    ----------
    instrument_ids : list[str]
        List of instrument IDs to trade.
    bar_types : list[str], default []
        Bar type strings for price subscription (injected by CLI).
    execution_policy : str, default "PostLimitExecutionPolicy"
        Execution policy class name. Available:
        "MarketExecutionPolicy", "PostLimitExecutionPolicy".
        Production configuration should use the explicit
        "PostLimitExecutionPolicy" string.
    bracket : BracketConfig | None
        Bracket order config for TP/SL attachment. None to disable.
    """

    instrument_ids: list[str]
    bar_types: list[str] = []
    execution_policy: str = "PostLimitExecutionPolicy"
    bracket: BracketConfig | None = None
    min_rebalance_pct: float = 0.05
