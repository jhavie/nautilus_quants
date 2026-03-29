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
        Used in "fixed" position_mode.
    total_capital : float, default 100_000.0
        Total capital for portfolio allocation in quote currency (USDT).
        Used in "weighted" position_mode: target = total_capital × |weight|.
    composite_factor : str, default "composite"
        Name of the composite factor to use for ranking.
    rebalance_interval : int, default 1
        Rebalance every N signals.
    selection_policy : str, default "FMZSelectionPolicy"
        Selection policy class name. Available:
        "FMZSelectionPolicy", "TopKDropoutSelectionPolicy",
        "WorldQuantSelectionPolicy".
    position_mode : str, default "fixed"
        Position sizing mode:
        - "fixed": use position_value per instrument, skip HOLD (no resize).
        - "equal_weight": NAV × long_share / N, continuous resize.
        - "weighted": total_capital × |weight| from policy, continuous resize.
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
    venue_name : str | None
        Venue name for NAV lookup (required for "equal_weight" position_mode).
    long_share : float, default 0.5
        Fraction of NAV allocated to long leg (equal_weight mode).
    min_rebalance_pct : float, default 0.05
        Minimum position value delta (as fraction of current value) to
        trigger a resize. Prevents excessive micro-adjustments.
    delay : int, default 1
        WorldQuant: data delay (0=current, 1=previous period).
    decay : int, default 0
        WorldQuant: linear decay window (0=no decay).
    neutralization : str, default "MARKET"
        WorldQuant: "MARKET" or "NONE".
    truncation : float, default 0.0
        WorldQuant: max weight per instrument (0.0=no truncation).
    enable_long : bool, default True
        WorldQuant: include long positions.
    enable_short : bool, default True
        WorldQuant: include short positions.
    """

    position_value: float = 300.0
    total_capital: float = 100_000.0
    composite_factor: str = "composite"
    rebalance_interval: int = 1
    selection_policy: str = "FMZSelectionPolicy"
    position_mode: str = "fixed"
    n_long: int = 40
    n_short: int = 40
    topk_long: int | None = None
    topk_short: int | None = None
    n_drop_long: int | None = None
    n_drop_short: int | None = None
    venue_name: str | None = None
    long_share: float = 0.5
    min_rebalance_pct: float = 0.05
    # WorldQuant BRAIN params
    delay: int = 1
    decay: int = 0
    neutralization: str = "MARKET"
    truncation: float = 0.0
    enable_long: bool = True
    enable_short: bool = True


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
    min_rebalance_pct : float, default 0.05
        Minimum position value delta (as fraction) to trigger resize.
    """

    instrument_ids: list[str]
    bar_types: list[str] = []
    execution_policy: str = "PostLimitExecutionPolicy"
    bracket: BracketConfig | None = None
    min_rebalance_pct: float = 0.05
