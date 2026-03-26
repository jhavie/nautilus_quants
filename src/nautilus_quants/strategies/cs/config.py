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
    n_long : int, default 40
        Number of instruments to long (lowest composite factor values).
    n_short : int, default 40
        Number of instruments to short (highest composite factor values).
    position_value : float, default 300.0
        Fixed position value per instrument in quote currency (USDT).
    composite_factor : str, default "composite"
        Name of the composite factor to use for ranking.
    rebalance_interval : int, default 1
        Rebalance every N signals.
    """

    n_long: int = 40
    n_short: int = 40
    position_value: float = 300.0
    composite_factor: str = "composite"
    rebalance_interval: int = 1


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
