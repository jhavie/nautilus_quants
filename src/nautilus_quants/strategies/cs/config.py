# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Configuration classes for CS strategy components."""

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


class CSStrategyConfig(StrategyConfig, frozen=True):
    """
    Configuration for CSStrategy.

    Parameters
    ----------
    instrument_ids : list[str]
        List of instrument IDs to trade.
    bar_types : list[str], default []
        Bar type strings for price subscription (injected by CLI).
    execution_mode : str, default "market"
        Execution policy: "market", "anchor" (future), "post_limit" (future).
    exposure_policy : str, default "close_first"
        Exposure ordering: "close_first" or "open_first".
    """

    instrument_ids: list[str]
    bar_types: list[str] = []
    execution_mode: str = "market"
    exposure_policy: str = "close_first"
