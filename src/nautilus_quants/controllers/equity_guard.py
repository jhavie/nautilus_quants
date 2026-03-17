"""EquityGuardController - simulated liquidation for backtesting.

Monitors MTM equity at regular intervals. When equity drops below
min_equity_ratio × initial_balance, stops guarded strategies
(triggering their on_stop → close_all_positions).

Usage (YAML engine.controller config):
    controller:
      controller_path: "nautilus_quants.controllers.equity_guard:EquityGuardController"
      config_path: "nautilus_quants.controllers.equity_guard:EquityGuardControllerConfig"
      config:
        interval: "1h"
        venue_name: "BINANCE"
        currency: "USDT"
        min_equity_ratio: 0.1
        guarded_strategy_ids: []
"""

from __future__ import annotations

from nautilus_trader.config import ControllerConfig
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Currency
from nautilus_trader.trading.controller import Controller
from nautilus_trader.trading.trader import Trader

from nautilus_quants.utils.bar_spec import parse_interval_to_timedelta
from nautilus_quants.utils.equity import compute_mtm_equity


class EquityGuardControllerConfig(ControllerConfig, kw_only=True, frozen=True):
    """Configuration for EquityGuardController.

    Parameters
    ----------
    interval : str, default "1h"
        Equity check interval (e.g., "1h", "4h", "8h").
    venue_name : str, default "SIM"
        Venue name.
    currency : str, default "USD"
        Settlement currency.
    min_equity_ratio : float, default 0.1
        Minimum equity/initial_balance ratio before halting.
    guarded_strategy_ids : list[str], default []
        Strategy IDs to stop on breach. Empty = all strategies.
    """

    interval: str = "1h"
    venue_name: str = "SIM"
    currency: str = "USD"
    min_equity_ratio: float = 0.1
    guarded_strategy_ids: list[str] = []


class EquityGuardController(Controller):
    """Simulated liquidation via Controller.

    Monitors MTM equity at regular intervals. When equity drops below
    min_equity_ratio × initial_balance, stops the guarded strategies
    (triggering their on_stop → close_all_positions).
    """

    def __init__(self, config: EquityGuardControllerConfig, trader: Trader) -> None:
        super().__init__(trader=trader, config=config)
        self._venue = Venue(config.venue_name)
        self._currency = Currency.from_str(config.currency)
        self._initial_balance: float | None = None
        self._halted: bool = False

    def on_start(self) -> None:
        """Register equity check timer and record initial balance."""
        interval_td = parse_interval_to_timedelta(self.config.interval)
        self.clock.set_timer(
            name="equity_guard",
            interval=interval_td,
            callback=self._on_check,
        )
        # Record initial balance as baseline
        self._initial_balance = compute_mtm_equity(
            self.portfolio,
            self._venue,
            self._currency,
        )
        self.log.info(
            f"EquityGuardController started: "
            f"initial_balance={self._initial_balance}, "
            f"min_equity_ratio={self.config.min_equity_ratio}, "
            f"guarded={self.config.guarded_strategy_ids or 'ALL'}"
        )

    def _on_check(self, event: object) -> None:
        """Timer callback: check equity ratio and halt if breached."""
        if self._halted:
            return
        if self._initial_balance is None or self._initial_balance <= 0:
            return

        equity = compute_mtm_equity(self.portfolio, self._venue, self._currency)
        if equity is None:
            return

        ratio = equity / self._initial_balance
        if ratio < self.config.min_equity_ratio:
            self._halt_guarded_strategies(equity, ratio)

    def _halt_guarded_strategies(self, equity: float, ratio: float) -> None:
        """Stop guarded strategies when equity ratio breached."""
        self._halted = True
        self.log.warning(
            f"EquityGuardController triggered: equity={equity:.2f}, "
            f"ratio={ratio:.4f} < {self.config.min_equity_ratio}. "
            f"Stopping guarded strategies."
        )
        guarded_ids = set(self.config.guarded_strategy_ids)
        for strategy in self._trader.strategies():
            if not guarded_ids or str(strategy.id) in guarded_ids:
                if strategy.is_running:
                    self.stop_strategy(strategy)
                    self.log.info(f"Stopped strategy: {strategy.id}")
