"""EquityGuardController - dual-layer equity protection with rolling drawdown.

Layer 1 (Rolling Drawdown): Monitors MTM equity against the peak within a
configurable window. When current/peak drops below drawdown_ratio, stops
guarded strategies and starts a cooldown timer for automatic restart.

Layer 2 (Absolute Guard): When equity drops below min_equity_ratio of the
initial balance, permanently stops strategies (no cooldown restart).

Usage (YAML engine.controller config):
    controller:
      controller_path: "nautilus_quants.controllers.equity_guard:EquityGuardController"
      config_path: "nautilus_quants.controllers.equity_guard:EquityGuardControllerConfig"
      config:
        interval: "1h"
        venue_name: "OKX"
        currency: "USDT"
        drawdown_window: "72h"
        drawdown_ratio: 0.7
        min_equity_ratio: 0.3
        cooldown_period: "24h"
        guarded_strategy_ids: []
"""

from __future__ import annotations

from collections import deque

from nautilus_trader.config import ControllerConfig
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Currency
from nautilus_trader.trading.controller import Controller
from nautilus_trader.trading.strategy import Strategy
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
    drawdown_window : str, default ""
        Rolling window for peak detection (e.g., "72h", "30d").
        Empty string = all-time peak (no pruning).
    drawdown_ratio : float, default 0.7
        Drawdown threshold: halt when current/peak < this value.
    min_equity_ratio : float, default 0.1
        Absolute guard: halt permanently when current/initial < this value.
    guarded_strategy_ids : list[str], default []
        Strategy IDs to stop on breach. Empty = all strategies.
    cooldown_period : str, default ""
        Cooldown before restarting after drawdown halt (e.g., "24h").
        Empty = no restart. Only applies to drawdown halt, not absolute guard.
    """

    interval: str = "1h"
    venue_name: str = "SIM"
    currency: str = "USD"
    drawdown_window: str = ""
    drawdown_ratio: float = 0.7
    min_equity_ratio: float = 0.1
    guarded_strategy_ids: list[str] = []
    cooldown_period: str = ""


class EquityGuardController(Controller):
    """Dual-layer equity protection with rolling drawdown and absolute guard.

    Layer 1 — Rolling Drawdown:
        Tracks equity history in a deque. On each check, finds peak within
        the configured window (or all-time if window is empty). Halts when
        current/peak < drawdown_ratio. Sets cooldown timer for restart.

    Layer 2 — Absolute Guard:
        Halts permanently when current/initial_balance < min_equity_ratio.
        No cooldown, no restart.
    """

    def __init__(self, config: EquityGuardControllerConfig, trader: Trader) -> None:
        super().__init__(trader=trader, config=config)
        self._venue = Venue(config.venue_name)
        self._currency = Currency.from_str(config.currency)
        self._initial_balance: float | None = None
        self._halted: bool = False
        self._stopped_strategies: list[Strategy] = []
        self._equity_history: deque[tuple[int, float]] = deque()

    def on_start(self) -> None:
        """Register equity check timer and record initial balance."""
        interval_td = parse_interval_to_timedelta(self.config.interval)
        self.clock.set_timer(
            name="equity_guard",
            interval=interval_td,
            callback=self._on_check,
        )
        self._initial_balance = compute_mtm_equity(
            self.portfolio, self._venue, self._currency,
        )
        if self._initial_balance is not None:
            self._equity_history.append(
                (self.clock.timestamp_ns(), self._initial_balance),
            )
        self.log.info(
            f"EquityGuardController started: "
            f"initial_balance={self._initial_balance}, "
            f"drawdown_window={self.config.drawdown_window or 'all-time'}, "
            f"drawdown_ratio={self.config.drawdown_ratio}, "
            f"min_equity_ratio={self.config.min_equity_ratio}, "
            f"cooldown={self.config.cooldown_period or 'disabled'}, "
            f"guarded={self.config.guarded_strategy_ids or 'ALL'}"
        )

    def _on_check(self, event: object) -> None:
        """Timer callback: dual-layer equity check."""
        if self._halted:
            return
        if self._initial_balance is None or self._initial_balance <= 0:
            return

        equity = compute_mtm_equity(self.portfolio, self._venue, self._currency)
        if equity is None:
            return

        now_ns = self.clock.timestamp_ns()
        self._equity_history.append((now_ns, equity))

        # Layer 2: Absolute guard (highest priority, permanent halt)
        abs_ratio = equity / self._initial_balance
        if abs_ratio < self.config.min_equity_ratio:
            self.log.warning(
                f"ABSOLUTE_GUARD: equity={equity:.2f}, "
                f"ratio={abs_ratio:.4f} < {self.config.min_equity_ratio}"
            )
            self._halt_guarded_strategies(equity, abs_ratio, permanent=True)
            return

        # Layer 1: Rolling drawdown
        if self.config.drawdown_window:
            window_td = parse_interval_to_timedelta(self.config.drawdown_window)
            window_ns = int(window_td.total_seconds() * 1_000_000_000)
            cutoff_ns = now_ns - window_ns
            while self._equity_history and self._equity_history[0][0] < cutoff_ns:
                self._equity_history.popleft()

        if not self._equity_history:
            return

        peak = max(eq for _, eq in self._equity_history)
        dd_ratio = equity / peak
        if dd_ratio < self.config.drawdown_ratio:
            self.log.warning(
                f"ROLLING_DRAWDOWN: equity={equity:.2f}, "
                f"peak={peak:.2f} "
                f"({self.config.drawdown_window or 'all-time'}), "
                f"ratio={dd_ratio:.4f} < {self.config.drawdown_ratio}"
            )
            self._halt_guarded_strategies(equity, dd_ratio, permanent=False)

    def _halt_guarded_strategies(
        self,
        equity: float,
        ratio: float,
        *,
        permanent: bool = False,
    ) -> None:
        """Stop guarded strategies.

        Parameters
        ----------
        permanent : bool
            If True, no cooldown timer is set (absolute guard).
        """
        self._halted = True
        guarded_ids = set(self.config.guarded_strategy_ids)
        self._stopped_strategies = []
        for strategy in self._trader.strategies():
            if not guarded_ids or str(strategy.id) in guarded_ids:
                if strategy.is_running:
                    self.stop_strategy(strategy)
                    self._stopped_strategies.append(strategy)
                    self.log.info(f"Stopped strategy: {strategy.id}")

        if not permanent and self.config.cooldown_period:
            cooldown_td = parse_interval_to_timedelta(self.config.cooldown_period)
            cooldown_ns = int(cooldown_td.total_seconds() * 1_000_000_000)
            self.clock.set_time_alert_ns(
                name="equity_guard_cooldown",
                alert_time_ns=self.clock.timestamp_ns() + cooldown_ns,
                callback=self._on_cooldown_expired,
            )
            self.log.info(
                f"Cooldown timer set: {self.config.cooldown_period}. "
                f"Strategies will restart after cooldown."
            )

    def _on_cooldown_expired(self, event: object) -> None:
        """Cooldown timer callback: restart previously stopped strategies."""
        self.log.info("Cooldown expired. Restarting strategies.")
        for strategy in self._stopped_strategies:
            self.start_strategy(strategy)
            self.log.info(f"Restarted strategy: {strategy.id}")

        # Reset baseline and history
        new_equity = compute_mtm_equity(
            self.portfolio, self._venue, self._currency,
        )
        self._initial_balance = new_equity
        self._stopped_strategies = []
        self._halted = False
        self._equity_history.clear()
        if new_equity is not None:
            self._equity_history.append(
                (self.clock.timestamp_ns(), new_equity),
            )
        self.log.info(
            f"EquityGuardController reset: new_baseline={self._initial_balance}"
        )
