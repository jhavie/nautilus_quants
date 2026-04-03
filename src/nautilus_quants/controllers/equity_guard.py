# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""EquityGuardController - multi-rule equity protection with rolling drawdown.

Supports multiple DrawdownRule entries, each with its own window, threshold,
and cooldown. On each check, all rules are evaluated independently. If any
rule triggers, strategies are halted with the longest cooldown among triggered
rules (or permanently if any triggered rule has no cooldown).

Layer 2 (Absolute Guard) remains a single min_equity_ratio threshold that
permanently halts strategies when equity drops below that ratio of the
initial balance.

Usage (YAML engine.controller config):
    controller:
      controller_path: "nautilus_quants.controllers.equity_guard:EquityGuardController"
      config_path: "nautilus_quants.controllers.equity_guard:EquityGuardControllerConfig"
      config:
        interval: "1h"
        venue_name: "OKX"
        currency: "USDT"
        min_equity_ratio: 0.3
        guarded_strategy_ids: []
        drawdown_rules:
          - name: "monthly"
            drawdown_window: "30d"
            drawdown_ratio: 0.8
            cooldown_period: "72h"
          - name: "daily"
            drawdown_window: "1d"
            drawdown_ratio: 0.95
            cooldown_period: "24h"
"""

from __future__ import annotations

from collections import deque

import msgspec
from nautilus_trader.config import ControllerConfig
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Currency
from nautilus_trader.trading.controller import Controller
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.trading.trader import Trader

from nautilus_quants.utils.bar_spec import parse_interval_to_timedelta
from nautilus_quants.utils.equity import compute_mtm_equity


class DrawdownRule(msgspec.Struct, kw_only=True, frozen=True):
    """A single drawdown protection rule.

    Parameters
    ----------
    name : str, default ""
        Human-readable name for logging (auto-numbered if empty).
    drawdown_window : str, default ""
        Rolling window for peak detection (e.g., "72h", "30d").
        Empty string = all-time peak (no pruning).
    drawdown_ratio : float, default 0.7
        Halt when current/peak < this value.
    cooldown_period : str, default ""
        Cooldown before restarting after halt (e.g., "24h").
        Empty = permanent halt (no restart).
    """

    name: str = ""
    drawdown_window: str = ""
    drawdown_ratio: float = 0.7
    cooldown_period: str = ""


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
        Absolute guard: halt permanently when current/initial < this value.
    guarded_strategy_ids : list[str], default []
        Strategy IDs to stop on breach. Empty = all strategies.
    drawdown_rules : list[DrawdownRule], default []
        List of drawdown rules. Each rule specifies a window, threshold,
        and cooldown. All rules are evaluated independently on each check.
        When multiple rules trigger, the longest cooldown is used.
        If any triggered rule has no cooldown, halt is permanent.
    """

    interval: str = "1h"
    venue_name: str = "SIM"
    currency: str = "USD"
    min_equity_ratio: float = 0.1
    guarded_strategy_ids: list[str] = []
    drawdown_rules: list[DrawdownRule] = []


class EquityGuardController(Controller):
    """Multi-rule equity protection with rolling drawdown and absolute guard.

    Layer 1 — Drawdown Rules:
        Evaluates all configured DrawdownRule entries on each check.
        Each rule tracks equity peak within its own window and triggers
        when current/peak drops below the rule's drawdown_ratio.
        When multiple rules trigger, the longest cooldown is used.

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

        # Resolve rules with auto-naming
        self._rules: tuple[DrawdownRule, ...] = tuple(
            DrawdownRule(
                name=r.name or f"rule_{i}",
                drawdown_window=r.drawdown_window,
                drawdown_ratio=r.drawdown_ratio,
                cooldown_period=r.cooldown_period,
            )
            if not r.name
            else r
            for i, r in enumerate(config.drawdown_rules)
        )

        # Pre-compute max window for global history pruning.
        # None means all-time (no pruning) — when any rule has empty window.
        self._max_window_ns: int | None = self._compute_max_window_ns()

    def _compute_max_window_ns(self) -> int | None:
        if not self._rules:
            return 0
        if any(not r.drawdown_window for r in self._rules):
            return None  # All-time: keep everything
        max_td = max(
            parse_interval_to_timedelta(r.drawdown_window) for r in self._rules
        )
        return int(max_td.total_seconds() * 1_000_000_000)

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
        rules_desc = ", ".join(
            f"{r.name}(w={r.drawdown_window or 'all-time'}, "
            f"r={r.drawdown_ratio}, cd={r.cooldown_period or 'permanent'})"
            for r in self._rules
        ) or "none"
        self.log.info(
            f"EquityGuardController started: "
            f"initial_balance={self._initial_balance}, "
            f"min_equity_ratio={self.config.min_equity_ratio}, "
            f"rules=[{rules_desc}], "
            f"guarded={self.config.guarded_strategy_ids or 'ALL'}"
        )

    def _on_check(self, event: object) -> None:
        """Timer callback: evaluate absolute guard then all drawdown rules."""
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

        # No drawdown rules configured — only absolute guard applies
        if not self._rules:
            return

        # Global history pruning to max window
        if self._max_window_ns is not None and self._max_window_ns > 0:
            cutoff_ns = now_ns - self._max_window_ns
            while self._equity_history and self._equity_history[0][0] < cutoff_ns:
                self._equity_history.popleft()

        if not self._equity_history:
            return

        # Layer 1: Evaluate all drawdown rules
        triggered: list[tuple[DrawdownRule, float, float]] = []  # (rule, peak, ratio)
        for rule in self._rules:
            peak = self._peak_in_window(now_ns, rule.drawdown_window)
            if peak is None or peak <= 0:
                continue
            dd_ratio = equity / peak
            if dd_ratio < rule.drawdown_ratio:
                triggered.append((rule, peak, dd_ratio))

        if not triggered:
            return

        # Log all triggered rules
        worst_ratio = min(dd for _, _, dd in triggered)
        for rule, peak, dd_ratio in triggered:
            self.log.warning(
                f"DRAWDOWN_BREACH [{rule.name}]: equity={equity:.2f}, "
                f"peak={peak:.2f} "
                f"({rule.drawdown_window or 'all-time'}), "
                f"ratio={dd_ratio:.4f} < {rule.drawdown_ratio}"
            )

        # Determine effective cooldown: permanent if any has no cooldown,
        # otherwise use the longest cooldown among triggered rules.
        if any(not r.cooldown_period for r, _, _ in triggered):
            self._halt_guarded_strategies(equity, worst_ratio, permanent=True)
        else:
            cooldowns = [
                parse_interval_to_timedelta(r.cooldown_period) for r, _, _ in triggered
            ]
            longest = max(cooldowns)
            longest_idx = cooldowns.index(longest)
            cooldown_str = triggered[longest_idx][0].cooldown_period
            self._halt_guarded_strategies(
                equity, worst_ratio, cooldown_period=cooldown_str,
            )

    def _peak_in_window(self, now_ns: int, window: str) -> float | None:
        """Return peak equity within the given window, or all-time if empty."""
        if not self._equity_history:
            return None
        if not window:
            return max(eq for _, eq in self._equity_history)
        window_td = parse_interval_to_timedelta(window)
        window_ns = int(window_td.total_seconds() * 1_000_000_000)
        cutoff_ns = now_ns - window_ns
        in_window = [eq for ts, eq in self._equity_history if ts >= cutoff_ns]
        return max(in_window) if in_window else None

    def _halt_guarded_strategies(
        self,
        equity: float,
        ratio: float,
        *,
        permanent: bool = False,
        cooldown_period: str = "",
    ) -> None:
        """Stop guarded strategies.

        Parameters
        ----------
        permanent : bool
            If True, no cooldown timer is set (absolute guard or rule
            without cooldown).
        cooldown_period : str
            Cooldown duration string (e.g., "24h"). Only used when
            permanent is False.
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

        if not permanent and cooldown_period:
            cooldown_td = parse_interval_to_timedelta(cooldown_period)
            cooldown_ns = int(cooldown_td.total_seconds() * 1_000_000_000)
            self.clock.set_time_alert_ns(
                name="equity_guard_cooldown",
                alert_time_ns=self.clock.timestamp_ns() + cooldown_ns,
                callback=self._on_cooldown_expired,
            )
            self.log.info(
                f"Cooldown timer set: {cooldown_period}. "
                f"Strategies will restart after cooldown."
            )

    def _on_cooldown_expired(self, event: object) -> None:
        """Cooldown timer callback: resume previously stopped strategies."""
        new_equity = compute_mtm_equity(
            self.portfolio, self._venue, self._currency,
        )
        if new_equity is None:
            self.log.warning(
                "Cannot retrieve equity at cooldown expiry, staying halted.",
            )
            return

        self.log.info("Cooldown expired. Resuming strategies.")
        for strategy in self._stopped_strategies:
            strategy.resume()
            self.log.info(f"Resumed strategy: {strategy.id}")

        self._initial_balance = new_equity
        self._stopped_strategies = []
        self._halted = False
        self._equity_history.clear()
        self._equity_history.append(
            (self.clock.timestamp_ns(), new_equity),
        )
        self.log.info(
            f"EquityGuardController reset: new_baseline={self._initial_balance}"
        )
