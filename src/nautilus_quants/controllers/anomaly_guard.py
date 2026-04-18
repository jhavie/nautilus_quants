# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""AnomalyGuardController - multi-rule anomaly protection with countdown confirmation.

Supersedes the former ``EquityGuardController``. Supports three rule kinds:

- ``equity_drawdown`` — rolling-window drawdown on either mark-to-market equity
  (``balance_total + unrealized_pnl``) or realized balance only
  (``balance_total``). Configurable per rule via ``equity_basis``.
- ``position_count`` — long/short position count deviates from planned counts
  beyond tolerance; reads ``snapshot:venue`` from the cache.
- ``notional_drift`` — per-position notional value drifts outside
  ``[target * min_drift_ratio, target * max_drift_ratio]`` for at least
  ``min_violators`` positions; reads ``snapshot:venue`` from the cache.

Rules use sustained-periods debouncing: a rule must produce ``sustained_periods``
consecutive check hits before it becomes eligible to fire a countdown. Once any
rule becomes eligible, a single global countdown timer starts. While the
countdown runs, re-checks continue; if ``cancel_on_recovery`` is true and no
rule is eligible anymore, the countdown is cancelled. When the countdown
expires, the controller picks the longest cooldown among rules still eligible
(or a permanent halt if any such rule has no cooldown) and stops all guarded
strategies. On cooldown expiry, strategies are resumed and per-rule state is
reset.

Usage (YAML engine.controller config)::

    controller:
      controller_path: "nautilus_quants.controllers.anomaly_guard:AnomalyGuardController"
      config_path: "nautilus_quants.controllers.anomaly_guard:AnomalyGuardControllerConfig"
      config:
        interval: "30m"
        venue_name: "OKX"
        currency: "USDT"
        guarded_strategy_ids: []
        snapshot_staleness_secs: 120
        countdown_period: "5m"
        cancel_on_recovery: true
        anomaly_rules:
          - name: "position_count"
            kind: "position_count"
            expected_long: 20
            expected_short: 20
            tolerance: 2
            sustained_periods: 3
            cooldown_period: "2h"
          - name: "per_position_notional"
            kind: "notional_drift"
            target_notional_usd: 200.0
            max_drift_ratio: 1.5
            min_drift_ratio: 0.5
            min_violators: 3
            sustained_periods: 2
            cooldown_period: "4h"
          - name: "monthly"
            kind: "equity_drawdown"
            equity_basis: "mark_to_market"
            drawdown_window: "30d"
            drawdown_ratio: 0.8
            sustained_periods: 1
            cooldown_period: "72h"
          - name: "daily"
            kind: "equity_drawdown"
            equity_basis: "realized_balance"
            drawdown_window: "1d"
            drawdown_ratio: 0.95
            sustained_periods: 1
            cooldown_period: "24h"
"""

from __future__ import annotations

import json
from collections import deque
from typing import Any

import msgspec
from nautilus_trader.config import ControllerConfig
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Currency
from nautilus_trader.portfolio.base import PortfolioFacade
from nautilus_trader.trading.controller import Controller
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.trading.trader import Trader

from nautilus_quants.utils.bar_spec import parse_interval_to_timedelta
from nautilus_quants.utils.cache_keys import SNAPSHOT_VENUE_CACHE_KEY
from nautilus_quants.utils.equity import compute_mtm_equity

KIND_EQUITY_DRAWDOWN = "equity_drawdown"
KIND_POSITION_COUNT = "position_count"
KIND_NOTIONAL_DRIFT = "notional_drift"

BASIS_MARK_TO_MARKET = "mark_to_market"
BASIS_REALIZED_BALANCE = "realized_balance"

_COUNTDOWN_TIMER_NAME = "anomaly_guard_countdown"
_COOLDOWN_TIMER_NAME = "anomaly_guard_cooldown"


class AnomalyRule(msgspec.Struct, kw_only=True, frozen=True):
    """A single anomaly protection rule.

    Common parameters
    -----------------
    name : str
        Human-readable rule name (auto-numbered if empty).
    kind : str
        One of ``equity_drawdown`` / ``position_count`` / ``notional_drift``.
    sustained_periods : int, default 1
        Consecutive check hits required before this rule becomes eligible
        to fire a countdown.
    cooldown_period : str, default ""
        Cooldown duration after halt (e.g., ``"24h"``). Empty → permanent halt.

    Kind-specific parameters — ``equity_drawdown``
    ---------------------------------------------
    equity_basis : str, default ``mark_to_market``
        ``mark_to_market`` → ``balance_total + unrealized_pnl``.
        ``realized_balance`` → ``balance_total`` only (ignore unrealized).
    drawdown_window : str, default ""
        Rolling window for peak detection (e.g., ``"72h"``, ``"30d"``).
        Empty = all-time peak (no pruning).
    drawdown_ratio : float, default 0.7
        Hit when ``current_equity / peak_in_window < this``.

    Kind-specific parameters — ``position_count``
    --------------------------------------------
    expected_long : int, default 0
    expected_short : int, default 0
    tolerance : int, default 0
        Hit when ``|actual_long - expected_long| > tolerance``
        or ``|actual_short - expected_short| > tolerance``.

    Kind-specific parameters — ``notional_drift``
    --------------------------------------------
    target_notional_usd : float, default 0.0
        Planned per-position notional value.
    max_drift_ratio : float, default 0.0
        Upper bound expressed as ``target * max_drift_ratio``. 0 disables.
    min_drift_ratio : float, default 0.0
        Lower bound expressed as ``target * min_drift_ratio``. 0 disables.
    min_violators : int, default 1
        Minimum number of positions outside the band to count as a hit.
    """

    name: str = ""
    kind: str = ""
    sustained_periods: int = 1
    cooldown_period: str = ""
    # equity_drawdown
    equity_basis: str = BASIS_MARK_TO_MARKET
    drawdown_window: str = ""
    drawdown_ratio: float = 0.7
    # position_count
    expected_long: int = 0
    expected_short: int = 0
    tolerance: int = 0
    # notional_drift
    target_notional_usd: float = 0.0
    max_drift_ratio: float = 0.0
    min_drift_ratio: float = 0.0
    min_violators: int = 1


class AnomalyGuardControllerConfig(ControllerConfig, kw_only=True, frozen=True):
    """Configuration for AnomalyGuardController.

    Parameters
    ----------
    interval : str, default "30m"
        Check interval (e.g., ``"30m"``, ``"1h"``).
    venue_name : str, default "SIM"
        Venue name used for equity lookup.
    currency : str, default "USD"
        Settlement currency used for equity lookup.
    guarded_strategy_ids : list[str], default []
        Strategy IDs to stop on breach. Empty = all strategies.
    snapshot_staleness_secs : int, default 120
        Max age of ``snapshot:venue`` before it is considered stale and
        position/notional rules are skipped for that tick.
    countdown_period : str, default "5m"
        Countdown confirmation window between the first sustained hit and
        actual halt.
    cancel_on_recovery : bool, default True
        When True, an active countdown is cancelled if all rules return to
        a non-eligible state (``consecutive_hits < sustained_periods``).
    anomaly_rules : list[AnomalyRule], default []
        Rule definitions evaluated on each check.
    """

    interval: str = "30m"
    venue_name: str = "SIM"
    currency: str = "USD"
    guarded_strategy_ids: list[str] = []
    snapshot_staleness_secs: int = 120
    countdown_period: str = "5m"
    cancel_on_recovery: bool = True
    anomaly_rules: list[AnomalyRule] = []


class AnomalyGuardController(Controller):
    """Multi-rule anomaly protection controller."""

    def __init__(
        self,
        config: AnomalyGuardControllerConfig,
        trader: Trader,
    ) -> None:
        super().__init__(trader=trader, config=config)
        self._venue = Venue(config.venue_name)
        self._currency = Currency.from_str(config.currency)

        self._rules: tuple[AnomalyRule, ...] = tuple(
            r if r.name else msgspec.structs.replace(r, name=f"rule_{i}") for i, r in enumerate(config.anomaly_rules)
        )

        # Per-rule state
        self._hits: dict[str, int] = {r.name: 0 for r in self._rules}
        self._equity_history: dict[str, deque[tuple[int, float]]] = {
            r.name: deque() for r in self._rules if r.kind == KIND_EQUITY_DRAWDOWN
        }

        # Global countdown / halt state
        self._countdown_active: bool = False
        self._halted: bool = False
        self._stopped_strategies: list[Strategy] = []

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def on_start(self) -> None:
        """Register the periodic check timer."""
        interval_td = parse_interval_to_timedelta(self.config.interval)
        self.clock.set_timer(
            name="anomaly_guard",
            interval=interval_td,
            callback=self._on_check,
        )
        rules_desc = (
            ", ".join(
                f"{r.name}[{r.kind},sustained={r.sustained_periods}," f"cd={r.cooldown_period or 'permanent'}]"
                for r in self._rules
            )
            or "none"
        )
        self.log.info(
            f"AnomalyGuardController started: interval={self.config.interval}, "
            f"countdown={self.config.countdown_period}, "
            f"cancel_on_recovery={self.config.cancel_on_recovery}, "
            f"guarded={self.config.guarded_strategy_ids or 'ALL'}, "
            f"rules=[{rules_desc}]"
        )

    # -------------------------------------------------------------------------
    # Periodic check
    # -------------------------------------------------------------------------

    def _on_check(self, event: object) -> None:
        """Timer callback: evaluate every rule and manage countdown state."""
        if self._halted:
            return
        if not self._rules:
            return

        now_ns = self.clock.timestamp_ns()
        snapshot_venue = self._read_venue_snapshot(now_ns)

        # Evaluate every rule: record equity samples for drawdown rules,
        # update consecutive-hit counter based on the rule's evaluation.
        for rule in self._rules:
            hit, diag = self._evaluate_rule(rule, now_ns, snapshot_venue)
            if hit:
                self._hits[rule.name] += 1
                self.log.warning(
                    f"ANOMALY_HIT [{rule.name}/{rule.kind}]: " f"hits={self._hits[rule.name]}/{rule.sustained_periods} {diag}"
                )
            else:
                if self._hits[rule.name] > 0:
                    self.log.info(f"ANOMALY_CLEAR [{rule.name}/{rule.kind}]: reset hits")
                self._hits[rule.name] = 0

        any_eligible = any(self._hits[r.name] >= r.sustained_periods for r in self._rules)

        if not self._countdown_active and any_eligible:
            self._start_countdown(now_ns)
        elif self._countdown_active and not any_eligible and self.config.cancel_on_recovery:
            self._cancel_countdown()

    # -------------------------------------------------------------------------
    # Rule evaluation
    # -------------------------------------------------------------------------

    def _evaluate_rule(
        self,
        rule: AnomalyRule,
        now_ns: int,
        snapshot_venue: dict[str, Any] | None,
    ) -> tuple[bool, str]:
        if rule.kind == KIND_EQUITY_DRAWDOWN:
            return self._eval_equity_drawdown(rule, now_ns)
        if rule.kind == KIND_POSITION_COUNT:
            return self._eval_position_count(rule, snapshot_venue)
        if rule.kind == KIND_NOTIONAL_DRIFT:
            return self._eval_notional_drift(rule, snapshot_venue)
        self.log.warning(f"Unknown rule kind: {rule.kind} ({rule.name})")
        return False, ""

    def _eval_equity_drawdown(self, rule: AnomalyRule, now_ns: int) -> tuple[bool, str]:
        equity = self._compute_equity_for_basis(rule.equity_basis)
        if equity is None:
            return False, ""

        history = self._equity_history[rule.name]
        history.append((now_ns, equity))

        # Prune to window (all-time → keep everything).
        if rule.drawdown_window:
            window_td = parse_interval_to_timedelta(rule.drawdown_window)
            window_ns = int(window_td.total_seconds() * 1_000_000_000)
            cutoff_ns = now_ns - window_ns
            while history and history[0][0] < cutoff_ns:
                history.popleft()

        if not history:
            return False, ""

        peak = max(eq for _, eq in history)
        if peak <= 0:
            return False, ""
        ratio = equity / peak
        if ratio < rule.drawdown_ratio:
            return True, (
                f"equity={equity:.2f} ({rule.equity_basis}), "
                f"peak={peak:.2f} (w={rule.drawdown_window or 'all-time'}), "
                f"ratio={ratio:.4f} < {rule.drawdown_ratio}"
            )
        return False, ""

    def _eval_position_count(self, rule: AnomalyRule, snapshot_venue: dict[str, Any] | None) -> tuple[bool, str]:
        if snapshot_venue is None:
            return False, ""
        summary = snapshot_venue.get("summary") or {}
        actual_long = int(summary.get("long_count", 0))
        actual_short = int(summary.get("short_count", 0))
        long_dev = abs(actual_long - rule.expected_long)
        short_dev = abs(actual_short - rule.expected_short)
        if long_dev > rule.tolerance or short_dev > rule.tolerance:
            return True, (
                f"long={actual_long}/{rule.expected_long} (dev={long_dev}), "
                f"short={actual_short}/{rule.expected_short} (dev={short_dev}), "
                f"tolerance={rule.tolerance}"
            )
        return False, ""

    def _eval_notional_drift(self, rule: AnomalyRule, snapshot_venue: dict[str, Any] | None) -> tuple[bool, str]:
        if snapshot_venue is None or rule.target_notional_usd <= 0:
            return False, ""
        positions = snapshot_venue.get("positions") or []
        max_bound = rule.target_notional_usd * rule.max_drift_ratio if rule.max_drift_ratio > 0 else float("inf")
        min_bound = rule.target_notional_usd * rule.min_drift_ratio if rule.min_drift_ratio > 0 else 0.0
        violators = 0
        worst: list[str] = []
        for pos in positions:
            notional = float(pos.get("notional_value", 0.0))
            if notional > max_bound or notional < min_bound:
                violators += 1
                if len(worst) < 3:
                    worst.append(f"{pos.get('instrument_id', '?')}={notional:.1f}")
        if violators >= rule.min_violators:
            return True, (
                f"violators={violators}/{rule.min_violators}, "
                f"target={rule.target_notional_usd:.1f}, "
                f"band=[{min_bound:.1f},{max_bound:.1f}], "
                f"worst=[{', '.join(worst)}]"
            )
        return False, ""

    # -------------------------------------------------------------------------
    # Equity / snapshot helpers
    # -------------------------------------------------------------------------

    def _compute_equity_for_basis(self, basis: str) -> float | None:
        if basis == BASIS_REALIZED_BALANCE:
            return _realized_balance(self.portfolio, self._venue, self._currency)
        # Default & mark_to_market: balance + unrealized.
        return compute_mtm_equity(self.portfolio, self._venue, self._currency)

    def _read_venue_snapshot(self, now_ns: int) -> dict[str, Any] | None:
        needs_snapshot = any(r.kind in (KIND_POSITION_COUNT, KIND_NOTIONAL_DRIFT) for r in self._rules)
        if not needs_snapshot:
            return None
        raw = self.cache.get(SNAPSHOT_VENUE_CACHE_KEY)
        if raw is None:
            self.log.warning(f"snapshot:venue missing from cache; skipping position/notional rules")
            return None
        try:
            data = json.loads(raw)
        except Exception as exc:
            self.log.warning(f"snapshot:venue decode failed: {exc}")
            return None
        ts_wall = int(data.get("ts_wall", 0))
        age_ns = now_ns - ts_wall
        max_age_ns = self.config.snapshot_staleness_secs * 1_000_000_000
        if age_ns > max_age_ns:
            self.log.warning(
                f"snapshot:venue stale by {age_ns / 1e9:.1f}s " f"(>{self.config.snapshot_staleness_secs}s); skipping"
            )
            return None
        return data

    # -------------------------------------------------------------------------
    # Countdown / halt / cooldown
    # -------------------------------------------------------------------------

    def _start_countdown(self, now_ns: int) -> None:
        countdown_td = parse_interval_to_timedelta(self.config.countdown_period)
        countdown_ns = int(countdown_td.total_seconds() * 1_000_000_000)
        alert_time_ns = now_ns + countdown_ns
        self._countdown_active = True
        self.clock.set_time_alert_ns(
            name=_COUNTDOWN_TIMER_NAME,
            alert_time_ns=alert_time_ns,
            callback=self._on_countdown_expired,
        )
        triggering = [r.name for r in self._rules if self._hits[r.name] >= r.sustained_periods]
        self.log.warning(f"ANOMALY_COUNTDOWN_START: {self.config.countdown_period} " f"triggered_by={triggering}")

    def _cancel_countdown(self) -> None:
        self._countdown_active = False
        try:
            self.clock.cancel_time_alert(_COUNTDOWN_TIMER_NAME)
        except Exception:
            # Older Nautilus variants may not expose cancel_time_alert; the
            # expiry callback will no-op via the flag below.
            pass
        self.log.info("ANOMALY_COUNTDOWN_CANCEL: rules recovered")

    def _on_countdown_expired(self, event: object) -> None:
        if not self._countdown_active or self._halted:
            return
        self._countdown_active = False

        eligible = [r for r in self._rules if self._hits[r.name] >= r.sustained_periods]
        if not eligible:
            # Safety net — recovered during race with timer callback.
            self.log.info("ANOMALY_COUNTDOWN_EXPIRED: no rule eligible; skipping halt")
            return

        names = [r.name for r in eligible]
        if any(not r.cooldown_period for r in eligible):
            self.log.warning(f"ANOMALY_HALT permanent: triggered_by={names}")
            self._halt_guarded_strategies(permanent=True)
            return

        cooldowns = [(r, parse_interval_to_timedelta(r.cooldown_period)) for r in eligible]
        longest_rule, _ = max(cooldowns, key=lambda pair: pair[1])
        self.log.warning(
            f"ANOMALY_HALT cooldown={longest_rule.cooldown_period} " f"(from {longest_rule.name}): triggered_by={names}"
        )
        self._halt_guarded_strategies(cooldown_period=longest_rule.cooldown_period)

    def _halt_guarded_strategies(
        self,
        *,
        permanent: bool = False,
        cooldown_period: str = "",
    ) -> None:
        """Stop guarded strategies and optionally set a cooldown timer."""
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
                name=_COOLDOWN_TIMER_NAME,
                alert_time_ns=self.clock.timestamp_ns() + cooldown_ns,
                callback=self._on_cooldown_expired,
            )
            self.log.info(f"Cooldown timer set: {cooldown_period}. " f"Strategies will restart after cooldown.")

    def _on_cooldown_expired(self, event: object) -> None:
        """Cooldown timer callback: resume strategies and reset rule state."""
        self.log.info("Cooldown expired. Resuming strategies.")
        for strategy in self._stopped_strategies:
            strategy.resume()
            self.log.info(f"Resumed strategy: {strategy.id}")

        self._stopped_strategies = []
        self._halted = False
        self._countdown_active = False
        self._hits = {r.name: 0 for r in self._rules}
        for history in self._equity_history.values():
            history.clear()
        self.log.info("AnomalyGuardController reset after cooldown")


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------


def _realized_balance(
    portfolio: PortfolioFacade,
    venue: Venue,
    currency: Currency,
) -> float | None:
    """Return the settled (realized) account balance, excluding unrealized PnL."""
    account = portfolio.account(venue)
    if account is None:
        return None
    balance = account.balance_total(currency)
    if balance is None:
        return None
    return balance.as_double()
