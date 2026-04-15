# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
RiskModelActor - Periodically computes risk model output and writes to Cache.

Responsibilities:
- Subscribe to bars directly (same pattern as FactorEngineActor)
- Load extra_data (funding_rate / open_interest / btc_close / quote_volume / ...)
- Drive an embedded FactorEngine that computes portfolio-defined risk variables
  (defined in portfolio.yaml under ``fundamental.variables``)
- Every ``update_interval_bars`` flushes, run Statistical and/or Fundamental
  risk models over the last ``lookback_bars`` observations
- Write the serialized RiskModelOutput to Nautilus Cache so that:
  - OptimizedSelectionPolicy reads it via ``cache.get(RISK_MODEL_STATE_CACHE_KEY)``
  - SnapshotAggregatorActor reads it for exposure monitoring (Grafana)

Design:
- **Portfolio-autonomous.** Risk variables (``returns`` / ``btc_beta`` / ...)
  are defined in portfolio.yaml, not in the alpha factor config. The alpha
  FactorEngineActor is unaffected — this actor runs its own FactorEngine
  instance tuned for risk-modeling inputs.
- **Cache writer, not msgbus publisher.** Keeps DecisionEngineActor unaware
  of risk data, preserving Separation of Concerns (CLAUDE.md Constitution V).
- **Parallel double-run.** If ``risk_model.type == "parallel"``, both
  Statistical and Fundamental models run each update; results are written
  to sibling cache keys (:statistical / :fundamental) plus the main
  ``RISK_MODEL_STATE_CACHE_KEY``.

Constitution Compliance:
- Extends Nautilus Actor (Principle I)
- Frozen ActorConfig (Principle I)
- Configuration-driven via portfolio.yaml (Principle II)
- No order submission, no direct trading (Principle V)
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from nautilus_trader.common.actor import Actor
from nautilus_trader.common.config import ActorConfig
from nautilus_trader.common.events import TimeEvent
from nautilus_trader.model.data import Bar, FundingRateUpdate
from nautilus_trader.model.identifiers import InstrumentId

from nautilus_quants.actors.factor_engine import (
    _detect_extra_bar_fields,
    _extract_bar_data,
)
from nautilus_quants.common.bar_subscription import BarSubscriptionMixin
from nautilus_quants.factors.engine.extra_data import (
    ExtraDataConfig,
    load_extra_data_config,
    load_lookup,
)
from nautilus_quants.factors.engine.factor_engine import FactorEngine
from nautilus_quants.portfolio.config import PortfolioConfig, load_portfolio_config
from nautilus_quants.portfolio.risk_model.fundamental import FundamentalRiskModel
from nautilus_quants.portfolio.risk_model.statistical import StatisticalRiskModel
from nautilus_quants.portfolio.types import serialize_risk_output
from nautilus_quants.utils.cache_keys import (
    RISK_MODEL_STATE_CACHE_KEY,
    RISK_MODEL_STATE_FUNDAMENTAL_CACHE_KEY,
    RISK_MODEL_STATE_STATISTICAL_CACHE_KEY,
)

logger = logging.getLogger(__name__)


class RiskModelActorConfig(ActorConfig, frozen=True):
    """Configuration for RiskModelActor.

    Parameters
    ----------
    portfolio_config_path : str
        Path to portfolio.yaml (risk_model + optimizer + constraints).
        The ``fundamental.variables`` block defines the risk variable
        expressions computed by this actor's embedded FactorEngine.
    update_interval_bars : int
        Recompute risk models every N flushes. Default 6 = one full trading
        day at 4h bars.
    bar_types : list[str]
        Bar type strings to subscribe to (injected by CLI from data config).
    extra_data_path : str
        Path to the extra_data YAML (typically shared with FactorEngineActor).
        Empty disables extra_data loading.
    max_history : int
        Rolling window size for the embedded FactorEngine's buffer.
    flush_timeout_secs : int
        Seconds to wait for all instruments before force-flushing (0 disables).
    """

    portfolio_config_path: str
    update_interval_bars: int = 6
    bar_types: list[str] = []
    extra_data_path: str = ""
    max_history: int = 500
    flush_timeout_secs: int = 5


class RiskModelActor(BarSubscriptionMixin, Actor):
    """Risk model estimator: subscribes to bars, drives embedded FactorEngine,
    writes serialized RiskModelOutput to Nautilus Cache.

    Buffers are per-timestamp ordered dicts capped at ``lookback_bars``.
    Every ``update_interval_bars`` flushes, the buffer is unpacked into a
    (T, N) returns DataFrame plus an (T*N, K) exposures frame, and the
    active model(s) run.
    """

    def __init__(self, config: RiskModelActorConfig) -> None:
        super().__init__(config)
        self._actor_config = config
        self._portfolio_config: PortfolioConfig | None = None

        self._risk_engine: FactorEngine | None = None
        self._stat_model: StatisticalRiskModel | None = None
        self._barra_model: FundamentalRiskModel | None = None

        # Buffers: ts_ns → data
        self._returns_buffer: OrderedDict[int, dict[str, float]] = OrderedDict()
        self._exposures_buffer: OrderedDict[int, dict[str, dict[str, float]]] = OrderedDict()
        # Market cap per instrument (from variables["market_cap"] each flush)
        self._market_cap_latest: dict[str, float] = {}

        # Bar pipeline state (mirrors FactorEngineActor flush-gating)
        self._expected_instruments: set[str] = set()
        self._pending_instruments: set[str] = set()
        self._pending_ts: int = 0
        self._last_flushed_ts: int = 0
        self._extra_fields_detected: bool = False

        # Extra data state
        self._latest_funding_rates: dict[str, float] = {}
        self._parquet_lookups: dict[str, dict[str, dict[int, dict[str, float]]]] = {}
        self._bar_field_lookups: dict[str, dict[str, dict[int, float]]] = {}
        self._broadcast_configs: list[ExtraDataConfig] = []
        self._broadcast_resolved: dict[str, str] = {}

        # Recompute schedule
        self._flush_counter: int = 0
        self._last_recompute_ts: int = 0
        self._warned_no_returns: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        """Load portfolio config, build models + engine, subscribe to bars."""
        path = Path(self._actor_config.portfolio_config_path)
        if not path.is_file():
            raise FileNotFoundError(f"portfolio_config_path not found: {path}")
        self._portfolio_config = load_portfolio_config(path)
        cfg = self._portfolio_config.risk_model

        if cfg.type in ("statistical", "parallel"):
            self._stat_model = StatisticalRiskModel(cfg.statistical)
        if cfg.type in ("fundamental", "parallel"):
            self._barra_model = FundamentalRiskModel(cfg.fundamental)
        if self._stat_model is None and self._barra_model is None:
            raise ValueError(f"invalid risk_model.type: {cfg.type!r}")

        # Build the embedded FactorEngine. Risk variables are registered as
        # expression factors (not engine-level variables) so flush_and_compute
        # returns them in the results dict.
        self._risk_engine = FactorEngine(max_history=self._actor_config.max_history)
        for var in cfg.fundamental.variables:
            self._risk_engine.register_expression_factor(
                name=var.name,
                expression=var.expression,
                description=var.description,
            )

        # Subscribe to bars (mirrors FactorEngineActor pattern).
        if not self._actor_config.bar_types:
            self.log.error(
                "RiskModelActor: bar_types not configured. Ensure the backtest "
                "CLI injects bar_types from the data config."
            )
            return
        self._subscribe_bar_types(self._actor_config.bar_types)
        self._expected_instruments = set(self._bar_type_to_inst_id.values())

        # Extra data: funding_rate (catalog), broadcast (btc_close/eth_close),
        # parquet lookups (open_interest), bar fields (quote_volume).
        self._setup_extra_data()

        self.log.info(
            f"RiskModelActor started: "
            f"type={cfg.type}, active_model={cfg.active_model}, "
            f"lookback_bars={cfg.statistical.lookback_bars}, "
            f"update_interval_bars={self._actor_config.update_interval_bars}, "
            f"variables={[v.name for v in cfg.fundamental.variables]}, "
            f"instruments={len(self._expected_instruments)}"
        )

    def on_stop(self) -> None:
        # Flush any residual bars so the last window lands in the buffer.
        if self._pending_ts > 0 and self._pending_instruments and self._risk_engine is not None:
            self._flush_and_update(self._pending_ts)
            self._pending_instruments.clear()
            self._pending_ts = 0
        self._cancel_flush_alert()

        self.log.info(
            f"RiskModelActor stopped. "
            f"flush_count={self._flush_counter}, "
            f"last_recompute_ts={self._last_recompute_ts}, "
            f"returns_buffer_len={len(self._returns_buffer)}"
        )

    # ------------------------------------------------------------------
    # Bar pipeline (mirrors FactorEngineActor)
    # ------------------------------------------------------------------

    def on_bar(self, bar: Bar) -> None:
        """Accumulate bars into the embedded FactorEngine, flush per-timestamp."""
        instrument_id = self._resolve_bar(bar)
        if instrument_id is None or self._risk_engine is None:
            return

        # Auto-detect BinanceBar extra fields on first bar (same as FactorEngineActor)
        if not self._extra_fields_detected:
            extra = _detect_extra_bar_fields(bar)
            self._extra_fields_detected = True
            if extra:
                existing = list(self._risk_engine._buffer._extra_fields)
                merged = list(dict.fromkeys(extra + existing))
                self._risk_engine.set_extra_fields(merged)

        ts = bar.ts_event
        if self._last_flushed_ts > 0 and ts <= self._last_flushed_ts:
            return

        # Timestamp advanced → force-flush residual from previous ts
        if self._pending_ts > 0 and ts > self._pending_ts:
            if self._pending_instruments:
                self._cancel_flush_alert()
                self._flush_and_update(self._pending_ts)
            else:
                self._cancel_flush_alert()
            self._pending_instruments.clear()
            self._pending_ts = 0

        bar_data = _extract_bar_data(bar)

        # Inject latest funding rate (forward-fill)
        if instrument_id in self._latest_funding_rates:
            bar_data["funding_rate"] = self._latest_funding_rates[instrument_id]

        # Inject parquet-sourced fields
        for _field_name, inst_map in self._parquet_lookups.items():
            inst_data = inst_map.get(instrument_id)
            if inst_data is None:
                continue
            ts_data = inst_data.get(ts)
            if ts_data:
                bar_data.update(ts_data)

        # Inject bar-field lookups (e.g. quote_volume via parquet)
        for field_name, lookup in self._bar_field_lookups.items():
            if field_name not in bar_data and instrument_id in lookup:
                val = lookup[instrument_id].get(ts)
                if val is not None:
                    bar_data[field_name] = val

        self._risk_engine.on_bar(instrument_id, bar_data, ts)
        self._pending_ts = ts
        self._pending_instruments.add(instrument_id)

        # All instruments reported → immediate flush
        if self._pending_instruments == self._expected_instruments:
            self._cancel_flush_alert()
            self._flush_and_update(ts)
            self._pending_instruments.clear()
            self._pending_ts = 0
        elif self._actor_config.flush_timeout_secs > 0 and len(self._pending_instruments) == 1:
            self._set_flush_alert(ts)

    def on_data(self, data: object) -> None:
        """Handle non-bar events (reserved for future extensions)."""
        # Bars come via on_bar; funding rate comes via on_funding_rate.
        # No other subscriptions are required.
        return

    def on_funding_rate(self, funding_rate: FundingRateUpdate) -> None:
        """Cache latest funding rate for per-bar injection (forward-fill)."""
        inst_id = str(funding_rate.instrument_id)
        self._latest_funding_rates[inst_id] = float(funding_rate.rate)

    # ------------------------------------------------------------------
    # Extra data setup
    # ------------------------------------------------------------------

    def _setup_extra_data(self) -> None:
        """Load extra_data YAML and configure per-source consumption."""
        if not self._actor_config.extra_data_path:
            return
        try:
            configs = load_extra_data_config(self._actor_config.extra_data_path)
        except Exception as exc:
            self.log.warning(f"Failed to load extra_data config: {exc}")
            return

        extra_buffer_fields: list[str] = []
        for cfg in configs:
            if cfg.source == "catalog":
                for inst_id_str in self._expected_instruments:
                    self.subscribe_funding_rates(InstrumentId.from_str(inst_id_str))
                extra_buffer_fields.append(cfg.name)
            elif cfg.source == "broadcast":
                self._broadcast_configs.append(cfg)
                extra_buffer_fields.append(cfg.name)
            elif cfg.source in ("bar", "parquet"):
                extra_buffer_fields.append(cfg.name)
                try:
                    lookup = load_lookup(cfg, list(self._expected_instruments))
                    if not lookup:
                        continue
                    if cfg.source == "bar":
                        self._bar_field_lookups[cfg.name] = lookup
                    else:
                        self._parquet_lookups[cfg.name] = lookup
                except Exception as exc:
                    self.log.warning(f"Extra data '{cfg.name}' load failed: {exc}")

        if extra_buffer_fields and self._risk_engine is not None:
            existing = list(self._risk_engine._buffer._extra_fields)
            new_fields = [f for f in extra_buffer_fields if f not in existing]
            if new_fields:
                self._risk_engine.set_extra_fields(list(existing) + new_fields)

    def _inject_broadcast_staged(self, ts: int) -> None:
        """Inject broadcast fields (btc_close, eth_close) before flush."""
        if self._risk_engine is None:
            return
        if not self._broadcast_resolved and self._broadcast_configs:
            for cfg in self._broadcast_configs:
                pattern = cfg.instruments[0] if cfg.instruments else ""
                if not pattern:
                    continue
                pattern_upper = pattern.upper()
                matched = next(
                    (i for i in self._expected_instruments if i.upper().startswith(pattern_upper)),
                    None,
                )
                if matched:
                    self._broadcast_resolved[cfg.name] = matched
        for cfg in self._broadcast_configs:
            matched = self._broadcast_resolved.get(cfg.name)
            if matched:
                self._risk_engine._buffer.inject_staged_field(ts, cfg.name, matched)

    # ------------------------------------------------------------------
    # Flush timeout (one-shot)
    # ------------------------------------------------------------------

    @property
    def _flush_alert_name(self) -> str:
        return f"risk_flush_timeout_{self.id}"

    def _set_flush_alert(self, ts: int) -> None:
        self._cancel_flush_alert()
        alert_ns = int(
            self.clock.timestamp_ns() + self._actor_config.flush_timeout_secs * 1_000_000_000
        )
        self.clock.set_time_alert(
            name=self._flush_alert_name,
            alert_time=pd.Timestamp(alert_ns, unit="ns"),
            callback=self._on_flush_timeout,
        )

    def _cancel_flush_alert(self) -> None:
        try:
            self.clock.cancel_timer(self._flush_alert_name)
        except Exception:
            pass

    def _on_flush_timeout(self, event: TimeEvent) -> None:
        if self._pending_ts > 0 and self._pending_instruments:
            self._flush_and_update(self._pending_ts)
            self._pending_instruments.clear()
            self._pending_ts = 0

    # ------------------------------------------------------------------
    # Flush → buffer → (periodic) recompute
    # ------------------------------------------------------------------

    def _flush_and_update(self, ts: int) -> None:
        """Flush the embedded engine and push its output into our buffers."""
        if self._risk_engine is None:
            return

        if self._broadcast_configs:
            self._inject_broadcast_staged(ts)

        try:
            risk_values = self._risk_engine.flush_and_compute(ts)
        except Exception as exc:
            self.log.warning(f"risk_engine flush failed at ts={ts}: {exc}")
            return

        self._last_flushed_ts = max(self._last_flushed_ts, ts)
        self._flush_counter += 1

        self._buffer_returns(ts, risk_values)
        self._buffer_exposures(ts, risk_values)
        self._update_market_cap_latest(risk_values)
        self._trim_buffers()

        if self._flush_counter % self._actor_config.update_interval_bars == 0:
            self._try_recompute(ts)

        self._warn_if_returns_missing()

    def _buffer_returns(
        self,
        ts: int,
        risk_values: dict[str, dict[str, float]],
    ) -> None:
        vals = risk_values.get("returns")
        if not vals:
            return
        self._returns_buffer[ts] = {k: float(v) for k, v in vals.items()}

    def _buffer_exposures(
        self,
        ts: int,
        risk_values: dict[str, dict[str, float]],
    ) -> None:
        if self._portfolio_config is None:
            return
        cfg = self._portfolio_config.risk_model.fundamental
        if not cfg.factors:
            return
        bucket: dict[str, dict[str, float]] = {}
        for spec in cfg.factors:
            var_vals = risk_values.get(spec.variable)
            if not var_vals:
                continue
            bucket[spec.name] = {k: float(v) for k, v in var_vals.items()}
        if bucket:
            self._exposures_buffer[ts] = bucket

    def _update_market_cap_latest(
        self,
        risk_values: dict[str, dict[str, float]],
    ) -> None:
        """Read the ``market_cap`` variable (if defined) for WLS weights."""
        if self._portfolio_config is None:
            return
        if self._portfolio_config.risk_model.fundamental.wls_weight_source != "market_cap":
            return
        mc_map = risk_values.get("market_cap") or {}
        for inst, v in mc_map.items():
            if np.isfinite(v) and v > 0:
                self._market_cap_latest[inst] = float(v)

    def _trim_buffers(self) -> None:
        if self._portfolio_config is None:
            return
        lookback = self._portfolio_config.risk_model.statistical.lookback_bars
        while len(self._returns_buffer) > lookback:
            self._returns_buffer.popitem(last=False)
        while len(self._exposures_buffer) > lookback:
            self._exposures_buffer.popitem(last=False)

    def _warn_if_returns_missing(self) -> None:
        """One-shot WARN if `returns` variable never populates the buffer."""
        if self._warned_no_returns or self._portfolio_config is None:
            return
        min_hist = self._portfolio_config.risk_model.statistical.min_history_bars
        if self._flush_counter > min_hist and not self._returns_buffer:
            self.log.warning(
                f"RiskModelActor: no `returns` values produced by risk_engine "
                f"after {self._flush_counter} flushes. Check that "
                f"portfolio.yaml fundamental.variables contains a `returns` entry."
            )
            self._warned_no_returns = True

    # ------------------------------------------------------------------
    # Recompute & publish
    # ------------------------------------------------------------------

    def _try_recompute(self, ts: int) -> None:
        if self._portfolio_config is None:
            return
        cfg = self._portfolio_config.risk_model
        min_hist = cfg.statistical.min_history_bars
        if len(self._returns_buffer) < min_hist:
            return

        returns_df = self._assemble_returns_frame()
        if returns_df is None or returns_df.empty:
            return

        if self._stat_model is not None:
            try:
                out_stat = self._stat_model.fit(returns=returns_df, timestamp_ns=ts)
                self.cache.add(
                    RISK_MODEL_STATE_STATISTICAL_CACHE_KEY,
                    serialize_risk_output(out_stat),
                )
            except Exception as exc:
                self.log.warning(f"Statistical fit failed: {exc}")

        if self._barra_model is not None:
            try:
                exposures_df = self._assemble_exposures_frame(returns_df)
                weights_vec = self._build_wls_weights(list(returns_df.columns))
                out_fund = self._barra_model.fit(
                    returns=returns_df,
                    timestamp_ns=ts,
                    exposures=exposures_df,
                    weights=weights_vec,
                )
                self.cache.add(
                    RISK_MODEL_STATE_FUNDAMENTAL_CACHE_KEY,
                    serialize_risk_output(out_fund),
                )
            except Exception as exc:
                self.log.warning(f"Fundamental fit failed: {exc}")

        active_key = (
            RISK_MODEL_STATE_FUNDAMENTAL_CACHE_KEY
            if cfg.active_model == "fundamental"
            else RISK_MODEL_STATE_STATISTICAL_CACHE_KEY
        )
        active_payload = self.cache.get(active_key)
        if active_payload is not None:
            self.cache.add(RISK_MODEL_STATE_CACHE_KEY, active_payload)
            self._last_recompute_ts = ts
            self.log.info(
                f"Risk model updated @ ts={ts}, "
                f"active={cfg.active_model}, "
                f"payload_bytes={len(active_payload)}"
            )

    def _assemble_returns_frame(self) -> pd.DataFrame | None:
        if not self._returns_buffer:
            return None
        df = pd.DataFrame.from_dict(self._returns_buffer, orient="index")
        df = df.sort_index()
        min_valid = max(2, int(0.5 * len(df)))
        keep_cols = [c for c in df.columns if df[c].notna().sum() >= min_valid]
        if not keep_cols:
            return None
        return df[keep_cols]

    def _assemble_exposures_frame(
        self,
        returns_df: pd.DataFrame,
    ) -> pd.DataFrame | None:
        if not self._exposures_buffer or self._portfolio_config is None:
            return None
        factor_names = [f.name for f in self._portfolio_config.risk_model.fundamental.factors]
        if not factor_names:
            return None

        instruments = list(returns_df.columns)
        rows: list[tuple[int, str, list[float]]] = []
        for ts in returns_df.index:
            ts_key = int(ts) if not isinstance(ts, int) else ts
            exp = self._exposures_buffer.get(ts_key, {})
            for inst in instruments:
                values = [float(exp.get(fn, {}).get(inst, np.nan)) for fn in factor_names]
                rows.append((ts_key, inst, values))
        if not rows:
            return None
        index = pd.MultiIndex.from_tuples([(r[0], r[1]) for r in rows], names=["ts", "inst"])
        data = np.array([r[2] for r in rows], dtype=np.float64)
        return pd.DataFrame(data, index=index, columns=factor_names)

    def _build_wls_weights(self, instruments: list[str]) -> np.ndarray | None:
        if not self._market_cap_latest:
            return None
        w = np.array(
            [max(self._market_cap_latest.get(inst, 0.0), 0.0) for inst in instruments],
            dtype=np.float64,
        )
        return np.sqrt(w)
