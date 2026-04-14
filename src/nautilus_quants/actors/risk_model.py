# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
RiskModelActor - Periodically computes risk model output and writes to Cache.

Responsibilities:
- Subscribe to FactorValues (published by FactorEngineActor / RegimeCompositeActor)
- Buffer returns + named factor exposures per timestamp
- Every ``update_interval_bars`` bars, run Statistical and/or Fundamental
  risk models over the last ``lookback_bars`` observations
- Write the serialized RiskModelOutput to Nautilus Cache so that:
  - OptimizedSelectionPolicy reads it via ``cache.get(RISK_MODEL_STATE_CACHE_KEY)``
  - SnapshotAggregatorActor reads it for exposure monitoring (Grafana)

Decisions:
- **Cache writer, not msgbus publisher.** Keeps DecisionEngineActor unaware of
  risk data, preserving Separation of Concerns (CLAUDE.md Constitution V).
- **Subscribes only to FactorValues.** Returns are read from a ``returns``
  variable defined in the FactorEngine factor_config_path; exposures are read
  from other variables named in portfolio.yaml (e.g. "btc_beta", "funding_rate").
- **Parallel double-run.** If config.type == "parallel", both Statistical and
  Fundamental models run each update; results are written to sibling cache
  keys (:statistical / :fundamental) plus the main RISK_MODEL_STATE_CACHE_KEY.

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
from nautilus_trader.model.data import DataType
from nautilus_trader.model.identifiers import ClientId

from nautilus_quants.factors.factor_values import FactorValues
from nautilus_quants.portfolio.config import PortfolioConfig, load_portfolio_config
from nautilus_quants.portfolio.risk_model.fundamental import FundamentalRiskModel
from nautilus_quants.portfolio.risk_model.statistical import StatisticalRiskModel
from nautilus_quants.portfolio.types import RiskModelOutput, serialize_risk_output
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
        Path to portfolio.yaml (risk_model + optimizer + constraints + monitor).
    update_interval_bars : int
        Recompute risk models every N bars. Default 6 = one full trading day at 4h.
    returns_variable : str
        Variable name in FactorValues that holds per-bar returns. Must be defined
        by FactorEngine's factor_config_path. Suggested definition:
        ``returns: close / delay(close, 1) - 1``.
    """

    portfolio_config_path: str
    update_interval_bars: int = 6
    returns_variable: str = "returns"


class RiskModelActor(Actor):
    """Risk model estimator: subscribes to FactorValues, writes Cache.

    Data buffers are per-timestamp ordered dicts capped at ``lookback_bars``.
    On each update trigger, the buffer is unpacked into a (T, N) returns DataFrame
    and a (T*N, K) exposures DataFrame, and the active model(s) run.
    """

    def __init__(self, config: RiskModelActorConfig) -> None:
        super().__init__(config)
        self._actor_config = config
        self._portfolio_config: PortfolioConfig | None = None

        self._stat_model: StatisticalRiskModel | None = None
        self._barra_model: FundamentalRiskModel | None = None

        # Buffers: ts_ns → data
        # returns_buffer[ts] = {inst_id: return_value}
        self._returns_buffer: OrderedDict[int, dict[str, float]] = OrderedDict()
        # exposures_buffer[ts][factor_name] = {inst_id: exposure_value}
        self._exposures_buffer: OrderedDict[int, dict[str, dict[str, float]]] = OrderedDict()

        # Market cap proxy cache: inst_id → rolling (quote_volume * close) mean.
        # EWM α sourced from portfolio.yaml fundamental.market_cap_ewm_alpha.
        self._market_cap_proxy: dict[str, float] = {}
        self._market_cap_ewm_alpha: float = 0.1  # populated from config in on_start

        self._bar_counter: int = 0
        self._last_recompute_ts: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        """Load portfolio config, build models, subscribe to FactorValues."""
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

        self._market_cap_ewm_alpha = cfg.fundamental.market_cap_ewm_alpha

        self.subscribe_data(
            DataType(FactorValues),
            client_id=ClientId(self.id.value),
        )
        self.log.info(
            f"RiskModelActor started: "
            f"type={cfg.type}, "
            f"active_model={cfg.active_model}, "
            f"lookback_bars={cfg.statistical.lookback_bars}, "
            f"update_interval_bars={self._actor_config.update_interval_bars}"
        )

    def on_stop(self) -> None:
        self.log.info(
            f"RiskModelActor stopped. "
            f"total_bars={self._bar_counter}, "
            f"last_recompute_ts={self._last_recompute_ts}"
        )

    # ------------------------------------------------------------------
    # Data path
    # ------------------------------------------------------------------

    def on_data(self, data: object) -> None:
        """Buffer FactorValues then trigger periodic recompute."""
        if not isinstance(data, FactorValues):
            return

        ts = int(data.ts_event)
        factors = data.factors
        self._buffer_returns(ts, factors)
        self._buffer_exposures(ts, factors)
        self._update_market_cap_proxy(factors)
        self._trim_buffers()

        self._bar_counter += 1
        if self._bar_counter % self._actor_config.update_interval_bars != 0:
            return
        self._try_recompute(ts)

    def _buffer_returns(self, ts: int, factors: dict[str, dict[str, float]]) -> None:
        returns_var = self._actor_config.returns_variable
        vals = factors.get(returns_var)
        if not vals:
            return
        self._returns_buffer[ts] = {k: float(v) for k, v in vals.items()}

    def _buffer_exposures(
        self,
        ts: int,
        factors: dict[str, dict[str, float]],
    ) -> None:
        if self._portfolio_config is None:
            return
        cfg = self._portfolio_config.risk_model.fundamental
        if not cfg.factors:
            return
        bucket: dict[str, dict[str, float]] = {}
        for spec in cfg.factors:
            var_vals = factors.get(spec.variable)
            if not var_vals:
                continue
            bucket[spec.name] = {k: float(v) for k, v in var_vals.items()}
        if bucket:
            self._exposures_buffer[ts] = bucket

    def _update_market_cap_proxy(
        self,
        factors: dict[str, dict[str, float]],
    ) -> None:
        """Maintain EWM estimate of market_cap = quote_volume × close.

        Reads "quote_volume" and "close" variables if present; falls back to
        any existing configured proxy variable (e.g. "log_market_cap").
        """
        if self._portfolio_config is None:
            return
        cfg = self._portfolio_config.risk_model.fundamental
        if cfg.wls_weight_source != "market_cap":
            return

        qv_map = factors.get("quote_volume") or {}
        close_map = factors.get("close") or {}
        if qv_map and close_map:
            alpha = self._market_cap_ewm_alpha
            for inst in qv_map.keys() | close_map.keys():
                qv = qv_map.get(inst)
                close = close_map.get(inst)
                if qv is None or close is None:
                    continue
                sample = float(qv) * float(close)
                if not np.isfinite(sample) or sample <= 0:
                    continue
                prev = self._market_cap_proxy.get(inst)
                self._market_cap_proxy[inst] = (
                    sample if prev is None else (1 - alpha) * prev + alpha * sample
                )

    def _trim_buffers(self) -> None:
        """Keep only the last lookback_bars timestamps."""
        if self._portfolio_config is None:
            return
        lookback = self._portfolio_config.risk_model.statistical.lookback_bars
        while len(self._returns_buffer) > lookback:
            self._returns_buffer.popitem(last=False)
        while len(self._exposures_buffer) > lookback:
            self._exposures_buffer.popitem(last=False)

    # ------------------------------------------------------------------
    # Recompute and publish
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

        # --- Statistical model ---
        if self._stat_model is not None:
            try:
                out_stat = self._stat_model.fit(
                    returns=returns_df,
                    timestamp_ns=ts,
                )
                self.cache.add(
                    RISK_MODEL_STATE_STATISTICAL_CACHE_KEY,
                    serialize_risk_output(out_stat),
                )
            except Exception as exc:
                self.log.warning(f"Statistical fit failed: {exc}")

        # --- Fundamental model ---
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

        # --- Main key points to the active model's payload ---
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
        """Unpack returns buffer into a (T, N) DataFrame (NaN-filled)."""
        if not self._returns_buffer:
            return None
        df = pd.DataFrame.from_dict(self._returns_buffer, orient="index")
        df = df.sort_index()
        # Drop instruments with insufficient observations
        min_valid = max(2, int(0.5 * len(df)))
        keep_cols = [c for c in df.columns if df[c].notna().sum() >= min_valid]
        if not keep_cols:
            return None
        return df[keep_cols]

    def _assemble_exposures_frame(
        self,
        returns_df: pd.DataFrame,
    ) -> pd.DataFrame | None:
        """Build MultiIndex (timestamp, instrument) exposures DataFrame."""
        if not self._exposures_buffer:
            return None
        if self._portfolio_config is None:
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
        if not self._market_cap_proxy:
            return None
        w = np.array(
            [max(self._market_cap_proxy.get(inst, 0.0), 0.0) for inst in instruments],
            dtype=np.float64,
        )
        return np.sqrt(w)
