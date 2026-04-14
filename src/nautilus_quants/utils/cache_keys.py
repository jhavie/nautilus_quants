# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Cache key constants for inter-component data transfer via engine.cache."""

# Cache key for strategy → report metadata transfer
POSITION_METADATA_CACHE_KEY = "position_metadata"

# Cache key for EquitySnapshotActor → ReportGenerator MTM equity transfer
EQUITY_SNAPSHOTS_CACHE_KEY = "equity_snapshots"

# Cache key for EquitySnapshotActor → ReportGenerator per-instrument market values
POSITION_MARKET_VALUES_CACHE_KEY = "position_market_values"

# Cache key for FactorEngineActor → ReportGenerator factor values transfer
FACTOR_VALUES_CACHE_KEY = "factor_values"

# Cache key for PostLimitExecAlgorithm → ReportGenerator execution states transfer
EXECUTION_STATES_CACHE_KEY = "execution_states"

# ---------------------------------------------------------------------------
# Snapshot keys for Grafana monitoring (written by SnapshotAggregatorActor)
# All stored as JSON under trader-{TRADER_ID}:general:{key} in Redis
# ---------------------------------------------------------------------------

SNAPSHOT_VENUE_CACHE_KEY = "snapshot:venue"
SNAPSHOT_EXECUTION_CACHE_KEY = "snapshot:execution"
SNAPSHOT_FACTOR_CACHE_KEY = "snapshot:factor"
SNAPSHOT_STRATEGY_CACHE_KEY = "snapshot:strategy"
SNAPSHOT_HEALTH_CACHE_KEY = "snapshot:health"

# Risk model snapshot (Fundamental/Statistical exposures + breaches)
SNAPSHOT_RISK_CACHE_KEY = "snapshot:risk"

# DecisionEngineActor → SnapshotAggregatorActor strategy config metadata
STRATEGY_CONFIG_CACHE_KEY = "snapshot:strategy_config"

# ---------------------------------------------------------------------------
# RiskModelActor → OptimizedSelectionPolicy + SnapshotAggregatorActor
# Serialized RiskModelOutput (JSON bytes); see portfolio.types.serialize_risk_output.
# ---------------------------------------------------------------------------

RISK_MODEL_STATE_CACHE_KEY = "risk_model:state"
RISK_MODEL_STATE_STATISTICAL_CACHE_KEY = "risk_model:state:statistical"
RISK_MODEL_STATE_FUNDAMENTAL_CACHE_KEY = "risk_model:state:fundamental"
