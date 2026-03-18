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
