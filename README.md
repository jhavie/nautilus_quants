# Nautilus Quants

Quantitative trading framework built on [NautilusTrader](https://nautilustrader.io) for crypto multi-factor research, backtesting, and execution.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    nautilus_quants                       │
├──────────┬──────────┬──────────┬──────────┬─────────────┤
│   Data   │ Factors  │  Alpha   │ Backtest │ Strategies  │
│ Pipeline │  Engine  │ Analysis │  Runner  │             │
├──────────┴──────────┴──────────┴──────────┴─────────────┤
│              NautilusTrader 1.222.0                      │
└─────────────────────────────────────────────────────────┘
```

## Modules

### Data Pipeline (`nautilus_quants.data`)

End-to-end market data ingestion with full validation and transformation:

- **Binance K-line**: Async download with checkpoint resume, duplicate/gap detection, CSV → Parquet
- **Tardis tick data**: Trade-level data from tardis.dev, multi-symbol concurrent download, CSV.gz → NautilusTrader Parquet catalog

Pipeline flow: `Download → Validate → Process → Transform`

### Factor Engine (`nautilus_quants.factors`)

Unified panel-based (timestamps × instruments) factor computation engine:

- **Expression parser**: Alpha101-style DSL via Lark grammar (`rank(ts_mean(close, 20))`)
- **45 built-in Alpha101 factors**: Crypto-adapted (no VWAP/IndNeutralize/market cap)
- **Operators**: Time-series (`ts_sum`, `ts_rank`, `correlation`, `decay_linear`), cross-sectional (`rank`, `scale`, `quantile`), math (`log`, `sign`, `power`)
- **Real-time**: FactorEngineActor publishes FactorValues as CustomData for live strategies

### Alpha Analysis (`nautilus_quants.alpha`)

Factor quality evaluation using alphalens-reloaded:

- IC / ICIR analysis with Newey-West correction
- Quantile returns analysis
- Parallel evaluation via ProcessPoolExecutor
- Visualization and reporting

### Backtest (`nautilus_quants.backtest`)

Configuration-driven backtesting on NautilusTrader BacktestNode:

- Native BacktestRunConfig YAML format
- Multiple strategy types: breakout, factor, cross-sectional, FMZ, WorldQuant
- Enhanced reporting via QuantStats
- Custom fill models (signal-close fill)

### Strategies (`nautilus_quants.strategies`)

| Strategy | Description |
|----------|-------------|
| `breakout` | Breakout signal detection and execution |
| `factor` | Factor-driven signals (consumes FactorValues) |
| `cross_sectional` | Cross-sectional ranking-based allocation |
| `fmz` | FMZ exchange strategies |
| `worldquant` | WorldQuant research flows |

## Quick Start

### Installation

```bash
# Python >= 3.12 required
pip install -e .

# With dev dependencies
pip install -e ".[dev]"
```

### CLI Usage

#### Data Pipeline — Binance K-line

```bash
# Download 4h K-line data
python -m nautilus_quants.data download \
  -c config/examples/data.yaml \
  -s BTCUSDT,ETHUSDT \
  -t 4h \
  --start-date 2025-01-01 \
  --end-date 2025-03-31

# Full pipeline (download → validate → process → transform)
python -m nautilus_quants.data run -c config/examples/data.yaml

# Check pipeline status
python -m nautilus_quants.data status -c config/examples/data.yaml
```

#### Data Pipeline — Tardis Tick Data

```bash
# Download 1-day trade ticks (3 coins, ~87 MB compressed)
python -m nautilus_quants.data tardis-download \
  -c config/examples/tardis_data.yaml \
  --from-date 2025-03-01 \
  --to-date 2025-03-02

# Transform CSV.gz to NautilusTrader Parquet catalog
python -m nautilus_quants.data tardis-transform \
  -c config/examples/tardis_data.yaml

# Dry-run (preview without executing)
python -m nautilus_quants.data --dry-run tardis-download \
  -c config/examples/tardis_data.yaml
```

#### Backtest

```bash
# Run backtest from config
python -m nautilus_quants.backtest run config/examples/backtest.yaml

# Validate config without running
python -m nautilus_quants.backtest run config/examples/backtest.yaml --dry-run

# List available strategies
python -m nautilus_quants.backtest list
```

#### Alpha Factor Analysis

```bash
# Run factor analysis
python -m nautilus_quants.alpha analyze config/examples/alpha_analysis.yaml -v
```

#### Factor Registry

```bash
# Register factors from YAML into the DuckDB registry
python -m nautilus_quants.alpha register config/cs/factors.yaml --source alpha101

# List all registered factors
python -m nautilus_quants.alpha list

# Filter by status, sort by absolute ICIR
python -m nautilus_quants.alpha list --status active --sort abs_icir

# Inspect factor details and version history
python -m nautilus_quants.alpha inspect alpha044

# Change factor status (candidate → active → archived)
python -m nautilus_quants.alpha status alpha044 active

# Export active factors to a production-ready factors.yaml
python -m nautilus_quants.alpha export-factors \
  --context-id alpha101 \
  --method equal \
  --top 30 \
  --transform cs_rank \
  -o config/live/factors.yaml
```

## Configuration

All parameters are YAML-driven. No hardcoded values in source code.

```
config/
├── examples/           # Getting started configs
│   ├── data.yaml       # Binance K-line pipeline
│   ├── tardis_data.yaml # Tardis tick data
│   ├── backtest.yaml   # Example backtest
│   └── factors.yaml    # 45 Alpha101 factors
├── factor/             # Factor strategy configs
├── cross_sectional/    # Cross-sectional configs
├── fmz/                # FMZ-specific configs
└── worldquant/         # WorldQuant configs
```

## Project Structure

```
src/nautilus_quants/
├── data/               # Data pipeline
│   ├── cli.py          # CLI entry point (9 commands)
│   ├── config.py       # Pipeline + Tardis config dataclasses
│   ├── download/       # Binance + Tardis downloaders
│   ├── validate/       # Integrity & consistency checks
│   ├── process/        # Dedup, gap fill, OHLC fix
│   └── transform/      # CSV → Parquet (Binance + Tardis)
├── factors/            # Factor computation engine
│   ├── engine/         # Buffer + Evaluator (panel-based)
│   ├── expression/     # Lark parser + AST
│   ├── operators/      # TS / CS / Math operators
│   └── builtin/        # 45 Alpha101 factors
├── alpha/              # Factor analysis (alphalens)
│   ├── cli.py          # Analysis + Registry CLI
│   ├── analysis/       # Evaluator + report generation
│   └── registry/       # DuckDB factor registry (CRUD, versioning, export)
├── backtest/           # Backtest framework
│   ├── cli.py          # Backtest CLI
│   ├── runner.py       # BacktestNode executor
│   └── models/         # Custom fill models
├── strategies/         # Strategy implementations
│   ├── breakout/
│   ├── factor/
│   ├── cross_sectional/
│   ├── fmz/
│   └── worldquant/
├── actors/             # NautilusTrader Actors
│   ├── factor_engine.py # Real-time factor computation
│   └── equity_snapshot.py
└── common/             # Shared utilities
    ├── anchor_price_execution.py
    └── bar_subscription.py
```

## Dependencies

| Category | Package | Purpose |
|----------|---------|---------|
| Core | nautilus_trader 1.222.0 | Backtesting & execution |
| Data | python-binance, tardis-dev | Market data sources |
| Factors | lark | Expression parser |
| Analysis | alphalens-reloaded, scipy | Factor quality evaluation |
| Reporting | quantstats, plotly | Performance visualization |
| Registry | duckdb | Factor library management (optional) |
| CLI | click, tqdm | Command-line interface |

## Development

```bash
# Run tests
pytest tests/ -x -q

# Run data module tests only
pytest tests/unit/data/ -x -q

# Format code
black --line-length 100 src/ tests/
isort --profile black --line-length 100 src/ tests/
```

## License

MIT
