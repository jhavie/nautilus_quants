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

- IC / ICIR analysis with Newey-West correction (20+ metrics)
- Factor signal quality: Monotonicity, IC Linearity, IC AR(1), IC Half-Life, Win Rate, Coverage
- Parallel evaluation via ProcessPoolExecutor
- Auto-persist to DuckDB registry (configurable via YAML)

### Alpha Mining (`nautilus_quants.alpha.mining`)

LLM-driven alpha factor discovery using Claude Code CLI (`claude -p`):

- **Hypothesis-first generation**: LLM outputs market hypothesis before writing expressions
- **DSL-native**: Generates expressions directly in the project's Alpha101-style DSL (62 operators)
- **Anti-duplication**: Injects all previous expressions into prompt to avoid repetition
- **Iterative feedback**: Top factors' IC/ICIR from previous rounds guide next generation
- **Auto-validation**: `parse_expression()` + `expression_hash()` for syntax check and dedup
- **Full logging**: Each round saves prompt, response, validated factors, and analysis results

### Factor Registry (`nautilus_quants.alpha.registry`)

DuckDB-backed factor lifecycle management with multi-environment support:

- **5 tables**: `factors` (core) → `alpha_analysis_metrics` (1:N) ↔ `backtest_run_metrics` (M:N via `backtest_factors`) + `configs_snapshot`
- **Auto-persist**: `alpha analyze` and `backtest run` auto-register factors and save metrics
- **Multi-env**: `test.duckdb` / `dev.duckdb` / `prod.duckdb` (separate DB files per environment)
- **Config as JSON**: Full YAML configs stored in `configs_snapshot` for reproducibility
- **Parametric factors**: Different params = different `factor_id`, grouped by `prototype` field

```yaml
# Add to any analysis/backtest YAML to enable auto-persist
registry:
  env: test
  db_dir: logs/registry
  enabled: true
```

- **Declarative composite**: `composite` section in factors YAML auto-generates normalize + weighted combination; only base factors are registered

**CLI Commands (`python -m nautilus_quants.alpha`):**

| Command | Description | Example |
|---------|-------------|---------|
| `analyze` | Run factor analysis + auto-persist to DB | `analyze config/cs/alpha_101.yaml` |
| `mine` | LLM-driven alpha factor mining via Claude CLI | `mine config/cs/alpha_mining.yaml --rounds 3` |
| `metrics` | Show all metrics for a factor (IC, ICIR, t(NW), skew, kurtosis, AR1...) | `metrics alpha101_alpha044_8h` |
| `list` | List registered factors | `list --prototype alpha044 --source alpha101` |
| `inspect` | Factor details + analysis metrics + backtests | `inspect alpha101_alpha044_8h` |
| `backtests` | List backtest runs with linked factors (incl. started_at) | `backtests --factor-id alpha101_alpha044_8h` |
| `config` | Show config snapshot linked to a backtest run | `config <backtest_id> --type all` |
| `status` | Change factor status (candidate/active/archived) | `status alpha101_alpha044_8h active` |
| `register` | Register factors from YAML without analysis | `register config/cs/factors.yaml` |
| `export-factors` | Export active factors to YAML with composite | `export-factors -o output.yaml --method icir_weight` |
| `promote` | Score, dedup, decorrelate & promote factors across envs | `promote --config config/examples/scoring.yaml` |
| `mine` | LLM-driven alpha factor mining via Claude Code CLI | `mine config/cs/alpha_mining.yaml --rounds 3` |
| `audit` | Audit registry for expression duplicates & prototype issues | `audit --env test` |
| `backfill` | Backfill expression_hash, prototype, parameters | `backfill --execute --env test` |
| `dedup` | Remove duplicate factors by expression hash | `dedup --execute --env test` |
| `regime` | Regime-conditional IC analysis (Jump Model vs EMA) | `regime config/cs/regime_llm_claude.yaml -v` |

**Backtest CLI (`python -m nautilus_quants.backtest`):**

| Command | Description | Example |
|---------|-------------|---------|
| `run` | Execute backtest + auto-persist to DB | `run config/cs/backtest.yaml` |
| `validate` | Validate config without executing | `validate config/cs/backtest.yaml` |
| `list` | List available strategies | `list -v` |

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

#### Alpha Factor Mining (LLM)

```bash
# Mine factors (5 rounds × 8 factors = ~40 candidates)
python -m nautilus_quants.alpha mine config/cs/alpha_101.yaml

# Mine with hypothesis direction
python -m nautilus_quants.alpha mine config/cs/alpha_101.yaml \
  --hypothesis "volume-price divergence predicts reversal in crypto"

# Generate only (skip IC analysis)
python -m nautilus_quants.alpha mine config/cs/alpha_101.yaml --no-analyze --rounds 3

# Use opus model for deeper reasoning
python -m nautilus_quants.alpha mine config/cs/alpha_101.yaml --model opus

# Review mined factors
python -m nautilus_quants.alpha list --env test --source llm_mining
python -m nautilus_quants.alpha promote --source-env test --target-env dev
```

#### Alpha Factor Tuning (Optuna)

Three-dimensional search (parameters / operators / variables) over eligible
factors pulled from the registry. Writes `register_top_k` variants per
prototype into `test.duckdb`; downstream `alpha promote` decides which ones
ship to `dev`.

```bash
# Tune all eligible factors (batch mode; config selects candidates + search space)
python -m nautilus_quants.alpha tune config/cs/alpha_mining_diversified.yaml

# Tune a single ad-hoc expression (no registry lookup)
python -m nautilus_quants.alpha tune config/cs/alpha_mining_diversified.yaml \
  --expression 'ts_mean(returns, 10)' --trials 50

# Tune only factors sharing one prototype
python -m nautilus_quants.alpha tune config/cs/alpha_mining_diversified.yaml \
  --prototype alpha044

# Dry-run (optimize + preview, no DB writes)
python -m nautilus_quants.alpha tune config/cs/alpha_mining_diversified.yaml \
  --no-register -v

# Resume after a crashed batch run — skip already-completed prototypes
python -m nautilus_quants.alpha tune config/cs/alpha_mining_diversified.yaml \
  --resume-from-dir logs/alpha_tune_xxx/20260414_004445 \
  --skip-already-tuned -v
```

**Resume options** (for long batches that may crash midway):

- `--resume-from-dir PATH`: Skip every prototype whose
  `proto_NNN_{label}/registration_summary.json` exists in an earlier run
  directory. This file is only written after `register_tuned_variants`
  returns successfully, so its presence is a precise signal of completion.
- `--skip-already-tuned`: Skip prototypes that already have
  `register_top_k` registered tune variants (`tag="tuned"` DB query,
  independent of any run directory). Use together with `--resume-from-dir`
  for maximum safety — the skip set is the **union** of both sources.
- Both are idempotent: partially-completed prototypes (fewer variants than
  `register_top_k`) are automatically retained so the missing ranks get
  filled in on the next run. `find_by_expression_hash` inside
  `register_tuned_variants` ensures no duplicate writes.

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
├── alpha/              # Factor analysis + mining
│   ├── cli.py          # Analysis & mining CLI
│   ├── analysis/       # Evaluator + report generation
│   └── mining/agent/   # LLM-driven factor mining (prompts + orchestrator)
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
