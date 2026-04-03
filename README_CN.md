# Nautilus Quants

基于 [NautilusTrader](https://nautilustrader.io) 构建的加密货币多因子量化交易框架，涵盖数据采集、因子计算、因子分析和回测执行全流程。

## 架构

```
┌─────────────────────────────────────────────────────────┐
│                    nautilus_quants                       │
├──────────┬──────────┬──────────┬──────────┬─────────────┤
│   数据   │   因子   │  Alpha   │   回测   │    策略     │
│   管道   │   引擎   │   分析   │   运行器 │             │
├──────────┴──────────┴──────────┴──────────┴─────────────┤
│              NautilusTrader 1.222.0                      │
└─────────────────────────────────────────────────────────┘
```

## 核心模块

### 数据管道 (`nautilus_quants.data`)

端到端的市场数据采集、验证和转换流水线：

- **Binance K线数据**：异步下载 + 断点续传 + 重复/缺口检测 + CSV → Parquet 转换
- **Tardis 逐笔成交数据**：从 tardis.dev 下载 trade tick，多币种并发下载（ThreadPoolExecutor），CSV.gz → NautilusTrader Parquet catalog

数据流：`下载 → 验证 → 处理 → 转换`

### 因子引擎 (`nautilus_quants.factors`)

统一的面板数据（时间戳 × 标的）因子计算引擎：

- **表达式解析器**：Alpha101 风格 DSL，基于 Lark 语法（如 `rank(ts_mean(close, 20))`）
- **45 个内置 Alpha101 因子**：针对加密货币适配（移除了 VWAP / 行业中性化 / 市值等依赖）
- **算子体系**：
  - 时序算子：`ts_sum`, `ts_rank`, `correlation`, `decay_linear`, `stddev` 等
  - 截面算子：`rank`, `scale`, `quantile` 等
  - 数学算子：`log`, `sign`, `power`, `abs` 等
- **实时计算**：FactorEngineActor 通过 CustomData 发布 FactorValues，支持实盘策略

### Alpha 分析 (`nautilus_quants.alpha`)

基于 alphalens-reloaded 的因子质量评估：

- IC / ICIR 分析（含 Newey-West 校正，20+ 指标）
- 因子信号质量：Monotonicity、IC Linearity、IC AR(1)、IC Half-Life、Win Rate、Coverage
- 基于 ProcessPoolExecutor 的并行评估
- 自动入库 DuckDB 注册表（通过 YAML 配置）

### 因子注册表 (`nautilus_quants.alpha.registry`)

基于 DuckDB 的因子全生命周期管理，支持多环境：

- **5 张表**：`factors`（核心）→ `alpha_analysis_metrics`（1:N）↔ `backtest_run_metrics`（M:N via `backtest_factors`）+ `configs_snapshot`
- **自动入库**：`alpha analyze` 和 `backtest run` 自动注册因子并保存指标
- **多环境**：`test.duckdb` / `dev.duckdb` / `prod.duckdb`（独立数据库文件）
- **配置快照**：完整 YAML 配置以 JSON 存入 `configs_snapshot`，支持完整还原
- **参数化因子**：不同参数 = 不同 `factor_id`，通过 `prototype` 字段分组

```yaml
# 在分析/回测 YAML 中添加以下配置启用自动入库
registry:
  env: test
  db_dir: logs/registry
  enabled: true
```

```bash
# CLI 命令
python -m nautilus_quants.alpha register config/cs/factors.yaml
python -m nautilus_quants.alpha list --prototype alpha044
python -m nautilus_quants.alpha inspect alpha101_alpha044_8h
python -m nautilus_quants.alpha metrics alpha101_alpha044_8h
```

### 回测 (`nautilus_quants.backtest`)

配置驱动的回测框架，基于 NautilusTrader BacktestNode：

- 原生 BacktestRunConfig YAML 格式
- 多策略类型支持：突破、因子、截面、FMZ、WorldQuant
- QuantStats 增强报告
- 自定义成交模型（信号收盘价成交）

### 策略 (`nautilus_quants.strategies`)

| 策略类型 | 说明 |
|----------|------|
| `breakout` | 突破信号检测与执行 |
| `factor` | 因子驱动信号（消费 FactorValues） |
| `cross_sectional` | 截面排序分配 |
| `fmz` | FMZ 交易所策略 |
| `worldquant` | WorldQuant 研究流程 |

## 快速开始

### 安装

```bash
# 需要 Python >= 3.12
pip install -e .

# 安装开发依赖
pip install -e ".[dev]"
```

### 命令行使用

#### 数据管道 — Binance K线

```bash
# 下载 4h K线数据
python -m nautilus_quants.data download \
  -c config/examples/data.yaml \
  -s BTCUSDT,ETHUSDT \
  -t 4h \
  --start-date 2025-01-01 \
  --end-date 2025-03-31

# 完整流水线（下载 → 验证 → 处理 → 转换）
python -m nautilus_quants.data run -c config/examples/data.yaml

# 查看数据状态
python -m nautilus_quants.data status -c config/examples/data.yaml

# 清理数据
python -m nautilus_quants.data clean --all --force -c config/examples/data.yaml
```

#### 数据管道 — Tardis 逐笔成交

```bash
# 下载 1 天 trade tick（3 个币种，约 87 MB 压缩）
python -m nautilus_quants.data tardis-download \
  -c config/examples/tardis_data.yaml \
  --from-date 2025-03-01 \
  --to-date 2025-03-02

# 指定币种下载
python -m nautilus_quants.data tardis-download \
  -c config/examples/tardis_data.yaml \
  -s BTCUSDT,ETHUSDT

# 转换 CSV.gz 为 NautilusTrader Parquet catalog
python -m nautilus_quants.data tardis-transform \
  -c config/examples/tardis_data.yaml

# 预览模式（不实际执行）
python -m nautilus_quants.data --dry-run tardis-download \
  -c config/examples/tardis_data.yaml
```

#### 回测

```bash
# 执行回测
python -m nautilus_quants.backtest run config/examples/backtest.yaml

# 预验证配置
python -m nautilus_quants.backtest run config/examples/backtest.yaml --dry-run

# 详细输出
python -m nautilus_quants.backtest run config/factor/backtest.yaml -v

# 查看可用策略
python -m nautilus_quants.backtest list
```

#### Alpha 因子分析

```bash
# 执行因子分析
python -m nautilus_quants.alpha analyze config/examples/alpha_analysis.yaml -v
```

## 配置系统

所有可调参数通过 YAML 配置文件管理，源码中不允许硬编码数值。

```
config/
├── examples/              # 入门示例配置
│   ├── data.yaml          # Binance K线管道配置
│   ├── tardis_data.yaml   # Tardis tick 数据配置
│   ├── backtest.yaml      # 回测示例
│   └── factors.yaml       # 45 个 Alpha101 因子定义
├── factor/                # 因子策略配置
├── cross_sectional/       # 截面策略配置
├── fmz/                   # FMZ 策略配置
└── worldquant/            # WorldQuant 策略配置
```

### 因子配置示例 (`factors.yaml`)

```yaml
metadata:
  name: "alpha101_45_crypto"
  version: "1.0.0"

variables:
  returns: "delta(close, 1) / delay(close, 1)"

factors:
  alpha001:
    expression: "rank(ts_argmax(signed_power(...), 5)) - 0.5"
    description: "Conditional signed power rank"
    category: "volatility"

  alpha012:
    expression: "sign(delta(volume, 1)) * -1 * delta(close, 1)"
    description: "Volume-price divergence reversal"
    category: "volume"
```

### Tardis 数据配置示例 (`tardis_data.yaml`)

```yaml
download:
  exchange: "binance-futures"
  api_key_env: "TARDIS_API_KEY"    # 环境变量名或直接填 TD.xxx key
  data_types:
    - trades
  symbols:
    - BTCUSDT
    - ETHUSDT
    - SOLUSDT
  from_date: "2025-03-01"
  to_date: "2025-03-02"
  max_symbol_workers: 3

paths:
  raw_data: "data/raw/tardis"
  catalog: "data/catalog"
```

## 项目结构

```
src/nautilus_quants/
├── data/               # 数据管道
│   ├── cli.py          # CLI 入口（9 个命令）
│   ├── config.py       # 管道 + Tardis 配置 dataclass
│   ├── download/       # Binance + Tardis 下载器
│   ├── validate/       # 完整性和一致性检查
│   ├── process/        # 去重、补缺、OHLC 修复
│   └── transform/      # CSV → Parquet（Binance + Tardis）
├── factors/            # 因子计算引擎
│   ├── engine/         # Buffer + Evaluator（面板数据模式）
│   ├── expression/     # Lark 解析器 + AST
│   ├── operators/      # 时序 / 截面 / 数学算子
│   └── builtin/        # 45 个 Alpha101 因子
├── alpha/              # 因子分析（alphalens）
│   ├── cli.py          # 分析 CLI
│   └── analysis/       # 评估器 + 报告生成
├── backtest/           # 回测框架
│   ├── cli.py          # 回测 CLI
│   ├── runner.py       # BacktestNode 执行器
│   └── models/         # 自定义成交模型
├── strategies/         # 策略实现
│   ├── breakout/       # 突破策略
│   ├── factor/         # 因子策略
│   ├── cross_sectional/# 截面策略
│   ├── fmz/            # FMZ 策略
│   └── worldquant/     # WorldQuant 策略
├── actors/             # NautilusTrader Actors
│   ├── factor_engine.py # 实时因子计算
│   └── equity_snapshot.py
└── common/             # 共享工具
    ├── anchor_price_execution.py
    └── bar_subscription.py
```

## 依赖

| 类别 | 包 | 用途 |
|------|-----|------|
| 核心 | nautilus_trader 1.222.0 | 回测与执行引擎 |
| 数据 | python-binance, tardis-dev | 市场数据源 |
| 因子 | lark | 表达式解析器 |
| 分析 | alphalens-reloaded, scipy | 因子质量评估 |
| 报告 | quantstats, plotly | 绩效可视化 |
| CLI | click, tqdm | 命令行界面 |

## 开发

```bash
# 运行全部测试
pytest tests/ -x -q

# 仅运行 data 模块测试
pytest tests/unit/data/ -x -q

# 代码格式化
black --line-length 100 src/ tests/
isort --profile black --line-length 100 src/ tests/
```

## 常见陷阱

- NaN 传播：`spearmanr` / IC 计算中需注意 NaN 处理
- t-statistic 膨胀：因子自相关导致的显著性虚高
- 截面 vs 时序因子：混淆 `rank`（截面）和 `ts_rank`（时序）的语义
- 因子符号方向：确保因子值与预期收益方向一致
- 退市币种：回测中需处理僵尸仓位

## License

MIT
