# CLAUDE.md - Nautilus Quants Development Guidelines

## Git 工作流

- **永不直推 develop 或 main**（已设 branch protection）
- PR 合并一律用 `gh pr merge --squash`
- worktree 目录名 = 分支名去掉斜杠（如 `feature/025-xxx` → `~/Sync/worktrees/feature-025-xxx`）

### 分支命名

| 类型 | 格式 | 编号规则 | 示例 |
|------|------|---------|------|
| feature | `feature/{NNN}-{description}` | 三位递增编号 | `feature/025-cicd-deploy-ghcr` |
| fix | `fix/{description}` | 无编号 | `fix/deploy-ignore-backtest` |
| refactor | `refactor/{description}` | 无编号 | `refactor/extract-shared-utils` |
| improvement | `improvement/{description}` | 无编号 | `improvement/docker-size-optimize` |
| chore | `chore/{description}` | 无编号 | `chore/update-claude-md` |

**只有 feature 分支使用递增编号**，当前最新编号：031

### 合并流程

```
feature/xxx ──PR(squash)──> develop ──PR(squash)──> main ──tag vX.Y.Z──> deploy
fix/xxx     ──PR(squash)──> develop ──PR(squash)──> main
```

### Commit 规范 (Conventional Commits)

格式：`{type}({scope}): {description}`

类型：`feat`, `fix`, `refactor`, `chore`, `docs`, `test`, `ci`
常用 scope：`ci`, `deps`, `data`, `backtest`, `factors`, `strategies`, `live`, `execution`

## CI/CD 部署

```
gh workflow run deploy.yml --ref main                      # testnet（默认）
gh workflow run deploy.yml --ref main -f environment=prod  # 正式环境
```

### 环境分离

| 环境 | 容器名 | 配置 | .env |
|------|--------|------|------|
| testnet | `nautilus-testnet-15m` | `okx_testnet_15m.yaml` | `.env` |
| prod | `nautilus-prod` | `okx_fmz.yaml` | `.env.prod` |

### Workflow 文件

- **ci.yml**：PR to main → `--extra all` → 全部测试（排除 network/slow）
- **deploy.yml**：手动触发 → `--extra dev` → live 测试 → Docker build → GHCR push → SSH deploy

### 关键信息

| 项目 | 值 |
|------|-----|
| 镜像 | `ghcr.io/jhavie/nautilus_quants:latest` / `:vX.Y.Z` |
| 服务器 | `aliyun-sg-server` (8.222.149.160) |
| 服务器目录 | `/root/workspace/nautilus_quants/` |
| GitHub Secrets | `DEPLOY_HOST`, `DEPLOY_SSH_KEY`, `GHCR_PAT` |

### 日常操作

```bash
# feature → develop → main → tag → deploy
gh pr create --base develop --head feature/xxx
gh pr merge <id> --squash
gh pr create --base main --head develop
gh pr merge <id> --squash
gh release create vX.Y.Z --target main
gh workflow run deploy.yml --ref main
gh run watch <id>
```

## 依赖分组 (pyproject.toml)

- **core**: nautilus_trader, numpy, pandas, pyyaml, click, lark, scipy
- **data**: tardis-dev, tqdm, PySocks, python-binance, tenacity
- **backtest**: quantstats, alphalens-reloaded, plotly
- **dev**: pytest, pytest-asyncio, black, isort, mypy
- **all**: data + backtest + dev

## 模块架构

- **core（部署需要）**：`live/`, `strategies/`, `actors/`, `execution/`, `factors/`, `utils/`, `common/`, `controllers/`
- **可选（部署不需要）**：`data/`, `backtest/`, `alpha/`
- 共享工具在 `utils/`（cache_keys, protocols, equity, registry, bar_spec）
- live 核心代码零 backtest 依赖

## 代码规范

- **行宽**: 100（black, isort）
- **Python**: >=3.12
- **文件头**: `# Copyright (c) 2025 nautilus_quants` + `# SPDX-License-Identifier: MIT`
- **配置驱动**: YAML 配置 + dataclass/msgspec，源码中不允许数值字面量
- **Nautilus 集成**: Actor/Strategy 用 frozen config（`ActorConfig, frozen=True`）
- **类型注解**: 公开 API 必须

## 开发约束 (Constitution)

### I. Nautilus-Native First
所有交易逻辑必须优先使用 NautilusTrader 原生组件（Actor/Strategy/Indicator 扩展基类，数据类型用 InstrumentId/Bar/QuoteTick，事件用 MessageBus）。仅在 Nautilus 缺少所需功能时才自定义。

### II. Configuration-Driven
所有运行时参数必须外化到 YAML 配置文件，源码中禁止可调参数的数值字面量。

### III. Test-First (NON-NEGOTIABLE)
TDD：先写失败测试，再实现到测试通过。PR 不通过测试不可合并。

### IV. Type Safety
所有函数签名和类属性必须有类型注解。Optional 类型必须显式处理。

### V. Separation of Concerns
- Indicators：纯计算，无副作用
- Actors：管理状态/筛选，不直接下单
- Strategies：接收信号做决策，不计算信号
- Execution：订单生命周期管理，独立于策略

### 禁止事项
- 自定义事件系统（用 Nautilus MessageBus）
- 热路径中的同步阻塞 I/O
- Actor/Strategy 实例外的全局可变状态
- 未经 Nautilus adapter 封装的第三方交易 API

## 本地参考目录

NautilusTrader 源码和文档在本地可供参考：

| 目录 | 内容 |
|------|------|
| `/Users/joe/Sync/nautilus_trader/docs/` | NautilusTrader 官方文档 |
| `/Users/joe/Sync/nautilus_trader/examples/` | 官方示例策略和 Actor |
| `/Users/joe/Sync/nautilus_trader/crates/` | Rust 核心引擎源码 |

查阅 adapter 实现、OrderBook API、Position 模型等以 `nautilus_trader` 源码为准。

## 调试协议

- 先读取日志文件（logs/），不要仅凭代码推断根因
- top 3 假设按可能性排序，确认后再修改
- 修复失败时暂停重新审视，不要叠加 workaround
- 只修改明确要求的文件，发现无关问题只提及不修改

## 领域上下文

- 基于 NautilusTrader 的加密货币多因子量化策略系统
- 数据流：交易所 API → Factor Engine → Strategy → Execution Algorithm → Exchange
- 常见陷阱：NaN 传播、因子符号方向错误、退市币种僵尸仓位、截面 vs 时序因子混淆
