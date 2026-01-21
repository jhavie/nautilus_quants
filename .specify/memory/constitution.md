<!--
================================================================================
SYNC IMPACT REPORT
================================================================================
Version Change: 1.0.0 → 1.1.0

Modified Principles: None

Added Sections:
- Development Workflow: Expanded Branch Naming Convention with explicit patterns
- Development Workflow: Added Feature Development Lifecycle

Removed Sections: None

Templates Requiring Updates:
- .specify/templates/plan-template.md: ✅ Compatible (Branch field exists)
- .specify/templates/spec-template.md: ✅ Compatible (Feature Branch field exists)
- .specify/templates/tasks-template.md: ✅ Compatible (No changes needed)

Follow-up TODOs: None
================================================================================
-->

# Nautilus Quants Constitution

## Core Principles

### I. Nautilus-Native First

All trading logic MUST leverage Nautilus Trader's native components before creating custom implementations. Custom code is permitted only when Nautilus lacks required functionality.

**Rationale**: Nautilus Trader provides battle-tested, high-performance components. Reinventing these introduces bugs, maintenance burden, and performance degradation.

**Compliance Criteria**:
- Every Actor, Strategy, and Indicator MUST extend Nautilus base classes
- Data types MUST use Nautilus-native types (InstrumentId, Bar, QuoteTick) where available
- Event handling MUST use Nautilus event system, not custom pub/sub

### II. Configuration-Driven Behavior

All runtime parameters MUST be externalized to YAML configuration files. No magic numbers or hardcoded values in source code.

**Rationale**: Quantitative strategies require rapid parameter iteration. Hardcoded values prevent backtesting variations and production tuning without code changes.

**Compliance Criteria**:
- Trading parameters MUST reside in `config/trading.yaml`
- Strategy parameters MUST reside in `config/strategy.yaml`
- Risk parameters MUST reside in `config/risk.yaml`
- Backtest parameters MUST reside in `config/backtest.yaml`
- Source code MUST NOT contain numeric literals for tunable parameters

### III. Test-First Development (NON-NEGOTIABLE)

All features MUST follow TDD: write failing tests first, then implement until tests pass.

**Rationale**: Trading systems handle real capital. Bugs have direct financial consequences. TDD ensures correctness before deployment.

**Compliance Criteria**:
- Unit tests MUST exist in `tests/unit/` for all indicators and pure logic
- Integration tests MUST exist in `tests/integration/` for strategy-actor interactions
- Tests MUST fail before implementation begins (Red-Green-Refactor)
- No PR merges without passing test suite (`pytest tests/`)

### IV. Type Safety

All code MUST be fully typed with Python type hints and pass mypy strict mode.

**Rationale**: Trading systems require precision. Runtime type errors can cause incorrect order sizes, wrong instruments, or missed signals.

**Compliance Criteria**:
- All function signatures MUST include type annotations
- All class attributes MUST include type annotations
- Code MUST pass `mypy --strict` without errors
- Optional types MUST be explicitly handled (no implicit None)

### V. Separation of Concerns

Components MUST maintain clear boundaries: Indicators compute signals, Actors manage state/screening, Strategies make trading decisions, Execution handles order lifecycle.

**Rationale**: Clean separation enables independent testing, backtesting of components, and swapping implementations without cascading changes.

**Compliance Criteria**:
- Indicators MUST be pure computational units with no side effects
- Actors MUST NOT directly place orders
- Strategies MUST receive signals, not compute them
- Execution logic MUST be isolated from strategy logic

## Technology Constraints

**Language/Runtime**: Python ≥3.12
**Core Framework**: Nautilus Trader 1.222.0
**Data Processing**: pandas ≥2.3.0, numpy >=1.26.4
**Configuration**: PyYAML ≥6.0
**Visualization**: Plotly ≥5.18.0 (Nautilus native charts)
**Testing**: pytest ≥7.0.0, pytest-asyncio ≥0.21.0
**Code Quality**: black, isort, mypy

**Prohibited**:
- Custom event systems (use Nautilus MessageBus)
- Synchronous blocking I/O in hot paths
- Global mutable state outside Actor/Strategy instances
- Third-party trading APIs not wrapped by Nautilus adapters

## Development Workflow

### Branch Naming Convention

All branches MUST follow the pattern: `<type>/<issue-id>-<short-description>`

| Type | Purpose | Example |
|------|---------|---------|
| `feature/` | New functionality | `feature/12-bollinger-indicator` |
| `fix/` | Bug fixes | `fix/15-order-fill-calculation` |
| `refactor/` | Code restructuring (no behavior change) | `refactor/18-extract-base-strategy` |
| `docs/` | Documentation only | `docs/20-api-reference` |
| `test/` | Test additions/improvements | `test/22-backtest-edge-cases` |
| `chore/` | Maintenance tasks | `chore/25-update-dependencies` |

**Naming Rules**:
- `<issue-id>` MUST reference a GitHub issue number (create issue first if none exists)
- `<short-description>` MUST use lowercase kebab-case (e.g., `add-rsi-indicator`)
- Description MUST be 2-5 words summarizing the change
- No trailing numbers or version suffixes

**Protected Branches**:
- `main`: Production-ready code, requires PR with passing CI
- `develop`: Integration branch for features awaiting release

### Feature Development Lifecycle

```
1. Create Issue     → Document requirement in GitHub Issues
2. Create Branch    → git checkout -b feature/<issue-id>-<description>
3. Write Spec       → /speckit.specify (if complex feature)
4. Write Tests      → TDD: tests MUST fail initially
5. Implement        → Code until tests pass
6. Self-Review      → Run full CI locally (pytest, mypy, black, isort)
7. Create PR        → Reference issue, include backtest results if strategy
8. Code Review      → Address feedback, ensure Constitution compliance
9. Merge            → Squash merge to develop, delete branch
10. Release         → Periodic merge develop → main with version tag
```

### Code Review Requirements

1. All changes MUST pass CI checks (tests, mypy, black, isort)
2. Strategy changes MUST include backtest results demonstrating no regression
3. Risk parameter changes MUST include justification and impact analysis
4. New indicators MUST include unit tests with edge cases (empty data, single bar, etc.)

### Commit Standards

- Commits MUST be atomic (one logical change per commit)
- Commit messages MUST follow conventional commits format:
  ```
  <type>(<scope>): <description>

  [optional body]

  [optional footer]
  ```
- Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`
- Scope: component name (e.g., `indicators`, `actors`, `strategies`, `execution`)
- Breaking changes MUST include `BREAKING CHANGE:` in footer

**Examples**:
```
feat(indicators): add Bollinger Bands indicator

Implements BB with configurable period and std deviation multiplier.
Closes #12

fix(execution): correct position size calculation for futures

Previous calculation ignored contract multiplier.
Closes #15
```

## Governance

This Constitution supersedes all other development practices and conventions. All code reviews MUST verify compliance with these principles.

### Amendment Process

1. Propose amendment via pull request modifying this file
2. Document rationale and migration plan for existing code
3. Require maintainer approval
4. Update CONSTITUTION_VERSION according to semantic versioning:
   - MAJOR: Principle removal or incompatible redefinition
   - MINOR: New principle or material expansion
   - PATCH: Clarifications and wording improvements

### Compliance Review

- Weekly: Automated CI validates type safety and test coverage
- Per-PR: Manual review against Constitution principles
- Monthly: Audit for configuration drift and hardcoded values

**Version**: 1.1.0 | **Ratified**: 2025-01-21 | **Last Amended**: 2025-01-21
