# Feature Specification: Backtest Module

**Feature Branch**: `002-backtest-module`
**Created**: 2026-01-21
**Status**: Draft
**Input**: User description: "我需要一个回测模块 取代掉目前文件内的backtest python文件为文件夹 backtest为独立模块 不与其它模块耦合 具有读取回测配置 执行回测 写入日志 利用nautilus_trader原生能力输出相关的report"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Run Backtest from Configuration File (Priority: P1)

A quant researcher wants to run a backtest by providing a YAML configuration file that specifies the strategy, data sources, venue settings, and output preferences. The system reads the configuration, executes the backtest using nautilus_trader's native BacktestEngine, and generates comprehensive reports.

**Why this priority**: This is the core functionality that enables users to run backtests without writing code. Configuration-driven backtesting is essential for rapid strategy iteration and reproducibility.

**Independent Test**: Can be fully tested by providing a valid YAML config file and verifying that the backtest completes with expected reports generated in the output directory.

**Acceptance Scenarios**:

1. **Given** a valid backtest configuration file with strategy, venue, and data settings, **When** the user runs the backtest command, **Then** the system executes the backtest and generates output reports in the specified directory.
2. **Given** a configuration file with missing required fields, **When** the user attempts to run the backtest, **Then** the system provides a clear error message indicating which fields are missing.
3. **Given** a configuration file referencing non-existent data files, **When** the user runs the backtest, **Then** the system reports which data files could not be found before attempting execution.

---

### User Story 2 - Generate Performance Reports (Priority: P1)

After a backtest completes, the user wants to receive comprehensive performance reports including orders, fills, positions, and account balance history. These reports should be saved as CSV files for further analysis.

**Why this priority**: Performance reports are essential for evaluating strategy effectiveness and are tightly coupled with the core backtest execution.

**Independent Test**: Can be tested by running a backtest with a simple strategy and verifying that all expected report CSV files are created with correct data structure.

**Acceptance Scenarios**:

1. **Given** a completed backtest with trading activity, **When** report generation is triggered, **Then** orders report, fills report, positions report, and account report CSV files are created in the output directory.
2. **Given** a completed backtest with no trades, **When** report generation is triggered, **Then** the system generates reports indicating no trading activity rather than failing.
3. **Given** an output directory that doesn't exist, **When** reports are generated, **Then** the system creates the directory structure automatically.

---

### User Story 3 - Generate Interactive Tearsheet (Priority: P2)

After a backtest completes, the user wants to generate an interactive HTML tearsheet with equity curves, drawdown charts, returns distribution, and performance statistics visualizations.

**Why this priority**: Visual tearsheets provide intuitive understanding of strategy performance and are valuable for stakeholder communication, but text reports (P1) provide the essential numerical data.

**Independent Test**: Can be tested by running a backtest and verifying an HTML file is generated that opens correctly in a browser with interactive charts.

**Acceptance Scenarios**:

1. **Given** a completed backtest with positions and returns data, **When** tearsheet generation is requested, **Then** an HTML file is created containing equity curve, drawdown chart, and performance statistics.
2. **Given** plotly is not installed, **When** tearsheet generation is requested, **Then** the system logs a warning with installation instructions and continues without failing.
3. **Given** a custom theme is specified in configuration, **When** the tearsheet is generated, **Then** the visualization uses the specified theme colors and styling.

---

### User Story 4 - View Performance Statistics (Priority: P2)

The user wants to see calculated performance statistics including PnL, returns, Sharpe ratio, max drawdown, win rate, and other key metrics after the backtest completes.

**Why this priority**: Statistics provide quantitative assessment of strategy performance and complement the CSV reports, enabling quick evaluation without opening files.

**Independent Test**: Can be tested by running a backtest and verifying statistics are logged and/or returned in a structured format.

**Acceptance Scenarios**:

1. **Given** a completed backtest with closed positions, **When** performance statistics are calculated, **Then** the system provides PnL (total and percentage), Sharpe ratio, max drawdown, win rate, and profit factor.
2. **Given** a multi-currency portfolio, **When** statistics are requested for a specific currency, **Then** the system returns statistics for that currency.
3. **Given** a backtest with no completed trades, **When** statistics are calculated, **Then** appropriate zero/null values are returned rather than calculation errors.

---

### User Story 5 - Configure Logging Behavior (Priority: P3)

The user wants to control the verbosity and destination of backtest logs, including the ability to save logs to files for debugging and audit purposes.

**Why this priority**: Logging is important for debugging and auditing but is secondary to core backtest execution and reporting functionality.

**Independent Test**: Can be tested by running a backtest with different logging configurations and verifying log output matches the specified settings.

**Acceptance Scenarios**:

1. **Given** a configuration with log_level set to "DEBUG", **When** the backtest runs, **Then** detailed debug messages are output.
2. **Given** a configuration with a log file path specified, **When** the backtest runs, **Then** logs are written to the specified file.
3. **Given** no logging configuration specified, **When** the backtest runs, **Then** the system uses sensible defaults (INFO level, console output).

---

### Edge Cases

- What happens when the data file format is unsupported or corrupted?
- How does the system handle a backtest that runs out of memory with large datasets?
- What happens when the strategy raises an exception during execution?
- How does the system handle timezone mismatches between data and configuration?
- What happens when the output directory is read-only?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Module MUST be a standalone Python package (`backtest/`) that can be imported independently without requiring other nautilus_quants modules.
- **FR-002**: Module MUST read backtest configuration from YAML files with validation of required fields.
- **FR-003**: Module MUST support configuring venues with account type, starting balances, leverage, and fee models.
- **FR-004**: Module MUST support loading market data from Parquet files and converting to nautilus_trader Bar/Tick formats.
- **FR-005**: Module MUST execute backtests using nautilus_trader's native BacktestEngine.
- **FR-006**: Module MUST generate orders report using nautilus_trader's ReportProvider.generate_orders_report().
- **FR-007**: Module MUST generate fills report using nautilus_trader's ReportProvider.generate_fills_report().
- **FR-008**: Module MUST generate positions report using nautilus_trader's ReportProvider.generate_positions_report().
- **FR-009**: Module MUST generate account report using nautilus_trader's ReportProvider.generate_account_report().
- **FR-010**: Module MUST calculate performance statistics using nautilus_trader's PortfolioAnalyzer (PnL, returns, Sharpe ratio, max drawdown, win rate, etc.).
- **FR-011**: Module MUST support generating HTML tearsheets using nautilus_trader's create_tearsheet() with configurable themes.
- **FR-012**: Module MUST support nautilus_trader's LoggingConfig for configurable log levels and output destinations.
- **FR-013**: Module MUST save all reports to a configurable output directory.
- **FR-014**: Module MUST provide a CLI entry point for running backtests from command line.
- **FR-015**: Module MUST provide a Python API for programmatic backtest execution.
- **FR-016**: Module MUST support time range filtering (start_date, end_date) for backtest data.
- **FR-017**: Module MUST support multiple instruments in a single backtest run.
- **FR-018**: Module MUST gracefully handle optional dependencies (plotly for tearsheets) with clear error messages.

### Key Entities

- **BacktestConfig**: Represents the complete configuration for a backtest run including venue settings, data sources, strategy configuration, logging preferences, and output settings.
- **BacktestRunner**: Orchestrates backtest execution, coordinating engine setup, data loading, strategy attachment, execution, and report generation.
- **ReportGenerator**: Handles generation of all output reports (CSV, HTML, statistics) using nautilus_trader's native capabilities.
- **DataLoader**: Responsible for loading and converting market data from various file formats to nautilus_trader data types.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can run a complete backtest by providing only a YAML configuration file, without writing Python code.
- **SC-002**: All generated CSV reports contain the same data fields as nautilus_trader's native ReportProvider output.
- **SC-003**: Performance statistics output includes at minimum: Total PnL, PnL%, Sharpe Ratio, Max Drawdown, Win Rate, and Profit Factor.
- **SC-004**: HTML tearsheets render correctly in modern browsers (Chrome, Firefox, Safari) with interactive chart functionality.
- **SC-005**: Module can be imported and used without importing any other nautilus_quants modules (verified by import test).
- **SC-006**: Backtest with 1 year of hourly data for 5 instruments completes within 60 seconds on standard hardware.
- **SC-007**: Clear error messages are provided for all common configuration mistakes within 5 seconds of execution start.

## Assumptions

- Python version 3.12 or higher is available (per project constitution).
- nautilus_trader version 1.222.0 or compatible is installed.
- Input data is in Parquet format with standard OHLCV columns (open, high, low, close, volume).
- Users have basic familiarity with YAML configuration syntax.
- The module will focus on crypto perpetual contracts initially, with extensibility for other instrument types.
- Plotly is an optional dependency; tearsheet generation gracefully degrades without it.

## Out of Scope

- Live trading execution (handled by separate execution module).
- Strategy development (strategies are provided by users or other modules).
- Data downloading/fetching (handled by data pipeline module).
- Optimization/parameter sweep functionality (potential future enhancement).
- Database storage of backtest results (files only for this version).
