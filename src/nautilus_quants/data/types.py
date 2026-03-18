"""
Core data types and entities for the data pipeline.

All entities are defined as dataclasses for immutability and type safety.
See specs/feature-1-binance-data-pipeline/data-model.md for full documentation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional


class ValidationSeverity(Enum):
    """Severity level for validation issues."""

    ERROR = "error"  # Blocks processing
    WARNING = "warning"  # Logged but continues


class ValidationCheckType(Enum):
    """Types of validation checks performed."""

    SCHEMA = "schema"
    DUPLICATES = "duplicates"
    GAPS = "gaps"
    OHLC_RELATIONSHIP = "ohlc_relationship"
    VOLUME = "volume"
    MONOTONIC = "monotonic"


@dataclass(frozen=True)
class RawKline:
    """Raw K-line data from Binance API.

    Attributes:
        timestamp: Unix timestamp in milliseconds
        open: Opening price
        high: Highest price
        low: Lowest price
        close: Closing price
        volume: Base asset volume
        quote_volume: Quote asset volume
        trades_count: Number of trades
        taker_buy_base_volume: Taker buy base asset volume
        taker_buy_quote_volume: Taker buy quote asset volume
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        timeframe: K-line interval (e.g., "1h")
        exchange: Source exchange
    """

    timestamp: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    quote_volume: Decimal
    trades_count: int
    taker_buy_base_volume: Decimal
    taker_buy_quote_volume: Decimal
    symbol: str
    timeframe: str
    exchange: str = "binance"


@dataclass
class ValidationIssue:
    """Single validation issue found in data.

    Attributes:
        check_type: Type of validation check that found this issue
        severity: ERROR or WARNING
        row_index: Row index where issue was found (None for file-level issues)
        timestamp: Affected timestamp (None for file-level issues)
        message: Human-readable description
        details: Additional context for the issue
    """

    check_type: ValidationCheckType
    severity: ValidationSeverity
    row_index: Optional[int]
    timestamp: Optional[int]
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Complete validation report for a data file.

    Attributes:
        symbol: Trading pair symbol
        timeframe: K-line interval
        file_path: Path to the validated file
        total_rows: Total number of rows in the file
        validated_at: When validation was performed
        passed: True if no ERROR-level issues
        issues: List of validation issues found
        error_count: Number of ERROR-level issues
        warning_count: Number of WARNING-level issues
        duplicate_count: Number of duplicate timestamps
        gap_count: Number of gaps detected
        invalid_ohlc_count: Number of invalid OHLC relationships
    """

    symbol: str
    timeframe: str
    file_path: str
    total_rows: int
    validated_at: datetime
    passed: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    error_count: int = 0
    warning_count: int = 0
    duplicate_count: int = 0
    gap_count: int = 0
    invalid_ohlc_count: int = 0


@dataclass(frozen=True)
class ProcessedKline:
    """Cleaned and validated K-line data.

    Attributes:
        timestamp: Unix timestamp in milliseconds
        open: Opening price
        high: Highest price
        low: Lowest price
        close: Closing price
        volume: Base asset volume
        quote_volume: Quote asset volume
        trades_count: Number of trades
        taker_buy_base_volume: Taker buy base asset volume
        taker_buy_quote_volume: Taker buy quote asset volume
        symbol: Trading pair symbol
        timeframe: K-line interval
        was_filled: True if this row was gap-filled
        original_row_index: Reference to raw data row index
    """

    timestamp: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    quote_volume: Decimal
    trades_count: int
    taker_buy_base_volume: Decimal
    taker_buy_quote_volume: Decimal
    symbol: str
    timeframe: str
    was_filled: bool = False
    original_row_index: Optional[int] = None


@dataclass
class ProcessingAction:
    """Single processing action taken on data.

    Attributes:
        action_type: Type of action (remove_duplicate, fill_gap, fix_ohlc)
        row_index: Row index where action was taken
        timestamp: Timestamp of the affected row
        description: Human-readable description
        before_value: Original values before modification
        after_value: New values after modification
    """

    action_type: str
    row_index: int
    timestamp: int
    description: str
    before_value: Optional[dict] = None
    after_value: Optional[dict] = None


@dataclass
class ProcessingReport:
    """Complete processing report for a data file.

    Attributes:
        symbol: Trading pair symbol
        timeframe: K-line interval
        input_file: Path to input raw file
        output_file: Path to output processed file
        processed_at: When processing was performed
        input_rows: Number of rows in input file
        output_rows: Number of rows in output file
        duplicates_removed: Count of duplicates removed
        gaps_filled: Count of gaps filled
        invalid_rows_removed: Count of invalid rows removed
        actions: Detailed list of processing actions for audit
    """

    symbol: str
    timeframe: str
    input_file: str
    output_file: str
    processed_at: datetime
    input_rows: int
    output_rows: int
    duplicates_removed: int
    gaps_filled: int
    invalid_rows_removed: int
    actions: list[ProcessingAction] = field(default_factory=list)


@dataclass
class DownloadCheckpoint:
    """Checkpoint for resumable downloads.

    Attributes:
        symbol: Trading pair symbol
        timeframe: K-line interval
        exchange: Source exchange
        market_type: "spot" or "futures"
        last_timestamp: Last successfully downloaded timestamp (Unix ms)
        last_updated: When checkpoint was last updated
        total_rows: Total rows downloaded so far
        start_date: Original request start date
        end_date: Original request end date
    """

    symbol: str
    timeframe: str
    exchange: str
    market_type: str
    last_timestamp: int
    last_updated: datetime
    total_rows: int
    start_date: str
    end_date: str


@dataclass
class PipelineRunContext:
    """Context for a complete pipeline run.

    Attributes:
        run_id: Unique run identifier (format: YYYYMMDD_HHMMSS)
        start_time: When the pipeline run started
        end_time: When the pipeline run ended (None if still running)
        config: Loaded configuration from config/data.yaml
        symbols: List of symbols being processed
        timeframes: List of timeframes being processed
        download_success: Whether download step succeeded
        validate_success: Whether validation step succeeded
        process_success: Whether processing step succeeded
        transform_success: Whether transform step succeeded
        log_dir: Directory for pipeline logs
        errors: List of error messages
    """

    run_id: str
    start_time: datetime
    config: dict
    symbols: list[str]
    timeframes: list[str]
    end_time: Optional[datetime] = None
    download_success: bool = False
    validate_success: bool = False
    process_success: bool = False
    transform_success: bool = False
    log_dir: str = ""
    errors: list[str] = field(default_factory=list)
