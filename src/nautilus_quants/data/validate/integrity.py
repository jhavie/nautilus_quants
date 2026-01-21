"""
Integrity validation for raw data files.

Checks file existence, schema correctness, and data types.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from nautilus_quants.data.types import (
    ValidationCheckType,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)


# Expected CSV columns for raw kline data
REQUIRED_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trades_count",
]

# Expected data types
COLUMN_TYPES = {
    "timestamp": "int64",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
    "quote_volume": "float64",
    "trades_count": "int64",
}


def validate_schema(df: pd.DataFrame) -> list[ValidationIssue]:
    """Validate DataFrame has required columns and types.

    Args:
        df: DataFrame to validate

    Returns:
        List of validation issues found
    """
    issues = []

    # Check required columns
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_columns:
        issues.append(
            ValidationIssue(
                check_type=ValidationCheckType.SCHEMA,
                severity=ValidationSeverity.ERROR,
                row_index=None,
                timestamp=None,
                message=f"Missing required columns: {missing_columns}",
                details={"missing_columns": list(missing_columns)},
            )
        )

    # Check data types (only for columns that exist)
    for col, expected_type in COLUMN_TYPES.items():
        if col in df.columns:
            actual_type = str(df[col].dtype)
            if actual_type != expected_type:
                # Try to convert and see if it works
                try:
                    if expected_type == "int64":
                        df[col].astype("int64")
                    elif expected_type == "float64":
                        df[col].astype("float64")
                except (ValueError, TypeError):
                    issues.append(
                        ValidationIssue(
                            check_type=ValidationCheckType.SCHEMA,
                            severity=ValidationSeverity.ERROR,
                            row_index=None,
                            timestamp=None,
                            message=f"Column '{col}' has type {actual_type}, expected {expected_type}",
                            details={
                                "column": col,
                                "actual_type": actual_type,
                                "expected_type": expected_type,
                            },
                        )
                    )

    return issues


def validate_file(
    file_path: Path | str,
    checks: Optional[list[ValidationCheckType]] = None,
) -> ValidationReport:
    """Validate a raw data CSV file.

    Performs integrity and consistency checks on the data file.

    Args:
        file_path: Path to CSV file
        checks: Specific checks to run (default: all)

    Returns:
        ValidationReport with issues found
    """
    from nautilus_quants.data.validate.consistency import (
        check_duplicates,
        check_gaps,
        check_monotonic,
        check_ohlc_relationships,
    )

    file_path = Path(file_path)
    issues: list[ValidationIssue] = []

    # Extract symbol and timeframe from file path
    # Expected format: .../SYMBOL/TIMEFRAME/SYMBOL_TIMEFRAME_*.csv
    try:
        symbol = file_path.parent.parent.name
        timeframe = file_path.parent.name
    except Exception:
        symbol = "UNKNOWN"
        timeframe = "UNKNOWN"

    # Check file exists
    if not file_path.exists():
        return ValidationReport(
            symbol=symbol,
            timeframe=timeframe,
            file_path=str(file_path),
            total_rows=0,
            validated_at=datetime.now(),
            passed=False,
            issues=[
                ValidationIssue(
                    check_type=ValidationCheckType.SCHEMA,
                    severity=ValidationSeverity.ERROR,
                    row_index=None,
                    timestamp=None,
                    message=f"File not found: {file_path}",
                )
            ],
            error_count=1,
        )

    # Load data
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return ValidationReport(
            symbol=symbol,
            timeframe=timeframe,
            file_path=str(file_path),
            total_rows=0,
            validated_at=datetime.now(),
            passed=False,
            issues=[
                ValidationIssue(
                    check_type=ValidationCheckType.SCHEMA,
                    severity=ValidationSeverity.ERROR,
                    row_index=None,
                    timestamp=None,
                    message=f"Failed to read CSV: {e}",
                )
            ],
            error_count=1,
        )

    total_rows = len(df)

    # Determine which checks to run
    if checks is None:
        checks = list(ValidationCheckType)

    # Run schema validation
    if ValidationCheckType.SCHEMA in checks:
        issues.extend(validate_schema(df))

    # Skip further checks if schema validation failed
    schema_errors = [i for i in issues if i.severity == ValidationSeverity.ERROR]
    if schema_errors:
        return ValidationReport(
            symbol=symbol,
            timeframe=timeframe,
            file_path=str(file_path),
            total_rows=total_rows,
            validated_at=datetime.now(),
            passed=False,
            issues=issues,
            error_count=len(schema_errors),
        )

    # Run consistency checks
    if ValidationCheckType.DUPLICATES in checks:
        issues.extend(check_duplicates(df))

    if ValidationCheckType.GAPS in checks:
        issues.extend(check_gaps(df, timeframe))

    if ValidationCheckType.MONOTONIC in checks:
        issues.extend(check_monotonic(df))

    if ValidationCheckType.OHLC_RELATIONSHIP in checks:
        issues.extend(check_ohlc_relationships(df))

    # Count issues by type
    error_count = sum(1 for i in issues if i.severity == ValidationSeverity.ERROR)
    warning_count = sum(1 for i in issues if i.severity == ValidationSeverity.WARNING)
    duplicate_count = sum(
        1 for i in issues if i.check_type == ValidationCheckType.DUPLICATES
    )
    gap_count = sum(1 for i in issues if i.check_type == ValidationCheckType.GAPS)
    invalid_ohlc_count = sum(
        1 for i in issues if i.check_type == ValidationCheckType.OHLC_RELATIONSHIP
    )

    return ValidationReport(
        symbol=symbol,
        timeframe=timeframe,
        file_path=str(file_path),
        total_rows=total_rows,
        validated_at=datetime.now(),
        passed=(error_count == 0),
        issues=issues,
        error_count=error_count,
        warning_count=warning_count,
        duplicate_count=duplicate_count,
        gap_count=gap_count,
        invalid_ohlc_count=invalid_ohlc_count,
    )
