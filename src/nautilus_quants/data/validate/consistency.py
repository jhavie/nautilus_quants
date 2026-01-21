"""
Consistency validation for raw data.

Checks for duplicates, gaps, OHLC relationships, and monotonic timestamps.
"""

import pandas as pd

from nautilus_quants.data.types import (
    ValidationCheckType,
    ValidationIssue,
    ValidationSeverity,
)


# Interval to milliseconds mapping
INTERVAL_MS = {
    "1m": 60 * 1000,
    "3m": 3 * 60 * 1000,
    "5m": 5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "30m": 30 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "2h": 2 * 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "6h": 6 * 60 * 60 * 1000,
    "8h": 8 * 60 * 60 * 1000,
    "12h": 12 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
    "3d": 3 * 24 * 60 * 60 * 1000,
    "1w": 7 * 24 * 60 * 60 * 1000,
}


def check_duplicates(df: pd.DataFrame) -> list[ValidationIssue]:
    """Check for duplicate timestamps.

    Args:
        df: DataFrame with 'timestamp' column

    Returns:
        List of validation issues for duplicates found
    """
    issues = []

    if "timestamp" not in df.columns:
        return issues

    # Find duplicate timestamps
    duplicates = df[df.duplicated(subset=["timestamp"], keep=False)]
    duplicate_timestamps = duplicates["timestamp"].unique()

    for ts in duplicate_timestamps:
        dup_rows = df[df["timestamp"] == ts]
        first_idx = dup_rows.index[0]
        dup_count = len(dup_rows)

        issues.append(
            ValidationIssue(
                check_type=ValidationCheckType.DUPLICATES,
                severity=ValidationSeverity.WARNING,
                row_index=int(first_idx),
                timestamp=int(ts),
                message=f"Duplicate timestamp found ({dup_count} occurrences)",
                details={
                    "first_occurrence": int(first_idx),
                    "count": dup_count,
                    "all_indices": [int(i) for i in dup_rows.index.tolist()],
                },
            )
        )

    return issues


def check_gaps(
    df: pd.DataFrame,
    timeframe: str,
    max_gap_bars: int = 3,
) -> list[ValidationIssue]:
    """Check for gaps in timestamp sequence.

    Args:
        df: DataFrame with 'timestamp' column
        timeframe: Expected interval (e.g., '1h', '4h')
        max_gap_bars: Gaps larger than this are reported as errors

    Returns:
        List of validation issues for gaps found
    """
    issues = []

    if "timestamp" not in df.columns or len(df) < 2:
        return issues

    # Get expected interval in milliseconds
    interval_ms = INTERVAL_MS.get(timeframe, 3600000)  # Default 1h

    # Calculate differences
    timestamps = df["timestamp"].sort_values()
    diffs = timestamps.diff().dropna()

    for idx, diff in diffs.items():
        if diff > interval_ms:
            gap_bars = int(diff / interval_ms) - 1
            prev_idx = int(idx) - 1
            prev_ts = int(timestamps.iloc[prev_idx]) if prev_idx >= 0 else None
            curr_ts = int(timestamps.iloc[int(idx)])

            severity = (
                ValidationSeverity.ERROR
                if gap_bars > max_gap_bars
                else ValidationSeverity.WARNING
            )

            issues.append(
                ValidationIssue(
                    check_type=ValidationCheckType.GAPS,
                    severity=severity,
                    row_index=int(idx),
                    timestamp=curr_ts,
                    message=f"Gap of {gap_bars} bars detected",
                    details={
                        "gap_bars": gap_bars,
                        "gap_ms": int(diff),
                        "expected_ms": interval_ms,
                        "prev_timestamp": prev_ts,
                        "curr_timestamp": curr_ts,
                    },
                )
            )

    return issues


def check_monotonic(df: pd.DataFrame) -> list[ValidationIssue]:
    """Check that timestamps are strictly increasing.

    Args:
        df: DataFrame with 'timestamp' column

    Returns:
        List of validation issues for non-monotonic timestamps
    """
    issues = []

    if "timestamp" not in df.columns or len(df) < 2:
        return issues

    # Check if timestamps are strictly increasing
    timestamps = df["timestamp"]
    diffs = timestamps.diff().dropna()

    non_increasing = diffs[diffs <= 0]

    for idx in non_increasing.index:
        prev_ts = int(timestamps.iloc[int(idx) - 1])
        curr_ts = int(timestamps.iloc[int(idx)])

        issues.append(
            ValidationIssue(
                check_type=ValidationCheckType.MONOTONIC,
                severity=ValidationSeverity.ERROR,
                row_index=int(idx),
                timestamp=curr_ts,
                message=f"Timestamp not strictly increasing (prev: {prev_ts}, curr: {curr_ts})",
                details={
                    "prev_timestamp": prev_ts,
                    "curr_timestamp": curr_ts,
                    "diff_ms": curr_ts - prev_ts,
                },
            )
        )

    return issues


def check_ohlc_relationships(df: pd.DataFrame) -> list[ValidationIssue]:
    """Check OHLC price relationships.

    Validates:
    - high >= max(open, close)
    - low <= min(open, close)
    - low <= high
    - All prices > 0
    - volume >= 0

    Args:
        df: DataFrame with OHLC columns

    Returns:
        List of validation issues for invalid OHLC relationships
    """
    issues = []

    required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    if not all(col in df.columns for col in required_cols):
        return issues

    for idx, row in df.iterrows():
        ts = int(row["timestamp"])
        o, h, low, c = row["open"], row["high"], row["low"], row["close"]
        v = row["volume"]

        # Check high >= max(open, close)
        if h < max(o, c):
            issues.append(
                ValidationIssue(
                    check_type=ValidationCheckType.OHLC_RELATIONSHIP,
                    severity=ValidationSeverity.ERROR,
                    row_index=int(idx),
                    timestamp=ts,
                    message=f"High ({h}) < max(open, close) ({max(o, c)})",
                    details={"open": o, "high": h, "low": low, "close": c},
                )
            )

        # Check low <= min(open, close)
        if low > min(o, c):
            issues.append(
                ValidationIssue(
                    check_type=ValidationCheckType.OHLC_RELATIONSHIP,
                    severity=ValidationSeverity.ERROR,
                    row_index=int(idx),
                    timestamp=ts,
                    message=f"Low ({low}) > min(open, close) ({min(o, c)})",
                    details={"open": o, "high": h, "low": low, "close": c},
                )
            )

        # Check low <= high
        if low > h:
            issues.append(
                ValidationIssue(
                    check_type=ValidationCheckType.OHLC_RELATIONSHIP,
                    severity=ValidationSeverity.ERROR,
                    row_index=int(idx),
                    timestamp=ts,
                    message=f"Low ({low}) > High ({h})",
                    details={"open": o, "high": h, "low": low, "close": c},
                )
            )

        # Check prices > 0
        if any(p <= 0 for p in [o, h, low, c]):
            issues.append(
                ValidationIssue(
                    check_type=ValidationCheckType.OHLC_RELATIONSHIP,
                    severity=ValidationSeverity.ERROR,
                    row_index=int(idx),
                    timestamp=ts,
                    message="Price <= 0 detected",
                    details={"open": o, "high": h, "low": low, "close": c},
                )
            )

        # Check volume >= 0
        if v < 0:
            issues.append(
                ValidationIssue(
                    check_type=ValidationCheckType.VOLUME,
                    severity=ValidationSeverity.ERROR,
                    row_index=int(idx),
                    timestamp=ts,
                    message=f"Negative volume: {v}",
                    details={"volume": v},
                )
            )

    return issues
