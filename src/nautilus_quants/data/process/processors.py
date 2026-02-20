"""
Data processing module for cleaning raw data.

Provides functions to remove duplicates, fill gaps, and fix invalid OHLC rows.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

from nautilus_quants.data.types import (
    ProcessingAction,
    ProcessingReport,
    ValidationReport,
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


@dataclass
class ProcessConfig:
    """Configuration for data processing."""

    remove_duplicates: bool = True
    keep_duplicate: Literal["first", "last"] = "last"
    fill_small_gaps: bool = True
    max_gap_bars: int = 3
    remove_invalid_ohlc: bool = True


def remove_duplicates(
    df: pd.DataFrame,
    keep: Literal["first", "last"] = "last",
) -> tuple[pd.DataFrame, list[ProcessingAction]]:
    """Remove duplicate timestamps from DataFrame.

    Args:
        df: DataFrame with 'timestamp' column
        keep: Which duplicate to keep ('first' or 'last')

    Returns:
        Tuple of (cleaned DataFrame, list of actions taken)
    """
    actions = []

    if "timestamp" not in df.columns:
        return df, actions

    # Find duplicates before removal
    duplicates = df[df.duplicated(subset=["timestamp"], keep=False)]
    duplicate_timestamps = duplicates["timestamp"].unique()

    for ts in duplicate_timestamps:
        dup_rows = df[df["timestamp"] == ts]
        kept_idx = dup_rows.index[0] if keep == "first" else dup_rows.index[-1]
        removed_indices = [i for i in dup_rows.index if i != kept_idx]

        for removed_idx in removed_indices:
            actions.append(
                ProcessingAction(
                    action_type="remove_duplicate",
                    row_index=int(removed_idx),
                    timestamp=int(ts),
                    description=f"Removed duplicate timestamp, kept {keep} occurrence",
                    before_value=df.loc[removed_idx].to_dict(),
                    after_value=None,
                )
            )

    # Remove duplicates
    df_clean = df.drop_duplicates(subset=["timestamp"], keep=keep).reset_index(
        drop=True
    )

    return df_clean, actions


def fill_gaps(
    df: pd.DataFrame,
    timeframe: str,
    max_gap_bars: int = 3,
) -> tuple[pd.DataFrame, list[ProcessingAction]]:
    """Fill small gaps in timestamp sequence using forward fill.

    Args:
        df: DataFrame with timestamp and OHLCV columns
        timeframe: Expected interval (e.g., '1h', '4h')
        max_gap_bars: Maximum gap size to fill (larger gaps are not filled)

    Returns:
        Tuple of (gap-filled DataFrame, list of actions taken)
    """
    actions = []

    if "timestamp" not in df.columns or len(df) < 2:
        return df, actions

    # Get expected interval in milliseconds
    interval_ms = INTERVAL_MS.get(timeframe, 3600000)

    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Find gaps and fill them
    new_rows = []
    for i in range(1, len(df)):
        prev_ts = df.loc[i - 1, "timestamp"]
        curr_ts = df.loc[i, "timestamp"]
        gap_ms = curr_ts - prev_ts

        if gap_ms > interval_ms:
            gap_bars = int(gap_ms / interval_ms) - 1

            if gap_bars <= max_gap_bars:
                # Fill the gap with forward-filled values
                prev_row = df.loc[i - 1]

                for j in range(1, gap_bars + 1):
                    fill_ts = prev_ts + j * interval_ms

                    # Create filled row using previous close as all OHLC
                    filled_row = {
                        "timestamp": fill_ts,
                        "open": prev_row["close"],
                        "high": prev_row["close"],
                        "low": prev_row["close"],
                        "close": prev_row["close"],
                        "volume": 0.0,
                        "quote_volume": 0.0,
                        "trades_count": 0,
                        "taker_buy_base_volume": 0.0,
                        "taker_buy_quote_volume": 0.0,
                    }

                    # Add was_filled column if not present
                    if "was_filled" in df.columns or len(new_rows) > 0:
                        filled_row["was_filled"] = True

                    new_rows.append(filled_row)

                    actions.append(
                        ProcessingAction(
                            action_type="fill_gap",
                            row_index=-1,  # Will be determined after insert
                            timestamp=int(fill_ts),
                            description=f"Filled gap ({j}/{gap_bars}) using forward fill",
                            before_value=None,
                            after_value=filled_row,
                        )
                    )

    # Add was_filled column to original data
    if new_rows:
        if "was_filled" not in df.columns:
            df = df.copy()
            df["was_filled"] = False

        # Ensure new rows have was_filled=True
        for row in new_rows:
            row["was_filled"] = True

        # Append new rows and sort
        new_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_df], ignore_index=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        # Ensure was_filled is boolean, filling any NaN with False
        df["was_filled"] = df["was_filled"].fillna(False).astype(bool)

    return df, actions


def remove_invalid_ohlc(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[ProcessingAction]]:
    """Remove rows with invalid OHLC relationships.

    Invalid conditions:
    - high < max(open, close)
    - low > min(open, close)
    - low > high
    - Any price <= 0
    - Negative volume

    Args:
        df: DataFrame with OHLC columns

    Returns:
        Tuple of (cleaned DataFrame, list of actions taken)
    """
    actions = []

    required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    if not all(col in df.columns for col in required_cols):
        return df, actions

    invalid_indices = []

    for idx, row in df.iterrows():
        o, h, low, c = row["open"], row["high"], row["low"], row["close"]
        v = row["volume"]

        is_invalid = False
        reason = []

        if h < max(o, c):
            is_invalid = True
            reason.append(f"high ({h}) < max(open, close)")

        if low > min(o, c):
            is_invalid = True
            reason.append(f"low ({low}) > min(open, close)")

        if low > h:
            is_invalid = True
            reason.append(f"low ({low}) > high ({h})")

        if any(p <= 0 for p in [o, h, low, c]):
            is_invalid = True
            reason.append("price <= 0")

        if v < 0:
            is_invalid = True
            reason.append(f"negative volume ({v})")

        if is_invalid:
            invalid_indices.append(idx)
            actions.append(
                ProcessingAction(
                    action_type="remove_invalid_ohlc",
                    row_index=int(idx),
                    timestamp=int(row["timestamp"]),
                    description=f"Removed invalid row: {', '.join(reason)}",
                    before_value=row.to_dict(),
                    after_value=None,
                )
            )

    # Remove invalid rows
    df_clean = df.drop(index=invalid_indices).reset_index(drop=True)

    return df_clean, actions


def process_data(
    input_path: Path | str,
    output_path: Path | str,
    validation_report: Optional[ValidationReport] = None,
    config: Optional[ProcessConfig] = None,
) -> ProcessingReport:
    """Process raw data based on validation report.

    Applies configured processing steps:
    1. Remove duplicates
    2. Fill small gaps
    3. Remove invalid OHLC rows

    Args:
        input_path: Path to raw CSV file
        output_path: Path for processed output
        validation_report: Validation results (optional, for logging)
        config: Processing configuration

    Returns:
        ProcessingReport with actions taken
    """
    if config is None:
        config = ProcessConfig()

    input_path = Path(input_path)
    output_path = Path(output_path)

    # Extract symbol and timeframe from path
    try:
        symbol = input_path.parent.parent.name
        timeframe = input_path.parent.name
    except Exception:
        symbol = "UNKNOWN"
        timeframe = "UNKNOWN"

    # Load data
    df = pd.read_csv(input_path)
    input_rows = len(df)

    all_actions: list[ProcessingAction] = []
    duplicates_removed = 0
    gaps_filled = 0
    invalid_removed = 0

    # Step 1: Remove duplicates
    if config.remove_duplicates:
        df, actions = remove_duplicates(df, keep=config.keep_duplicate)
        duplicates_removed = len(actions)
        all_actions.extend(actions)

    # Step 2: Fill gaps
    if config.fill_small_gaps:
        df, actions = fill_gaps(df, timeframe, max_gap_bars=config.max_gap_bars)
        gaps_filled = len(actions)
        all_actions.extend(actions)

    # Step 3: Remove invalid OHLC
    if config.remove_invalid_ohlc:
        df, actions = remove_invalid_ohlc(df)
        invalid_removed = len(actions)
        all_actions.extend(actions)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save processed data
    df.to_csv(output_path, index=False)

    return ProcessingReport(
        symbol=symbol,
        timeframe=timeframe,
        input_file=str(input_path),
        output_file=str(output_path),
        processed_at=datetime.now(),
        input_rows=input_rows,
        output_rows=len(df),
        duplicates_removed=duplicates_removed,
        gaps_filled=gaps_filled,
        invalid_rows_removed=invalid_removed,
        actions=all_actions,
    )
