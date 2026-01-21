"""
Validation module for checking data integrity and consistency.

Provides:
- validate_file: Main entry point for validating a raw data file
- Integrity checks: Schema, types, file format
- Consistency checks: Duplicates, gaps, OHLC relationships
"""

from nautilus_quants.data.validate.consistency import (
    check_duplicates,
    check_gaps,
    check_ohlc_relationships,
)
from nautilus_quants.data.validate.integrity import validate_file, validate_schema

__all__ = [
    "validate_file",
    "validate_schema",
    "check_duplicates",
    "check_gaps",
    "check_ohlc_relationships",
]
