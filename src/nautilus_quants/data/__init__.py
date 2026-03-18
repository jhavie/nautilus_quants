"""
Data pipeline module for downloading, validating, processing, and transforming market data.

This module is completely decoupled from trading/strategy modules and provides:
- Download: Fetch historical K-line data from Binance
- Validate: Check data integrity and consistency
- Process: Clean data (remove duplicates, fill gaps)
- Transform: Convert to Nautilus Trader Parquet format
"""

from nautilus_quants.data.types import (
    DownloadCheckpoint,
    PipelineRunContext,
    ProcessedKline,
    ProcessingAction,
    ProcessingReport,
    RawKline,
    ValidationCheckType,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)

__all__ = [
    "RawKline",
    "ValidationReport",
    "ValidationIssue",
    "ValidationCheckType",
    "ValidationSeverity",
    "ProcessedKline",
    "ProcessingReport",
    "ProcessingAction",
    "DownloadCheckpoint",
    "PipelineRunContext",
]
