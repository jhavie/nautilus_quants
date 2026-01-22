"""Reporting utilities for backtest module."""

from datetime import datetime
from pathlib import Path


def generate_run_id() -> str:
    """Generate unique run ID in format YYYYMMDD_HHMMSS.

    Returns:
        Run ID string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_output_directory(base_dir: str | Path, run_id: str) -> Path:
    """Create output directory for backtest run.

    Args:
        base_dir: Base output directory
        run_id: Run identifier

    Returns:
        Path to created directory
    """
    output_dir = Path(base_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
