"""
Reporting infrastructure for the data pipeline.

Generates run IDs, writes reports in text and JSON formats.
"""

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


def generate_run_id() -> str:
    """Generate a unique run ID in format YYYYMMDD_HHMMSS."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_log_dir(base_path: Path | str, run_id: str) -> Path:
    """Create and return the log directory for a pipeline run.

    Args:
        base_path: Base logs directory (e.g., 'logs/data_pipeline')
        run_id: Unique run identifier

    Returns:
        Path to the created log directory
    """
    log_dir = Path(base_path) / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


class ReportWriter:
    """Writer for pipeline reports in text and JSON formats."""

    def __init__(self, log_dir: Path | str):
        """Initialize report writer.

        Args:
            log_dir: Directory to write reports to
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def write_step_report(
        self,
        step_name: str,
        run_id: str,
        summary: dict[str, Any],
        details_by_symbol: Optional[list[dict]] = None,
        errors: Optional[list[str]] = None,
    ) -> Path:
        """Write a step report (download, validate, process, transform).

        Args:
            step_name: Name of the pipeline step
            run_id: Run identifier
            summary: Summary statistics
            details_by_symbol: Optional per-symbol details
            errors: Optional list of errors

        Returns:
            Path to the written report file
        """
        report_path = self.log_dir / f"{step_name}_report.txt"
        lines = [
            "=" * 70,
            f"{step_name.upper()} REPORT",
            "=" * 70,
            "",
            f"Run ID: {run_id}",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "SUMMARY",
            "-" * 40,
        ]

        for key, value in summary.items():
            lines.append(f"  {key}: {value}")

        if details_by_symbol:
            lines.extend(["", "DETAILS BY SYMBOL", "-" * 40])
            # Header
            lines.append(
                f"{'Symbol':<14} {'Files':<8} {'Size (MB)':<12} {'Time (s)':<10} {'Status'}"
            )
            lines.append("-" * 60)
            for detail in details_by_symbol:
                status = "✓" if detail.get("success", True) else "✗"
                lines.append(
                    f"{detail.get('symbol', 'N/A'):<14} "
                    f"{detail.get('files', 0):<8} "
                    f"{detail.get('size_mb', 0):<12.1f} "
                    f"{detail.get('time_seconds', 0):<10.1f} "
                    f"{status}"
                )

        if errors:
            lines.extend(["", "ERRORS", "-" * 40])
            for error in errors:
                lines.append(f"  - {error}")

        lines.append("")

        with open(report_path, "w") as f:
            f.write("\n".join(lines))

        return report_path

    def write_summary_report(
        self,
        run_id: str,
        start_time: datetime,
        end_time: datetime,
        config: dict[str, Any],
        step_results: list[dict[str, Any]],
        errors: Optional[list[str]] = None,
    ) -> Path:
        """Write the overall summary report.

        Args:
            run_id: Run identifier
            start_time: Pipeline start time
            end_time: Pipeline end time
            config: Configuration used
            step_results: Results from each step
            errors: Optional list of errors

        Returns:
            Path to the written summary file
        """
        duration = (end_time - start_time).total_seconds()
        duration_str = (
            f"{duration / 60:.1f} minutes"
            if duration >= 60
            else f"{duration:.1f} seconds"
        )

        all_success = all(step.get("success", False) for step in step_results)
        status = "SUCCESS" if all_success else "FAILED"

        report_path = self.log_dir / f"summary_{run_id}.txt"
        lines = [
            "=" * 70,
            "DATA PIPELINE SUMMARY REPORT",
            "=" * 70,
            "",
            f"Run ID: {run_id}",
            f"Start: {start_time.isoformat()}",
            f"End: {end_time.isoformat()}",
            f"Duration: {duration_str}",
            f"Status: {status}",
            "",
            "CONFIGURATION",
            "-" * 40,
            f"  Symbols: {len(config.get('download', {}).get('symbols', []))}",
            f"  Date range: {config.get('download', {}).get('start_date', 'N/A')} to {config.get('download', {}).get('end_date', 'N/A')}",
            f"  Exchange: {config.get('download', {}).get('exchange', 'binance')}-{config.get('download', {}).get('market_type', 'futures')}",
            f"  Timeframes: {', '.join(config.get('download', {}).get('timeframes', []))}",
            "",
            "STEP RESULTS",
            "-" * 40,
        ]

        for step in step_results:
            status_icon = "✓" if step.get("success", False) else "✗"
            lines.append("")
            lines.append(
                f"  [{status_icon}] {step.get('step_name', 'UNKNOWN').upper()}"
            )
            lines.append(f"      Duration: {step.get('duration_seconds', 0):.1f}s")
            for key, value in step.get("summary", {}).items():
                lines.append(f"      {key}: {value}")

        if errors:
            lines.extend(["", "ERRORS", "-" * 40])
            for error in errors:
                lines.append(f"  - {error}")

        lines.extend(["", "=" * 70])

        with open(report_path, "w") as f:
            f.write("\n".join(lines))

        return report_path

    def write_details_json(
        self,
        run_id: str,
        start_time: datetime,
        end_time: datetime,
        config: dict[str, Any],
        steps: list[dict[str, Any]],
        errors: Optional[list[str]] = None,
    ) -> Path:
        """Write the detailed JSON report.

        Args:
            run_id: Run identifier
            start_time: Pipeline start time
            end_time: Pipeline end time
            config: Configuration used
            steps: Detailed step information
            errors: Optional list of errors

        Returns:
            Path to the written JSON file
        """
        duration = (end_time - start_time).total_seconds()
        all_success = all(step.get("success", False) for step in steps)

        details = {
            "run_id": run_id,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "elapsed_seconds": round(duration, 2),
            "success": all_success,
            "config": _serialize_for_json(config),
            "steps": [_serialize_for_json(step) for step in steps],
            "errors": errors or [],
        }

        report_path = self.log_dir / f"details_{run_id}.json"
        with open(report_path, "w") as f:
            json.dump(details, f, indent=2, default=str)

        return report_path


def _serialize_for_json(obj: Any) -> Any:
    """Serialize object for JSON output."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: _serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_serialize_for_json(item) for item in obj]
    return obj
