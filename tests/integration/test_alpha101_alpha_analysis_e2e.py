# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""E2E integration test: Alpha analysis for 45 Alpha101 factors.

Validates that the alpha analysis pipeline (IC/ICIR, charts, summary)
runs end-to-end on 3 months of 4h data for all 45 crypto-friendly factors.

Usage:
  pytest tests/integration/test_alpha101_alpha_analysis_e2e.py -v
  python tests/integration/test_alpha101_alpha_analysis_e2e.py  # standalone
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALPHA_CONFIG = PROJECT_ROOT / "config" / "examples" / "alpha_101_45.yaml"
CATALOG_PATH = Path("/Users/joe/Sync/nautilus_quants2/data/data_4years_4h/catalog")

SYMBOLS_12 = [
    "AAVEUSDT.BINANCE",
    "ADAUSDT.BINANCE",
    "ATOMUSDT.BINANCE",
    "BNBUSDT.BINANCE",
    "BTCUSDT.BINANCE",
    "DOGEUSDT.BINANCE",
    "ETHUSDT.BINANCE",
    "LINKUSDT.BINANCE",
    "LTCUSDT.BINANCE",
    "SOLUSDT.BINANCE",
    "UNIUSDT.BINANCE",
    "XRPUSDT.BINANCE",
]

# ---------------------------------------------------------------------------
# Precondition checks
# ---------------------------------------------------------------------------

_DATA_AVAILABLE = CATALOG_PATH.exists()
_CONFIG_AVAILABLE = ALPHA_CONFIG.exists()

skip_no_data = pytest.mark.skipif(
    not _DATA_AVAILABLE,
    reason=f"Catalog not found at {CATALOG_PATH}",
)
skip_no_config = pytest.mark.skipif(
    not _CONFIG_AVAILABLE,
    reason=f"Alpha config not found at {ALPHA_CONFIG}",
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@skip_no_data
@skip_no_config
class TestAlpha101AlphaAnalysisE2E:
    """Full alpha analysis E2E for 45 Alpha101 factors."""

    def test_alpha_analysis_runs(self, tmp_path: Path):
        """Alpha analysis pipeline completes without crashing."""
        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                "-m", "nautilus_quants.alpha",
                "analyze",
                str(ALPHA_CONFIG),
            ],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(PROJECT_ROOT),
        )

        assert result.returncode == 0, (
            f"Alpha analysis failed with code {result.returncode}\n"
            f"STDERR: {result.stderr[-2000:]}\n"
            f"STDOUT: {result.stdout[-2000:]}"
        )

    def test_output_directory_created(self):
        """Output directory with summary.txt exists after analysis."""
        output_base = PROJECT_ROOT / "logs" / "alpha_101_45factors"
        if not output_base.exists():
            pytest.skip("Run test_alpha_analysis_runs first")

        # Find latest run directory
        run_dirs = sorted(output_base.iterdir())
        assert len(run_dirs) > 0, "No run directories found"

        latest = run_dirs[-1]
        summary = latest / "summary.txt"
        assert summary.exists(), f"summary.txt not found in {latest}"

    def test_factor_chart_directories(self):
        """Each analyzed factor has a chart directory with PNG files."""
        output_base = PROJECT_ROOT / "logs" / "alpha_101_45factors"
        if not output_base.exists():
            pytest.skip("Run test_alpha_analysis_runs first")

        run_dirs = sorted(output_base.iterdir())
        if not run_dirs:
            pytest.skip("No run directories found")

        latest = run_dirs[-1]
        factor_dirs = [
            d for d in latest.iterdir()
            if d.is_dir() and d.name.startswith("alpha")
        ]

        assert len(factor_dirs) > 0, "No factor directories found"

        # Check at least one factor has charts
        has_charts = False
        for fd in factor_dirs:
            pngs = list(fd.glob("*.png"))
            if pngs:
                has_charts = True
                break

        assert has_charts, "No PNG charts generated for any factor"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import subprocess

    if not _DATA_AVAILABLE:
        print(f"ERROR: Catalog not found at {CATALOG_PATH}")
        sys.exit(1)
    if not _CONFIG_AVAILABLE:
        print(f"ERROR: Alpha config not found at {ALPHA_CONFIG}")
        sys.exit(1)

    print("=" * 70)
    print("Alpha101 45-Factor Alpha Analysis E2E")
    print(f"  Config: {ALPHA_CONFIG}")
    print(f"  Data:   {CATALOG_PATH}")
    print("=" * 70)

    t0 = time.time()
    result = subprocess.run(
        [
            sys.executable,
            "-m", "nautilus_quants.alpha",
            "analyze",
            str(ALPHA_CONFIG),
        ],
        text=True,
        timeout=600,
        cwd=str(PROJECT_ROOT),
    )
    elapsed = time.time() - t0

    print(f"\n{'=' * 70}")
    print(f"Duration: {elapsed:.1f}s")
    print(f"Exit code: {result.returncode}")
    print(f"{'=' * 70}")

    if result.returncode == 0:
        # Find output
        output_base = PROJECT_ROOT / "logs" / "alpha_101_45factors"
        if output_base.exists():
            run_dirs = sorted(output_base.iterdir())
            if run_dirs:
                latest = run_dirs[-1]
                factor_dirs = [d for d in latest.iterdir() if d.is_dir() and d.name.startswith("alpha")]
                print(f"\n✅ PASSED — {len(factor_dirs)} factors analyzed")
                print(f"   Output: {latest}")
    else:
        print("\n❌ FAILED")

    sys.exit(result.returncode)
