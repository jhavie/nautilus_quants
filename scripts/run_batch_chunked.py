#!/usr/bin/env python3
"""Run factor analysis in chunks of N factors to avoid long runtimes.

Usage:
    python scripts/run_batch_chunked.py [--chunk-size 20] [--clean]
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

CONFIGS = [
    ("Alpha101", "config/alpha_batch/factors_alpha101.yaml", "config/alpha_batch/analyze_alpha101.yaml"),
    ("TA", "config/alpha_batch/factors_ta.yaml", "config/alpha_batch/analyze_ta.yaml"),
    ("Alpha158", "config/alpha_batch/factors_alpha158.yaml", "config/alpha_batch/analyze_alpha158.yaml"),
    ("Alpha191", "config/alpha_batch/factors_alpha191.yaml", "config/alpha_batch/analyze_alpha191.yaml"),
]

ENV = "test"
CWD = Path(__file__).resolve().parent.parent


def run_chunked(
    lib_name: str,
    factors_path: str,
    analyze_path: str,
    chunk_size: int = 20,
) -> None:
    with open(CWD / factors_path) as f:
        factors_data = yaml.safe_load(f)
    with open(CWD / analyze_path) as f:
        analyze_data = yaml.safe_load(f)

    all_factor_names = list(factors_data.get("factors", {}).keys())
    total = len(all_factor_names)
    chunks = [all_factor_names[i:i + chunk_size] for i in range(0, total, chunk_size)]

    print(f"\n{'='*60}")
    print(f"{lib_name}: {total} factors in {len(chunks)} chunks of {chunk_size}")
    print(f"{'='*60}")

    for idx, chunk in enumerate(chunks):
        # Create temp factor YAML with only this chunk
        chunk_factors = {name: factors_data["factors"][name] for name in chunk}
        chunk_data = {
            "metadata": factors_data["metadata"],
            "factors": chunk_factors,
        }
        if "variables" in factors_data:
            chunk_data["variables"] = factors_data["variables"]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", prefix=f"{lib_name}_chunk{idx}_",
            dir=str(CWD / "config/alpha_batch"), delete=False,
        ) as tmp:
            yaml.dump(chunk_data, tmp, default_flow_style=False, sort_keys=False)
            tmp_factors_path = tmp.name

        # Create temp analyze YAML pointing to chunk factors
        chunk_analyze = dict(analyze_data)
        chunk_analyze["factor_config_path"] = tmp_factors_path

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", prefix=f"analyze_{lib_name}_chunk{idx}_",
            dir=str(CWD / "config/alpha_batch"), delete=False,
        ) as tmp:
            yaml.dump(chunk_analyze, tmp, default_flow_style=False, sort_keys=False)
            tmp_analyze_path = tmp.name

        print(f"\n  Chunk {idx+1}/{len(chunks)}: {len(chunk)} factors ({chunk[0]}..{chunk[-1]})")

        result = subprocess.run(
            [sys.executable, "-m", "nautilus_quants.alpha", "analyze",
             tmp_analyze_path, "--env", ENV, "-q"],
            cwd=str(CWD),
            capture_output=True, text=True,
        )

        # Print warnings
        for line in result.stderr.split("\n"):
            if "Warning" in line and "failed" in line:
                print(f"    {line.strip()}")
            elif "Registry" in line:
                print(f"    {line.strip()}")

        # Clean temp files
        Path(tmp_factors_path).unlink(missing_ok=True)
        Path(tmp_analyze_path).unlink(missing_ok=True)

        if result.returncode != 0:
            print(f"    ERROR (exit {result.returncode})")
            for line in result.stderr.split("\n")[-5:]:
                if line.strip():
                    print(f"    {line.strip()}")
        else:
            print(f"    OK")


def verify_registry() -> None:
    print(f"\n{'='*60}")
    print("Registry verification")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, "-c", """
from nautilus_quants.alpha.registry.database import RegistryDatabase
from nautilus_quants.alpha.registry.environment import resolve_env
env = resolve_env('test')
db = RegistryDatabase.for_environment(env)
factors = db.fetch_one('SELECT COUNT(*) FROM factors')
metrics = db.fetch_one('SELECT COUNT(*) FROM alpha_analysis_metrics')
distinct = db.fetch_one('SELECT COUNT(DISTINCT factor_id) FROM alpha_analysis_metrics')
no_metrics = db.fetch_all(
    'SELECT f.factor_id FROM factors f LEFT JOIN alpha_analysis_metrics m ON f.factor_id = m.factor_id WHERE m.factor_id IS NULL'
)
print(f'Factors registered: {factors[0]}')
print(f'Factors with metrics: {distinct[0]}')
print(f'Metric records: {metrics[0]}')
if no_metrics:
    print(f'Factors WITHOUT metrics: {[r[0] for r in no_metrics]}')
else:
    print('All factors have metrics!')
db.close()
"""],
        cwd=str(CWD),
        capture_output=True, text=True,
    )
    print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip())


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--clean", action="store_true", help="Delete test.duckdb before starting")
    parser.add_argument("--only", choices=["alpha101", "ta", "alpha158", "alpha191"], default=None)
    args = parser.parse_args()

    if args.clean:
        db_path = CWD / "logs/registry/test.duckdb"
        if db_path.exists():
            db_path.unlink()
            print("Deleted test.duckdb")

    for lib_name, factors_path, analyze_path in CONFIGS:
        if args.only and args.only != lib_name.lower():
            continue
        run_chunked(lib_name, factors_path, analyze_path, args.chunk_size)

    verify_registry()


if __name__ == "__main__":
    main()
