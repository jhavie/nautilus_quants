#!/usr/bin/env python3
# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Generate factor YAML configs from builtin factor dictionaries.

Usage:
    python scripts/generate_factor_yamls.py alpha101 [--ohlcv-only] [--no-magic] [-o config/alpha_batch/factors_alpha101.yaml]
    python scripts/generate_factor_yamls.py alpha101 --priority 20 -o config/alpha_batch/factors_alpha101_p1.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


# Known vwap-dependent Alpha101 factors
_ALPHA101_VWAP = {
    "alpha005", "alpha011", "alpha025", "alpha027", "alpha032", "alpha036",
    "alpha041", "alpha042", "alpha047", "alpha050", "alpha057", "alpha061",
    "alpha062", "alpha064", "alpha065", "alpha066", "alpha071", "alpha072",
    "alpha073", "alpha074", "alpha075", "alpha077", "alpha078", "alpha081",
    "alpha083", "alpha084", "alpha086", "alpha094", "alpha096", "alpha098",
}

# Magic number aggregation factors (hardcoded coefficients for linear combo)
_ALPHA101_MAGIC = {"alpha036"}

# High-priority factors for Batch 1 validation
_ALPHA101_PRIORITY = [
    "alpha044", "alpha033", "alpha017", "alpha026", "alpha055", "alpha002",
    "alpha039", "alpha003", "alpha001", "alpha006", "alpha012", "alpha020",
    "alpha028", "alpha046", "alpha053", "alpha054", "alpha101", "alpha004",
    "alpha009", "alpha010",
]


def load_alpha101() -> dict:
    from nautilus_quants.factors.builtin.alpha101 import ALPHA101_FACTORS
    return ALPHA101_FACTORS


def filter_factors(
    factors: dict,
    ohlcv_only: bool = True,
    no_magic: bool = True,
    priority: int | None = None,
    vwap_set: set | None = None,
    magic_set: set | None = None,
    priority_list: list | None = None,
) -> dict:
    """Filter factors by criteria."""
    result = {}
    for name, info in sorted(factors.items()):
        if ohlcv_only and vwap_set and name in vwap_set:
            continue
        if no_magic and magic_set and name in magic_set:
            continue
        result[name] = info

    if priority is not None and priority_list:
        # Select top-N from priority list that exist in filtered set
        selected = {}
        for pname in priority_list:
            if pname in result and len(selected) < priority:
                selected[pname] = result[pname]
        return selected

    return result


def generate_factors_yaml(
    factors: dict,
    source: str,
    description: str = "",
    variables: dict | None = None,
) -> dict:
    """Generate a factors.yaml compatible dict."""
    config = {
        "metadata": {
            "name": source,
            "version": "1.0",
            "description": description,
        },
    }
    if variables:
        config["variables"] = variables
    config["factors"] = {
        name: {
            "expression": info["expression"],
            "description": info.get("description", ""),
            "category": info.get("category", ""),
        }
        for name, info in factors.items()
    }
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate factor YAML configs")
    parser.add_argument("source", choices=["alpha101"])
    parser.add_argument("--ohlcv-only", action="store_true", default=True)
    parser.add_argument("--no-ohlcv-filter", action="store_true")
    parser.add_argument("--no-magic", action="store_true", default=True)
    parser.add_argument("--priority", type=int, default=None, help="Select top-N priority factors")
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()

    ohlcv_only = not args.no_ohlcv_filter

    if args.source == "alpha101":
        raw = load_alpha101()
        filtered = filter_factors(
            raw,
            ohlcv_only=ohlcv_only,
            no_magic=args.no_magic,
            priority=args.priority,
            vwap_set=_ALPHA101_VWAP,
            magic_set=_ALPHA101_MAGIC,
            priority_list=_ALPHA101_PRIORITY,
        )
        # Alpha101 common variables
        variables = {
            "returns": "delta(close, 1) / delay(close, 1)",
        }
        config = generate_factors_yaml(
            filtered,
            source=f"alpha101{'_p1' if args.priority else ''}",
            description=f"Alpha101 OHLCV-only factors ({len(filtered)} factors)",
            variables=variables,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"Generated {len(filtered)} factors → {args.output}")


if __name__ == "__main__":
    main()
