#!/usr/bin/env python3
"""
Entry script for the Binance data pipeline CLI.

Usage:
    python scripts/data/pipeline.py [OPTIONS] COMMAND [ARGS]...

Commands:
    run         Execute full pipeline
    download    Download raw data from Binance
    validate    Validate data integrity
    process     Process/clean data
    transform   Transform to Parquet format
    status      Show pipeline status
    clean       Clean generated files

See --help for more information.
"""

from nautilus_quants.data.cli import main

if __name__ == "__main__":
    main()
