# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for factor cache (save / load / hash / validation)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nautilus_quants.factors.cache import (
    compute_cache_key,
    compute_config_hash,
    has_cache,
    load_as_factor_series,
    load_as_snapshots,
    load_cache_metadata,
    save_factor_cache,
    save_snapshots_as_cache,
    validate_cache,
)
from nautilus_quants.factors.config import FactorConfig, FactorDefinition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_factor_config(
    expression: str = "-correlation(high, rank(volume), 5)",
) -> FactorConfig:
    return FactorConfig(
        name="test",
        version="1.0",
        variables={"returns": "delta(close, 1) / delay(close, 1)"},
        factors=[
            FactorDefinition(name="alpha044", expression=expression),
        ],
    )


def _make_factor_series() -> dict[str, pd.Series]:
    """Create a minimal factor_series in FactorEvaluator output format."""
    dates = pd.to_datetime(["2022-01-01", "2022-01-01", "2022-01-02", "2022-01-02"])
    assets = ["BTCUSDT.BINANCE", "ETHUSDT.BINANCE"] * 2
    index = pd.MultiIndex.from_arrays([dates, assets], names=["date", "asset"])
    return {
        "alpha044": pd.Series([0.5, -0.3, 0.2, 0.1], index=index, dtype=float),
    }


def _make_snapshots() -> list[tuple[int, dict[str, dict[str, float]]]]:
    """Create minimal actor snapshots."""
    ts1 = int(pd.Timestamp("2022-01-01").value)
    ts2 = int(pd.Timestamp("2022-01-02").value)
    return [
        (ts1, {"alpha044": {"BTCUSDT.BINANCE": 0.5, "ETHUSDT.BINANCE": -0.3}}),
        (ts2, {"alpha044": {"BTCUSDT.BINANCE": 0.2, "ETHUSDT.BINANCE": 0.1}}),
    ]


# ---------------------------------------------------------------------------
# Cache key tests
# ---------------------------------------------------------------------------


class TestComputeCacheKey:
    def test_deterministic(self) -> None:
        config = _make_factor_config()
        key1 = compute_cache_key(config, "4h", ["BTCUSDT.BINANCE"], "/data/catalog")
        key2 = compute_cache_key(config, "4h", ["BTCUSDT.BINANCE"], "/data/catalog")
        assert key1 == key2

    def test_changes_on_expression(self) -> None:
        config1 = _make_factor_config("-correlation(high, rank(volume), 5)")
        config2 = _make_factor_config("-correlation(high, rank(volume), 10)")
        key1 = compute_cache_key(config1, "4h", ["BTCUSDT.BINANCE"], "/data/catalog")
        key2 = compute_cache_key(config2, "4h", ["BTCUSDT.BINANCE"], "/data/catalog")
        assert key1 != key2

    def test_changes_on_bar_spec(self) -> None:
        config = _make_factor_config()
        key1 = compute_cache_key(config, "1h", ["BTCUSDT.BINANCE"], "/data/catalog")
        key2 = compute_cache_key(config, "4h", ["BTCUSDT.BINANCE"], "/data/catalog")
        assert key1 != key2

    def test_instrument_order_irrelevant(self) -> None:
        config = _make_factor_config()
        key1 = compute_cache_key(
            config, "4h", ["BTCUSDT.BINANCE", "ETHUSDT.BINANCE"], "/data/catalog",
        )
        key2 = compute_cache_key(
            config, "4h", ["ETHUSDT.BINANCE", "BTCUSDT.BINANCE"], "/data/catalog",
        )
        assert key1 == key2


# ---------------------------------------------------------------------------
# compute_config_hash tests
# ---------------------------------------------------------------------------


class TestComputeConfigHash:
    def test_deterministic(self) -> None:
        config = _make_factor_config()
        assert compute_config_hash(config) == compute_config_hash(config)

    def test_changes_on_expression(self) -> None:
        h1 = compute_config_hash(_make_factor_config("rank(volume)"))
        h2 = compute_config_hash(_make_factor_config("rank(close)"))
        assert h1 != h2

    def test_ignores_description(self) -> None:
        c1 = FactorConfig(
            factors=[FactorDefinition(name="f", expression="close", description="A")],
        )
        c2 = FactorConfig(
            factors=[FactorDefinition(name="f", expression="close", description="B")],
        )
        assert compute_config_hash(c1) == compute_config_hash(c2)

    def test_length(self) -> None:
        h = compute_config_hash(_make_factor_config())
        assert len(h) == 16  # truncated hex


# ---------------------------------------------------------------------------
# has_cache tests
# ---------------------------------------------------------------------------


class TestHasCache:
    def test_empty_dir(self, tmp_path: pd.Timestamp) -> None:
        assert has_cache(tmp_path) is False

    def test_nonexistent_dir(self, tmp_path: pd.Timestamp) -> None:
        assert has_cache(tmp_path / "nonexistent") is False

    def test_with_parquet(self, tmp_path: pd.Timestamp) -> None:
        save_factor_cache(_make_factor_series(), tmp_path)
        assert has_cache(tmp_path) is True


# ---------------------------------------------------------------------------
# Save/load roundtrip: factor_series format
# ---------------------------------------------------------------------------


class TestFactorSeriesRoundtrip:
    def test_roundtrip(self, tmp_path: pd.Timestamp) -> None:
        original = _make_factor_series()
        save_factor_cache(original, tmp_path, config_hash="abc123")

        loaded = load_as_factor_series(tmp_path)

        assert set(loaded.keys()) == set(original.keys())
        for name in original:
            pd.testing.assert_series_equal(
                loaded[name].sort_index(),
                original[name].sort_index(),
                check_names=False,
            )

    def test_metadata_written(self, tmp_path: pd.Timestamp) -> None:
        save_factor_cache(
            _make_factor_series(),
            tmp_path,
            factor_config_path="config/cs/factors.yaml",
            config_hash="abc123",
        )
        meta = load_cache_metadata(tmp_path)
        assert meta["config_hash"] == "abc123"
        assert meta["factor_names"] == ["alpha044"]
        assert meta["instrument_count"] == 2
        assert meta["timestamp_count"] == 2
        assert sorted(meta["instruments"]) == [
            "BTCUSDT.BINANCE", "ETHUSDT.BINANCE",
        ]
        assert "bar_spec" not in meta


# ---------------------------------------------------------------------------
# Save/load roundtrip: snapshots format
# ---------------------------------------------------------------------------


class TestSnapshotsRoundtrip:
    def test_roundtrip_from_factor_series(self, tmp_path: pd.Timestamp) -> None:
        """Save as factor_series, load as snapshots."""
        save_factor_cache(_make_factor_series(), tmp_path)
        snapshots = load_as_snapshots(tmp_path)

        # 2 timestamps
        assert len(snapshots) == 2
        # Each ts has alpha044
        for ts, factors in snapshots.items():
            assert "alpha044" in factors
            assert len(factors["alpha044"]) == 2  # 2 instruments

    def test_roundtrip_from_snapshots(self, tmp_path: pd.Timestamp) -> None:
        """Save as snapshots, load as snapshots."""
        original = _make_snapshots()
        save_snapshots_as_cache(original, tmp_path)
        loaded = load_as_snapshots(tmp_path)

        for ts, factors in original:
            assert ts in loaded
            for fname, fvals in factors.items():
                for inst, val in fvals.items():
                    assert loaded[ts][fname][inst] == pytest.approx(val)

    def test_snapshots_to_factor_series(self, tmp_path: pd.Timestamp) -> None:
        """Save as snapshots, load as factor_series."""
        save_snapshots_as_cache(_make_snapshots(), tmp_path)
        loaded = load_as_factor_series(tmp_path)

        assert "alpha044" in loaded
        assert len(loaded["alpha044"]) == 4  # 2 timestamps × 2 instruments

    def test_snapshots_metadata_has_instruments(self, tmp_path: pd.Timestamp) -> None:
        save_snapshots_as_cache(
            _make_snapshots(), tmp_path, config_hash="deadbeef",
        )
        meta = load_cache_metadata(tmp_path)
        assert meta["config_hash"] == "deadbeef"
        assert sorted(meta["instruments"]) == [
            "BTCUSDT.BINANCE", "ETHUSDT.BINANCE",
        ]


# ---------------------------------------------------------------------------
# validate_cache tests
# ---------------------------------------------------------------------------


class TestValidateCache:
    def test_valid_cache(self, tmp_path: pd.Timestamp) -> None:
        config = _make_factor_config()
        h = compute_config_hash(config)
        save_factor_cache(_make_factor_series(), tmp_path, config_hash=h)

        valid, warnings = validate_cache(tmp_path, h)
        assert valid is True
        assert warnings == []

    def test_config_hash_mismatch(self, tmp_path: pd.Timestamp) -> None:
        save_factor_cache(
            _make_factor_series(), tmp_path, config_hash="old_hash",
        )

        valid, warnings = validate_cache(tmp_path, "new_hash")
        assert valid is False
        assert len(warnings) == 1
        assert "mismatch" in warnings[0].lower()

    def test_missing_instruments_warning(self, tmp_path: pd.Timestamp) -> None:
        config = _make_factor_config()
        h = compute_config_hash(config)
        save_factor_cache(_make_factor_series(), tmp_path, config_hash=h)

        valid, warnings = validate_cache(
            tmp_path, h,
            expected_instruments={"BTCUSDT.BINANCE", "ETHUSDT.BINANCE", "SOLUSDT.BINANCE"},
        )
        assert valid is True  # still usable
        assert len(warnings) == 1
        assert "SOLUSDT.BINANCE" in warnings[0]

    def test_superset_instruments_no_warning(self, tmp_path: pd.Timestamp) -> None:
        config = _make_factor_config()
        h = compute_config_hash(config)
        save_factor_cache(_make_factor_series(), tmp_path, config_hash=h)

        valid, warnings = validate_cache(
            tmp_path, h,
            expected_instruments={"BTCUSDT.BINANCE"},  # subset of cached
        )
        assert valid is True
        assert warnings == []

    def test_empty_stored_hash_is_valid(self, tmp_path: pd.Timestamp) -> None:
        """Backward compat: old caches without config_hash pass validation."""
        save_factor_cache(_make_factor_series(), tmp_path, config_hash="")

        valid, warnings = validate_cache(tmp_path, "any_hash")
        assert valid is True


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_snapshots(self, tmp_path: pd.Timestamp) -> None:
        save_snapshots_as_cache([], tmp_path)
        assert has_cache(tmp_path) is False

    def test_nan_values_excluded(self, tmp_path: pd.Timestamp) -> None:
        """NaN values in factor_series should be excluded from snapshots."""
        dates = pd.to_datetime(["2022-01-01", "2022-01-01"])
        assets = ["BTCUSDT.BINANCE", "ETHUSDT.BINANCE"]
        index = pd.MultiIndex.from_arrays([dates, assets], names=["date", "asset"])
        series = {"alpha044": pd.Series([0.5, np.nan], index=index)}

        save_factor_cache(series, tmp_path)
        snapshots = load_as_snapshots(tmp_path)

        ts = list(snapshots.keys())[0]
        assert "BTCUSDT.BINANCE" in snapshots[ts]["alpha044"]
        assert "ETHUSDT.BINANCE" not in snapshots[ts]["alpha044"]
