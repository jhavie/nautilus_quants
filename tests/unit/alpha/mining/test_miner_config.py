# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for MiningConfig and DirectionConfig parsing."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from nautilus_quants.alpha.mining.agent.miner import (
    DirectionConfig,
    MiningConfig,
)
from nautilus_quants.factors.expression.complexity import ComplexityConstraints


@pytest.fixture()
def tmp_yaml(tmp_path: Path):
    """Helper to write a temp YAML and return its path."""

    def _write(content: str) -> Path:
        p = tmp_path / "test_mining.yaml"
        p.write_text(textwrap.dedent(content), encoding="utf-8")
        return p

    return _write


# ── constraints parsing ──────────────────────────────────────────


class TestConstraintsParsing:

    def test_no_constraints_uses_defaults(self, tmp_yaml):
        p = tmp_yaml("""\
            mining:
              source: "test"
        """)
        cfg = MiningConfig.from_yaml(p)
        assert cfg.constraints == ComplexityConstraints()

    def test_partial_constraints(self, tmp_yaml):
        p = tmp_yaml("""\
            mining:
              constraints:
                max_depth: 4
                max_window: 500
        """)
        cfg = MiningConfig.from_yaml(p)
        assert cfg.constraints.max_depth == 4
        assert cfg.constraints.max_window == 500
        assert cfg.constraints.max_char_length == 200  # default

    def test_full_constraints(self, tmp_yaml):
        p = tmp_yaml("""\
            mining:
              constraints:
                max_char_length: 150
                max_node_count: 20
                max_depth: 5
                max_func_nesting: 3
                max_variables: 4
                max_window: 360
                max_numeric_ratio: 0.2
        """)
        cfg = MiningConfig.from_yaml(p)
        assert cfg.constraints.max_char_length == 150
        assert cfg.constraints.max_node_count == 20
        assert cfg.constraints.max_depth == 5
        assert cfg.constraints.max_func_nesting == 3
        assert cfg.constraints.max_variables == 4
        assert cfg.constraints.max_window == 360
        assert cfg.constraints.max_numeric_ratio == 0.2


# ── directions parsing ───────────────────────────────────────────


class TestDirectionsParsing:

    def test_no_directions_empty_tuple(self, tmp_yaml):
        p = tmp_yaml("""\
            mining:
              source: "test"
        """)
        cfg = MiningConfig.from_yaml(p)
        assert cfg.directions == ()

    def test_single_direction(self, tmp_yaml):
        p = tmp_yaml("""\
            mining:
              directions:
                price_volume:
                  theme: "量价因子"
                  operators: [ts_mean, delta, cs_rank]
                  variables: [close, volume]
                  tags: [price, volume]
                  rounds: 3
        """)
        cfg = MiningConfig.from_yaml(p)
        assert len(cfg.directions) == 1
        d = cfg.directions[0]
        assert d.name == "price_volume"
        assert d.theme == "量价因子"
        assert d.operators == ("ts_mean", "delta", "cs_rank")
        assert d.variables == ("close", "volume")
        assert d.tags == ("price", "volume")
        assert d.rounds == 3

    def test_multiple_directions(self, tmp_yaml):
        p = tmp_yaml("""\
            mining:
              directions:
                momentum:
                  theme: "动量因子"
                volatility:
                  theme: "波动率因子"
                  operators: [ts_std]
                onchain:
                  theme: "链上因子"
                  variables: [close, open_interest]
        """)
        cfg = MiningConfig.from_yaml(p)
        assert len(cfg.directions) == 3
        names = [d.name for d in cfg.directions]
        assert "momentum" in names
        assert "volatility" in names
        assert "onchain" in names

    def test_direction_missing_theme_raises(self, tmp_yaml):
        p = tmp_yaml("""\
            mining:
              directions:
                bad_direction:
                  operators: [ts_mean]
        """)
        with pytest.raises(ValueError, match="theme"):
            MiningConfig.from_yaml(p)

    def test_direction_optional_fields_default(self, tmp_yaml):
        p = tmp_yaml("""\
            mining:
              directions:
                minimal:
                  theme: "测试"
        """)
        cfg = MiningConfig.from_yaml(p)
        d = cfg.directions[0]
        assert d.operators == ()
        assert d.variables == ()
        assert d.tags == ()
        assert d.rounds == 0

    def test_directions_are_tuples(self, tmp_yaml):
        p = tmp_yaml("""\
            mining:
              directions:
                test:
                  theme: "test"
                  operators: [a, b]
                  variables: [c, d]
                  tags: [e]
        """)
        cfg = MiningConfig.from_yaml(p)
        d = cfg.directions[0]
        assert isinstance(d.operators, tuple)
        assert isinstance(d.variables, tuple)
        assert isinstance(d.tags, tuple)
        assert isinstance(cfg.directions, tuple)


# ── existing fields backward compat ──────────────────────────────


class TestBackwardCompat:

    def test_existing_fields_unchanged(self, tmp_yaml):
        p = tmp_yaml("""\
            bar_spec: "1h"
            mining:
              source: "llm_test"
              factors_per_round: 10
              model: "opus"
              proxy: "http://localhost:9999"
              parallel: 5
              output_dir: "logs/test"
        """)
        cfg = MiningConfig.from_yaml(p)
        assert cfg.bar_spec == "1h"
        assert cfg.source == "llm_test"
        assert cfg.factors_per_round == 10
        assert cfg.model == "opus"
        assert cfg.proxy == "http://localhost:9999"
        assert cfg.parallel == 5

    def test_cli_overrides_still_work(self, tmp_yaml):
        p = tmp_yaml("""\
            mining:
              factors_per_round: 8
              model: "sonnet"
        """)
        cfg = MiningConfig.from_yaml(
            p, factors_per_round=12, model="opus", hypothesis="test"
        )
        assert cfg.factors_per_round == 12
        assert cfg.model == "opus"
        assert cfg.hypothesis == "test"

    def test_proxy_omitted_defaults_to_empty(self, tmp_yaml):
        """Omitting mining.proxy yields empty string; _call_claude will skip env injection."""
        p = tmp_yaml("""\
            mining:
              source: "llm_test"
        """)
        cfg = MiningConfig.from_yaml(p)
        assert cfg.proxy == ""

    def test_proxy_explicit_empty_string_means_no_proxy(self, tmp_yaml):
        """Explicitly setting mining.proxy to "" also yields empty string."""
        p = tmp_yaml("""\
            mining:
              proxy: ""
        """)
        cfg = MiningConfig.from_yaml(p)
        assert cfg.proxy == ""


# ── DirectionConfig frozen ───────────────────────────────────────


class TestDirectionConfig:

    def test_frozen(self):
        d = DirectionConfig(name="test", theme="test")
        with pytest.raises(AttributeError):
            d.theme = "changed"  # type: ignore[misc]
