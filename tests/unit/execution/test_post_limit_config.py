# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for PostLimitExecAlgorithmConfig."""

from __future__ import annotations

from nautilus_trader.model.identifiers import ExecAlgorithmId

from nautilus_quants.execution.post_limit.config import PostLimitExecAlgorithmConfig


class TestPostLimitExecAlgorithmConfig:
    def test_default_values(self) -> None:
        config = PostLimitExecAlgorithmConfig()

        assert config.exec_algorithm_id == ExecAlgorithmId("PostLimit")
        assert config.offset_ticks == 0
        assert config.timeout_secs == 15.0
        assert config.max_chase_attempts == 3
        assert config.chase_step_ticks == 1
        assert config.fallback_to_market is True
        assert config.post_only is True
        assert config.max_post_only_retries == 3
        assert config.enable_residual_sweep is True
        assert config.residual_sweep_min_notional_fallback == 5.0

    def test_custom_values(self) -> None:
        config = PostLimitExecAlgorithmConfig(
            exec_algorithm_id=ExecAlgorithmId("CustomPostLimit"),
            offset_ticks=-2,
            timeout_secs=30.0,
            max_chase_attempts=5,
            chase_step_ticks=2,
            fallback_to_market=False,
            post_only=False,
            max_post_only_retries=1,
            enable_residual_sweep=False,
            residual_sweep_min_notional_fallback=10.0,
        )

        assert config.exec_algorithm_id == ExecAlgorithmId("CustomPostLimit")
        assert config.offset_ticks == -2
        assert config.timeout_secs == 30.0
        assert config.max_chase_attempts == 5
        assert config.chase_step_ticks == 2
        assert config.fallback_to_market is False
        assert config.post_only is False
        assert config.max_post_only_retries == 1
        assert config.enable_residual_sweep is False
        assert config.residual_sweep_min_notional_fallback == 10.0

    def test_config_is_frozen(self) -> None:
        config = PostLimitExecAlgorithmConfig()

        try:
            config.offset_ticks = 1  # type: ignore[misc]
            assert False, "Mutation should fail for frozen config"
        except (AttributeError, TypeError):
            pass
