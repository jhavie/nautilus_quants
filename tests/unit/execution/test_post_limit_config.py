# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for PostLimitExecAlgorithmConfig."""

from __future__ import annotations

from nautilus_trader.model.identifiers import ExecAlgorithmId

from nautilus_quants.execution.post_limit.config import PostLimitExecAlgorithmConfig


class TestPostLimitExecAlgorithmConfig:
    """Test configuration defaults and overrides."""

    def test_default_values(self) -> None:
        config = PostLimitExecAlgorithmConfig()
        assert config.exec_algorithm_id == ExecAlgorithmId("PostLimit")
        assert config.offset_ticks == 0
        assert config.timeout_secs == 15.0
        assert config.max_chase_attempts == 3
        assert config.chase_step_ticks == 1
        assert config.fallback_to_market is True
        assert config.post_only is True

    def test_custom_values(self) -> None:
        config = PostLimitExecAlgorithmConfig(
            offset_ticks=2,
            timeout_secs=30.0,
            max_chase_attempts=5,
            chase_step_ticks=2,
            fallback_to_market=False,
            post_only=False,
        )
        assert config.offset_ticks == 2
        assert config.timeout_secs == 30.0
        assert config.max_chase_attempts == 5
        assert config.chase_step_ticks == 2
        assert config.fallback_to_market is False
        assert config.post_only is False

    def test_custom_algorithm_id(self) -> None:
        config = PostLimitExecAlgorithmConfig(
            exec_algorithm_id=ExecAlgorithmId("CustomPostLimit"),
        )
        assert config.exec_algorithm_id == ExecAlgorithmId("CustomPostLimit")

    def test_frozen_immutability(self) -> None:
        config = PostLimitExecAlgorithmConfig()
        # msgspec frozen=True prevents attribute modification
        # This should raise an error on mutation attempt
        try:
            config.offset_ticks = 5  # type: ignore[misc]
            assert False, "Should have raised"
        except (AttributeError, TypeError):
            pass

    def test_negative_offset_ticks(self) -> None:
        """Negative offset_ticks (retreat from BBO) is valid."""
        config = PostLimitExecAlgorithmConfig(offset_ticks=-1)
        assert config.offset_ticks == -1

    def test_zero_timeout(self) -> None:
        """Zero timeout means immediate chase/fallback."""
        config = PostLimitExecAlgorithmConfig(timeout_secs=0.0)
        assert config.timeout_secs == 0.0

    def test_zero_max_chase(self) -> None:
        """Zero max chase means no chasing, immediate market fallback."""
        config = PostLimitExecAlgorithmConfig(max_chase_attempts=0)
        assert config.max_chase_attempts == 0

    def test_default_max_post_only_retries(self) -> None:
        """Default max_post_only_retries should be 3."""
        config = PostLimitExecAlgorithmConfig()
        assert config.max_post_only_retries == 3

    def test_zero_max_post_only_retries(self) -> None:
        """Zero disables POST_ONLY retry (immediate market fallback)."""
        config = PostLimitExecAlgorithmConfig(max_post_only_retries=0)
        assert config.max_post_only_retries == 0

    def test_custom_max_post_only_retries(self) -> None:
        """Custom max_post_only_retries value."""
        config = PostLimitExecAlgorithmConfig(max_post_only_retries=5)
        assert config.max_post_only_retries == 5
