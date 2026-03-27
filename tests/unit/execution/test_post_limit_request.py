# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for typed PostLimit request parsing."""

from __future__ import annotations

import pytest

from nautilus_quants.execution.post_limit.request import (
    PostLimitRequest,
    PostLimitRequestError,
)


class TestPostLimitRequest:
    def test_parse_valid_request(self) -> None:
        request = PostLimitRequest.parse(
            {
                "anchor_px": "50000.5",
                "timeout_secs": "7.5",
                "max_chase_attempts": "2",
                "chase_step_ticks": "3",
                "post_only": "true",
                "target_quote_quantity": "1000",
                "contract_multiplier": "0.1",
            },
        )

        assert request.anchor_px == 50000.5
        assert request.timeout_secs == 7.5
        assert request.max_chase_attempts == 2
        assert request.chase_step_ticks == 3
        assert request.post_only is True
        assert request.target_quote_quantity == 1000.0
        assert request.contract_multiplier == 0.1

    def test_missing_anchor_px_fails(self) -> None:
        with pytest.raises(PostLimitRequestError, match="Missing required"):
            PostLimitRequest.parse({"timeout_secs": "3"})

    def test_invalid_boolean_fails(self) -> None:
        with pytest.raises(PostLimitRequestError, match="boolean-like"):
            PostLimitRequest.parse({"anchor_px": "100", "post_only": "maybe"})

    def test_non_positive_target_quote_quantity_fails(self) -> None:
        with pytest.raises(PostLimitRequestError, match="must be > 0.0"):
            PostLimitRequest.parse({"anchor_px": "100", "target_quote_quantity": "0"})

    def test_empty_params_fail_fast(self) -> None:
        with pytest.raises(PostLimitRequestError, match="required for PostLimit"):
            PostLimitRequest.parse(None)
