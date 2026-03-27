# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Typed request parsing for PostLimit execution parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


class PostLimitRequestError(ValueError):
    """Raised when exec_algorithm_params are missing or invalid."""


@dataclass(frozen=True)
class PostLimitRequest:
    """Typed request parsed from a primary order's exec_algorithm_params."""

    anchor_px: float
    timeout_secs: float | None = None
    max_chase_attempts: int | None = None
    chase_step_ticks: int | None = None
    post_only: bool | None = None
    target_quote_quantity: float | None = None
    contract_multiplier: float = 1.0

    @classmethod
    def parse(cls, params: Mapping[str, object] | None) -> PostLimitRequest:
        if not params:
            raise PostLimitRequestError("`exec_algorithm_params` are required for PostLimit")

        anchor_px = _parse_required_positive_float(params, "anchor_px")

        timeout_secs = _parse_optional_float(params, "timeout_secs", minimum=0.0)
        max_chase_attempts = _parse_optional_int(params, "max_chase_attempts", minimum=0)
        chase_step_ticks = _parse_optional_int(params, "chase_step_ticks", minimum=0)
        post_only = _parse_optional_bool(params, "post_only")
        target_quote_quantity = _parse_optional_float(
            params,
            "target_quote_quantity",
            minimum=0.0,
            strictly_positive=True,
        )
        contract_multiplier = _parse_optional_float(
            params,
            "contract_multiplier",
            minimum=0.0,
            strictly_positive=True,
        )

        return cls(
            anchor_px=anchor_px,
            timeout_secs=timeout_secs,
            max_chase_attempts=max_chase_attempts,
            chase_step_ticks=chase_step_ticks,
            post_only=post_only,
            target_quote_quantity=target_quote_quantity,
            contract_multiplier=contract_multiplier or 1.0,
        )


def _parse_required_positive_float(params: Mapping[str, object], key: str) -> float:
    value = params.get(key)
    if value is None:
        raise PostLimitRequestError(f"Missing required exec_algorithm_params[{key!r}]")

    parsed = _coerce_float(value, key)
    if parsed <= 0:
        raise PostLimitRequestError(f"exec_algorithm_params[{key!r}] must be > 0")
    return parsed


def _parse_optional_float(
    params: Mapping[str, object],
    key: str,
    *,
    minimum: float,
    strictly_positive: bool = False,
) -> float | None:
    value = params.get(key)
    if value is None:
        return None

    parsed = _coerce_float(value, key)
    if strictly_positive and parsed <= minimum:
        raise PostLimitRequestError(f"exec_algorithm_params[{key!r}] must be > {minimum}")
    if not strictly_positive and parsed < minimum:
        raise PostLimitRequestError(f"exec_algorithm_params[{key!r}] must be >= {minimum}")
    return parsed


def _parse_optional_int(params: Mapping[str, object], key: str, *, minimum: int) -> int | None:
    value = params.get(key)
    if value is None:
        return None

    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise PostLimitRequestError(
            f"exec_algorithm_params[{key!r}] must be an integer",
        ) from exc

    if parsed < minimum:
        raise PostLimitRequestError(f"exec_algorithm_params[{key!r}] must be >= {minimum}")
    return parsed


def _parse_optional_bool(params: Mapping[str, object], key: str) -> bool | None:
    value = params.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False

    raise PostLimitRequestError(
        f"exec_algorithm_params[{key!r}] must be a boolean-like value",
    )


def _coerce_float(value: object, key: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise PostLimitRequestError(
            f"exec_algorithm_params[{key!r}] must be a float-like value",
        ) from exc
