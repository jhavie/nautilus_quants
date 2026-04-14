# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Search space construction for factor tuning.

Derives ``ParamSpec`` / ``OperatorSlot`` / ``VariableSlot`` lists from an
expression string, and rebuilds expressions once the optimiser samples a
configuration. Every AST walk piggy-backs on the existing parser /
normaliser in ``nautilus_quants.factors.expression`` so parsing is never
re-implemented.

Public API
----------
- ``expand_config_params`` — resolve ``parameters:`` references to numeric
  literals before template extraction.
- ``classify_parameters`` — tag each placeholder from
  ``expression_template()`` (window, threshold, …).
- ``detect_operator_slots`` — build ``OperatorSlot`` instances for every
  substitutable operator position in the expression.
- ``detect_variable_slots`` — build ``VariableSlot`` instances filtered by
  availability and scope compatibility.
- ``reconstruct_expression`` — apply variable / operator / numeric choices
  back into a canonical expression string via AST rewriting.
- ``OPERATOR_GROUPS`` / ``VARIABLE_GROUPS`` — static catalogues of
  interchangeable operators and variables.
- ``build_search_space`` — one-stop helper that returns the template, numeric
  specs, operator slots, and variable slots for an expression.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from nautilus_quants.alpha.tuning.config import (
    PARAM_TYPE_COEFFICIENT,
    PARAM_TYPE_SIGN,
    PARAM_TYPE_THRESHOLD,
    PARAM_TYPE_WINDOW,
    VAR_SCOPE_BROADCAST,
    VAR_SCOPE_PER_INSTRUMENT,
    OperatorAlternative,
    OperatorSlot,
    ParamSpec,
    VariableGroup,
    VariableSlot,
)
from nautilus_quants.factors.expression.ast import (
    ASTNode,
    BinaryOpNode,
    FunctionCallNode,
    NumberNode,
    StringNode,
    TernaryNode,
    UnaryOpNode,
    VariableNode,
)
from nautilus_quants.factors.expression.normalize import expression_template, normalize_expression
from nautilus_quants.factors.expression.parser import parse_expression

# ── Operator catalogue ──────────────────────────────────────────────────────

# Time-series operators whose *last* positional argument is a rolling window.
_TS_WINDOW_OPERATORS: frozenset[str] = frozenset(
    {
        "ts_mean",
        "ts_sum",
        "ts_std",
        "ts_min",
        "ts_max",
        "ts_rank",
        "ts_argmax",
        "ts_argmin",
        "delta",
        "delay",
        "correlation",
        "covariance",
        "decay_linear",
        "ts_product",
        "ts_skew",
        "ts_slope",
        "ts_rsquare",
        "ts_residual",
        "ema",
        "wq_ts_rank",
        "wq_ts_argmax",
        "wq_ts_argmin",
        "stddev",
        "sma",
        "ts_delta",
        "ts_delay",
        "ts_corr",
        "ts_covariance",
        "ts_std_dev",
        "ts_decay_linear",
    }
)

# ``identity`` is a synthetic name meaning "drop the wrapper". Any alternative
# whose ``args_template`` equals ``{inner}`` is treated this way during
# reconstruction.
_IDENTITY_NAME = "identity"

# Operator substitution groups for single-argument cross-sectional wrappers.
# These are the groups the MVP actually substitutes. The rendering logic below
# supports multi-arg variants, but we stay conservative by default — users
# can extend ``OPERATOR_GROUPS`` or pass overrides via
# ``TuneConfig.search_space_overrides`` when they need more.
OPERATOR_GROUPS: dict[str, tuple[OperatorAlternative, ...]] = {
    # Outer normalisation: rich alternative set incl. winsorize/clip_quantile.
    "cs_normalize": (
        OperatorAlternative("cs_rank"),
        OperatorAlternative("cs_zscore"),
        OperatorAlternative(
            "normalize",
            "({inner}, {use_std}, {limit})",
            extra_params=(
                ParamSpec(
                    "use_std",
                    PARAM_TYPE_COEFFICIENT,
                    original_value=1.0,
                    values=(0.0, 1.0),
                ),
                ParamSpec(
                    "limit",
                    PARAM_TYPE_THRESHOLD,
                    original_value=0.0,
                    values=(0.0, 0.1),
                ),
            ),
        ),
        OperatorAlternative(
            "winsorize",
            "({inner}, {std_mult})",
            extra_params=(
                ParamSpec(
                    "std_mult",
                    PARAM_TYPE_COEFFICIENT,
                    original_value=4.0,
                    values=(2.0, 3.0, 4.0, 5.0),
                ),
            ),
        ),
        OperatorAlternative(
            "clip_quantile",
            "({inner}, {lower}, {upper})",
            extra_params=(
                ParamSpec(
                    "lower",
                    PARAM_TYPE_THRESHOLD,
                    original_value=0.05,
                    values=(0.02, 0.05, 0.1),
                ),
                ParamSpec(
                    "upper",
                    PARAM_TYPE_THRESHOLD,
                    original_value=0.95,
                    values=(0.9, 0.95, 0.98),
                ),
            ),
        ),
        OperatorAlternative("cs_demean"),
        OperatorAlternative("cs_scale"),
        OperatorAlternative(_IDENTITY_NAME, "{inner}"),
    ),
    # Intermediate ranking (e.g. rank(volume) inside correlation).
    "cs_ranking": (
        OperatorAlternative("cs_rank"),
        OperatorAlternative("cs_zscore"),
        OperatorAlternative(_IDENTITY_NAME, "{inner}"),
    ),
}

# Members of each group, including *aliases* we should treat as substitutable
# (e.g. a call to ``rank(x)`` is really ``cs_rank(x)``).
_GROUP_ALIASES: dict[str, frozenset[str]] = {
    "cs_normalize": frozenset(
        {
            "cs_rank",
            "cs_zscore",
            "normalize",
            "winsorize",
            "clip_quantile",
            "cs_demean",
            "cs_scale",
            "rank",
            "zscore",
            "scale",
            "demean",
        }
    ),
    "cs_ranking": frozenset(
        {
            "cs_rank",
            "cs_zscore",
            "rank",
            "zscore",
        }
    ),
}


def _canonicalise_op_name(name: str) -> str:
    """Map an alias to the canonical operator name used inside OPERATOR_GROUPS."""
    aliases = {
        "rank": "cs_rank",
        "zscore": "cs_zscore",
        "scale": "cs_scale",
        "demean": "cs_demean",
    }
    return aliases.get(name, name)


# ── Variable catalogue ──────────────────────────────────────────────────────


VARIABLE_GROUPS: dict[str, VariableGroup] = {
    "price_ohlc": VariableGroup(
        members=("open", "high", "low", "close"),
        scope=VAR_SCOPE_PER_INSTRUMENT,
        description="K-line open / high / low / close",
    ),
    "volume_like": VariableGroup(
        members=("volume", "quote_volume"),
        scope=VAR_SCOPE_PER_INSTRUMENT,
        description="Trade volume (base vs USDT-quote)",
    ),
    "benchmark_price": VariableGroup(
        members=("btc_close", "eth_close"),
        scope=VAR_SCOPE_BROADCAST,
        description="Market benchmark prices",
    ),
    "benchmark_returns": VariableGroup(
        members=("btc_returns", "eth_returns"),
        scope=VAR_SCOPE_BROADCAST,
        description="Market benchmark returns",
    ),
    "benchmark_vol": VariableGroup(
        members=("btc_vol", "eth_vol"),
        scope=VAR_SCOPE_BROADCAST,
        description="Market benchmark volatility",
    ),
    "beta": VariableGroup(
        members=("btc_beta", "eth_beta"),
        scope=VAR_SCOPE_PER_INSTRUMENT,
        description="Instrument sensitivity to benchmark",
    ),
    "returns": VariableGroup(
        members=("returns", "ret1", "ret5", "ret20"),
        scope=VAR_SCOPE_PER_INSTRUMENT,
        description="Different-window price returns",
    ),
    "funding_rate": VariableGroup(
        members=("funding_rate", "san_funding_rate"),
        scope=VAR_SCOPE_PER_INSTRUMENT,
        description="Funding rate (catalog vs SanAPI source)",
    ),
    "open_interest": VariableGroup(
        members=("open_interest", "san_open_interest"),
        scope=VAR_SCOPE_PER_INSTRUMENT,
        description="Open interest (two data sources)",
    ),
    "social_volume": VariableGroup(
        members=("san_volume_usd", "san_social_volume"),
        scope=VAR_SCOPE_PER_INSTRUMENT,
        description="Attention metrics (USD volume vs social mentions)",
    ),
}

_VARIABLE_TO_GROUP: dict[str, str] = {}
for _gname, _group in VARIABLE_GROUPS.items():
    for _member in _group.members:
        _VARIABLE_TO_GROUP.setdefault(_member, _gname)


# ── Parameter context extraction ───────────────────────────────────────────


@dataclass(frozen=True)
class _ParamContext:
    """Where a numeric placeholder sits in the AST."""

    index: int
    value: float
    parent_func: str | None
    arg_pos: int | None
    is_last_arg: bool
    top_level_sign: bool


def _extract_param_contexts(
    node: ASTNode,
    *,
    contexts: list[_ParamContext] | None = None,
    counter: list[int] | None = None,
    parent_func: str | None = None,
    arg_pos: int | None = None,
    total_args: int = 0,
    top_level_sign_slot: bool = False,
) -> list[_ParamContext]:
    """DFS the normalised AST, recording metadata for every ``NumberNode``."""
    if contexts is None:
        contexts = []
    if counter is None:
        counter = [0]

    if isinstance(node, NumberNode):
        is_last = parent_func is not None and arg_pos == total_args - 1
        contexts.append(
            _ParamContext(
                index=counter[0],
                value=node.value,
                parent_func=parent_func,
                arg_pos=arg_pos,
                is_last_arg=is_last,
                top_level_sign=top_level_sign_slot,
            )
        )
        counter[0] += 1
        return contexts

    if isinstance(node, (StringNode, VariableNode)):
        return contexts

    if isinstance(node, UnaryOpNode):
        _extract_param_contexts(node.operand, contexts=contexts, counter=counter)
        return contexts

    if isinstance(node, BinaryOpNode):
        left_is_sign = node.operator == "*" and isinstance(node.left, NumberNode) and abs(abs(node.left.value) - 1.0) < 1e-12
        _extract_param_contexts(
            node.left,
            contexts=contexts,
            counter=counter,
            top_level_sign_slot=left_is_sign,
        )
        _extract_param_contexts(node.right, contexts=contexts, counter=counter)
        return contexts

    if isinstance(node, TernaryNode):
        _extract_param_contexts(node.condition, contexts=contexts, counter=counter)
        _extract_param_contexts(node.true_expr, contexts=contexts, counter=counter)
        _extract_param_contexts(node.false_expr, contexts=contexts, counter=counter)
        return contexts

    if isinstance(node, FunctionCallNode):
        n = len(node.arguments)
        for i, arg in enumerate(node.arguments):
            _extract_param_contexts(
                arg,
                contexts=contexts,
                counter=counter,
                parent_func=node.name,
                arg_pos=i,
                total_args=n,
            )
        return contexts

    return contexts  # pragma: no cover — unknown node


# ── Standard search spaces ──────────────────────────────────────────────────


_STANDARD_WINDOWS: tuple[float, ...] = (
    2.0,
    3.0,
    5.0,
    8.0,
    10.0,
    15.0,
    20.0,
    30.0,
    40.0,
    60.0,
    80.0,
    120.0,
    180.0,
    240.0,
)


def _window_search_space(original: float) -> tuple[float, ...]:
    """Geometric window ladder clipped to ±3 octaves around the original."""
    if original <= 0:
        return _STANDARD_WINDOWS
    lo = original / 8.0
    hi = original * 8.0
    candidates = tuple(w for w in _STANDARD_WINDOWS if lo <= w <= hi)
    rounded = float(round(original))
    if rounded not in candidates:
        candidates = tuple(sorted({*candidates, rounded}))
    return candidates


# Standard threshold candidates for fraction-style parameters (clip_quantile
# bounds, normalize limit, etc.). The 1-decimal grid 0.1..0.9 covers every
# economically meaningful cutoff for a quantile clip; we add 0.05 / 0.95 as
# the de-facto "tight clip" outliers because they appear all the time in
# real factor configs and rounding 0.05 → 0.1 loses a useful distinction.
_STANDARD_THRESHOLDS: tuple[float, ...] = (
    0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95,
)


def _threshold_search_space(original: float) -> tuple[float, ...]:
    """Categorical 0.1-step ladder for fraction parameters.

    The returned tuple is centred on the rounded original (snap to nearest
    grid point), then expanded ±3 grid steps. Out-of-grid originals (e.g.
    0.476) are first snapped to the nearest grid value (→ 0.5).
    """
    # Snap original to the closest grid point so an LLM-mined oddity like
    # 0.476 doesn't pollute the search with non-canonical magnitudes.
    snapped = min(_STANDARD_THRESHOLDS, key=lambda x: abs(x - original))
    try:
        idx = _STANDARD_THRESHOLDS.index(snapped)
    except ValueError:
        return _STANDARD_THRESHOLDS

    # Window of ±3 grid steps around the snapped original.
    lo = max(0, idx - 3)
    hi = min(len(_STANDARD_THRESHOLDS), idx + 4)
    return _STANDARD_THRESHOLDS[lo:hi]


# Standard coefficient magnitudes for non-window, non-threshold numeric
# params (multipliers like ``5 * x``, exponents like ``power(x, 2)``).
# Limited to economically meaningful values — Optuna sampling 4.567 in a
# log-scale range is the kind of magic-number p-hacking we explicitly
# want to avoid.
_STANDARD_COEFFICIENT_MAGNITUDES: tuple[float, ...] = (
    0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0,
)


def _coefficient_search_space(original: float) -> tuple[float, ...]:
    """Categorical ladder for coefficient parameters with sign preservation.

    Behaviour:
    - ``original = 0`` → return ``(0.0,)`` (structural zero, not tunable).
    - ``original > 0`` → snap |value| to ``_STANDARD_COEFFICIENT_MAGNITUDES``,
      return ±3 grid steps **all positive** (preserving the sign).
    - ``original < 0`` → same window, all negative.

    Sign preservation matters: an LLM-mined ``-2 * btc_beta`` carries
    structural meaning (long short-beta names); flipping it to ``+2`` would
    invert the entire factor direction — that is a *different factor*, not
    a parameter tweak.
    """
    if original == 0:
        return (0.0,)

    sign = -1.0 if original < 0 else 1.0
    magnitude = abs(original)

    snapped = min(
        _STANDARD_COEFFICIENT_MAGNITUDES, key=lambda x: abs(x - magnitude)
    )
    try:
        idx = _STANDARD_COEFFICIENT_MAGNITUDES.index(snapped)
    except ValueError:
        return tuple(sign * c for c in _STANDARD_COEFFICIENT_MAGNITUDES)

    lo = max(0, idx - 3)
    hi = min(len(_STANDARD_COEFFICIENT_MAGNITUDES), idx + 4)
    window = _STANDARD_COEFFICIENT_MAGNITUDES[lo:hi]
    return tuple(sign * c for c in window)


# ── AST helpers ────────────────────────────────────────────────────────────


def _serialize_ast(node: ASTNode) -> str:
    """Serialize an AST back to an expression string (non-canonical)."""
    if isinstance(node, NumberNode):
        v = node.value
        if v == int(v) and abs(v) < 1e15:
            return str(int(v))
        return f"{v:g}"
    if isinstance(node, StringNode):
        return f'"{node.value}"'
    if isinstance(node, VariableNode):
        return node.name
    if isinstance(node, UnaryOpNode):
        return f"{node.operator}({_serialize_ast(node.operand)})"
    if isinstance(node, BinaryOpNode):
        return f"({_serialize_ast(node.left)} {node.operator} " f"{_serialize_ast(node.right)})"
    if isinstance(node, TernaryNode):
        return (
            f"({_serialize_ast(node.condition)} ? " f"{_serialize_ast(node.true_expr)} : " f"{_serialize_ast(node.false_expr)})"
        )
    if isinstance(node, FunctionCallNode):
        args = ", ".join(_serialize_ast(a) for a in node.arguments)
        return f"{node.name}({args})"
    raise TypeError(f"Unknown AST node: {type(node).__name__}")  # pragma: no cover


def _substitute_variables_by_name(
    node: ASTNode,
    values: dict[str, float],
) -> ASTNode:
    if isinstance(node, VariableNode):
        if node.name in values:
            return NumberNode(values[node.name])
        return node
    if isinstance(node, (NumberNode, StringNode)):
        return node
    if isinstance(node, UnaryOpNode):
        return UnaryOpNode(
            node.operator,
            _substitute_variables_by_name(node.operand, values),
        )
    if isinstance(node, BinaryOpNode):
        return BinaryOpNode(
            node.operator,
            _substitute_variables_by_name(node.left, values),
            _substitute_variables_by_name(node.right, values),
        )
    if isinstance(node, TernaryNode):
        return TernaryNode(
            _substitute_variables_by_name(node.condition, values),
            _substitute_variables_by_name(node.true_expr, values),
            _substitute_variables_by_name(node.false_expr, values),
        )
    if isinstance(node, FunctionCallNode):
        return FunctionCallNode(
            node.name,
            tuple(_substitute_variables_by_name(arg, values) for arg in node.arguments),
        )
    return node  # pragma: no cover


# ── Public API ──────────────────────────────────────────────────────────────


def expand_config_params(
    expression: str,
    parameters: dict[str, Any],
) -> str:
    """Substitute ``parameters:`` names with numeric literals in-place.

    Only names bound to numeric scalars are inlined. Non-numeric parameters
    (strings, lists) remain as ``VariableNode`` references and will be
    resolved at runtime by the evaluator.
    """
    if not parameters:
        return normalize_expression(expression)

    numeric_params: dict[str, float] = {}
    for name, value in parameters.items():
        if isinstance(value, bool):
            numeric_params[name] = float(value)
        elif isinstance(value, (int, float)):
            numeric_params[name] = float(value)

    if not numeric_params:
        return normalize_expression(expression)

    ast = parse_expression(expression)
    substituted = _substitute_variables_by_name(ast, numeric_params)
    return normalize_expression(_serialize_ast(substituted))


def classify_parameters(
    template: str,
    values: list[float] | tuple[float, ...],
    expression: str | None = None,
) -> tuple[ParamSpec, ...]:
    """Classify every ``p{i}`` placeholder in ``template``.

    Classification order:

    1. A value of ±1.0 at the top-level sign position → ``sign`` (not tuned).
    2. Last positional argument of a TS rolling operator → ``window``.
    3. Value in (0, 1) and not a window → ``threshold``.
    4. Otherwise → ``coefficient`` (log-scaled range).
    """
    source_expr = expression or template
    ast = parse_expression(source_expr)
    from nautilus_quants.factors.expression.normalize import _normalize

    normalised = _normalize(ast)
    contexts = _extract_param_contexts(normalised)

    if len(contexts) != len(values):
        raise ValueError("Parameter context count mismatch: expected " f"{len(values)} placeholders, got {len(contexts)}")

    specs: list[ParamSpec] = []
    for ctx, value in zip(contexts, values):
        name = f"p{ctx.index}"
        if ctx.top_level_sign and abs(abs(value) - 1.0) < 1e-12:
            specs.append(
                ParamSpec(
                    name=name,
                    param_type=PARAM_TYPE_SIGN,
                    original_value=value,
                )
            )
            continue

        if ctx.parent_func in _TS_WINDOW_OPERATORS and ctx.is_last_arg:
            specs.append(
                ParamSpec(
                    name=name,
                    param_type=PARAM_TYPE_WINDOW,
                    original_value=value,
                    values=_window_search_space(value),
                )
            )
            continue

        if 0.0 < value < 1.0:
            # Threshold params (clip_quantile bounds, normalize limit, ...)
            # are *categorical* on a 1-decimal grid. Two reasons:
            # 1) Avoid magic-number overfitting — Optuna would otherwise
            #    chase precise values like 0.476 / 0.95805 that look
            #    p-hacked and don't generalise.
            # 2) Constrain the search budget — clip thresholds have a
            #    natural granularity (5%, 10%, 20% are the only meaningful
            #    distinctions for a quantile cutoff).
            specs.append(
                ParamSpec(
                    name=name,
                    param_type=PARAM_TYPE_THRESHOLD,
                    original_value=value,
                    values=_threshold_search_space(value),
                )
            )
            continue

        # Coefficients (non-window non-threshold non-sign numerics) get a
        # categorical ladder centred on the rounded original magnitude.
        # Sign is preserved — flipping +2 → -2 changes the factor's
        # direction, which is a structural choice, not a parameter tweak.
        specs.append(
            ParamSpec(
                name=name,
                param_type=PARAM_TYPE_COEFFICIENT,
                original_value=value,
                values=_coefficient_search_space(value),
            )
        )

    return tuple(specs)


# ── Operator slot detection ─────────────────────────────────────────────────


def _is_at_top_level(ast: ASTNode, target: FunctionCallNode) -> bool:
    """Return True iff ``target`` is the root, modulo a leading ``-1 * …``.

    The normaliser turns ``-expr`` into ``-1 * expr``; we treat the root
    of the right-hand side of that multiply as "top level" too.
    """
    if ast is target:
        return True
    if (
        isinstance(ast, BinaryOpNode)
        and ast.operator == "*"
        and isinstance(ast.left, NumberNode)
        and abs(abs(ast.left.value) - 1.0) < 1e-12
        and ast.right is target
    ):
        return True
    return False


def detect_operator_slots(expression: str) -> tuple[OperatorSlot, ...]:
    """Find every substitutable operator position in ``expression``.

    Detection rules:

    - The *outermost* substitutable function call (root or the RHS of
      ``-1 * …``) is labelled ``cs_normalize`` and gets the full
      alternative set (winsorize, clip_quantile, identity, …).
    - Every other call whose canonical name is in ``_GROUP_ALIASES[
      "cs_ranking"]`` becomes a ``cs_ranking`` slot with three alternatives.
    - Only single-argument cross-sectional wrappers are recognised. Multi-
      argument operator substitutions (e.g. correlation ↔ covariance) stay
      opt-in via ``search_space_overrides`` in the tune config.

    The returned slots preserve the AST node identity via a *DFS index*
    embedded in ``slot_id`` (``op_0``, ``op_1``, …), matching the order in
    which ``reconstruct_expression`` rewrites them.
    """
    # Parse in normalised form so unary minus is already ``-1 * ...``.
    from nautilus_quants.factors.expression.normalize import _normalize

    raw_ast = parse_expression(expression)
    ast = _normalize(raw_ast)

    slots: list[OperatorSlot] = []
    counter = [0]

    def walk(node: ASTNode, depth: int = 0) -> None:
        if isinstance(node, FunctionCallNode):
            canonical = _canonicalise_op_name(node.name)

            # Only single-arg cross-sectional wrappers are considered.
            is_single_arg_cs = len(node.arguments) == 1 and canonical in _GROUP_ALIASES["cs_normalize"]

            if is_single_arg_cs:
                top_level = _is_at_top_level(ast, node)
                group = "cs_normalize" if top_level else "cs_ranking"
                slots.append(
                    OperatorSlot(
                        slot_id=f"op_{counter[0]}",
                        current_op=canonical,
                        group=group,
                        alternatives=OPERATOR_GROUPS[group],
                        inner_expr=_serialize_ast(node.arguments[0]),
                        position="outer" if top_level else "inner",
                    )
                )
                counter[0] += 1

            for arg in node.arguments:
                walk(arg, depth + 1)
            return

        if isinstance(node, UnaryOpNode):
            walk(node.operand, depth + 1)
            return
        if isinstance(node, BinaryOpNode):
            walk(node.left, depth + 1)
            walk(node.right, depth + 1)
            return
        if isinstance(node, TernaryNode):
            walk(node.condition, depth + 1)
            walk(node.true_expr, depth + 1)
            walk(node.false_expr, depth + 1)
            return

    walk(ast)
    return tuple(slots)


# ── Variable slot detection ────────────────────────────────────────────────


def detect_variable_slots(
    expression: str,
    available_vars: set[str] | None = None,
    derived_vars: set[str] | None = None,
) -> tuple[VariableSlot, ...]:
    """Scan ``expression`` for variables belonging to a ``VariableGroup``.

    Parameters
    ----------
    expression
        Factor expression.
    available_vars
        Variables currently provided by the runtime (OHLCV base + ExtraData
        variables). Used to filter alternatives so tuning cannot propose a
        field the runtime will not deliver. ``None`` disables the filter.
    derived_vars
        User-defined variables in the factor config (``variables:`` section).
        These are skipped because they are synthesised from base variables;
        substituting *their* definitions is out of scope.
    """
    ast = parse_expression(expression)
    derived = derived_vars or set()
    occurrences: dict[str, list[tuple[int, ...]]] = {}

    def walk(node: ASTNode, path: tuple[int, ...]) -> None:
        if isinstance(node, VariableNode):
            if node.name in derived:
                return
            if node.name not in _VARIABLE_TO_GROUP:
                return
            occurrences.setdefault(node.name, []).append(path)
            return
        if isinstance(node, (NumberNode, StringNode)):
            return
        if isinstance(node, UnaryOpNode):
            walk(node.operand, path + (0,))
            return
        if isinstance(node, BinaryOpNode):
            walk(node.left, path + (0,))
            walk(node.right, path + (1,))
            return
        if isinstance(node, TernaryNode):
            walk(node.condition, path + (0,))
            walk(node.true_expr, path + (1,))
            walk(node.false_expr, path + (2,))
            return
        if isinstance(node, FunctionCallNode):
            for i, arg in enumerate(node.arguments):
                walk(arg, path + (i,))
            return

    walk(ast, ())

    slots: list[VariableSlot] = []
    for var_idx, (var_name, positions) in enumerate(sorted(occurrences.items())):
        group_name = _VARIABLE_TO_GROUP[var_name]
        group = VARIABLE_GROUPS[group_name]
        candidates = tuple(m for m in group.members if m != var_name)
        if available_vars is not None:
            candidates = tuple(m for m in candidates if m in available_vars)
        if not candidates:
            continue
        slots.append(
            VariableSlot(
                slot_id=f"var_{var_idx}",
                current_var=var_name,
                group_name=group_name,
                alternatives=(var_name,) + candidates,
                positions=tuple(positions),
                scope=group.scope,
            )
        )
    return tuple(slots)


# ── Expression reconstruction ──────────────────────────────────────────────


def _format_number(value: float) -> str:
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:g}"


def _build_new_function_call(
    original: FunctionCallNode,
    chosen_op: OperatorAlternative,
    extras: dict[str, float],
) -> ASTNode:
    """Build a new FunctionCallNode (or raw inner) for the chosen alternative.

    The single-argument rendering pathway is the only one the MVP supports.
    For ``identity``, we return the original inner node unchanged. For all
    others, we construct ``FunctionCallNode(new_name, (inner, …extras))``.
    """
    inner = original.arguments[0]
    if chosen_op.name == _IDENTITY_NAME:
        return inner

    args: list[ASTNode] = [inner]
    for spec in chosen_op.extra_params:
        value = float(extras.get(spec.name, spec.original_value))
        args.append(NumberNode(value))
    return FunctionCallNode(chosen_op.name, tuple(args))


def _apply_operator_choices_ast(
    ast: ASTNode,
    choices: dict[str, str],
    slots: tuple[OperatorSlot, ...],
    extras: dict[str, dict[str, float]],
) -> ASTNode:
    """AST-level operator substitution.

    Re-runs the same DFS as ``detect_operator_slots`` to locate each slot's
    ``FunctionCallNode`` and rebuilds the tree with the chosen operator.
    """
    if not choices or not slots:
        return ast

    slot_map = {slot.slot_id: slot for slot in slots}
    from nautilus_quants.factors.expression.normalize import _normalize

    normalised = _normalize(ast)
    counter = [0]

    def rebuild(node: ASTNode) -> ASTNode:
        if isinstance(node, FunctionCallNode):
            canonical = _canonicalise_op_name(node.name)
            is_single_arg_cs = len(node.arguments) == 1 and canonical in _GROUP_ALIASES["cs_normalize"]
            if is_single_arg_cs:
                slot_id = f"op_{counter[0]}"
                counter[0] += 1
                slot = slot_map.get(slot_id)
                if slot is not None:
                    chosen = choices.get(slot_id)
                    if chosen is not None and chosen != slot.current_op:
                        alt = slot.alt_by_name(chosen)
                        if alt is not None:
                            # Rebuild inner first so deeper slots also rotate.
                            rebuilt_inner = rebuild(node.arguments[0])
                            mock = FunctionCallNode(canonical, (rebuilt_inner,))
                            return _build_new_function_call(mock, alt, extras.get(slot_id, {}))
            return FunctionCallNode(
                node.name,
                tuple(rebuild(arg) for arg in node.arguments),
            )
        if isinstance(node, UnaryOpNode):
            return UnaryOpNode(node.operator, rebuild(node.operand))
        if isinstance(node, BinaryOpNode):
            return BinaryOpNode(node.operator, rebuild(node.left), rebuild(node.right))
        if isinstance(node, TernaryNode):
            return TernaryNode(
                rebuild(node.condition),
                rebuild(node.true_expr),
                rebuild(node.false_expr),
            )
        return node

    return rebuild(normalised)


def _apply_variable_choices_ast(
    ast: ASTNode,
    choices: dict[str, str],
    slots: tuple[VariableSlot, ...],
) -> ASTNode:
    """AST-level variable substitution. Every ``VariableNode`` whose name
    matches a slot's ``current_var`` becomes the chosen alternative.
    """
    if not choices or not slots:
        return ast
    rename: dict[str, str] = {}
    for slot in slots:
        chosen = choices.get(slot.slot_id)
        if chosen is None or chosen == slot.current_var:
            continue
        if chosen not in slot.alternatives:
            raise ValueError(
                f"Variable choice '{chosen}' for slot {slot.slot_id} is not " f"in the allowed set {slot.alternatives}"
            )
        rename[slot.current_var] = chosen
    if not rename:
        return ast

    def rewrite(node: ASTNode) -> ASTNode:
        if isinstance(node, VariableNode):
            new_name = rename.get(node.name, node.name)
            return VariableNode(new_name)
        if isinstance(node, (NumberNode, StringNode)):
            return node
        if isinstance(node, UnaryOpNode):
            return UnaryOpNode(node.operator, rewrite(node.operand))
        if isinstance(node, BinaryOpNode):
            return BinaryOpNode(node.operator, rewrite(node.left), rewrite(node.right))
        if isinstance(node, TernaryNode):
            return TernaryNode(
                rewrite(node.condition),
                rewrite(node.true_expr),
                rewrite(node.false_expr),
            )
        if isinstance(node, FunctionCallNode):
            return FunctionCallNode(node.name, tuple(rewrite(a) for a in node.arguments))
        return node

    return rewrite(ast)


_P_PLACEHOLDER_RE = re.compile(r"(?<![A-Za-z0-9_])p(\d+)(?![A-Za-z0-9_])")


def _apply_numeric_params(template: str, params: dict[str, float]) -> str:
    """Substitute ``p0``, ``p1``, … placeholders with numeric literals.

    Performed on the textual template (not the AST) because the placeholders
    masquerade as ``VariableNode`` instances when parsed — rewriting via the
    parser would require inventing a ``NumberNode`` per placeholder. A regex
    is both simpler and faster.
    """
    if not params:
        return template

    def repl(match: re.Match[str]) -> str:
        key = f"p{match.group(1)}"
        if key not in params:
            return match.group(0)
        return _format_number(float(params[key]))

    return _P_PLACEHOLDER_RE.sub(repl, template)


def reconstruct_expression(
    template: str,
    params: dict[str, float],
    *,
    op_choices: dict[str, str] | None = None,
    operator_slots: tuple[OperatorSlot, ...] | list[OperatorSlot] = (),
    var_choices: dict[str, str] | None = None,
    variable_slots: tuple[VariableSlot, ...] | list[VariableSlot] = (),
    op_extra_params: dict[str, dict[str, float]] | None = None,
) -> str:
    """Rebuild an expression from the tuning template + sampled choices.

    Steps (order matters — subtle):

    1. Apply operator rewrites on the *template* AST. The template still
       contains ``p{i}`` as ``VariableNode`` names, so inner structure is
       fully preserved; rewrites see untouched DFS indices.
    2. Apply variable rewrites on the post-operator AST.
    3. Serialize AST and substitute numeric placeholders last so the
       resulting expression normalises cleanly.
    4. Run through ``normalize_expression`` for canonical form.
    """
    template_ast = parse_expression(template)

    operator_slots_tuple = tuple(operator_slots)
    op_choices = op_choices or {}
    op_extra_params = op_extra_params or {}
    ast_after_ops = _apply_operator_choices_ast(template_ast, op_choices, operator_slots_tuple, op_extra_params)

    variable_slots_tuple = tuple(variable_slots)
    var_choices = var_choices or {}
    ast_after_vars = _apply_variable_choices_ast(ast_after_ops, var_choices, variable_slots_tuple)

    serialised = _serialize_ast(ast_after_vars)
    with_numbers = _apply_numeric_params(serialised, params)
    return normalize_expression(with_numbers)


# ── Convenience wrapper ────────────────────────────────────────────────────


def build_search_space(
    expression: str,
    *,
    parameters: dict[str, Any] | None = None,
    available_vars: Iterable[str] | None = None,
    derived_vars: Iterable[str] | None = None,
    tune_numeric: bool = True,
    tune_operators: bool = False,
    tune_variables: bool = False,
) -> tuple[
    str,
    tuple[ParamSpec, ...],
    tuple[OperatorSlot, ...],
    tuple[VariableSlot, ...],
]:
    """One-stop helper — resolve config parameters, classify numeric
    placeholders, detect operator and variable slots.

    Dimensions that are disabled return empty tuples regardless of what the
    expression contains, matching the YAML ``dimensions`` switches.
    """
    resolved = expand_config_params(expression, parameters or {})
    template, values = expression_template(resolved)

    numeric_specs: tuple[ParamSpec, ...] = ()
    if tune_numeric:
        numeric_specs = classify_parameters(template, values, resolved)

    operator_slots: tuple[OperatorSlot, ...] = ()
    if tune_operators:
        operator_slots = detect_operator_slots(template)

    variable_slots: tuple[VariableSlot, ...] = ()
    if tune_variables:
        avail = set(available_vars) if available_vars is not None else None
        derived_set = set(derived_vars) if derived_vars is not None else None
        variable_slots = detect_variable_slots(template, available_vars=avail, derived_vars=derived_set)

    return template, numeric_specs, operator_slots, variable_slots
