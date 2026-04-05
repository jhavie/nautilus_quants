# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Prompt construction for LLM-based alpha factor mining.

Builds structured prompts for Claude Code CLI (`claude -p`) that generate
factor expressions in the project's DSL. Adapted from AlphaAgent's
hypothesis-driven approach with crypto-specific guidelines.
"""

from __future__ import annotations


# ── JSON Schema for --json-schema flag ────────────────────────────────────

_FACTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "hypothesis": {"type": "string"},
        "expression": {"type": "string"},
        "description": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["hypothesis", "expression", "description", "tags"],
}

RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "factors": {
            "type": "object",
            "additionalProperties": _FACTOR_SCHEMA,
        },
    },
    "required": ["factors"],
}


def get_json_schema() -> dict:
    """Return the JSON Schema for ``claude --json-schema``."""
    return RESPONSE_SCHEMA


# ── Operator reference (static, mirrors operators/*.py registries) ────────

_OPERATOR_REFERENCE = """\
### Time-Series Operators (column-wise, require window parameter)
- ts_mean(x, window)        — Rolling mean over *window* bars
- ts_sum(x, window)         — Rolling sum
- ts_std(x, window)         — Rolling standard deviation (sample)
- ts_min(x, window)         — Rolling minimum
- ts_max(x, window)         — Rolling maximum
- ts_rank(x, window)        — Percentile rank of latest value in window [1/d, 1]
- ts_argmax(x, window)      — Bars since rolling max (1-indexed from oldest)
- ts_argmin(x, window)      — Bars since rolling min
- ts_skew(x, window)        — Rolling skewness
- ts_product(x, window)     — Rolling product
- delta(x, window)          — x[t] - x[t-window]
- delay(x, window)          — x[t-window]  (lagged value)
- correlation(x, y, window) — Rolling Pearson correlation between x and y
- covariance(x, y, window)  — Rolling covariance between x and y
- decay_linear(x, window)   — Linearly weighted moving average (recent bars weighted more)
- ema(x, window)            — Exponential moving average (alpha = 2/(window+1))
- ts_slope(x, window)       — Rolling linear regression slope (trend strength)
- ts_rsquare(x, window)     — Rolling R-squared (0=noise, 1=perfect linear trend)
- ts_residual(x, window)    — Deviation of last value from regression line
- ts_percentile(x, window, q) — Rolling q-th quantile (e.g. q=0.5 for median)

### Cross-Sectional Operators (row-wise, across all instruments at each timestamp)
- cs_rank(x)                — Percentile rank across instruments [1/n, 1]
- rank(x)                   — Alias for cs_rank
- cs_zscore(x)              — Z-score normalization across instruments
- cs_scale(x)               — Scale so |sum| = 1
- cs_demean(x)              — Subtract cross-sectional mean
- cs_max(x)                 — Cross-sectional max (broadcast)
- cs_min(x)                 — Cross-sectional min (broadcast)
- normalize(x)              — Demean (optionally divide by std)
- winsorize(x, std=4)       — Clip to ±std multiples
- scale_down(x)             — Normalize to [0, 1]
- clip_quantile(x, lower=0.2, upper=0.8) — Clip to quantile range
- vector_neut(x, y)         — Residual of x regressed on y (neutralization)

### Math Operators (element-wise)
- log(x)                    — Natural logarithm (NaN for x ≤ 0)
- sign(x)                   — Sign function (-1, 0, 1)
- abs(x)                    — Absolute value
- sqrt(x)                   — Square root
- power(x, exp)             — x^exp
- signed_power(x, exp)      — sign(x) × |x|^exp
- exp(x)                    — Exponential e^x
- floor(x), ceil(x)        — Floor / ceiling
- round(x, decimals=0)      — Round to N decimals
- max(a, b), min(a, b)      — Element-wise max / min
- if_else(cond, true_val, false_val) — Ternary selection
- is_nan(x)                 — Returns 1.0 if NaN, else 0.0
- replace_zero(x, eps=0.0001) — Replace exact zeros with epsilon

### Ternary Operator (inline syntax)
- condition ? true_expr : false_expr

### Arithmetic & Comparison
- +, -, *, /, ^ (power)
- ==, !=, <, >, <=, >=
- && (and), || (or), ! (not)"""


def _get_window_guide(bar_spec: str) -> str:
    """Map bar frequency to human-readable window guide."""
    if bar_spec == "1h":
        return (
            "Window mapping (1h bars): "
            "24=1day, 168=1week, 720=1month, 2160=3months"
        )
    if bar_spec == "4h":
        return (
            "Window mapping (4h bars): "
            "6=1day, 42=1week, 180=1month, 540=3months"
        )
    return f"Bar frequency: {bar_spec}"


_CONSTRUCTION_RULES = """\
## Factor Construction Rules (CRITICAL — follow strictly)

1. **Never use raw prices directly.** Always use relative changes or rankings:
   - GOOD: `delta(close, 1) / delay(close, 1)`, `cs_rank(close)`
   - BAD:  `close`, `close - open`  (scale differs across instruments)

2. **Prevent division by zero.** Add epsilon to denominators:
   - GOOD: `delta(volume, 6) / replace_zero(delay(volume, 6))`
   - BAD:  `delta(volume, 6) / delay(volume, 6)`

3. **Cross-sectional normalization.** Wrap final output in `cs_rank()` or `cs_zscore()`:
   - This ensures factors are comparable across instruments with different scales

4. **Robust over precise.** Prefer rank-based measures over raw values:
   - `ts_rank(x, w)` over `ts_mean(x, w)` — resistant to outliers
   - `cs_rank(x)` over raw `x` — removes scale differences

5. **Crypto-specific:**
   - 24/7 market — no overnight gaps, no weekend effects
   - High volatility — use shorter windows (6-42 bars) for responsiveness
   - OHLCV + funding_rate + open_interest available
   - funding_rate: settlement every 8h (00:00/08:00/16:00 UTC), positive=longs pay shorts
   - open_interest: total open positions in base asset, 4h granularity
   - {window_guide}
   - **CRITICAL:** With 4h bars, funding_rate only changes every 2 bars (forward-filled).

6. **Avoid strict equalities.** Use ranges instead of `==`:
   - BAD:  `ts_min(low, 10) == delay(ts_min(low, 10), 1)`
   - GOOD: `abs(ts_min(low, 10) - delay(ts_min(low, 10), 1)) < ts_std(low, 20) * 0.1`

7. **Expression complexity.** Aim for 2-4 operator nesting levels.
   - Too simple: `cs_rank(close)` (trivial, unlikely to have alpha)
   - Too complex: 6+ nested calls (overfitting risk, slow to compute)

8. **Each factor must be independent.** Do not reference other factor names."""


_AVAILABLE_VARIABLES = (
    "close, open, high, low, volume, returns, funding_rate, open_interest\n"
    "- returns = delta(close,1)/delay(close,1), pre-computed\n"
    "- funding_rate = 8-hour perpetual funding rate from Bybit "
    "(typically ±0.01%, forward-filled across bars)\n"
    "- open_interest = total open interest in base asset units from Bybit "
    "(e.g. BTC quantity, 4h granularity)"
)


# ── Public API ────────────────────────────────────────────────────────────


def build_generation_prompt(
    round_num: int,
    num_factors: int,
    bar_spec: str,
    previous_factors: list[dict],
    top_factors: list[dict],
    hypothesis: str | None = None,
) -> str:
    """Build the factor generation prompt for ``claude -p``.

    Args:
        round_num: Current mining round (1-indexed).
        num_factors: Number of factors to generate this round.
        bar_spec: Bar frequency (e.g. "4h", "1h").
        previous_factors: All previously generated factors
            ``[{name, expression, ic_mean, icir}]``.
        top_factors: Best factors so far (by ICIR), same schema.
        hypothesis: Optional user-provided hypothesis direction.

    Returns:
        Complete prompt string for ``claude -p``.
    """
    sections: list[str] = []

    # ── Role ──
    sections.append(
        "You are a quantitative researcher mining alpha factors for "
        "crypto perpetual futures (cross-sectional strategy). "
        "Generate factors as DSL expressions that predict short-term "
        "relative returns across a universe of 100+ crypto instruments."
    )

    # ── DSL reference ──
    sections.append("## Available Operators\n" + _OPERATOR_REFERENCE)

    # ── Available variables ──
    sections.append(f"## Available Variables\n{_AVAILABLE_VARIABLES}")

    # ── Construction rules ──
    rules = _CONSTRUCTION_RULES.replace(
        "{window_guide}", _get_window_guide(bar_spec),
    )
    sections.append(rules)

    # ── Hypothesis direction ──
    if hypothesis:
        sections.append(
            f"## Research Direction\n"
            f"Focus on this hypothesis: {hypothesis}\n"
            f"Generate factors that test different aspects of this idea."
        )
    elif round_num == 1:
        sections.append(
            "## Research Direction\n"
            "This is the first round. Explore diverse alpha sources:\n"
            "- Momentum / mean-reversion at different horizons\n"
            "- Volume-price relationships (divergence, confirmation)\n"
            "- Volatility regime changes\n"
            "- Intraday patterns (open-close, high-low range)\n"
            "- Cross-sectional relative strength"
        )

    # ── Anti-duplication ──
    if previous_factors:
        expr_list = "\n".join(
            f"  - {f['expression']}" for f in previous_factors
        )
        sections.append(
            "## Already Generated Expressions (DO NOT duplicate)\n"
            f"The following {len(previous_factors)} expressions have already "
            "been generated. You MUST create structurally different factors:\n"
            f"{expr_list}"
        )

    # ── Performance feedback ──
    if top_factors:
        feedback_lines = []
        for f in top_factors[:5]:
            ic = f.get("ic_mean", {})
            icir = f.get("icir", {})
            ic_str = ", ".join(f"{k}={v:.4f}" for k, v in ic.items()) if ic else "n/a"
            icir_str = ", ".join(f"{k}={v:.3f}" for k, v in icir.items()) if icir else "n/a"
            feedback_lines.append(
                f"  - {f['expression']}\n"
                f"    IC: {ic_str}  |  ICIR: {icir_str}"
            )
        sections.append(
            "## Best Factors So Far (generate variations of these)\n"
            + "\n".join(feedback_lines)
        )

    # ── Output instructions ──
    sections.append(
        f"## Task\n"
        f"Generate exactly {num_factors} NEW alpha factor expressions.\n"
        f"For each factor, provide:\n"
        f"- **hypothesis**: The market intuition (1-2 sentences, "
        f"explain WHY this should predict returns)\n"
        f"- **expression**: A valid DSL expression using ONLY the "
        f"operators and variables listed above\n"
        f"- **description**: Brief technical description\n"
        f"- **tags**: Category tags (e.g. momentum, reversal, volume, "
        f"volatility, correlation)\n\n"
        f"Use snake_case for factor names. "
        f"Each factor must be independent and structurally unique."
    )

    return "\n\n".join(sections)
