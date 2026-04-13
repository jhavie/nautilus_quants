# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Prompt construction for LLM-based alpha factor mining.

Builds structured prompts for Claude Code CLI (`claude -p`) that generate
factor expressions in the project's DSL. Adapted from AlphaAgent's
hypothesis-driven approach with crypto-specific guidelines.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nautilus_quants.factors.expression.complexity import ComplexityConstraints


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

### Factor Cutting Operators (research-driven signal purification)
- rolling_selmean_top(x, y, d, n)  — Mean of x for top-n y values in d-bar window
- rolling_selmean_btm(x, y, d, n)  — Mean of x for bottom-n y values in d-bar window
- rolling_selmean_diff(x, y, d, n) — Top mean minus bottom mean (core cutting operator)
- ts_max_to_min(x, d)              — Rolling amplitude: ts_max(x,d) - ts_min(x,d)
- diff_sign(x, d)                  — Deviation direction: sign(x - ts_mean(x,d))
- ts_meanrank(x, d)                — Cross-sectional rank → time-series mean

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
- fill_nan(x, value)          — Replace NaN with a constant

### Ternary Operator (inline syntax)
- condition ? true_expr : false_expr

### Arithmetic & Comparison
- +, -, *, /, ^ (power)
- ==, !=, <, >, <=, >=
- && (and), || (or), ! (not)"""


def _get_window_guide(bar_spec: str) -> str:
    """Map bar frequency to human-readable window guide."""
    if bar_spec == "1h":
        return "Window mapping (1h bars): " "24=1day, 168=1week, 720=1month, 2160=3months"
    if bar_spec == "4h":
        return "Window mapping (4h bars): " "6=1day, 42=1week, 180=1month, 540=3months"
    return f"Bar frequency: {bar_spec}"


_CONSTRUCTION_RULES = """\
## Factor Construction Rules (CRITICAL — follow strictly)

1. **Never use raw prices directly.** Always use relative changes or rankings:
   - GOOD: `delta(close, 1) / delay(close, 1)`, `cs_rank(close)`, `(close - delay(close, 1)) / replace_zero(ts_std(close, 20))`
   - BAD:  `close`, `close - open`  (scale differs across instruments)

2. **Prevent division by zero.** Add epsilon to denominators:
   - GOOD: `delta(volume, 6) / replace_zero(delay(volume, 6))`
   - BAD:  `delta(volume, 6) / delay(volume, 6)`

3. **Cross-sectional processing** — choose based on signal characteristics:
   | Signal type | Operator | When to use |
   |-------------|----------|-------------|
   | Heavy tails / extreme outliers | `cs_rank(expr)` | volume ratios, OI spikes, social bursts |
   | Magnitude matters (don't lose info) | `cs_zscore(expr)` | returns, FR, beta — keeps relative scale |
   | Moderate outliers | `winsorize(expr, 3)` | returns, correlation values |
   | Need market-neutral | `vector_neut(expr, btc_returns)` | any signal with BTC exposure |
   | Demean only (preserve scale) | `cs_demean(expr)` | interaction terms, residual signals |
   | Isolate extremes only | `clip_quantile(expr, 0.1, 0.9)` | signals where middle is noise |
   | Bounded interaction term | `scale_down(expr, 0.5)` | signal × signal products |
   | Already bounded (e.g. corr) | no wrapper needed | correlation [-1,1], ts_rank [0,1] |
   - Pick ONE outer wrapper. NEVER nest: `winsorize(cs_rank(...))` is wrong
   - Do NOT wrap with `normalize()` — the composite pipeline handles that
   - DIVERSITY MATTERS: don't default to `cs_rank` for every factor — see Wrapper Diversity section below

4. **Use diverse operators** in intermediate steps:
   - `ts_rank(x, w)` — rank in time, resistant to outliers
   - `cs_demean(x)` — remove cross-sectional average (isolate relative signal)
   - `cs_rank(x)` — fine as intermediate input (e.g. `correlation(high, cs_rank(volume), 5)`)
   - `scale_down(x, 0.5)` — map to [-0.5, 0.5] range, useful for interaction terms

5. **Crypto-specific:**
   - 24/7 market — no overnight gaps, no weekend effects
   - High volatility — use shorter windows (6-42 bars) for responsiveness
   - OHLCV available, plus extra data variables depending on config
   - funding_rate (Bybit): 8h settlement (forward-filled, changes every 2 bars at 4h)
   - san_funding_rate (SanAPI): cross-exchange aggregated, native 4h granularity
   - open_interest (Bybit): single-exchange, base asset (token) units
   - san_open_interest (SanAPI): cross-exchange aggregated, USD units
   - **Tip:** `san_open_interest / close` converts USD → token base units, removing price co-movement
   - {window_guide}

6. **Market factor variables:**
   - btc_close/eth_close are broadcast (same value across all instruments)
   - btc_beta measures how much an altcoin moves with BTC; high beta = high BTC sensitivity
   - Use btc_beta for market-neutral strategies: `vector_neut(returns, btc_returns)`
   - Combine beta with momentum: `winsorize(btc_beta * returns, 3)` (beta-weighted momentum)
   - vwap deviation: `(close - vwap) / replace_zero(vwap)` (intraday mean reversion)

7. **Avoid strict equalities.** Use ranges instead of `==`:
   - BAD:  `ts_min(low, 10) == delay(ts_min(low, 10), 1)`
   - GOOD: `abs(ts_min(low, 10) - delay(ts_min(low, 10), 1)) < ts_std(low, 20) * 0.1`

8. **Expression complexity.** Aim for 2-4 operator nesting levels.
   - Too simple: `winsorize(close, 3)` (trivial, unlikely to have alpha)
   - Too complex: 6+ nested calls (overfitting risk, slow to compute)

9. **Each factor must be independent.** Do not reference other factor names."""


_AVAILABLE_VARIABLES = (
    "close, open, high, low, volume, returns, "
    "funding_rate, open_interest, quote_volume, "
    "san_funding_rate, san_open_interest, "
    "san_volume_usd, san_social_volume, "
    "btc_close, eth_close, btc_returns, eth_returns, "
    "btc_vol, eth_vol, btc_beta, eth_beta, vwap\n"
    "- returns = delta(close,1)/delay(close,1), pre-computed\n"
    "- funding_rate = Bybit 8h perpetual funding rate (forward-filled, ±0.01%)\n"
    "- open_interest = Bybit open interest in base asset (token) units\n"
    "- san_funding_rate = cross-exchange aggregated funding rate, native 4h\n"
    "- san_open_interest = cross-exchange aggregated open interest (USD units);\n"
    "  divide by close to get token-base units: san_open_interest / close\n"
    "- san_volume_usd = cross-exchange ROLLING 24h trading volume (USD, sampled every 4h);\n"
    "  NOT a 4h bucket — use cs_rank for attention ranking, or rate-of-change for surges\n"
    "- san_social_volume = total social mentions across Twitter/Reddit/Telegram/4chan etc;\n"
    "  sudden spikes often precede price moves — use delta/delay over raw counts\n"
    "- quote_volume = traded value in USDT (intra-bar turnover)\n"
    "- vwap = quote_volume / volume, volume-weighted average price\n"
    "- btc_close = BTC close price broadcast to all instruments\n"
    "- eth_close = ETH close price broadcast to all instruments\n"
    "- btc_returns = BTC log-returns, pre-computed\n"
    "- eth_returns = ETH log-returns, pre-computed\n"
    "- btc_vol = ts_std(btc_returns, 42), BTC 7-day realized volatility\n"
    "- eth_vol = ts_std(eth_returns, 42), ETH 7-day realized volatility\n"
    "- btc_beta = rolling beta to BTC (42 bars), measures sensitivity to BTC moves\n"
    "- eth_beta = rolling beta to ETH (42 bars), measures sensitivity to ETH moves"
)


# ── Public API ────────────────────────────────────────────────────────────


def _filter_operator_reference(subset: tuple[str, ...]) -> str:
    """Filter operator reference to only include *subset* operators.

    Always keeps the Ternary and Arithmetic sections.
    Handles combined bullet lines like ``floor(x), ceil(x)`` and
    ``max(a, b), min(a, b)`` by matching any name on the line.
    """
    names = frozenset(subset)
    lines = _OPERATOR_REFERENCE.strip().split("\n")
    result: list[str] = []
    current_header: str | None = None
    header_emitted = False

    for line in lines:
        if line.startswith("###"):
            # Always include Ternary / Arithmetic sections.
            if "Ternary" in line or "Arithmetic" in line:
                result.append(line)
                current_header = line
                header_emitted = True
                continue
            current_header = line
            header_emitted = False
            continue

        stripped = line.strip()
        if stripped.startswith("- "):
            # Extract ALL function names on this line: "floor(x), ceil(x)" → {floor, ceil}
            line_names = set(re.findall(r"(\w+)\(", stripped))
            if line_names & names:
                if not header_emitted and current_header:
                    result.append(current_header)
                    header_emitted = True
                result.append(line)
        elif header_emitted:
            # Non-operator lines under an emitted section.
            result.append(line)

    return "\n".join(result)


def build_generation_prompt(
    round_num: int,
    num_factors: int,
    bar_spec: str,
    previous_factors: list[dict],
    top_factors: list[dict],
    hypothesis: str | None = None,
    *,
    theme: str | None = None,
    operator_subset: tuple[str, ...] | None = None,
    variable_subset: tuple[str, ...] | None = None,
    constraints: ComplexityConstraints | None = None,
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
        theme: Broad theme for diversified mode (e.g. "量价因子").
            Takes priority over *hypothesis*.
        operator_subset: Restrict prompt to these operators only.
        variable_subset: Restrict prompt to these variables only.
        constraints: Complexity constraints to advertise in prompt.

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
    if operator_subset:
        sections.append("## Available Operators\n" + _filter_operator_reference(operator_subset))
    else:
        sections.append("## Available Operators\n" + _OPERATOR_REFERENCE)

    # ── Available variables ──
    if variable_subset:
        vars_str = ", ".join(variable_subset)
        extras: list[str] = []
        if "returns" in variable_subset:
            extras.append("- returns = delta(close,1)/delay(close,1), pre-computed")
        if "funding_rate" in variable_subset:
            extras.append(
                "- funding_rate = Bybit 8h perpetual funding rate " "(forward-filled, ±0.01%)"
            )
        if "san_funding_rate" in variable_subset:
            extras.append(
                "- san_funding_rate = cross-exchange aggregated funding rate, "
                "native 4h granularity (no forward-fill artifacts)"
            )
        if "open_interest" in variable_subset:
            extras.append("- open_interest = Bybit open interest in base asset (token) units")
        if "san_open_interest" in variable_subset:
            extras.append(
                "- san_open_interest = cross-exchange aggregated OI (USD units); "
                "use san_open_interest / close for token-base units"
            )
        if "san_volume_usd" in variable_subset:
            extras.append(
                "- san_volume_usd = cross-exchange ROLLING 24h trading volume (USD, "
                "sampled every 4h); NOT a 4h bucket, use cs_rank for attention ranking"
            )
        if "san_social_volume" in variable_subset:
            extras.append(
                "- san_social_volume = total social mentions across platforms (count); "
                "use delta/delay rather than raw counts, spikes often lead price"
            )
        var_block = vars_str + ("\n" + "\n".join(extras) if extras else "")
        sections.append(f"## Available Variables\n{var_block}")
    else:
        sections.append(f"## Available Variables\n{_AVAILABLE_VARIABLES}")

    # ── Construction rules ──
    rules = _CONSTRUCTION_RULES.replace(
        "{window_guide}",
        _get_window_guide(bar_spec),
    )
    sections.append(rules)

    # ── Hard complexity constraints ──
    if constraints:
        sections.append(
            "## Hard Constraints (expressions violating ANY will be REJECTED)\n"
            f"- Maximum expression length: {constraints.max_char_length} characters\n"
            f"- Maximum AST node count: {constraints.max_node_count} nodes\n"
            f"- Maximum AST depth: {constraints.max_depth} levels\n"
            f"- Maximum function nesting: {constraints.max_func_nesting} levels\n"
            f"- Maximum distinct variables: {constraints.max_variables}\n"
            f"- Maximum window parameter: {constraints.max_window} bars\n"
            f"- Maximum numeric literal ratio: {constraints.max_numeric_ratio:.0%} "
            f"of AST nodes\n"
            f"- Optimal expression length: 50-150 characters (best generalization)\n"
            f"- Expressions >150 chars almost ALWAYS overfit on crypto data.\n"
            f"  These are enforced by an AST validator. PREFER simpler expressions."
        )

    # ── Research direction: theme > hypothesis > default ──
    if theme:
        sections.append(
            f"## Research Direction\n"
            f"Theme: {theme}\n"
            f"Explore diverse hypotheses within this theme.\n"
            f"For each factor, propose a specific, testable hypothesis "
            f"explaining why it should predict short-term relative returns.\n"
            f"Ensure different factors explore different sub-patterns "
            f"within the theme."
        )
    elif hypothesis:
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
        expr_list = "\n".join(f"  - {f['expression']}" for f in previous_factors)
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
                f"  - {f['expression']}\n" f"    IC: {ic_str}  |  ICIR: {icir_str}"
            )
        sections.append(
            "## Best Factors So Far (generate variations of these)\n" + "\n".join(feedback_lines)
        )

    # ── Wrapper Diversity Guidance ──
    sections.append(
        "## Wrapper Diversity Guidance (IMPORTANT — read before generating)\n\n"
        "Past mining batches show >90% of factors used `cs_rank` as the outer "
        "wrapper, while `cs_zscore`, `cs_demean`, `scale_down`, `clip_quantile` "
        "were never tried. This monoculture creates highly correlated factors "
        "and over-fits to rank-based noise.\n\n"
        "**Match wrapper to signal characteristics:**\n"
        "- Heavy-tailed (volume ratios, OI spikes, social bursts) → `cs_rank`\n"
        "- Bounded with meaningful magnitude (returns, FR, beta, vwap dev) "
        "→ `cs_zscore`\n"
        "- Already in [-1,1] or [0,1] (correlation, ts_rank) → no wrapper\n"
        "- Want middle suppressed, only tails predict → `clip_quantile(x, 0.1, 0.9)`\n"
        "- Need market-neutral → `vector_neut(x, btc_returns)`\n"
        "- Demean preserving scale (interaction terms) → `cs_demean(x)`\n"
        "- Bounded interaction term → `scale_down(x, 0.5)`\n"
        "- Moderate outlier clipping (preserves order) → `winsorize(x, 3)`\n\n"
        "**Aim for ≤50% `cs_rank` outer wrappers** across your generated batch. "
        "If you've already used `cs_rank` for several factors, switch to "
        "`cs_zscore` / `cs_demean` / `winsorize` / `vector_neut` for the next.\n"
    )

    # ── Factor Cutting Techniques ──
    sections.append(
        "## Factor Cutting Techniques (因子切割论)\n\n"
        "You can refine factors by conditioning on market microstructure "
        "dimensions. This isolates the regime where the factor works best, "
        "improving signal purity.\n\n"
        "### Native cutting operators (preferred — precise, no zero-dilution):\n"
        "- `rolling_selmean_diff(returns, volume, 20, 5)` — "
        "High-volume days return minus low-volume days return\n"
        "- `rolling_selmean_diff(returns, ts_std(returns, 6), 20, 5)` — "
        "High-volatility return minus low-volatility return\n"
        "- `rolling_selmean_diff(returns, funding_rate, 20, 5)` — "
        "Regime split by funding rate level\n\n"
        "### Single-side cutting (top-only or bottom-only — when only one regime predicts):\n"
        "- `rolling_selmean_top(returns, volume, 20, 5)` — "
        "Mean return on the 5 highest-volume days only "
        "(use when low-volume regime is just noise, top alone has alpha)\n"
        "- `rolling_selmean_btm(returns, ts_std(returns, 6), 20, 5)` — "
        "Mean return on the 5 calmest days only "
        "(isolates low-vol regime alpha; symmetric `_diff` mixes regimes)\n"
        "- `rolling_selmean_top(san_funding_rate, san_open_interest, 24, 6)` — "
        "FR on the highest-OI bars (crowded long crowding signal)\n\n"
        "### Other cutting operators (often forgotten — use these to diversify):\n"
        "- `cs_rank(ts_max_to_min(close, 20) / replace_zero(ts_mean(close, 20)))` — "
        "relative amplitude (range/mean), captures volatility regime\n"
        "- `cs_zscore(diff_sign(volume, 20) * returns)` — "
        "return signed by 'is volume above 20-bar mean' (regime-conditioned momentum)\n"
        "- `cs_rank(ts_meanrank(returns, 20))` — "
        "average cross-sectional rank position over 20 bars (smoothed momentum, "
        "less noisy than instantaneous `cs_rank(returns)`)\n"
        "- `ts_max_to_min(san_funding_rate, 24) - ts_max_to_min(san_funding_rate, 6)` — "
        "long-window vs short-window FR amplitude divergence\n\n"
        "### if_else cutting (alternative — more flexible conditions):\n"
        "- Conditional: `if_else(cs_rank(ts_mean(volume, 6)) > 0.5, F, 0)`\n"
        "- Differential: `if_else(C, F, 0) - if_else(!C, F, 0)` "
        "(high regime minus low regime)\n\n"
        "### When to use cutting:\n"
        "- When you believe a factor works better in specific market "
        "conditions (high volume, high volatility, positive funding)\n"
        "- The subtraction acts as implicit standardization — "
        "the low-regime baseline removes noise\n"
        "- The output is already cross-sectionally comparable — "
        "additional wrapping is optional"
    )

    # ── Cross-Sectional Processing Cookbook ──
    sections.append(
        "## Cross-Sectional Processing Cookbook\n\n"
        "### Market neutralization (remove BTC/ETH beta exposure):\n"
        "- `vector_neut(ts_slope(close, 20), btc_returns)` — BTC-neutral trend\n"
        "- `vector_neut(returns, btc_returns)` — idiosyncratic return\n\n"
        "### Interaction terms (combine two signals):\n"
        "- `cs_demean(returns) * cs_rank(volume)` — volume-weighted relative return\n"
        "- `cs_rank(funding_rate) * cs_rank(open_interest)` — FR × OI interaction\n\n"
        "### SanAPI derivatives (cross-exchange aggregated):\n"
        "- `san_open_interest / close` — convert USD OI to token units "
        "(removes price co-movement for cleaner cross-sectional ranking)\n"
        "- `rank(delta(san_open_interest / close, 6))` — 1-day OI growth rank\n"
        "- `rank(san_funding_rate) - rank(returns(6))` — FR-return divergence\n"
        "- `cs_rank(san_funding_rate) * cs_rank(san_open_interest / close)` — "
        "crowded positioning signal\n\n"
        "### SanAPI attention signals (exclusive cross-exchange/social data):\n"
        "- `cs_rank(delta(san_social_volume, 6) / replace_zero(delay(san_social_volume, 6)))` "
        "— 24h social burst (precedes price)\n"
        "- `correlation(san_social_volume, volume, 20)` "
        "— social-volume sync (attention translates to trading)\n"
        "- `cs_rank(san_volume_usd) - cs_rank(quote_volume)` "
        "— Binance-vs-aggregate volume dispersion\n"
        "- `rank(san_volume_usd) / rank(replace_zero(san_open_interest / close))` "
        "— turnover vs positioning ratio\n"
        "- `cs_rank(san_social_volume) - cs_rank(san_volume_usd)` "
        "— social attention premium (hype without flow)\n\n"
        "### Wrapper variations (avoid cs_rank monoculture):\n"
        "- `cs_zscore(returns - ts_mean(returns, 20))` "
        "— z-score deviation, preserves magnitude (vs rank's equal-spaced compression)\n"
        "- `cs_zscore(ts_slope(close, 20))` "
        "— z-score trend strength (alternative to `cs_rank` for bounded signals)\n"
        "- `cs_demean(funding_rate) * volume` "
        "— demean before interaction (preserves scale, vs `cs_rank * cs_rank` rank-only)\n"
        "- `winsorize(btc_beta * returns, 3)` "
        "— winsorized beta-weighted return (alternative to `cs_rank`)\n"
        "- `clip_quantile(funding_rate, 0.05, 0.95)` "
        "— FR tail signal: drop middle 90%, keep extremes\n"
        "- `cs_zscore(san_open_interest / close) - cs_zscore(volume)` "
        "— z-score difference (vs rank difference, keeps intensity info)\n\n"
        "### Tail signal extraction:\n"
        "- `clip_quantile(correlation(returns, volume, 20), 0.1, 0.9)` "
        "— focus on extreme correlations\n"
        "- `scale_down(ts_std(returns, 20), 0.5)` — bounded volatility signal\n\n"
        "### Regime conditioning (from cutting operators):\n"
        "- `rolling_selmean_diff(returns, volume, 20, 5)` "
        "— high-volume vs low-volume return\n"
        "- `if_else(funding_rate > 0, returns, 0) - "
        "if_else(funding_rate < 0, returns, 0)` — FR regime split\n\n"
        "Use these patterns as building blocks. Combine and vary them creatively."
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
