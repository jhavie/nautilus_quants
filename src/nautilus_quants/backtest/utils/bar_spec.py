"""Bar specification parsing utilities."""

from nautilus_trader.model.data import BarSpecification
from nautilus_trader.model.enums import BarAggregation, PriceType


def parse_timeframe(tf: str) -> tuple[int, str]:
    """解析时间框架字符串为 (step, unit)

    用于构造 bar aggregation 的 @ 语法源部分。

    Args:
        tf: 时间框架 (e.g., "1m", "1h", "4h", "1d")

    Returns:
        (step, unit) tuple, e.g., (1, "MINUTE"), (4, "HOUR")

    Examples:
        >>> parse_timeframe("1m")
        (1, "MINUTE")
        >>> parse_timeframe("4h")
        (4, "HOUR")
        >>> parse_timeframe("1d")
        (1, "DAY")
    """
    tf_lower = tf.lower()

    if tf_lower.endswith("m"):
        return int(tf_lower[:-1]), "MINUTE"
    elif tf_lower.endswith("h"):
        return int(tf_lower[:-1]), "HOUR"
    elif tf_lower.endswith("d"):
        return int(tf_lower[:-1]), "DAY"
    else:
        raise ValueError(f"Invalid timeframe: {tf}. Use format like '1m', '1h', '4h', '1d'")


def parse_bar_spec(bar_spec: str) -> BarSpecification:
    """Parse simplified or native bar spec format.

    Simplified: "1h", "4h", "1m", "15m", "1d"
    Native: "1-HOUR-LAST", "15-MINUTE-LAST"

    Args:
        bar_spec: Bar specification string

    Returns:
        BarSpecification object

    Raises:
        ValueError: If bar_spec format is invalid
    """
    # Check if native format (contains hyphen with HOUR/MINUTE/DAY)
    if "-" in bar_spec and any(
        x in bar_spec.upper() for x in ["HOUR", "MINUTE", "DAY"]
    ):
        # Native format: "1-HOUR-LAST"
        parts = bar_spec.upper().split("-")
        step = int(parts[0])
        agg_str = parts[1]
        price_type = parts[2] if len(parts) > 2 else "LAST"

        aggregation = getattr(BarAggregation, agg_str)
        price = getattr(PriceType, price_type)

        return BarSpecification(step=step, aggregation=aggregation, price_type=price)

    # Simplified format: "1h", "4h", "1m", "15m", "1d"
    bar_spec_lower = bar_spec.lower()

    if bar_spec_lower.endswith("m"):
        aggregation = BarAggregation.MINUTE
        step = int(bar_spec_lower[:-1])
    elif bar_spec_lower.endswith("h"):
        aggregation = BarAggregation.HOUR
        step = int(bar_spec_lower[:-1])
    elif bar_spec_lower.endswith("d"):
        aggregation = BarAggregation.DAY
        step = int(bar_spec_lower[:-1])
    else:
        raise ValueError(
            f"Invalid bar_spec format: {bar_spec}. "
            "Use simplified format (1m, 5m, 1h, 4h, 1d) or native format (1-HOUR-LAST)"
        )

    return BarSpecification(
        step=step,
        aggregation=aggregation,
        price_type=PriceType.LAST,
    )


def format_bar_spec(bar_spec: str, internal: bool = False) -> str:
    """Format simplified bar spec to Nautilus native format.

    Args:
        bar_spec: Simplified bar spec (e.g., "1h", "1m", "4h")
        internal: If True, use INTERNAL aggregation source, else EXTERNAL

    Returns:
        Native format string (e.g., "1-HOUR-LAST-INTERNAL")

    Examples:
        >>> format_bar_spec("1h")
        "1-HOUR-LAST-EXTERNAL"
        >>> format_bar_spec("1h", internal=True)
        "1-HOUR-LAST-INTERNAL"
        >>> format_bar_spec("1m")
        "1-MINUTE-LAST-EXTERNAL"
    """
    spec = parse_bar_spec(bar_spec)

    agg_name = BarAggregation(spec.aggregation).name
    price_name = PriceType(spec.price_type).name
    source = "INTERNAL" if internal else "EXTERNAL"

    return f"{spec.step}-{agg_name}-{price_name}-{source}"
