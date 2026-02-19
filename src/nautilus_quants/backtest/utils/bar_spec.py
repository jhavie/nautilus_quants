"""Bar specification parsing utilities."""

from datetime import timedelta

from nautilus_trader.model.data import BarSpecification
from nautilus_trader.model.enums import BarAggregation, PriceType


def parse_timeframe(tf: str) -> tuple[int, str]:
    """Parse timeframe string into ``(step, unit)``.

    Args:
        tf: Timeframe (e.g., "1m", "1h", "4h", "1d")

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


def parse_interval_to_timedelta(interval: str) -> timedelta:
    """Convert interval string to timedelta.

    Args:
        interval: Interval string (e.g., "1m", "1h", "4h", "8h", "1d")

    Returns:
        Corresponding timedelta object

    Examples:
        >>> parse_interval_to_timedelta("1h")
        datetime.timedelta(seconds=3600)
        >>> parse_interval_to_timedelta("8h")
        datetime.timedelta(seconds=28800)
    """
    step, unit = parse_timeframe(interval)
    if unit == "MINUTE":
        return timedelta(minutes=step)
    if unit == "HOUR":
        return timedelta(hours=step)
    if unit == "DAY":
        return timedelta(days=step)
    raise ValueError(f"Unsupported interval unit: {unit}")


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


def format_bar_spec(bar_spec: str, internal: bool = False, include_source: bool = True) -> str:
    """Format simplified bar spec to Nautilus native format.

    Args:
        bar_spec: Simplified bar spec (e.g., "1h", "1m", "4h")
        internal: If True, use INTERNAL aggregation source, else EXTERNAL
        include_source: If True, include source suffix (-INTERNAL/-EXTERNAL)

    Returns:
        Native format string (e.g., "1-HOUR-LAST-EXTERNAL" or "1-HOUR-LAST")

    Examples:
        >>> format_bar_spec("1h")
        "1-HOUR-LAST-EXTERNAL"
        >>> format_bar_spec("1h", internal=True)
        "1-HOUR-LAST-INTERNAL"
        >>> format_bar_spec("1m", include_source=False)
        "1-MINUTE-LAST"
    """
    spec = parse_bar_spec(bar_spec)

    agg_name = BarAggregation(spec.aggregation).name
    price_name = PriceType(spec.price_type).name

    if include_source:
        source = "INTERNAL" if internal else "EXTERNAL"
        return f"{spec.step}-{agg_name}-{price_name}-{source}"
    else:
        return f"{spec.step}-{agg_name}-{price_name}"
