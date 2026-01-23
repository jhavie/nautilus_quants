"""Interval parsing utilities for bar aggregation."""


def parse_interval_to_bar_spec(interval: str) -> tuple[int, str]:
    """Parse interval string (e.g., '1h', '4h') to (step, aggregation) tuple."""
    interval = interval.lower().strip()

    if interval.endswith("m"):
        return (int(interval[:-1]), "MINUTE")
    elif interval.endswith("h"):
        return (int(interval[:-1]) * 60, "MINUTE")
    elif interval.endswith("d"):
        return (int(interval[:-1]), "DAY")
    else:
        raise ValueError(f"Invalid interval: '{interval}'. Use: 1m, 5m, 1h, 4h, 1d")


def create_aggregated_bar_type(
    instrument_id: str,
    interval: str,
    source_bar_spec: str = "1-MINUTE-EXTERNAL",
    price_type: str = "LAST",
) -> tuple[str, str]:
    """Create bar type strings for NautilusTrader aggregation subscription.

    Returns (bar_type_str, subscribe_str) for use with subscribe_bars().
    """
    step, aggregation = parse_interval_to_bar_spec(interval)
    bar_type_str = f"{instrument_id}-{step}-{aggregation}-{price_type}-INTERNAL"
    subscribe_str = f"{bar_type_str}@{source_bar_spec}"
    return bar_type_str, subscribe_str


def interval_to_minutes(interval: str) -> int:
    """Convert interval string to total minutes."""
    step, aggregation = parse_interval_to_bar_spec(interval)
    if aggregation == "MINUTE":
        return step
    elif aggregation == "DAY":
        return step * 1440
    raise ValueError(f"Unknown aggregation: {aggregation}")
