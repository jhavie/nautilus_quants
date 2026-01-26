"""Common utilities for strategies - bar aggregation helpers."""


def parse_interval_to_bar_spec(interval: str) -> tuple[int, str]:
    """Parse interval string (e.g., '1h', '4h') to (step, aggregation) tuple.

    Parameters
    ----------
    interval : str
        Interval string like '1m', '5m', '1h', '4h', '1d'.

    Returns
    -------
    tuple[int, str]
        Tuple of (step, aggregation) e.g., (60, 'MINUTE') for '1h'.

    Raises
    ------
    ValueError
        If interval format is invalid.
    """
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

    This creates the target bar type and subscription string for Nautilus
    native bar aggregation using the '@' syntax.

    Parameters
    ----------
    instrument_id : str
        Instrument ID string (e.g., 'ETHUSDT.BINANCE').
    interval : str
        Target interval (e.g., '1h', '4h', '1d').
    source_bar_spec : str, default '1-MINUTE-EXTERNAL'
        Source bar specification.
    price_type : str, default 'LAST'
        Price type for bars.

    Returns
    -------
    tuple[str, str]
        Tuple of (bar_type_str, subscribe_str).
        - bar_type_str: Target bar type (e.g., 'ETHUSDT.BINANCE-60-MINUTE-LAST-INTERNAL')
        - subscribe_str: Subscription string (e.g., '...INTERNAL@1-MINUTE-EXTERNAL')

    Example
    -------
    >>> create_aggregated_bar_type('ETHUSDT.BINANCE', '1h', '1-MINUTE-EXTERNAL')
    ('ETHUSDT.BINANCE-60-MINUTE-LAST-INTERNAL',
     'ETHUSDT.BINANCE-60-MINUTE-LAST-INTERNAL@1-MINUTE-EXTERNAL')
    """
    step, aggregation = parse_interval_to_bar_spec(interval)
    bar_type_str = f"{instrument_id}-{step}-{aggregation}-{price_type}-INTERNAL"
    subscribe_str = f"{bar_type_str}@{source_bar_spec}"
    return bar_type_str, subscribe_str


def interval_to_minutes(interval: str) -> int:
    """Convert interval string to total minutes.

    Parameters
    ----------
    interval : str
        Interval string like '1m', '5m', '1h', '4h', '1d'.

    Returns
    -------
    int
        Total minutes.

    Raises
    ------
    ValueError
        If interval format is invalid.
    """
    step, aggregation = parse_interval_to_bar_spec(interval)
    if aggregation == "MINUTE":
        return step
    elif aggregation == "DAY":
        return step * 1440
    raise ValueError(f"Unknown aggregation: {aggregation}")
