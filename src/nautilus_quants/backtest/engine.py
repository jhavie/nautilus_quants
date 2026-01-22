"""Backtest engine utilities.

This module provides utility functions for creating instruments.
The main engine setup is now handled by the high-level BacktestNode API.
"""

import time
from decimal import Decimal

from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.instruments import CryptoPerpetual


def create_crypto_perpetual(
    instrument_id: str,
    venue: str = "BINANCE",
    maker_fee: float = 0.0002,
    taker_fee: float = 0.0004,
    ts_init: int | None = None,
) -> CryptoPerpetual:
    """Create a crypto perpetual instrument.

    Args:
        instrument_id: Symbol (e.g., "BTCUSDT")
        venue: Exchange venue name
        maker_fee: Maker fee rate
        taker_fee: Taker fee rate
        ts_init: Initialization timestamp in nanoseconds (defaults to current time)

    Returns:
        CryptoPerpetual instrument
    """
    from nautilus_trader.model.identifiers import InstrumentId, Symbol
    from nautilus_trader.model.objects import Money, Price, Quantity

    # Use provided timestamp or current time
    if ts_init is None:
        ts_init = time.time_ns()

    symbol = Symbol(instrument_id)
    instrument = CryptoPerpetual(
        instrument_id=InstrumentId(symbol=symbol, venue=Venue(venue)),
        raw_symbol=symbol,
        base_currency=USDT,
        quote_currency=USDT,
        settlement_currency=USDT,
        is_inverse=False,
        price_precision=2,
        size_precision=3,
        price_increment=Price.from_str("0.01"),
        size_increment=Quantity.from_str("0.001"),
        max_quantity=Quantity.from_str("10000"),
        min_quantity=Quantity.from_str("0.001"),
        max_notional=Money(1_000_000, USDT),
        min_notional=Money(10, USDT),
        max_price=Price.from_str("1000000"),
        min_price=Price.from_str("0.01"),
        margin_init=Decimal("0.05"),
        margin_maint=Decimal("0.025"),
        maker_fee=Decimal(str(maker_fee)),
        taker_fee=Decimal(str(taker_fee)),
        ts_event=ts_init,
        ts_init=ts_init,
    )

    return instrument
