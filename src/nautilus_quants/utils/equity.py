# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Shared equity computation utilities."""

from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Currency
from nautilus_trader.portfolio.base import PortfolioFacade


def compute_mtm_equity(
    portfolio: PortfolioFacade,
    venue: Venue,
    currency: Currency,
) -> float | None:
    """Compute mark-to-market equity (balance + unrealized PnL).

    Args:
        portfolio: Portfolio facade for account and PnL access.
        venue: Trading venue.
        currency: Settlement currency.

    Returns:
        MTM equity value, or None if account or balance is unavailable.
    """
    account = portfolio.account(venue)
    if account is None:
        return None

    balance = account.balance_total(currency)
    if balance is None:
        return None

    unrealized_value = 0.0
    unrealized_dict = portfolio.unrealized_pnls(venue)
    if unrealized_dict:
        unrealized_money = unrealized_dict.get(currency)
        if unrealized_money is not None:
            unrealized_value = unrealized_money.as_double()

    return balance.as_double() + unrealized_value
