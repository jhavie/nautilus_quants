"""Binance venue configuration builder.

Thin config layer: reads VenueConfig + env vars → official Binance adapter Config objects.
Factory classes are re-exported directly from nautilus_trader.
"""

from __future__ import annotations

from nautilus_trader.adapters.binance.common.enums import BinanceAccountType
from nautilus_trader.adapters.binance.config import (
    BinanceDataClientConfig,
    BinanceExecClientConfig,
)
from nautilus_trader.adapters.binance.factories import (
    BinanceLiveDataClientFactory,
    BinanceLiveExecClientFactory,
)
from nautilus_trader.common.config import InstrumentProviderConfig

from nautilus_quants.live.config import VenueConfig
from nautilus_quants.live.exceptions import LiveConfigError
from nautilus_quants.live.utils.config_parser import _get_env_credential

# Re-export official factories
DATA_FACTORY = BinanceLiveDataClientFactory
EXEC_FACTORY = BinanceLiveExecClientFactory

# Map instrument_type + contract_type to BinanceAccountType
_ACCOUNT_TYPE_MAP: dict[tuple[str, str], BinanceAccountType] = {
    ("SWAP", "LINEAR"): BinanceAccountType.USDT_FUTURES,
    ("SWAP", "INVERSE"): BinanceAccountType.COIN_FUTURES,
    ("FUTURES", "LINEAR"): BinanceAccountType.USDT_FUTURES,
    ("FUTURES", "INVERSE"): BinanceAccountType.COIN_FUTURES,
    ("SPOT", "LINEAR"): BinanceAccountType.SPOT,
    ("SPOT", "INVERSE"): BinanceAccountType.SPOT,
    ("MARGIN", "LINEAR"): BinanceAccountType.MARGIN,
    ("MARGIN", "INVERSE"): BinanceAccountType.MARGIN,
}


def _resolve_account_type(venue: VenueConfig) -> BinanceAccountType:
    key = (venue.instrument_type.upper(), venue.contract_type.upper())
    if key not in _ACCOUNT_TYPE_MAP:
        raise LiveConfigError(
            f"Unsupported Binance instrument_type/contract_type combination: "
            f"{venue.instrument_type}/{venue.contract_type}. "
            f"Valid: {list(_ACCOUNT_TYPE_MAP.keys())}"
        )
    return _ACCOUNT_TYPE_MAP[key]


def build_binance_data_config(
    venue: VenueConfig,
    instrument_ids: list[str] | None = None,
) -> BinanceDataClientConfig:
    """Build official BinanceDataClientConfig from VenueConfig + env vars.

    Environment variables:
        BINANCE_API_KEY, BINANCE_API_SECRET

    Args:
        venue: Parsed venue configuration.
        instrument_ids: Optional list of instrument IDs to load.

    Returns:
        BinanceDataClientConfig ready for TradingNode.
    """
    api_key = _get_env_credential("BINANCE_API_KEY")
    api_secret = _get_env_credential("BINANCE_API_SECRET")

    account_type = _resolve_account_type(venue)

    load_ids = None
    if instrument_ids:
        load_ids = frozenset(f"{iid}.BINANCE" for iid in instrument_ids)

    provider_config = InstrumentProviderConfig(
        load_all=load_ids is None,
        load_ids=load_ids,
    )

    return BinanceDataClientConfig(
        api_key=api_key,
        api_secret=api_secret,
        account_type=account_type,
        testnet=venue.is_demo,
        instrument_provider=provider_config,
    )


def build_binance_exec_config(
    venue: VenueConfig,
    instrument_ids: list[str] | None = None,
) -> BinanceExecClientConfig:
    """Build official BinanceExecClientConfig from VenueConfig + env vars.

    Environment variables:
        BINANCE_API_KEY, BINANCE_API_SECRET

    Args:
        venue: Parsed venue configuration.
        instrument_ids: Optional list of instrument IDs to load.

    Returns:
        BinanceExecClientConfig ready for TradingNode.
    """
    api_key = _get_env_credential("BINANCE_API_KEY")
    api_secret = _get_env_credential("BINANCE_API_SECRET")

    account_type = _resolve_account_type(venue)

    load_ids = None
    if instrument_ids:
        load_ids = frozenset(f"{iid}.BINANCE" for iid in instrument_ids)

    provider_config = InstrumentProviderConfig(
        load_all=load_ids is None,
        load_ids=load_ids,
    )

    return BinanceExecClientConfig(
        api_key=api_key,
        api_secret=api_secret,
        account_type=account_type,
        testnet=venue.is_demo,
        instrument_provider=provider_config,
    )
