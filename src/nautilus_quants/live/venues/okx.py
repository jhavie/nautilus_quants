"""OKX venue configuration builder.

Thin config layer: reads VenueConfig + env vars → official OKX adapter Config objects.
Factory classes are re-exported directly from nautilus_trader.
"""

from __future__ import annotations

from nautilus_trader.adapters.okx.config import OKXDataClientConfig, OKXExecClientConfig
from nautilus_trader.adapters.okx.factories import (
    OKXLiveDataClientFactory,
    OKXLiveExecClientFactory,
)
from nautilus_trader.common.config import InstrumentProviderConfig
from nautilus_trader.core.nautilus_pyo3.okx import (
    OKXContractType,
    OKXInstrumentType,
    OKXMarginMode,
)

from nautilus_quants.live.config import VenueConfig
from nautilus_quants.live.exceptions import LiveConfigError
from nautilus_quants.live.utils.config_parser import _get_env_credential

# Re-export official factories
DATA_FACTORY = OKXLiveDataClientFactory
EXEC_FACTORY = OKXLiveExecClientFactory

_INSTRUMENT_TYPE_MAP: dict[str, OKXInstrumentType] = {
    "SWAP": OKXInstrumentType.SWAP,
    "SPOT": OKXInstrumentType.SPOT,
    "FUTURES": OKXInstrumentType.FUTURES,
    "MARGIN": OKXInstrumentType.MARGIN,
}

_CONTRACT_TYPE_MAP: dict[str, OKXContractType] = {
    "LINEAR": OKXContractType.LINEAR,
    "INVERSE": OKXContractType.INVERSE,
}

_MARGIN_MODE_MAP: dict[str, OKXMarginMode] = {
    "CROSS": OKXMarginMode.CROSS,
    "ISOLATED": OKXMarginMode.ISOLATED,
}


def _resolve_instrument_type(raw: str) -> OKXInstrumentType:
    key = raw.upper()
    if key not in _INSTRUMENT_TYPE_MAP:
        raise LiveConfigError(
            f"Unsupported OKX instrument_type: {raw}. "
            f"Valid: {list(_INSTRUMENT_TYPE_MAP.keys())}"
        )
    return _INSTRUMENT_TYPE_MAP[key]


def _resolve_contract_type(raw: str) -> OKXContractType:
    key = raw.upper()
    if key not in _CONTRACT_TYPE_MAP:
        raise LiveConfigError(
            f"Unsupported OKX contract_type: {raw}. "
            f"Valid: {list(_CONTRACT_TYPE_MAP.keys())}"
        )
    return _CONTRACT_TYPE_MAP[key]


def _resolve_margin_mode(raw: str) -> OKXMarginMode:
    key = raw.upper()
    if key not in _MARGIN_MODE_MAP:
        raise LiveConfigError(
            f"Unsupported OKX margin_mode: {raw}. "
            f"Valid: {list(_MARGIN_MODE_MAP.keys())}"
        )
    return _MARGIN_MODE_MAP[key]


def build_okx_data_config(
    venue: VenueConfig,
    instrument_ids: list[str] | None = None,
) -> OKXDataClientConfig:
    """Build official OKXDataClientConfig from VenueConfig + env vars.

    Environment variables:
        OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE

    Args:
        venue: Parsed venue configuration.
        instrument_ids: Optional list of instrument IDs to load.

    Returns:
        OKXDataClientConfig ready for TradingNode.
    """
    api_key = _get_env_credential("OKX_API_KEY")
    api_secret = _get_env_credential("OKX_API_SECRET")
    api_passphrase = _get_env_credential("OKX_API_PASSPHRASE")

    instrument_type = _resolve_instrument_type(venue.instrument_type)
    contract_type = _resolve_contract_type(venue.contract_type)

    # Build instrument provider config
    load_ids = None
    if instrument_ids:
        # Support both "BTC-USDT-SWAP" and "BTC-USDT-SWAP.OKX" formats
        load_ids = frozenset(
            iid if ".OKX" in iid else f"{iid}.OKX" for iid in instrument_ids
        )

    provider_config = InstrumentProviderConfig(
        load_all=load_ids is None,
        load_ids=load_ids,
    )

    return OKXDataClientConfig(
        api_key=api_key,
        api_secret=api_secret,
        api_passphrase=api_passphrase,
        instrument_types=(instrument_type,),
        contract_types=(contract_type,),
        is_demo=venue.is_demo,
        http_timeout_secs=venue.http_timeout_secs,
        instrument_provider=provider_config,
    )


def build_okx_exec_config(
    venue: VenueConfig,
    instrument_ids: list[str] | None = None,
) -> OKXExecClientConfig:
    """Build official OKXExecClientConfig from VenueConfig + env vars.

    Environment variables:
        OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE

    Args:
        venue: Parsed venue configuration.
        instrument_ids: Optional list of instrument IDs to load.

    Returns:
        OKXExecClientConfig ready for TradingNode.
    """
    api_key = _get_env_credential("OKX_API_KEY")
    api_secret = _get_env_credential("OKX_API_SECRET")
    api_passphrase = _get_env_credential("OKX_API_PASSPHRASE")

    instrument_type = _resolve_instrument_type(venue.instrument_type)
    contract_type = _resolve_contract_type(venue.contract_type)
    margin_mode = _resolve_margin_mode(venue.margin_mode)

    load_ids = None
    if instrument_ids:
        load_ids = frozenset(
            iid if ".OKX" in iid else f"{iid}.OKX" for iid in instrument_ids
        )

    provider_config = InstrumentProviderConfig(
        load_all=load_ids is None,
        load_ids=load_ids,
    )

    return OKXExecClientConfig(
        api_key=api_key,
        api_secret=api_secret,
        api_passphrase=api_passphrase,
        instrument_types=(instrument_type,),
        contract_types=(contract_type,),
        margin_mode=margin_mode,
        is_demo=venue.is_demo,
        http_timeout_secs=venue.http_timeout_secs,
        use_fills_channel=venue.use_fills_channel,
        instrument_provider=provider_config,
    )
