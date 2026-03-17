"""Configuration dataclasses for the live trading module."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VenueConfig:
    """Exchange venue configuration.

    Parameters
    ----------
    name : str
        Exchange name (e.g., "OKX", "BINANCE").
    instrument_type : str
        Type of instruments (e.g., "SWAP", "SPOT", "MARGIN").
    contract_type : str
        Contract type for derivatives ("LINEAR" or "INVERSE").
    margin_mode : str
        Margin mode ("CROSS" or "ISOLATED").
    is_demo : bool
        Whether to use the exchange demo/testnet environment.
    http_timeout_secs : int
        HTTP request timeout in seconds.
    use_fills_channel : bool
        Whether to use the fills WebSocket channel (OKX-specific).
    """

    name: str
    instrument_type: str = "SWAP"
    contract_type: str = "LINEAR"
    margin_mode: str = "CROSS"
    is_demo: bool = True
    http_timeout_secs: int = 10
    use_fills_channel: bool = False


@dataclass(frozen=True)
class InstrumentsConfig:
    """Instrument subscription configuration.

    Parameters
    ----------
    bar_spec : str
        Bar specification in simplified format (e.g., "1h", "4h").
    instrument_ids : list[str]
        List of instrument IDs (e.g., ["BTC-USDT-SWAP", "ETH-USDT-SWAP"]).
    """

    bar_spec: str
    instrument_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LiveConfig:
    """Top-level live trading configuration.

    Parameters
    ----------
    venue : VenueConfig
        Exchange venue configuration.
    instruments : InstrumentsConfig
        Instrument subscription configuration.
    """

    venue: VenueConfig
    instruments: InstrumentsConfig
