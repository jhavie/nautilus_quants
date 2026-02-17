"""Catalog data loader for alpha analysis.

Shared data loading infrastructure for both analysis and future mining modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from nautilus_trader.model.data import BarType, BarSpecification
from nautilus_trader.model.enums import BarAggregation, PriceType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.persistence.catalog import ParquetDataCatalog

if TYPE_CHECKING:
    from nautilus_trader.model.data import Bar


def _parse_bar_spec(bar_spec: str) -> tuple[int, BarAggregation]:
    """Parse bar spec string to step and aggregation.

    Args:
        bar_spec: Bar specification string (e.g., "1h", "4h", "15m")

    Returns:
        Tuple of (step, aggregation)
    """
    if bar_spec.endswith("m"):
        return int(bar_spec[:-1]), BarAggregation.MINUTE
    elif bar_spec.endswith("h"):
        return int(bar_spec[:-1]), BarAggregation.HOUR
    elif bar_spec.endswith("d"):
        return int(bar_spec[:-1]), BarAggregation.DAY
    else:
        raise ValueError(f"Unsupported bar spec: {bar_spec}")


class CatalogDataLoader:
    """Load bar data from Nautilus ParquetDataCatalog.

    Uses Nautilus-native catalog API to load Bar objects directly.
    """

    def __init__(self, catalog_path: str, bar_spec: str = "1h") -> None:
        self._catalog = ParquetDataCatalog(catalog_path)
        self._bar_spec = bar_spec

    def load_bars(self, instrument_ids: list[str]) -> dict[str, list[Bar]]:
        """Load bars for each instrument from catalog.

        Args:
            instrument_ids: List of instrument IDs (e.g., ["BTCUSDT.BINANCE"])

        Returns:
            Dict mapping instrument_id to list of Bar objects
        """
        step, aggregation = _parse_bar_spec(self._bar_spec)
        result: dict[str, list[Bar]] = {}

        for inst_id_str in instrument_ids:
            instrument_id = InstrumentId.from_str(inst_id_str)
            bar_spec = BarSpecification(
                step=step,
                aggregation=aggregation,
                price_type=PriceType.LAST,
            )
            bar_type = BarType(
                instrument_id=instrument_id,
                bar_spec=bar_spec,
            )
            bars = self._catalog.bars(bar_types=[str(bar_type)])
            result[inst_id_str] = list(bars) if bars is not None else []

        return result

    @staticmethod
    def bars_to_dataframe(bars: list[Bar]) -> pd.DataFrame:
        """Convert Bar list to DataFrame using Nautilus-native serialization.

        Uses ``Bar.to_dict()`` (Cython) for faster batch conversion than
        manual Python-side field extraction.

        Args:
            bars: List of Nautilus Bar objects

        Returns:
            DataFrame with columns [open, high, low, close, volume]
            and datetime index named 'timestamp'
        """
        from nautilus_trader.model.data import Bar as BarModel

        if not bars:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(BarModel.to_dict(b) for b in bars)
        df.index = pd.to_datetime(df["ts_event"], unit="ns")
        df.index.name = "timestamp"
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        for col in ohlcv_cols:
            df[col] = df[col].astype(float)
        return df[ohlcv_cols]
