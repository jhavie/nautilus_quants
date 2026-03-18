# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""BarSubscriptionMixin - Unified bar_type subscription management for Nautilus actors and strategies.

Replaces the O(N) list membership check in on_bar() with an O(1) dict lookup,
and pre-computes the instrument_id_str mapping to avoid repeated Cython str conversions.
"""

from __future__ import annotations

from nautilus_trader.model.data import Bar, BarType


class BarSubscriptionMixin:
    """Provide unified bar_type subscription management for Nautilus Strategy / Actor.

    Usage
    -----
    1. Add as the first base class before Strategy or Actor.
    2. Call ``self._subscribe_bar_types(self.config.bar_types)`` in ``on_start()``.
    3. Replace ``if bar.bar_type not in self._bar_types:`` guards with
       ``instrument_id = self._resolve_bar(bar); if instrument_id is None: return``.
    """

    def _subscribe_bar_types(self, bar_types_config: list[str]) -> None:
        """Parse, subscribe, and build BarType → instrument_id_str mapping.

        Must be called in ``on_start()`` before any bars arrive.

        Parameters
        ----------
        bar_types_config : list[str]
            List of bar type strings (e.g. ``["BTCUSDT.BINANCE-1-HOUR-LAST@EXTERNAL"]``).
        """
        self._bar_type_to_inst_id: dict[BarType, str] = {}
        for bar_type_str in bar_types_config:
            bar_type = BarType.from_str(bar_type_str)
            self._bar_type_to_inst_id[bar_type] = str(bar_type.instrument_id)
            self.subscribe_bars(bar_type)

    def _resolve_bar(self, bar: Bar) -> str | None:
        """O(1) guard: return instrument_id_str if bar type is subscribed, else None.

        Parameters
        ----------
        bar : Bar
            Incoming bar to check.

        Returns
        -------
        str | None
            Pre-computed ``instrument_id`` string if subscribed, otherwise ``None``.
        """
        return self._bar_type_to_inst_id.get(bar.bar_type)
