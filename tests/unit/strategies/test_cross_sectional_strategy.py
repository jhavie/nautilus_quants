# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Unit tests for CrossSectionalFactorStrategy event-time execution."""

from nautilus_quants.strategies.cross_sectional.strategy import (
    CrossSectionalFactorStrategy,
    CrossSectionalFactorStrategyConfig,
)


class TestCrossSectionalPendingExecution:
    """Pending rebalance behavior with event-time close snapshots."""

    def test_try_execute_pending_waits_until_snapshot_complete(self) -> None:
        config = CrossSectionalFactorStrategyConfig(
            instrument_ids=["A.BINANCE", "B.BINANCE"],
            n_positions=1,
            rebalance_period=1,
            monthly_position_update=False,
        )
        strategy = CrossSectionalFactorStrategy(config)
        strategy._pending_by_ts[100] = [("A.BINANCE", 0.1), ("B.BINANCE", 0.9)]

        captured: list[tuple[int, dict[str, float]]] = []

        def _capture_rebalance(
            sorted_instruments: list[tuple[str, float]],
            ts_event: int,
            execution_prices: dict[str, float],
        ) -> None:
            _ = sorted_instruments
            captured.append((ts_event, execution_prices))

        strategy._rebalance_positions_with_buffer = _capture_rebalance  # type: ignore[method-assign]

        strategy._price_book.record_close(100, "A.BINANCE", 10.0)
        strategy._try_execute_pending(100)
        assert captured == []

        strategy._price_book.record_close(100, "B.BINANCE", 20.0)
        strategy._try_execute_pending(100)
        assert captured == [(100, {"A.BINANCE": 10.0, "B.BINANCE": 20.0})]

    def test_rebalance_passes_exec_price_to_open_position(self) -> None:
        config = CrossSectionalFactorStrategyConfig(
            instrument_ids=["A.BINANCE", "B.BINANCE", "C.BINANCE", "D.BINANCE"],
            n_positions=1,
            rebalance_period=1,
            monthly_position_update=False,
        )
        strategy = CrossSectionalFactorStrategy(config)

        sorted_instruments = [
            ("A.BINANCE", 0.1),
            ("B.BINANCE", 0.2),
            ("C.BINANCE", 0.3),
            ("D.BINANCE", 0.4),
        ]
        execution_prices = {
            "A.BINANCE": 11.0,
            "B.BINANCE": 22.0,
            "C.BINANCE": 33.0,
            "D.BINANCE": 44.0,
        }

        captured: list[tuple[str, float]] = []

        def _capture_open(
            instrument_id_str: str,
            side,
            exec_price: float,
            rank: int | None = None,
            composite_value: float | None = None,
            ts_event: int | None = None,
        ) -> bool:
            _ = (side, rank, composite_value, ts_event)
            captured.append((instrument_id_str, exec_price))
            return True

        strategy._open_position = _capture_open  # type: ignore[method-assign]
        strategy._rebalance_positions_with_buffer(
            sorted_instruments=sorted_instruments,
            ts_event=123,
            execution_prices=execution_prices,
        )

        assert ("A.BINANCE", 11.0) in captured
        assert ("D.BINANCE", 44.0) in captured
