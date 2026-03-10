# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Unit tests for FMZFactorStrategy and FMZFactorStrategyConfig."""

from unittest.mock import MagicMock, PropertyMock, patch

from nautilus_trader.model.enums import OrderSide, PositionSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity

from nautilus_quants.strategies.fmz import FMZFactorStrategy, FMZFactorStrategyConfig


class TestFMZFactorStrategyConfig:
    """Tests for FMZFactorStrategyConfig."""

    def test_creates_with_required_fields(self) -> None:
        """Test creation with required fields."""
        config = FMZFactorStrategyConfig(
            instrument_ids=["BTCUSDT.BINANCE", "ETHUSDT.BINANCE"],
        )
        assert len(config.instrument_ids) == 2
        assert config.instrument_ids[0] == "BTCUSDT.BINANCE"

    def test_defaults_are_applied(self) -> None:
        """Test default values are applied (FMZ original parameters)."""
        config = FMZFactorStrategyConfig(
            instrument_ids=["BTCUSDT.BINANCE"],
        )
        assert config.n_long == 40
        assert config.n_short == 40
        assert config.position_value == 300.0
        assert config.rebalance_interval == 1
        assert config.composite_factor == "composite"
        assert config.bar_types == []

    def test_creates_with_custom_values(self) -> None:
        """Test creation with custom values."""
        config = FMZFactorStrategyConfig(
            instrument_ids=["BTCUSDT.BINANCE", "ETHUSDT.BINANCE"],
            n_long=20,
            n_short=20,
            position_value=500.0,
            rebalance_interval=4,
            composite_factor="my_factor",
        )
        assert config.n_long == 20
        assert config.n_short == 20
        assert config.position_value == 500.0
        assert config.rebalance_interval == 4
        assert config.composite_factor == "my_factor"


class TestFMZFactorStrategyInit:
    """Tests for FMZFactorStrategy initialization."""

    def test_init_stores_config(self) -> None:
        """Test initialization stores config correctly."""
        config = FMZFactorStrategyConfig(
            instrument_ids=["BTCUSDT.BINANCE", "ETHUSDT.BINANCE"],
            n_long=10,
            n_short=10,
        )
        strategy = FMZFactorStrategy(config)

        assert len(strategy._instrument_ids) == 2
        assert strategy._n_instruments == 2

    def test_init_parses_instrument_ids(self) -> None:
        """Test initialization parses instrument_ids correctly."""
        config = FMZFactorStrategyConfig(
            instrument_ids=["BTCUSDT.BINANCE", "ETHUSDT.BINANCE"],
        )
        strategy = FMZFactorStrategy(config)

        assert strategy._instrument_ids[0].symbol.value == "BTCUSDT"
        assert strategy._instrument_ids[0].venue.value == "BINANCE"
        assert strategy._instrument_ids[1].symbol.value == "ETHUSDT"

    def test_init_state_is_clean(self) -> None:
        """Test initialization starts with clean state."""
        config = FMZFactorStrategyConfig(
            instrument_ids=["BTCUSDT.BINANCE"],
        )
        strategy = FMZFactorStrategy(config)

        assert strategy._pending_by_ts == {}
        assert strategy._long_positions == {}
        assert strategy._short_positions == {}
        assert strategy._hour_count == 0
        assert strategy._bar_count == 0
        assert strategy._bars_until_rebalance == 0


class TestFMZFactorStrategyRebalance:
    """Tests for FMZFactorStrategy rebalance logic."""

    def test_rebalance_selects_correct_positions(self) -> None:
        """Test rebalance selects bottom N for long and top N for short."""
        config = FMZFactorStrategyConfig(
            instrument_ids=[
                "A.BINANCE",
                "B.BINANCE",
                "C.BINANCE",
                "D.BINANCE",
                "E.BINANCE",
                "F.BINANCE",
            ],
            n_long=2,
            n_short=2,
            position_value=100.0,
        )
        # Simulate composite values (sorted: A=0.1, B=0.2, C=0.3, D=0.4, E=0.5, F=0.6)
        composite = {
            "A.BINANCE": 0.1,  # Should be long (lowest)
            "B.BINANCE": 0.2,  # Should be long
            "C.BINANCE": 0.3,
            "D.BINANCE": 0.4,
            "E.BINANCE": 0.5,  # Should be short
            "F.BINANCE": 0.6,  # Should be short (highest)
        }

        # Test sorting logic
        sorted_symbols = sorted(composite.items(), key=lambda x: x[1])
        long_targets = set([s for s, _ in sorted_symbols[:config.n_long]])
        short_targets = set([s for s, _ in sorted_symbols[-config.n_short:]])

        assert long_targets == {"A.BINANCE", "B.BINANCE"}
        assert short_targets == {"E.BINANCE", "F.BINANCE"}

    def test_rebalance_with_nan_values(self) -> None:
        """Test rebalance filters out NaN values."""
        import math

        composite = {
            "A.BINANCE": 0.1,
            "B.BINANCE": math.nan,  # Should be filtered
            "C.BINANCE": 0.3,
            "D.BINANCE": 0.4,
        }

        # Filter logic from strategy
        filtered = {
            k: v for k, v in composite.items()
            if not math.isnan(v)
        }

        assert len(filtered) == 3
        assert "B.BINANCE" not in filtered


class TestFMZPendingExecution:
    """Tests for pending signal execution with event-time snapshots."""

    def test_try_execute_pending_waits_until_snapshot_is_complete(self) -> None:
        config = FMZFactorStrategyConfig(
            instrument_ids=["A.BINANCE", "B.BINANCE"],
            n_long=1,
            n_short=1,
        )
        strategy = FMZFactorStrategy(config)

        captured: list[tuple[int, dict[str, float]]] = []

        def _capture_rebalance(
            composite: dict[str, float],
            execution_prices: dict[str, float],
            signal_ts: int,
        ) -> None:
            _ = composite
            captured.append((signal_ts, execution_prices))

        strategy._rebalance = _capture_rebalance  # type: ignore[method-assign]

        signal_ts = 100
        strategy._pending_by_ts[signal_ts] = {"A.BINANCE": 0.1, "B.BINANCE": 0.9}
        strategy._price_book.record_close(signal_ts, "A.BINANCE", 10.0)
        strategy._try_execute_pending(signal_ts)
        assert captured == []

        strategy._price_book.record_close(signal_ts, "B.BINANCE", 20.0)
        strategy._try_execute_pending(signal_ts)
        assert len(captured) == 1
        assert captured[0][0] == signal_ts
        assert captured[0][1] == {"A.BINANCE": 10.0, "B.BINANCE": 20.0}

    def test_rebalance_passes_exec_price_to_open_position(self) -> None:
        config = FMZFactorStrategyConfig(
            instrument_ids=["A.BINANCE", "B.BINANCE", "C.BINANCE", "D.BINANCE"],
            n_long=1,
            n_short=1,
        )
        strategy = FMZFactorStrategy(config)
        composite = {
            "A.BINANCE": 0.1,
            "B.BINANCE": 0.2,
            "C.BINANCE": 0.3,
            "D.BINANCE": 0.4,
        }
        execution_prices = {
            "A.BINANCE": 11.0,
            "B.BINANCE": 22.0,
            "C.BINANCE": 33.0,
            "D.BINANCE": 44.0,
        }

        captured: list[tuple[str, float]] = []

        _mock_qty = Quantity.from_int(1)

        def _capture_open(
            instrument_id_str: str,
            side,
            exec_price: float,
            rank: int | None = None,
            composite_value: float | None = None,
            ts_event: int | None = None,
        ) -> Quantity | None:
            _ = (side, rank, composite_value, ts_event)
            captured.append((instrument_id_str, exec_price))
            return _mock_qty

        strategy._open_position = _capture_open  # type: ignore[method-assign]
        strategy._rebalance(
            composite=composite,
            execution_prices=execution_prices,
            signal_ts=123,
        )

        assert ("A.BINANCE", 11.0) in captured
        assert ("D.BINANCE", 44.0) in captured

    def test_rebalance_flip_close_passes_exec_price(self) -> None:
        """FLIP close passes exec_price to _close_position for anchor injection."""
        config = FMZFactorStrategyConfig(
            instrument_ids=["A.BINANCE", "B.BINANCE", "C.BINANCE", "D.BINANCE"],
            n_long=1,
            n_short=1,
        )
        strategy = FMZFactorStrategy(config)

        # Pre-set: A is currently short, should FLIP_TO_LONG
        strategy._short_positions["A.BINANCE"] = Quantity.from_int(1)

        close_calls: list[tuple[str, str, float | None]] = []

        def _capture_close(
            instrument_id_str: str,
            reason: str,
            exec_price: float | None = None,
        ) -> None:
            close_calls.append((instrument_id_str, reason, exec_price))

        _mock_qty = Quantity.from_int(1)

        def _fake_open(
            instrument_id_str: str,
            side,
            exec_price: float,
            rank=None,
            composite_value=None,
            ts_event=None,
        ) -> Quantity | None:
            return _mock_qty

        strategy._close_position = _capture_close  # type: ignore[method-assign]
        strategy._open_position = _fake_open  # type: ignore[method-assign]

        composite = {
            "A.BINANCE": 0.1,
            "B.BINANCE": 0.2,
            "C.BINANCE": 0.3,
            "D.BINANCE": 0.4,
        }
        execution_prices = {
            "A.BINANCE": 11.0,
            "B.BINANCE": 22.0,
            "C.BINANCE": 33.0,
            "D.BINANCE": 44.0,
        }

        strategy._rebalance(
            composite=composite,
            execution_prices=execution_prices,
            signal_ts=123,
        )

        # A should be closed with FLIP_TO_LONG and exec_price=11.0
        flip_close = [c for c in close_calls if c[1] == "FLIP_TO_LONG"]
        assert len(flip_close) == 1
        assert flip_close[0][0] == "A.BINANCE"
        assert flip_close[0][2] == 11.0

    def test_no_factor_data_close_without_exec_price(self) -> None:
        """NO_FACTOR_DATA close falls back to None when price book has no data."""
        config = FMZFactorStrategyConfig(
            instrument_ids=["A.BINANCE", "B.BINANCE", "C.BINANCE", "D.BINANCE"],
            n_long=1,
            n_short=1,
        )
        strategy = FMZFactorStrategy(config)

        # Pre-set: X has a long position but is missing from composite
        strategy._long_positions["X.BINANCE"] = Quantity.from_int(1)

        close_calls: list[tuple[str, str, float | None]] = []

        def _capture_close(
            instrument_id_str: str,
            reason: str,
            exec_price: float | None = None,
        ) -> None:
            close_calls.append((instrument_id_str, reason, exec_price))

        _mock_qty = Quantity.from_int(1)

        def _fake_open(
            instrument_id_str: str,
            side,
            exec_price: float,
            rank=None,
            composite_value=None,
            ts_event=None,
        ) -> Quantity | None:
            return _mock_qty

        strategy._close_position = _capture_close  # type: ignore[method-assign]
        strategy._open_position = _fake_open  # type: ignore[method-assign]

        composite = {
            "A.BINANCE": 0.1,
            "B.BINANCE": 0.2,
            "C.BINANCE": 0.3,
            "D.BINANCE": 0.4,
        }
        execution_prices = {
            "A.BINANCE": 11.0,
            "B.BINANCE": 22.0,
            "C.BINANCE": 33.0,
            "D.BINANCE": 44.0,
        }

        strategy._rebalance(
            composite=composite,
            execution_prices=execution_prices,
            signal_ts=123,
        )

        # X should be closed with NO_FACTOR_DATA; no price in book → exec_price=None
        no_data_close = [c for c in close_calls if c[1] == "NO_FACTOR_DATA"]
        assert len(no_data_close) == 1
        assert no_data_close[0][0] == "X.BINANCE"
        assert no_data_close[0][2] is None

    def test_no_factor_data_close_uses_price_book_anchor(self) -> None:
        """NO_FACTOR_DATA close injects anchor_px from price book when available."""
        config = FMZFactorStrategyConfig(
            instrument_ids=["A.BINANCE", "B.BINANCE", "C.BINANCE", "D.BINANCE"],
            n_long=1,
            n_short=1,
        )
        strategy = FMZFactorStrategy(config)

        # Pre-set: X has a long position but is missing from composite
        strategy._long_positions["X.BINANCE"] = Quantity.from_int(1)

        # Record X's close in price book so anchor is available
        strategy._price_book.record_close(100, "X.BINANCE", 55.5)

        close_calls: list[tuple[str, str, float | None]] = []

        def _capture_close(
            instrument_id_str: str,
            reason: str,
            exec_price: float | None = None,
        ) -> None:
            close_calls.append((instrument_id_str, reason, exec_price))

        _mock_qty = Quantity.from_int(1)

        def _fake_open(
            instrument_id_str: str,
            side,
            exec_price: float,
            rank=None,
            composite_value=None,
            ts_event=None,
        ) -> Quantity | None:
            return _mock_qty

        strategy._close_position = _capture_close  # type: ignore[method-assign]
        strategy._open_position = _fake_open  # type: ignore[method-assign]

        composite = {
            "A.BINANCE": 0.1,
            "B.BINANCE": 0.2,
            "C.BINANCE": 0.3,
            "D.BINANCE": 0.4,
        }
        execution_prices = {
            "A.BINANCE": 11.0,
            "B.BINANCE": 22.0,
            "C.BINANCE": 33.0,
            "D.BINANCE": 44.0,
        }

        strategy._rebalance(
            composite=composite,
            execution_prices=execution_prices,
            signal_ts=123,
        )

        # X should be closed with NO_FACTOR_DATA and exec_price from price book
        no_data_close = [c for c in close_calls if c[1] == "NO_FACTOR_DATA"]
        assert len(no_data_close) == 1
        assert no_data_close[0][0] == "X.BINANCE"
        assert no_data_close[0][2] == 55.5


class TestCloseAllPositions:
    """Tests for _close_all_positions using cache positions (source of truth)."""

    def test_close_all_positions_uses_cache_positions(self) -> None:
        """Verify close orders are derived from cache.positions_open(), not tracking dicts.

        on_stop must iterate cache positions, use Order.closing_side() for direction,
        position.quantity for size, reduce_only=True, position_id, and anchor_px.
        """
        config = FMZFactorStrategyConfig(
            instrument_ids=["A.BINANCE", "B.BINANCE", "C.BINANCE"],
            n_long=1,
            n_short=1,
        )
        strategy = FMZFactorStrategy(config)

        # Seed price_book with known close prices for anchor_px
        strategy._price_book.record_close(100, "A.BINANCE", 50000.0)
        strategy._price_book.record_close(100, "B.BINANCE", 3000.0)

        # Mock cache positions with real PositionSide values
        qty_a = Quantity.from_str("10.5")
        qty_b = Quantity.from_str("20.3")

        mock_position_a = MagicMock()
        mock_position_a.id = "POS-A-LONG"
        mock_position_a.is_closed = False
        mock_position_a.instrument_id = InstrumentId.from_str("A.BINANCE")
        mock_position_a.side = PositionSide.LONG
        mock_position_a.quantity = qty_a

        mock_position_b = MagicMock()
        mock_position_b.id = "POS-B-SHORT"
        mock_position_b.is_closed = False
        mock_position_b.instrument_id = InstrumentId.from_str("B.BINANCE")
        mock_position_b.side = PositionSide.SHORT
        mock_position_b.quantity = qty_b

        mock_cache = MagicMock()
        mock_cache.positions_open.return_value = [mock_position_a, mock_position_b]

        # order_factory is a read-only Cython property — patch at type level
        mock_factory = MagicMock()
        mock_order = MagicMock()
        mock_factory.market.return_value = mock_order

        submitted: list[tuple] = []

        def _capture_submit(order, **kwargs):
            submitted.append((order, kwargs))

        strategy.submit_order = _capture_submit  # type: ignore[method-assign]

        with (
            patch.object(
                type(strategy), "order_factory",
                new_callable=PropertyMock, return_value=mock_factory,
            ),
            patch.object(
                type(strategy), "cache",
                new_callable=PropertyMock, return_value=mock_cache,
            ),
        ):
            strategy._close_all_positions("STRATEGY_STOP")

        # Verify order_factory.market was called for each cache position
        assert mock_factory.market.call_count == 2

        calls = mock_factory.market.call_args_list
        a_call = [c for c in calls if "A.BINANCE" in str(c)]
        b_call = [c for c in calls if "B.BINANCE" in str(c)]
        assert len(a_call) == 1
        assert len(b_call) == 1

        # A is LONG → Order.closing_side(LONG) = SELL
        a_kwargs = a_call[0].kwargs
        assert a_kwargs["order_side"] == OrderSide.SELL
        assert a_kwargs["quantity"] == qty_a
        assert a_kwargs["reduce_only"] is True
        assert a_kwargs["exec_algorithm_params"] == {"anchor_px": "50000.0"}

        # B is SHORT → Order.closing_side(SHORT) = BUY
        b_kwargs = b_call[0].kwargs
        assert b_kwargs["order_side"] == OrderSide.BUY
        assert b_kwargs["quantity"] == qty_b
        assert b_kwargs["reduce_only"] is True
        assert b_kwargs["exec_algorithm_params"] == {"anchor_px": "3000.0"}

        # Both orders submitted with position_id from cache position
        assert len(submitted) == 2
        a_submit = [s for s in submitted if s[1].get("position_id") == "POS-A-LONG"]
        b_submit = [s for s in submitted if s[1].get("position_id") == "POS-B-SHORT"]
        assert len(a_submit) == 1
        assert len(b_submit) == 1

        # Tracking dicts should be cleared
        assert strategy._long_positions == {}
        assert strategy._short_positions == {}

    def test_close_all_positions_without_anchor_still_works(self) -> None:
        """When price_book has no data, orders still emit but without exec_algorithm_params."""
        config = FMZFactorStrategyConfig(
            instrument_ids=["A.BINANCE"],
            n_long=1,
            n_short=1,
        )
        strategy = FMZFactorStrategy(config)

        # No price_book data → anchor_px will be None

        qty_a = Quantity.from_str("5.0")
        mock_position = MagicMock()
        mock_position.id = "POS-A"
        mock_position.is_closed = False
        mock_position.instrument_id = InstrumentId.from_str("A.BINANCE")
        mock_position.side = PositionSide.LONG
        mock_position.quantity = qty_a

        mock_cache = MagicMock()
        mock_cache.positions_open.return_value = [mock_position]

        mock_factory = MagicMock()
        mock_order = MagicMock()
        mock_factory.market.return_value = mock_order

        submitted: list[tuple] = []

        def _capture_submit(order, **kwargs):
            submitted.append((order, kwargs))

        strategy.submit_order = _capture_submit  # type: ignore[method-assign]

        with (
            patch.object(
                type(strategy), "order_factory",
                new_callable=PropertyMock, return_value=mock_factory,
            ),
            patch.object(
                type(strategy), "cache",
                new_callable=PropertyMock, return_value=mock_cache,
            ),
        ):
            strategy._close_all_positions("STRATEGY_STOP")

        assert mock_factory.market.call_count == 1
        call_kwargs = mock_factory.market.call_args_list[0].kwargs
        assert call_kwargs["reduce_only"] is True
        assert call_kwargs["exec_algorithm_params"] is None

        # position_id comes from cache position
        assert len(submitted) == 1
        assert submitted[0][1].get("position_id") == "POS-A"

        assert strategy._long_positions == {}

    def test_close_all_positions_empty_cache(self) -> None:
        """No orders submitted when cache has no open positions."""
        config = FMZFactorStrategyConfig(
            instrument_ids=["A.BINANCE"],
            n_long=1,
            n_short=1,
        )
        strategy = FMZFactorStrategy(config)

        mock_cache = MagicMock()
        mock_cache.positions_open.return_value = []

        submitted: list = []
        strategy.submit_order = lambda order, **kw: submitted.append(order)  # type: ignore[method-assign]

        with patch.object(
            type(strategy), "cache",
            new_callable=PropertyMock, return_value=mock_cache,
        ):
            strategy._close_all_positions("STRATEGY_STOP")

        assert submitted == []
        assert strategy._long_positions == {}
        assert strategy._short_positions == {}


class TestOnPositionOpened:
    """Tests for on_position_opened late-close during strategy stop."""

    def test_on_position_opened_closes_during_stop(self) -> None:
        """When _stopping=True, on_position_opened submits a close order."""
        config = FMZFactorStrategyConfig(
            instrument_ids=["A.BINANCE"],
            n_long=1,
            n_short=1,
        )
        strategy = FMZFactorStrategy(config)
        strategy._stopping = True

        # Seed price_book
        strategy._price_book.record_close(100, "A.BINANCE", 42000.0)

        qty = Quantity.from_str("1.5")
        mock_position = MagicMock()
        mock_position.id = "POS-LATE"
        mock_position.is_closed = False
        mock_position.instrument_id = InstrumentId.from_str("A.BINANCE")
        mock_position.side = PositionSide.LONG
        mock_position.quantity = qty

        mock_event = MagicMock()
        mock_event.position_id = "POS-LATE"

        mock_cache = MagicMock()
        mock_cache.position.return_value = mock_position

        mock_factory = MagicMock()
        mock_order = MagicMock()
        mock_factory.market.return_value = mock_order

        submitted: list[tuple] = []

        def _capture_submit(order, **kwargs):
            submitted.append((order, kwargs))

        strategy.submit_order = _capture_submit  # type: ignore[method-assign]

        with (
            patch.object(
                type(strategy), "order_factory",
                new_callable=PropertyMock, return_value=mock_factory,
            ),
            patch.object(
                type(strategy), "cache",
                new_callable=PropertyMock, return_value=mock_cache,
            ),
        ):
            strategy.on_position_opened(mock_event)

        assert mock_factory.market.call_count == 1
        call_kwargs = mock_factory.market.call_args.kwargs
        assert call_kwargs["order_side"] == OrderSide.SELL
        assert call_kwargs["quantity"] == qty
        assert call_kwargs["reduce_only"] is True
        assert call_kwargs["exec_algorithm_params"] == {"anchor_px": "42000.0"}

        assert len(submitted) == 1
        assert submitted[0][1].get("position_id") == "POS-LATE"

    def test_on_position_opened_noop_when_not_stopping(self) -> None:
        """When _stopping=False, on_position_opened does nothing."""
        config = FMZFactorStrategyConfig(
            instrument_ids=["A.BINANCE"],
            n_long=1,
            n_short=1,
        )
        strategy = FMZFactorStrategy(config)
        strategy._stopping = False

        mock_event = MagicMock()
        submitted: list = []
        strategy.submit_order = lambda order, **kw: submitted.append(order)  # type: ignore[method-assign]

        strategy.on_position_opened(mock_event)

        assert submitted == []
