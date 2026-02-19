"""Unit tests for EquityGuardController."""

from unittest.mock import MagicMock, patch

import pytest

from nautilus_quants.controllers.equity_guard import (
    EquityGuardController,
    EquityGuardControllerConfig,
)


class TestEquityGuardControllerConfig:
    """Tests for EquityGuardControllerConfig."""

    def test_default_values(self) -> None:
        config = EquityGuardControllerConfig()
        assert config.interval == "1h"
        assert config.venue_name == "SIM"
        assert config.currency == "USD"
        assert config.min_equity_ratio == 0.1
        assert config.guarded_strategy_ids == []

    def test_custom_values(self) -> None:
        config = EquityGuardControllerConfig(
            interval="4h",
            venue_name="BINANCE",
            currency="USDT",
            min_equity_ratio=0.2,
            guarded_strategy_ids=["Strategy-001"],
        )
        assert config.interval == "4h"
        assert config.venue_name == "BINANCE"
        assert config.currency == "USDT"
        assert config.min_equity_ratio == 0.2
        assert config.guarded_strategy_ids == ["Strategy-001"]

    def test_config_is_frozen(self) -> None:
        config = EquityGuardControllerConfig()
        with pytest.raises(Exception):
            config.interval = "4h"  # type: ignore[misc]


class TestEquityGuardControllerOnCheck:
    """Tests for EquityGuardController._on_check behavior.

    Since Controller/Actor has Cython-level readonly 'config' attribute,
    we test _on_check and _halt_guarded_strategies as unbound methods
    with a mock 'self' object that mimics the controller's internal state.
    """

    def _make_mock_controller(
        self,
        min_equity_ratio: float = 0.1,
        guarded_strategy_ids: list[str] | None = None,
        initial_balance: float | None = 10000.0,
        halted: bool = False,
    ) -> MagicMock:
        """Create a mock that behaves like EquityGuardController."""
        mock = MagicMock(spec=EquityGuardController)
        mock.config = EquityGuardControllerConfig(
            interval="1h",
            venue_name="SIM",
            currency="USD",
            min_equity_ratio=min_equity_ratio,
            guarded_strategy_ids=guarded_strategy_ids or [],
        )
        mock._venue = MagicMock()
        mock._currency = MagicMock()
        mock._initial_balance = initial_balance
        mock._halted = halted
        mock._trader = MagicMock()
        mock._trader.strategies.return_value = []
        mock.portfolio = MagicMock()
        mock.log = MagicMock()
        mock.stop_strategy = MagicMock()
        return mock

    def test_on_check_no_halt_when_equity_healthy(self) -> None:
        """Test _on_check does NOT halt when equity is above threshold."""
        mock = self._make_mock_controller(min_equity_ratio=0.1)

        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
            return_value=5000.0,
        ):
            EquityGuardController._on_check(mock, MagicMock())

        assert not mock._halted
        mock.stop_strategy.assert_not_called()

    def test_on_check_halts_when_equity_below_threshold(self) -> None:
        """Test _on_check halts strategies when equity < min_ratio * initial."""
        mock = self._make_mock_controller(min_equity_ratio=0.1)

        mock_strategy = MagicMock()
        mock_strategy.id = "Strategy-001"
        mock_strategy.is_running = True
        mock._trader.strategies.return_value = [mock_strategy]

        # Use real _halt_guarded_strategies so _halted flag is set
        mock._halt_guarded_strategies = lambda equity, ratio: (
            EquityGuardController._halt_guarded_strategies(mock, equity, ratio)
        )

        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
            return_value=500.0,  # 5% < 10% threshold
        ):
            EquityGuardController._on_check(mock, MagicMock())

        assert mock._halted
        mock.stop_strategy.assert_called_once_with(mock_strategy)

    def test_on_check_skips_when_already_halted(self) -> None:
        """Test _on_check does nothing when already halted."""
        mock = self._make_mock_controller(halted=True)

        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
        ) as mock_equity:
            EquityGuardController._on_check(mock, MagicMock())

        mock_equity.assert_not_called()

    def test_on_check_skips_when_initial_balance_none(self) -> None:
        """Test _on_check safely exits when initial_balance is None."""
        mock = self._make_mock_controller(initial_balance=None)

        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
        ) as mock_equity:
            EquityGuardController._on_check(mock, MagicMock())

        mock_equity.assert_not_called()

    def test_on_check_skips_when_initial_balance_zero(self) -> None:
        """Test _on_check safely exits when initial_balance is zero."""
        mock = self._make_mock_controller(initial_balance=0.0)

        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
        ) as mock_equity:
            EquityGuardController._on_check(mock, MagicMock())

        mock_equity.assert_not_called()

    def test_on_check_skips_when_equity_none(self) -> None:
        """Test _on_check safely exits when compute_mtm_equity returns None."""
        mock = self._make_mock_controller()

        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
            return_value=None,
        ):
            EquityGuardController._on_check(mock, MagicMock())

        assert not mock._halted


class TestEquityGuardControllerHaltStrategies:
    """Tests for _halt_guarded_strategies targeting specific strategies."""

    def _make_mock_controller(
        self,
        guarded_strategy_ids: list[str] | None = None,
    ) -> MagicMock:
        mock = MagicMock(spec=EquityGuardController)
        mock.config = EquityGuardControllerConfig(
            guarded_strategy_ids=guarded_strategy_ids or [],
        )
        mock._halted = False
        mock._trader = MagicMock()
        mock.log = MagicMock()
        mock.stop_strategy = MagicMock()
        return mock

    def test_stops_all_strategies_when_empty_guarded_list(self) -> None:
        """Test all strategies stopped when guarded_strategy_ids is empty."""
        mock = self._make_mock_controller(guarded_strategy_ids=[])

        s1 = MagicMock()
        s1.id = "Strategy-001"
        s1.is_running = True
        s2 = MagicMock()
        s2.id = "Strategy-002"
        s2.is_running = True
        mock._trader.strategies.return_value = [s1, s2]

        EquityGuardController._halt_guarded_strategies(mock, equity=500.0, ratio=0.05)

        assert mock._halted
        assert mock.stop_strategy.call_count == 2

    def test_stops_only_guarded_strategies(self) -> None:
        """Test only specified strategies are stopped."""
        mock = self._make_mock_controller(guarded_strategy_ids=["Strategy-001"])

        s1 = MagicMock()
        s1.id = "Strategy-001"
        s1.__str__ = lambda self: "Strategy-001"
        s1.is_running = True

        s2 = MagicMock()
        s2.id = "Strategy-002"
        s2.__str__ = lambda self: "Strategy-002"
        s2.is_running = True

        mock._trader.strategies.return_value = [s1, s2]

        EquityGuardController._halt_guarded_strategies(mock, equity=500.0, ratio=0.05)

        assert mock._halted
        mock.stop_strategy.assert_called_once_with(s1)

    def test_skips_already_stopped_strategies(self) -> None:
        """Test does not stop strategies that are already not running."""
        mock = self._make_mock_controller(guarded_strategy_ids=[])

        s1 = MagicMock()
        s1.id = "Strategy-001"
        s1.is_running = False  # Already stopped
        mock._trader.strategies.return_value = [s1]

        EquityGuardController._halt_guarded_strategies(mock, equity=500.0, ratio=0.05)

        assert mock._halted
        mock.stop_strategy.assert_not_called()
