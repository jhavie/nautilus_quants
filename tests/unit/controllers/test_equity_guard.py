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
        assert config.cooldown_period == ""

    def test_custom_values(self) -> None:
        config = EquityGuardControllerConfig(
            interval="4h",
            venue_name="BINANCE",
            currency="USDT",
            min_equity_ratio=0.2,
            guarded_strategy_ids=["Strategy-001"],
            cooldown_period="24h",
        )
        assert config.interval == "4h"
        assert config.venue_name == "BINANCE"
        assert config.currency == "USDT"
        assert config.min_equity_ratio == 0.2
        assert config.guarded_strategy_ids == ["Strategy-001"]
        assert config.cooldown_period == "24h"

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


class TestEquityGuardControllerCooldown:
    """Tests for cooldown period: halt → wait → restart cycle."""

    def _make_mock_controller(
        self,
        min_equity_ratio: float = 0.7,
        cooldown_period: str = "24h",
        guarded_strategy_ids: list[str] | None = None,
        initial_balance: float | None = 10000.0,
    ) -> MagicMock:
        """Create a mock with cooldown-related attributes."""
        mock = MagicMock(spec=EquityGuardController)
        mock.config = EquityGuardControllerConfig(
            interval="1h",
            venue_name="OKX",
            currency="USDT",
            min_equity_ratio=min_equity_ratio,
            guarded_strategy_ids=guarded_strategy_ids or [],
            cooldown_period=cooldown_period,
        )
        mock._venue = MagicMock()
        mock._currency = MagicMock()
        mock._initial_balance = initial_balance
        mock._halted = False
        mock._stopped_strategies = []
        mock._trader = MagicMock()
        mock._trader.strategies.return_value = []
        mock.portfolio = MagicMock()
        mock.log = MagicMock()
        mock.stop_strategy = MagicMock()
        mock.start_strategy = MagicMock()
        mock.clock = MagicMock()
        mock.clock.timestamp_ns.return_value = 1_000_000_000_000
        return mock

    def test_halt_sets_cooldown_timer(self) -> None:
        """Test _halt_guarded_strategies sets a one-shot timer when cooldown is configured."""
        mock = self._make_mock_controller(cooldown_period="24h")

        s1 = MagicMock()
        s1.id = "CSStrategy-001"
        s1.is_running = True
        mock._trader.strategies.return_value = [s1]

        EquityGuardController._halt_guarded_strategies(
            mock, equity=6000.0, ratio=0.6,
        )

        assert mock._halted
        mock.clock.set_time_alert_ns.assert_called_once()
        call_kwargs = mock.clock.set_time_alert_ns.call_args
        assert call_kwargs.kwargs["name"] == "equity_guard_cooldown"
        # 24h = 86400 seconds = 86_400_000_000_000 nanoseconds
        expected_ns = 1_000_000_000_000 + 86_400_000_000_000
        assert call_kwargs.kwargs["alert_time_ns"] == expected_ns

    def test_halt_no_timer_when_cooldown_empty(self) -> None:
        """Test _halt_guarded_strategies does NOT set timer when cooldown is empty."""
        mock = self._make_mock_controller(cooldown_period="")

        s1 = MagicMock()
        s1.id = "CSStrategy-001"
        s1.is_running = True
        mock._trader.strategies.return_value = [s1]

        EquityGuardController._halt_guarded_strategies(
            mock, equity=6000.0, ratio=0.6,
        )

        assert mock._halted
        mock.clock.set_time_alert_ns.assert_not_called()

    def test_halt_records_stopped_strategies(self) -> None:
        """Test stopped strategies are recorded for later restart."""
        mock = self._make_mock_controller(cooldown_period="24h")

        s1 = MagicMock()
        s1.id = "CSStrategy-001"
        s1.is_running = True
        s2 = MagicMock()
        s2.id = "Other-001"
        s2.is_running = True
        mock._trader.strategies.return_value = [s1, s2]

        EquityGuardController._halt_guarded_strategies(
            mock, equity=6000.0, ratio=0.6,
        )

        assert len(mock._stopped_strategies) == 2

    def test_cooldown_expired_restarts_strategies(self) -> None:
        """Test _on_cooldown_expired restarts all previously stopped strategies."""
        mock = self._make_mock_controller(cooldown_period="24h")

        s1 = MagicMock()
        s1.id = "CSStrategy-001"
        s2 = MagicMock()
        s2.id = "CSStrategy-002"
        mock._stopped_strategies = [s1, s2]
        mock._halted = True

        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
            return_value=7000.0,
        ):
            EquityGuardController._on_cooldown_expired(mock, MagicMock())

        assert mock.start_strategy.call_count == 2
        mock.start_strategy.assert_any_call(s1)
        mock.start_strategy.assert_any_call(s2)

    def test_cooldown_expired_resets_state(self) -> None:
        """Test _on_cooldown_expired resets _halted and _initial_balance."""
        mock = self._make_mock_controller(cooldown_period="24h")
        mock._halted = True
        mock._initial_balance = 10000.0
        mock._stopped_strategies = [MagicMock()]

        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
            return_value=7000.0,
        ):
            EquityGuardController._on_cooldown_expired(mock, MagicMock())

        assert not mock._halted
        assert mock._initial_balance == 7000.0
        assert mock._stopped_strategies == []

    def test_multiple_cooldown_cycles(self) -> None:
        """Test halt → cooldown → restart → halt again works correctly."""
        mock = self._make_mock_controller(
            cooldown_period="24h", min_equity_ratio=0.7,
        )

        s1 = MagicMock()
        s1.id = "CSStrategy-001"
        s1.is_running = True
        mock._trader.strategies.return_value = [s1]

        # Cycle 1: halt
        EquityGuardController._halt_guarded_strategies(
            mock, equity=6000.0, ratio=0.6,
        )
        assert mock._halted
        assert len(mock._stopped_strategies) == 1

        # Cycle 1: cooldown expired → restart
        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
            return_value=6500.0,
        ):
            EquityGuardController._on_cooldown_expired(mock, MagicMock())
        assert not mock._halted
        assert mock._initial_balance == 6500.0

        # Cycle 2: halt again (equity drops below 70% of new baseline 6500)
        mock._stopped_strategies = []
        s1.is_running = True
        EquityGuardController._halt_guarded_strategies(
            mock, equity=4000.0, ratio=0.615,
        )
        assert mock._halted
        assert mock.clock.set_time_alert_ns.call_count == 2
