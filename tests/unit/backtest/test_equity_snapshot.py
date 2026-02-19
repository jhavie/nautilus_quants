"""Unit tests for EquitySnapshotActor and reports integration."""

import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from nautilus_quants.actors.equity_snapshot import (
    EquitySnapshotActor,
    EquitySnapshotActorConfig,
)
from nautilus_quants.backtest.config import ReportConfig, TearsheetConfig
from nautilus_quants.backtest.protocols import EQUITY_SNAPSHOTS_CACHE_KEY
from nautilus_quants.backtest.reports import ReportGenerator


class TestEquitySnapshotActorConfig:
    """Tests for EquitySnapshotActorConfig."""

    def test_default_values(self) -> None:
        config = EquitySnapshotActorConfig()
        assert config.interval == "8h"
        assert config.venue_name == "SIM"
        assert config.currency == "USD"

    def test_custom_values(self) -> None:
        config = EquitySnapshotActorConfig(
            interval="4h",
            venue_name="BINANCE",
            currency="USDT",
        )
        assert config.interval == "4h"
        assert config.venue_name == "BINANCE"
        assert config.currency == "USDT"

    def test_config_is_frozen(self) -> None:
        config = EquitySnapshotActorConfig()
        with pytest.raises(Exception):
            config.interval = "1h"  # type: ignore[misc]


class TestParseIntervalToTimedelta:
    """Tests for parse_interval_to_timedelta utility."""

    def test_parse_hours(self) -> None:
        from datetime import timedelta

        from nautilus_quants.backtest.utils.bar_spec import parse_interval_to_timedelta

        assert parse_interval_to_timedelta("8h") == timedelta(hours=8)

    def test_parse_minutes(self) -> None:
        from datetime import timedelta

        from nautilus_quants.backtest.utils.bar_spec import parse_interval_to_timedelta

        assert parse_interval_to_timedelta("30m") == timedelta(minutes=30)

    def test_parse_days(self) -> None:
        from datetime import timedelta

        from nautilus_quants.backtest.utils.bar_spec import parse_interval_to_timedelta

        assert parse_interval_to_timedelta("1d") == timedelta(days=1)

    def test_parse_invalid_raises(self) -> None:
        from nautilus_quants.backtest.utils.bar_spec import parse_interval_to_timedelta

        with pytest.raises(ValueError):
            parse_interval_to_timedelta("invalid")


class TestReportGeneratorEquityFromActorSnapshots:
    """Tests for ReportGenerator._build_equity_from_actor_snapshots."""

    @pytest.fixture
    def mock_engine(self) -> MagicMock:
        engine = MagicMock()
        engine.cache.orders.return_value = []
        engine.cache.positions.return_value = []
        engine.cache.accounts.return_value = []
        engine.portfolio.analyzer.get_performance_stats_pnls.return_value = {}
        engine.portfolio.analyzer.get_performance_stats_returns.return_value = {}
        engine.portfolio.analyzer.get_performance_stats_general.return_value = {}
        return engine

    @pytest.fixture
    def report_config(self) -> ReportConfig:
        return ReportConfig(
            output_dir="logs/backtest_runs",
            formats=["csv", "html"],
            tearsheet=TearsheetConfig(enabled=True),
        )

    def test_reads_actor_data_from_cache(
        self, mock_engine: MagicMock, tmp_path: Path, report_config: ReportConfig
    ) -> None:
        """Test _build_equity_from_actor_snapshots reads from cache."""
        # Simulate actor data in cache
        equity_points = [
            (1_000_000_000_000_000_000, 10000.0),  # ts_ns, equity
            (1_000_028_800_000_000_000, 10100.0),   # +8h
            (1_000_057_600_000_000_000, 10200.0),   # +16h
        ]
        mock_engine.cache.get.return_value = pickle.dumps(equity_points)

        generator = ReportGenerator(
            engine=mock_engine,
            output_dir=tmp_path,
            config=report_config,
        )

        result = generator._build_equity_from_actor_snapshots()

        assert result is not None
        assert len(result) == 3
        assert result.iloc[0] == 10000.0
        assert result.iloc[2] == 10200.0
        mock_engine.cache.get.assert_called_with(EQUITY_SNAPSHOTS_CACHE_KEY)

    def test_returns_none_when_no_cache_data(
        self, mock_engine: MagicMock, tmp_path: Path, report_config: ReportConfig
    ) -> None:
        """Test returns None when cache has no actor data."""
        mock_engine.cache.get.return_value = None

        generator = ReportGenerator(
            engine=mock_engine,
            output_dir=tmp_path,
            config=report_config,
        )

        result = generator._build_equity_from_actor_snapshots()
        assert result is None

    def test_returns_none_on_empty_equity_points(
        self, mock_engine: MagicMock, tmp_path: Path, report_config: ReportConfig
    ) -> None:
        """Test returns None when equity points list is empty."""
        mock_engine.cache.get.return_value = pickle.dumps([])

        generator = ReportGenerator(
            engine=mock_engine,
            output_dir=tmp_path,
            config=report_config,
        )

        result = generator._build_equity_from_actor_snapshots()
        assert result is None

    def test_returns_none_on_unpicklable_data(
        self, mock_engine: MagicMock, tmp_path: Path, report_config: ReportConfig
    ) -> None:
        """Test returns None when cache data is corrupted."""
        mock_engine.cache.get.return_value = b"not-pickle-data"

        generator = ReportGenerator(
            engine=mock_engine,
            output_dir=tmp_path,
            config=report_config,
        )

        result = generator._build_equity_from_actor_snapshots()
        assert result is None

    def test_returns_none_on_cache_exception(
        self, mock_engine: MagicMock, tmp_path: Path, report_config: ReportConfig
    ) -> None:
        """Test returns None when cache.get raises exception."""
        mock_engine.cache.get.side_effect = Exception("Cache error")

        generator = ReportGenerator(
            engine=mock_engine,
            output_dir=tmp_path,
            config=report_config,
        )

        result = generator._build_equity_from_actor_snapshots()
        assert result is None


class TestReportGeneratorGetReturnsSeries:
    """Tests for ReportGenerator._get_returns_series with actor data."""

    @pytest.fixture
    def mock_engine(self) -> MagicMock:
        engine = MagicMock()
        engine.cache.orders.return_value = []
        engine.cache.positions.return_value = []
        engine.cache.accounts.return_value = []
        engine.portfolio.analyzer.get_performance_stats_pnls.return_value = {}
        engine.portfolio.analyzer.get_performance_stats_returns.return_value = {}
        engine.portfolio.analyzer.get_performance_stats_general.return_value = {}
        return engine

    @pytest.fixture
    def report_config(self) -> ReportConfig:
        return ReportConfig(
            output_dir="logs/backtest_runs",
            formats=["csv"],
        )

    def test_uses_actor_data_when_available(
        self, mock_engine: MagicMock, tmp_path: Path, report_config: ReportConfig
    ) -> None:
        """Test _get_returns_series prefers actor data over realized equity."""
        # Create actor equity data spanning 3 days
        base_ns = pd.Timestamp("2024-01-01").value
        day_ns = 86400 * 1_000_000_000
        equity_points = [
            (base_ns, 10000.0),
            (base_ns + 8 * 3600 * 1_000_000_000, 10050.0),
            (base_ns + day_ns, 10100.0),
            (base_ns + day_ns + 8 * 3600 * 1_000_000_000, 10150.0),
            (base_ns + 2 * day_ns, 10200.0),
        ]
        mock_engine.cache.get.return_value = pickle.dumps(equity_points)

        generator = ReportGenerator(
            engine=mock_engine,
            output_dir=tmp_path,
            config=report_config,
        )

        returns = generator._get_returns_series()

        # Should have 2 daily returns (day 1 → day 2, day 2 → day 3)
        assert not returns.empty
        assert len(returns) == 2
        # Verify returns are positive
        assert all(r > 0 for r in returns)

    def test_falls_back_to_realized_equity(
        self, mock_engine: MagicMock, tmp_path: Path, report_config: ReportConfig
    ) -> None:
        """Test _get_returns_series falls back when no actor data."""
        # No actor data in cache
        mock_engine.cache.get.return_value = None

        # But provide realized account equity via ReportProvider
        mock_account = MagicMock()
        mock_engine.cache.accounts.return_value = [mock_account]

        # Create a simple account report DataFrame
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        account_df = pd.DataFrame(
            {"total": [10000.0, 10100.0, 10200.0]},
            index=dates,
        )

        generator = ReportGenerator(
            engine=mock_engine,
            output_dir=tmp_path,
            config=report_config,
        )

        with patch(
            "nautilus_trader.analysis.ReportProvider.generate_account_report",
            return_value=account_df,
        ):
            returns = generator._get_returns_series()

        assert not returns.empty
        assert len(returns) == 2

    def test_falls_back_when_actor_returns_empty(
        self, mock_engine: MagicMock, tmp_path: Path, report_config: ReportConfig
    ) -> None:
        """Fallback to realized equity when actor snapshots cannot produce returns."""
        # Actor has snapshots, but all points are within one day -> no daily returns.
        base_ns = pd.Timestamp("2024-01-01").value
        equity_points = [
            (base_ns, 10000.0),
            (base_ns + 8 * 3600 * 1_000_000_000, 10050.0),
        ]
        mock_engine.cache.get.return_value = pickle.dumps(equity_points)

        mock_account = MagicMock()
        mock_engine.cache.accounts.return_value = [mock_account]
        account_df = pd.DataFrame(
            {"total": [10000.0, 10100.0, 10200.0]},
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )

        generator = ReportGenerator(
            engine=mock_engine,
            output_dir=tmp_path,
            config=report_config,
        )

        with patch(
            "nautilus_trader.analysis.ReportProvider.generate_account_report",
            return_value=account_df,
        ):
            returns = generator._get_returns_series()

        assert not returns.empty
        assert len(returns) == 2

    def test_returns_empty_when_no_data(
        self, mock_engine: MagicMock, tmp_path: Path, report_config: ReportConfig
    ) -> None:
        """Test returns empty Series when neither actor nor account data."""
        mock_engine.cache.get.return_value = None
        mock_engine.cache.accounts.return_value = []

        generator = ReportGenerator(
            engine=mock_engine,
            output_dir=tmp_path,
            config=report_config,
        )

        returns = generator._get_returns_series()
        assert returns.empty


class TestBuildDailyReturnsFromEquity:
    """Tests for ReportGenerator._build_daily_returns_from_equity."""

    @pytest.fixture
    def generator(self, tmp_path: Path) -> ReportGenerator:
        engine = MagicMock()
        config = ReportConfig(formats=["csv"])
        return ReportGenerator(engine=engine, output_dir=tmp_path, config=config)

    def test_resamples_to_daily(self, generator: ReportGenerator) -> None:
        """Test equity resampled to daily frequency."""
        dates = pd.date_range("2024-01-01", periods=6, freq="8h")
        equity = pd.Series(
            [10000.0, 10050.0, 10100.0, 10150.0, 10200.0, 10250.0],
            index=dates,
        )

        returns = generator._build_daily_returns_from_equity(equity)

        # 6 points over ~2 days → 2 daily points → 1 return
        assert not returns.empty
        assert len(returns) == 1

    def test_empty_input_returns_empty(self, generator: ReportGenerator) -> None:
        returns = generator._build_daily_returns_from_equity(pd.Series(dtype=float))
        assert returns.empty

    def test_single_day_returns_empty(self, generator: ReportGenerator) -> None:
        dates = pd.date_range("2024-01-01", periods=3, freq="1h")
        equity = pd.Series([10000.0, 10050.0, 10100.0], index=dates)

        returns = generator._build_daily_returns_from_equity(equity)
        assert returns.empty

    def test_inf_returns_filtered(self, generator: ReportGenerator) -> None:
        """Test that inf returns from near-zero equity are filtered out."""
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        # Simulate equity dropping to near-zero then "recovering"
        # Day 2→3: 0.01→5000 = +inf-like pct_change
        equity = pd.Series([10000.0, 100.0, 0.01, 5000.0], index=dates)

        returns = generator._build_daily_returns_from_equity(equity)

        # Should contain no inf values
        assert not returns.isin([float("inf"), float("-inf")]).any()

    def test_zero_equity_produces_no_inf(self, generator: ReportGenerator) -> None:
        """Test that equity series with zero values doesn't produce inf returns."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        equity = pd.Series([10000.0, 5000.0, 0.0, 0.0, 3000.0], index=dates)

        returns = generator._build_daily_returns_from_equity(equity)

        # After dropna from resample, and inf filtering, no inf should remain
        assert not returns.isin([float("inf"), float("-inf")]).any()


class TestComputeMtmEquity:
    """Tests for compute_mtm_equity shared utility."""

    def test_returns_balance_plus_unrealized(self) -> None:
        from nautilus_quants.backtest.utils.equity import compute_mtm_equity

        portfolio = MagicMock()
        venue = MagicMock()
        currency = MagicMock()

        mock_balance = MagicMock()
        mock_balance.as_double.return_value = 10000.0

        mock_unrealized = MagicMock()
        mock_unrealized.as_double.return_value = -500.0

        portfolio.account.return_value = MagicMock()
        portfolio.account.return_value.balance_total.return_value = mock_balance
        portfolio.unrealized_pnls.return_value = {currency: mock_unrealized}

        result = compute_mtm_equity(portfolio, venue, currency)
        assert result == 9500.0

    def test_returns_none_when_no_account(self) -> None:
        from nautilus_quants.backtest.utils.equity import compute_mtm_equity

        portfolio = MagicMock()
        portfolio.account.return_value = None

        result = compute_mtm_equity(portfolio, MagicMock(), MagicMock())
        assert result is None

    def test_returns_none_when_no_balance(self) -> None:
        from nautilus_quants.backtest.utils.equity import compute_mtm_equity

        portfolio = MagicMock()
        portfolio.account.return_value = MagicMock()
        portfolio.account.return_value.balance_total.return_value = None

        result = compute_mtm_equity(portfolio, MagicMock(), MagicMock())
        assert result is None

    def test_returns_balance_when_no_unrealized(self) -> None:
        from nautilus_quants.backtest.utils.equity import compute_mtm_equity

        portfolio = MagicMock()
        mock_balance = MagicMock()
        mock_balance.as_double.return_value = 10000.0
        portfolio.account.return_value = MagicMock()
        portfolio.account.return_value.balance_total.return_value = mock_balance
        portfolio.unrealized_pnls.return_value = None

        result = compute_mtm_equity(portfolio, MagicMock(), MagicMock())
        assert result == 10000.0
