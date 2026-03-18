"""Tests for AnalysisReportGenerator."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from nautilus_quants.alpha.analysis.config import AlphaAnalysisConfig
from nautilus_quants.alpha.analysis.report import (
    AnalysisReportGenerator,
    CHART_REGISTRY,
    compute_ic_summary,
)


class TestChartRegistry:
    """Test chart registry completeness."""

    def test_registry_has_expected_charts(self):
        expected = {
            "quantile_returns_bar",
            "quantile_returns_violin",
            "cumulative_returns",
            "cumulative_returns_long_short",
            "quantile_spread",
            "returns_table",
            "ic_time_series",
            "ic_histogram",
            "ic_qq",
            "monthly_ic_heatmap",
            "turnover",
            "turnover_table",
            "factor_rank_autocorrelation",
            "event_study",
            "events_distribution",
            "quantile_statistics_table",
        }
        assert expected == set(CHART_REGISTRY.keys())

    def test_registry_values_are_callable(self):
        for name, func in CHART_REGISTRY.items():
            assert callable(func), f"{name} is not callable"


class TestAnalysisReportGenerator:
    """Test AnalysisReportGenerator."""

    def _make_config(self, output_dir: str, charts: list[str] | None = None) -> AlphaAnalysisConfig:
        return AlphaAnalysisConfig(
            catalog_path="/test",
            factor_config_path="test.yaml",
            instrument_ids=["A", "B"],
            charts=charts or ["ic_time_series"],
            output_dir=output_dir,
        )

    def _make_factor_data(self) -> pd.DataFrame:
        """Create minimal alphalens factor_data DataFrame."""
        dates = pd.date_range("2024-01-01", periods=10, freq="h")
        assets = ["A", "B", "C"]
        tuples = [(d, a) for d in dates for a in assets]
        idx = pd.MultiIndex.from_tuples(tuples, names=["date", "asset"])
        n = len(tuples)
        return pd.DataFrame(
            {
                "1": np.random.randn(n) * 0.01,
                "4": np.random.randn(n) * 0.01,
                "factor": np.random.randn(n),
                "factor_quantile": np.random.choice([1, 2, 3], n),
            },
            index=idx,
        )

    def test_generate_creates_output_dir(self, tmp_path: Path):
        config = self._make_config(str(tmp_path / "reports"))
        generator = AnalysisReportGenerator(config)

        output_dir = tmp_path / "reports" / "test_run"
        output_dir.mkdir(parents=True, exist_ok=True)

        factor_data = self._make_factor_data()
        ic = pd.DataFrame(
            {"1": np.random.randn(10) * 0.05, "4": np.random.randn(10) * 0.05},
            index=pd.date_range("2024-01-01", periods=10, freq="h"),
        )

        # Generate for one factor
        with patch("nautilus_quants.alpha.analysis.report.plt") as mock_plt:
            mock_fig = MagicMock()
            mock_plt.gcf.return_value = mock_fig
            generator.generate_factor_charts(
                factor_name="volume",
                factor_data=factor_data,
                output_dir=output_dir,
            )

        # Check factor subdirectory was created
        factor_dir = output_dir / "volume"
        assert factor_dir.exists()

    def test_generate_summary(self, tmp_path: Path):
        config = self._make_config(str(tmp_path / "reports"))
        generator = AnalysisReportGenerator(config)

        output_dir = tmp_path / "reports" / "test_run"
        output_dir.mkdir(parents=True, exist_ok=True)

        ic_results = {
            "volume": pd.DataFrame(
                {"1": [0.05, 0.03], "4": [0.04, 0.02]},
                index=pd.date_range("2024-01-01", periods=2, freq="h"),
            ),
        }

        summary_path = generator.generate_summary(ic_results, output_dir)
        assert summary_path.exists()
        content = summary_path.read_text()
        assert "volume" in content

    def test_print_summary_table(self, tmp_path: Path, capsys):
        config = self._make_config(str(tmp_path))
        generator = AnalysisReportGenerator(config)

        ic_results = {
            "volume": pd.DataFrame(
                {"1": [0.05, 0.03], "4": [0.04, 0.02]},
                index=pd.date_range("2024-01-01", periods=2, freq="h"),
            ),
        }

        generator.print_summary_table(ic_results)
        captured = capsys.readouterr()
        assert "volume" in captured.out

    def test_generate_summary_with_nan_ic(self, tmp_path: Path):
        """IC DataFrames with NaN should still produce valid t-stat/p-value."""
        config = self._make_config(str(tmp_path / "reports"))
        generator = AnalysisReportGenerator(config)

        output_dir = tmp_path / "reports" / "test_run"
        output_dir.mkdir(parents=True, exist_ok=True)

        # IC with NaN values (simulates momentum factor with warmup gaps)
        ic_results = {
            "momentum_3h": pd.DataFrame(
                {"1h": [0.05, np.nan, 0.03, -0.02, 0.01]},
                index=pd.date_range("2024-01-01", periods=5, freq="h"),
            ),
        }

        summary_path = generator.generate_summary(ic_results, output_dir)
        content = summary_path.read_text()

        assert "momentum_3h" in content
        assert "t(NW)=nan" not in content
        assert "p(NW)=nan" not in content
        assert "N=" in content

    def test_print_summary_table_with_nan_ic(self, tmp_path: Path, capsys):
        """print_summary_table should handle NaN IC values."""
        config = self._make_config(str(tmp_path))
        generator = AnalysisReportGenerator(config)

        ic_results = {
            "momentum_3h": pd.DataFrame(
                {"1h": [0.05, np.nan, 0.03, -0.02, 0.01]},
                index=pd.date_range("2024-01-01", periods=5, freq="h"),
            ),
        }

        generator.print_summary_table(ic_results)
        captured = capsys.readouterr()
        assert "momentum_3h" in captured.out

    def test_generate_summary_uses_nw_tstat(self, tmp_path: Path):
        """Summary should output Newey-West t-stat, not raw t-stat."""
        config = self._make_config(str(tmp_path / "reports"))
        generator = AnalysisReportGenerator(config)

        output_dir = tmp_path / "reports" / "test_run"
        output_dir.mkdir(parents=True, exist_ok=True)

        np.random.seed(42)
        ic_results = {
            "test_factor": pd.DataFrame(
                {"1h": np.random.randn(100) * 0.05},
                index=pd.date_range("2024-01-01", periods=100, freq="h"),
            ),
        }

        summary_path = generator.generate_summary(ic_results, output_dir)
        content = summary_path.read_text()

        # Should contain NW t-stat, not raw t-stat
        assert "t(NW)=" in content
        assert "p(NW)=" in content

    def test_generate_summary_n_value_consistency(self, tmp_path: Path):
        """N in summary should match non-NaN count of IC values."""
        config = self._make_config(str(tmp_path / "reports"))
        generator = AnalysisReportGenerator(config)

        output_dir = tmp_path / "reports" / "test_run"
        output_dir.mkdir(parents=True, exist_ok=True)

        ic_results = {
            "factor_a": pd.DataFrame(
                {"1h": [0.05, np.nan, 0.03, -0.02, 0.01, np.nan, -0.04]},
                index=pd.date_range("2024-01-01", periods=7, freq="h"),
            ),
        }

        summary_path = generator.generate_summary(ic_results, output_dir)
        content = summary_path.read_text()

        # N should be 5 (7 total - 2 NaN)
        assert "N=5" in content
        # N_eff should also be present
        assert "N_eff=" in content


class TestComputeIcSummary:
    """Test compute_ic_summary and cross-validate with alphalens."""

    def test_nan_handling(self):
        """compute_ic_summary should produce valid stats even with NaN."""
        ic_df = pd.DataFrame(
            {"1h": [0.05, np.nan, 0.03, -0.02, 0.01, np.nan, -0.04]},
            index=pd.date_range("2024-01-01", periods=7, freq="h"),
        )
        table = compute_ic_summary(ic_df)
        assert not np.isnan(table.loc["1h", "t-stat(IC)"])
        assert not np.isnan(table.loc["1h", "p-value(IC)"])
        assert not np.isnan(table.loc["1h", "t-stat(NW)"])
        assert not np.isnan(table.loc["1h", "p-value(NW)"])
        assert table.loc["1h", "N"] == 5  # 7 - 2 NaN
        assert "NaN Count" in table.columns
        assert table.loc["1h", "NaN Count"] == 2

    def test_no_nan_no_extra_column(self):
        """When no NaN, NaN Count column should not appear."""
        ic_df = pd.DataFrame(
            {"1h": [0.05, 0.03, -0.02, 0.01]},
            index=pd.date_range("2024-01-01", periods=4, freq="h"),
        )
        table = compute_ic_summary(ic_df)
        assert "NaN Count" not in table.columns

    def test_per_column_nan_independence(self):
        """Each period column should independently drop NaN (Issue 2).

        When 1h has 10 valid values and 4h has only 8 (2 trailing NaN),
        N for 1h should still be 10, not truncated to 8.
        """
        ic_df = pd.DataFrame(
            {
                "1h": [0.05, 0.03, -0.02, 0.01, 0.04, -0.01, 0.02, 0.03, -0.03, 0.01],
                "4h": [0.04, 0.02, -0.01, 0.03, 0.05, -0.02, 0.01, 0.02, np.nan, np.nan],
            },
            index=pd.date_range("2024-01-01", periods=10, freq="h"),
        )
        table = compute_ic_summary(ic_df)
        assert table.loc["1h", "N"] == 10
        assert table.loc["4h", "N"] == 8

    def test_zero_std_produces_nan_icir(self):
        """Risk-Adjusted IC should be NaN when IC std is zero (Issue 2 div-by-zero)."""
        ic_df = pd.DataFrame(
            {"1h": [0.05, 0.05, 0.05, 0.05]},
            index=pd.date_range("2024-01-01", periods=4, freq="h"),
        )
        table = compute_ic_summary(ic_df)
        assert np.isnan(table.loc["1h", "Risk-Adjusted IC"])

    def test_includes_nw_columns(self):
        """compute_ic_summary should have t-stat(NW), p-value(NW), and N_eff columns."""
        np.random.seed(42)
        ic_df = pd.DataFrame(
            {"1h": np.random.randn(50) * 0.05, "4h": np.random.randn(50) * 0.08},
            index=pd.date_range("2024-01-01", periods=50, freq="h"),
        )
        table = compute_ic_summary(ic_df)
        assert "t-stat(NW)" in table.columns
        assert "p-value(NW)" in table.columns
        assert "N_eff" in table.columns
        # NW values should be finite numbers
        for period in ["1h", "4h"]:
            assert np.isfinite(table.loc[period, "t-stat(NW)"])
            assert np.isfinite(table.loc[period, "p-value(NW)"])
            assert table.loc[period, "N_eff"] > 0

    def test_cross_validate_with_alphalens_no_nan(self):
        """When no NaN, compute_ic_summary should match alphalens exactly."""
        import alphalens.plotting as plotting

        np.random.seed(42)
        ic_df = pd.DataFrame(
            {
                "1h": np.random.randn(100) * 0.05,
                "4h": np.random.randn(100) * 0.08,
            },
            index=pd.date_range("2024-01-01", periods=100, freq="h"),
        )

        our_table = compute_ic_summary(ic_df)
        al_table = plotting.plot_information_table(ic_df, return_df=True)

        for col in ["IC Mean", "IC Std.", "Risk-Adjusted IC", "t-stat(IC)", "p-value(IC)"]:
            np.testing.assert_allclose(
                our_table[col].values,
                al_table[col].values,
                rtol=1e-10,
                err_msg=f"Mismatch in {col}",
            )


class TestNeweyWestTstat:
    """Test _newey_west_tstat helper function."""

    def test_reduces_significance_for_autocorrelated_data(self):
        """NW t-stat should be smaller than raw t-stat for autocorrelated IC."""
        from nautilus_quants.alpha.analysis.report import _newey_west_tstat

        np.random.seed(42)
        # Generate highly autocorrelated series (AR(1) with rho=0.9)
        n = 1000
        x = np.zeros(n)
        x[0] = np.random.randn()
        for i in range(1, n):
            x[i] = 0.9 * x[i - 1] + np.random.randn() * 0.1
        # Add a small positive mean
        x += 0.02

        series = pd.Series(x)
        nw_t, nw_p, n_eff = _newey_west_tstat(series)

        # Raw t-stat
        from scipy import stats as scipy_stats
        raw_t, raw_p = scipy_stats.ttest_1samp(x, 0)

        # NW should reduce significance (smaller |t|)
        assert abs(nw_t) < abs(raw_t), (
            f"NW t-stat ({nw_t:.2f}) should be smaller than raw ({raw_t:.2f}) "
            "for autocorrelated data"
        )
        # N_eff should be smaller than N
        assert 0 < n_eff < n, f"N_eff={n_eff} should be between 0 and {n}"

    def test_similar_to_raw_for_iid_data(self):
        """NW t-stat should be close to raw t-stat for i.i.d. data."""
        from nautilus_quants.alpha.analysis.report import _newey_west_tstat

        np.random.seed(42)
        x = np.random.randn(500) * 0.05
        series = pd.Series(x)
        nw_t, _, n_eff = _newey_west_tstat(series)

        from scipy import stats as scipy_stats
        raw_t, _ = scipy_stats.ttest_1samp(x, 0)

        # Should be roughly similar (within 30%)
        assert abs(nw_t - raw_t) / max(abs(raw_t), 1e-10) < 0.3, (
            f"NW ({nw_t:.2f}) and raw ({raw_t:.2f}) should be similar for i.i.d."
        )

    def test_insufficient_data(self):
        """Should return NaN for data with fewer than 2 observations."""
        from nautilus_quants.alpha.analysis.report import _newey_west_tstat

        assert np.isnan(_newey_west_tstat(pd.Series([1.0]))[0])
        assert np.isnan(_newey_west_tstat(pd.Series(dtype=float))[0])

    def test_hourly_data_uses_larger_lag(self):
        """For hourly data, freq-aware lag should dominate the classic formula."""
        from nautilus_quants.alpha.analysis.report import _newey_west_tstat

        np.random.seed(42)
        n = 5000  # ~208 days of hourly data
        # AR(1) with moderate autocorrelation
        x = np.zeros(n)
        x[0] = np.random.randn()
        for i in range(1, n):
            x[i] = 0.8 * x[i - 1] + np.random.randn() * 0.1
        x += 0.02

        hourly = pd.Series(x, index=pd.date_range("2024-01-01", periods=n, freq="h"))
        daily = pd.Series(x, index=pd.date_range("2024-01-01", periods=n, freq="D"))

        nw_t_hourly, _, _ = _newey_west_tstat(hourly)
        nw_t_daily, _, _ = _newey_west_tstat(daily)

        # Hourly should produce a *smaller* |t| because it uses more lags
        assert abs(nw_t_hourly) < abs(nw_t_daily), (
            f"Hourly t={nw_t_hourly:.2f} should be smaller than daily t={nw_t_daily:.2f} "
            "due to frequency-aware lag selection"
        )

    def test_returns_n_eff(self):
        """_newey_west_tstat should return a 3-tuple with n_eff."""
        from nautilus_quants.alpha.analysis.report import _newey_west_tstat

        np.random.seed(42)
        series = pd.Series(np.random.randn(100) * 0.05)
        result = _newey_west_tstat(series)
        assert len(result) == 3
        t_stat, p_value, n_eff = result
        assert isinstance(n_eff, int)
        assert n_eff > 0


class TestInferBarsPerDay:
    """Test _infer_bars_per_day helper."""

    def test_hourly(self):
        from nautilus_quants.alpha.analysis.report import _infer_bars_per_day

        idx = pd.date_range("2024-01-01", periods=100, freq="h")
        assert _infer_bars_per_day(idx) == 24

    def test_daily(self):
        from nautilus_quants.alpha.analysis.report import _infer_bars_per_day

        idx = pd.date_range("2024-01-01", periods=100, freq="D")
        assert _infer_bars_per_day(idx) == 1

    def test_15min(self):
        from nautilus_quants.alpha.analysis.report import _infer_bars_per_day

        idx = pd.date_range("2024-01-01", periods=100, freq="15min")
        assert _infer_bars_per_day(idx) == 96

    def test_non_datetime_index(self):
        from nautilus_quants.alpha.analysis.report import _infer_bars_per_day

        idx = pd.RangeIndex(100)
        assert _infer_bars_per_day(idx) == 1
