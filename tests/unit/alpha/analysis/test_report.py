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
        assert "t=nan" not in content
        assert "p=nan" not in content
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
