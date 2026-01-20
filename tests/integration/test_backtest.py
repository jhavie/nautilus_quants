"""
回测集成测试

需要 Nautilus Catalog 数据才能运行
"""

import pytest
from pathlib import Path

from nautilus_quants.execution.backtest import run_backtest, _get_default_symbols


class TestBacktestHelpers:
    """回测辅助函数测试"""

    def test_get_default_symbols(self):
        """测试默认币种列表"""
        symbols = _get_default_symbols()

        assert len(symbols) == 50
        assert "BTCUSDT" in symbols
        assert "ETHUSDT" in symbols
        assert all(s.endswith("USDT") for s in symbols)


class TestBacktestConfig:
    """回测配置测试"""

    def test_config_loading(self, tmp_path):
        """测试配置加载"""
        # 创建临时配置文件
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        (config_dir / "trading.yaml").write_text(
            """
capital:
  initial: 50000.0
position:
  max_positions: 5
  max_single_pct: 0.20
leverage:
  default: 2.0
"""
        )

        (config_dir / "strategy.yaml").write_text(
            """
universe:
  top_n: 5
  volume_factor:
    period: 7
signal:
  breakout:
    lookback_bars: 30
  trend_filter:
    sma_period: 100
"""
        )

        (config_dir / "risk.yaml").write_text(
            """
stop_loss:
  pct: 0.03
take_profit:
  pct: 0.06
"""
        )

        (config_dir / "backtest.yaml").write_text(
            """
period:
  start_date: "2024-01-01"
  end_date: "2024-06-30"
costs:
  taker_fee_pct: 0.0005
  maker_fee_pct: 0.0002
"""
        )

        from nautilus_quants.core.config import load_config

        configs = load_config(config_dir)

        assert configs["trading"]["capital"]["initial"] == 50000.0
        assert configs["strategy"]["universe"]["top_n"] == 5
        assert configs["risk"]["stop_loss"]["pct"] == 0.03
        assert configs["backtest"]["period"]["start_date"] == "2024-01-01"


# TODO: 完整回测测试需要准备 Nautilus Catalog 数据
#
# @pytest.mark.integration
# class TestBacktestExecution:
#     """回测执行测试 (需要数据)"""
#
#     @pytest.fixture
#     def catalog_path(self):
#         """Catalog 数据路径"""
#         return Path("data/catalog")
#
#     @pytest.mark.skipif(
#         not Path("data/catalog").exists(),
#         reason="Catalog data not available"
#     )
#     def test_run_backtest(self, catalog_path, tmp_path):
#         """测试运行回测"""
#         # 创建配置
#         config_dir = tmp_path / "config"
#         # ... 准备配置 ...
#
#         node = run_backtest(
#             config_dir=config_dir,
#             data_catalog_path=catalog_path,
#             candidate_symbols=["BTCUSDT", "ETHUSDT"],
#         )
#
#         results = node.run()
#         assert results is not None
