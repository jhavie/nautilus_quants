# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Alpha101 Built-in Factors — popbo/academic-aligned expressions.

Expressions restored to match popbo/alphas101.py (DolphinDB reference).
Uses academic-version operators: rank() = CS, ts_rank() = TS, scale() = CS.

Reference: https://arxiv.org/abs/1601.00991
popbo reference: WorldQuant_alpha101_code/alphas/alphas101.py

Skipped (IndNeutralize): 048, 056, 058, 059, 063, 067, 069, 070, 076,
    079, 080, 082, 087, 089, 090, 091, 093, 097, 100
"""

from __future__ import annotations


# Alpha101 factor expressions
# Reference: https://arxiv.org/abs/1601.00991

ALPHA101_FACTORS = {
    # Alpha#1: rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5
    "alpha001": {
        "expression": "rank(ts_argmax(signed_power(if_else(returns < 0, stddev(returns, 20), close), 2), 5)) - 0.5",
        "description": "Alpha#1: Conditional signed power rank",
        "category": "momentum",
    },
    # Alpha#2: (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
    "alpha002": {
        "expression": "-1 * correlation(rank(delta(log(volume), 2)), rank((close - open) / open), 6)",
        "description": "Alpha#2: Volume-price correlation",
        "category": "volume",
    },
    # Alpha#3: (-1 * correlation(rank(open), rank(volume), 10))
    "alpha003": {
        "expression": "-1 * correlation(rank(open), rank(volume), 10)",
        "description": "Alpha#3: Open-volume correlation",
        "category": "volume",
    },
    # Alpha#4: (-1 * Ts_Rank(rank(low), 9))
    "alpha004": {
        "expression": "-1 * ts_rank(rank(low), 9)",
        "description": "Alpha#4: TS rank of CS rank of low",
        "category": "reversion",
    },
    # Alpha#5: (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))
    "alpha005": {
        "expression": "rank(open - ts_sum(vwap, 10) / 10) * -1 * abs(rank(close - vwap))",
        "description": "Alpha#5: Open deviation from vwap rank",
        "category": "momentum",
    },
    # Alpha#6: (-1 * correlation(open, volume, 10))
    "alpha006": {
        "expression": "-1 * correlation(open, volume, 10)",
        "description": "Alpha#6: Open-volume correlation",
        "category": "volume",
    },
    # Alpha#7: ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1))
    "alpha007": {
        "expression": "if_else(ts_mean(volume, 20) < volume, -1 * ts_rank(abs(delta(close, 7)), 60) * sign(delta(close, 7)), -1)",
        "description": "Alpha#7: Volume conditional momentum",
        "category": "volume",
    },
    # Alpha#8: (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10))))
    "alpha008": {
        "expression": "-1 * rank(ts_sum(open, 5) * ts_sum(returns, 5) - delay(ts_sum(open, 5) * ts_sum(returns, 5), 10))",
        "description": "Alpha#8: Open-returns sum momentum",
        "category": "momentum",
    },
    # Alpha#9: ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))
    "alpha009": {
        "expression": "if_else(ts_min(delta(close, 1), 5) > 0, delta(close, 1), if_else(ts_max(delta(close, 1), 5) < 0, delta(close, 1), -1 * delta(close, 1)))",
        "description": "Alpha#9: Delta direction",
        "category": "momentum",
    },
    # Alpha#10: rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))
    "alpha010": {
        "expression": "rank(if_else(ts_min(delta(close, 1), 4) > 0, delta(close, 1), if_else(ts_max(delta(close, 1), 4) < 0, delta(close, 1), -1 * delta(close, 1))))",
        "description": "Alpha#10: Ranked delta direction",
        "category": "momentum",
    },
    # Alpha#11: ((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3)))
    "alpha011": {
        "expression": "(rank(ts_max(vwap - close, 3)) + rank(ts_min(vwap - close, 3))) * rank(delta(volume, 3))",
        "description": "Alpha#11: Vwap-close range rank times volume delta rank",
        "category": "momentum",
    },
    # Alpha#12: (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
    "alpha012": {
        "expression": "sign(delta(volume, 1)) * -1 * delta(close, 1)",
        "description": "Alpha#12: Volume delta sign times close delta",
        "category": "volume",
    },
    # Alpha#13: (-1 * rank(covariance(rank(close), rank(volume), 5)))
    "alpha013": {
        "expression": "-1 * rank(covariance(rank(close), rank(volume), 5))",
        "description": "Alpha#13: Rank covariance rank",
        "category": "volume",
    },
    # Alpha#14: ((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))
    "alpha014": {
        "expression": "-1 * rank(delta(returns, 3)) * correlation(open, volume, 10)",
        "description": "Alpha#14: Returns delta rank times open-volume correlation",
        "category": "volume",
    },
    # Alpha#15: (-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))
    "alpha015": {
        "expression": "-1 * ts_sum(rank(correlation(rank(high), rank(volume), 3)), 3)",
        "description": "Alpha#15: High-volume correlation rank sum",
        "category": "volume",
    },
    # Alpha#16: (-1 * rank(covariance(rank(high), rank(volume), 5)))
    "alpha016": {
        "expression": "-1 * rank(covariance(rank(high), rank(volume), 5))",
        "description": "Alpha#16: High-volume covariance rank",
        "category": "volume",
    },
    # Alpha#17: (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) * rank(ts_rank((volume / adv20), 5)))
    "alpha017": {
        "expression": "-1 * rank(ts_rank(close, 10)) * rank(delta(delta(close, 1), 1)) * rank(ts_rank(volume / ts_mean(volume, 20), 5))",
        "description": "Alpha#17: Rank triple product",
        "category": "momentum",
    },
    # Alpha#18: (-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10))))
    "alpha018": {
        "expression": "-1 * rank(stddev(abs(close - open), 5) + (close - open) + correlation(close, open, 10))",
        "description": "Alpha#18: Close-open deviation rank",
        "category": "volatility",
    },
    # Alpha#19: ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns, 250)))))
    "alpha019": {
        "expression": "-1 * sign((close - delay(close, 7)) + delta(close, 7)) * (1 + rank(1 + ts_sum(returns, 250)))",
        "description": "Alpha#19: 7-day change sign with returns sum",
        "category": "momentum",
    },
    # Alpha#20: (((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open - delay(low, 1))))
    "alpha020": {
        "expression": "-1 * rank(open - delay(high, 1)) * rank(open - delay(close, 1)) * rank(open - delay(low, 1))",
        "description": "Alpha#20: Open deviation from prior HLC rank product",
        "category": "momentum",
    },
    # Alpha#21: complex conditional on sma, stddev, volume/adv20
    "alpha021": {
        "expression": "if_else(ts_mean(close, 8) + stddev(close, 8) < ts_mean(close, 2), -1, if_else(ts_mean(close, 2) < ts_mean(close, 8) - stddev(close, 8), 1, if_else(volume / ts_mean(volume, 20) >= 1, 1, -1)))",
        "description": "Alpha#21: Mean-std breakout",
        "category": "reversion",
    },
    # Alpha#22: (-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))
    "alpha022": {
        "expression": "-1 * delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))",
        "description": "Alpha#22: High-volume correlation change",
        "category": "volume",
    },
    # Alpha#23: (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)
    "alpha023": {
        "expression": "if_else(ts_mean(high, 20) < high, -1 * delta(high, 2), 0)",
        "description": "Alpha#23: High above mean reversal",
        "category": "reversion",
    },
    # Alpha#24: conditional on delta(sma(close,100),100)/delay(close,100) <= 0.05
    "alpha024": {
        "expression": "if_else(delta(ts_mean(close, 100), 100) / delay(close, 100) <= 0.05, -1 * (close - ts_min(close, 100)), -1 * delta(close, 3))",
        "description": "Alpha#24: Mean change threshold",
        "category": "reversion",
    },
    # Alpha#25: rank(((((-1 * returns) * adv20) * vwap) * (high - close)))
    "alpha025": {
        "expression": "rank(-1 * returns * ts_mean(volume, 20) * vwap * (high - close))",
        "description": "Alpha#25: Volume-weighted returns rank",
        "category": "volume",
    },
    # Alpha#26: (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))
    "alpha026": {
        "expression": "-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)",
        "description": "Alpha#26: Max correlation rank",
        "category": "volume",
    },
    # Alpha#27: ((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1) : 1)
    "alpha027": {
        "expression": "if_else(rank(ts_mean(correlation(rank(volume), rank(vwap), 6), 2)) > 0.5, -1, 1)",
        "description": "Alpha#27: Volume-vwap correlation rank threshold",
        "category": "volume",
    },
    # Alpha#28: scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))
    "alpha028": {
        "expression": "scale(correlation(ts_mean(volume, 20), low, 5) + (high + low) / 2 - close)",
        "description": "Alpha#28: Volume-low correlation plus midpoint scaled",
        "category": "volume",
    },
    # Alpha#29: (min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close - 1), 5))))), 2), 1))))), 1), 5) + ts_rank(delay((-1 * returns), 6), 5))
    "alpha029": {
        "expression": "ts_min(rank(rank(scale(log(ts_sum(rank(rank(-1 * rank(delta(close - 1, 5)))), 2))))), 5) + ts_rank(delay(-1 * returns, 6), 5)",
        "description": "Alpha#29: Complex rank of delta",
        "category": "momentum",
    },
    # Alpha#30: (((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) + sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))
    "alpha030": {
        "expression": "(1 - rank(sign(close - delay(close, 1)) + sign(delay(close, 1) - delay(close, 2)) + sign(delay(close, 2) - delay(close, 3)))) * ts_sum(volume, 5) / ts_sum(volume, 20)",
        "description": "Alpha#30: Price sign rank volume ratio",
        "category": "volume",
    },
    # Alpha#31: ((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10)))) + rank((-1 * delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))
    "alpha031": {
        "expression": "rank(rank(rank(decay_linear(-1 * rank(rank(delta(close, 10))), 10)))) + rank(-1 * delta(close, 3)) + sign(scale(correlation(ts_mean(volume, 20), low, 12)))",
        "description": "Alpha#31: Triple rank delta with volume correlation",
        "category": "momentum",
    },
    # Alpha#32: (scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5), 230))))
    "alpha032": {
        "expression": "scale(ts_mean(close, 7) - close) + 20 * scale(correlation(vwap, delay(close, 5), 230))",
        "description": "Alpha#32: Mean to close ratio plus correlation",
        "category": "momentum",
    },
    # Alpha#33: rank((-1 * ((1 - (open / close))^1)))
    "alpha033": {
        "expression": "rank(-1 + open / close)",
        "description": "Alpha#33: Open to close ratio rank",
        "category": "momentum",
    },
    # Alpha#34: rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))
    "alpha034": {
        "expression": "rank(2 - rank(stddev(returns, 2) / stddev(returns, 5)) - rank(delta(close, 1)))",
        "description": "Alpha#34: Std ratio rank plus delta rank",
        "category": "volatility",
    },
    # Alpha#35: ((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 - Ts_Rank(returns, 32)))
    "alpha035": {
        "expression": "ts_rank(volume, 32) * (1 - ts_rank(close + high - low, 16)) * (1 - ts_rank(returns, 32))",
        "description": "Alpha#35: Volume rank times range rank",
        "category": "volume",
    },
    # Alpha#36: complex with rank and ts_rank
    "alpha036": {
        "expression": "2.21 * rank(correlation(close - open, delay(volume, 1), 15)) + 0.7 * rank(open - close) + 0.73 * rank(ts_rank(delay(-1 * returns, 6), 5)) + rank(abs(correlation(vwap, ts_mean(volume, 20), 6))) + 0.6 * rank((ts_mean(close, 200) - open) * (close - open))",
        "description": "Alpha#36: Composite correlation rank",
        "category": "volume",
    },
    # Alpha#37: (rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))
    "alpha037": {
        "expression": "rank(correlation(delay(open - close, 1), close, 200)) + rank(open - close)",
        "description": "Alpha#37: Open-close correlation rank",
        "category": "momentum",
    },
    # Alpha#38: ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))
    "alpha038": {
        "expression": "-1 * rank(ts_rank(open, 10)) * rank(close / open)",
        "description": "Alpha#38: Close rank ratio",
        "category": "reversion",
    },
    # Alpha#39: ((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 + rank(sum(returns, 250))))
    "alpha039": {
        "expression": "-1 * rank(delta(close, 7) * (1 - rank(decay_linear(volume / ts_mean(volume, 20), 9)))) * (1 + rank(ts_mean(returns, 250)))",
        "description": "Alpha#39: Delta times volume rank decay",
        "category": "volume",
    },
    # Alpha#40: ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))
    "alpha040": {
        "expression": "-1 * rank(stddev(high, 10)) * correlation(high, volume, 10)",
        "description": "Alpha#40: High std rank times correlation",
        "category": "volatility",
    },
    # Alpha#41: (((high * low)^0.5) - vwap)
    "alpha041": {
        "expression": "power(high * low, 0.5) - vwap",
        "description": "Alpha#41: HL geometric mean minus vwap",
        "category": "momentum",
    },
    # Alpha#42: (rank((vwap - close)) / rank((vwap + close)))
    "alpha042": {
        "expression": "rank(vwap - close) / rank(vwap + close)",
        "description": "Alpha#42: Vwap-close ratio rank",
        "category": "momentum",
    },
    # Alpha#43: (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))
    "alpha043": {
        "expression": "ts_rank(volume / ts_mean(volume, 20), 20) * ts_rank(-1 * delta(close, 7), 8)",
        "description": "Alpha#43: Volume ratio rank times delta rank",
        "category": "volume",
    },
    # Alpha#44: (-1 * correlation(high, rank(volume), 5))
    "alpha044": {
        "expression": "-1 * correlation(high, rank(volume), 5)",
        "description": "Alpha#44: High-volume rank correlation",
        "category": "volume",
    },
    # Alpha#45: (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) * rank(correlation(sum(close, 5), sum(close, 20), 2))))
    "alpha045": {
        "expression": "-1 * rank(ts_mean(delay(close, 5), 20)) * correlation(close, volume, 2) * rank(correlation(ts_sum(close, 5), ts_sum(close, 20), 2))",
        "description": "Alpha#45: Delayed mean times correlations",
        "category": "volume",
    },
    # Alpha#46: conditional on ((delay(close,20)-delay(close,10))/10 - (delay(close,10)-close)/10)
    "alpha046": {
        "expression": "if_else((delay(close, 20) - delay(close, 10)) / 10 - (delay(close, 10) - close) / 10 > 0.25, -1, if_else((delay(close, 20) - delay(close, 10)) / 10 - (delay(close, 10) - close) / 10 < 0, 1, -1 * delta(close, 1)))",
        "description": "Alpha#46: Long-term trend threshold",
        "category": "momentum",
    },
    # Alpha#47: ((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) / 5))) - rank((vwap - delay(vwap, 5))))
    "alpha047": {
        "expression": "((rank(1 / close) * volume / ts_mean(volume, 20)) * (high * rank(high - close) / ts_mean(high, 5))) - rank(vwap - delay(vwap, 5))",
        "description": "Alpha#47: High rank volume ratio vwap",
        "category": "volume",
    },
    # Alpha#49: conditional on inner < -0.1
    "alpha049": {
        "expression": "if_else((delay(close, 20) - delay(close, 10)) / 10 - (delay(close, 10) - close) / 10 < -0.1, 1, -1 * delta(close, 1))",
        "description": "Alpha#49: Long-term decline signal",
        "category": "reversion",
    },
    # Alpha#50: (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))
    "alpha050": {
        "expression": "-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5)",
        "description": "Alpha#50: Max rank correlation",
        "category": "volume",
    },
    # Alpha#51: conditional on inner < -0.05
    "alpha051": {
        "expression": "if_else((delay(close, 20) - delay(close, 10)) / 10 - (delay(close, 10) - close) / 10 < -0.05, 1, -1 * delta(close, 1))",
        "description": "Alpha#51: Long-term decline signal v2",
        "category": "reversion",
    },
    # Alpha#52: ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) - sum(returns, 20)) / 220))) * ts_rank(volume, 5))
    "alpha052": {
        "expression": "-1 * delta(ts_min(low, 5), 5) * rank((ts_sum(returns, 240) - ts_sum(returns, 20)) / 220) * ts_rank(volume, 5)",
        "description": "Alpha#52: Low min change with returns sum rank",
        "category": "momentum",
    },
    # Alpha#53: (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))
    "alpha053": {
        "expression": "-1 * delta(((close - low) - (high - close)) / replace_zero(close - low, 0.0001), 9)",
        "description": "Alpha#53: High-low close ratio delta",
        "category": "momentum",
    },
    # Alpha#54: ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))
    "alpha054": {
        "expression": "-1 * (low - close) * power(open, 5) / (replace_zero(low - high, -0.0001) * power(close, 5))",
        "description": "Alpha#54: Low-close open power ratio",
        "category": "momentum",
    },
    # Alpha#55: (-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)))), rank(volume), 6))
    "alpha055": {
        "expression": "-1 * correlation(rank((close - ts_min(low, 12)) / replace_zero(ts_max(high, 12) - ts_min(low, 12), 0.0001)), rank(volume), 6)",
        "description": "Alpha#55: Normalized close rank volume correlation",
        "category": "volume",
    },
    # Alpha#57: (0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))))
    "alpha057": {
        "expression": "0 - (close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2)",
        "description": "Alpha#57: Close-vwap normalized by argmax rank",
        "category": "momentum",
    },
    # Alpha#60: (0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) - scale(rank(ts_argmax(close, 10))))))
    "alpha060": {
        "expression": "0 - (2 * scale(rank(((close - low) - (high - close)) / replace_zero(high - low, 0.0001) * volume)) - scale(rank(ts_argmax(close, 10))))",
        "description": "Alpha#60: Combined rank product",
        "category": "volume",
    },
    # Alpha#61: (rank((vwap - ts_min(vwap, 16))) < rank(correlation(vwap, adv180, 18)))
    "alpha061": {
        "expression": "if_else(rank(vwap - ts_min(vwap, 16)) < rank(correlation(vwap, ts_mean(volume, 180), 18)), 1, 0)",
        "description": "Alpha#61: Vwap min rank vs correlation rank",
        "category": "volume",
    },
    # Alpha#62: ((rank(correlation(vwap, sum(adv20, 22), 10)) < rank(((rank(open) + rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1)
    "alpha062": {
        "expression": "(rank(correlation(vwap, ts_mean(ts_mean(volume, 20), 22), 10)) < rank((rank(open) + rank(open) < rank((high + low) / 2) + rank(high)))) * -1",
        "description": "Alpha#62: Vwap-volume correlation rank vs open rank",
        "category": "volume",
    },
    # Alpha#64: ((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 13), sum(adv120, 13), 17)) < rank(delta(((((high + low) / 2) * 0.178404) + (vwap * (1 - 0.178404))), 4))) * -1)
    "alpha064": {
        "expression": "if_else(rank(correlation(ts_mean(open * 0.178404 + low * 0.821596, 13), ts_mean(ts_mean(volume, 120), 13), 17)) < rank(delta(((high + low) / 2 * 0.178404 + vwap * 0.821596), 4)), -1, 0)",
        "description": "Alpha#64: Open-low sum correlation vs midpoint delta",
        "category": "volume",
    },
    # Alpha#65: ((rank(correlation(((open * 0.00817205) + (vwap * 0.99182795)), sma(adv60, 9), 6)) < rank((open - ts_min(open, 14)))) * -1)
    "alpha065": {
        "expression": "if_else(rank(correlation(open * 0.00817205 + vwap * 0.99182795, ts_mean(ts_mean(volume, 60), 9), 6)) < rank(open - ts_min(open, 14)), -1, 0)",
        "description": "Alpha#65: Open-vwap combo correlation vs open min",
        "category": "volume",
    },
    # Alpha#66: ((rank(decay_linear(delta(vwap, 4), 7)) + ts_rank(decay_linear(((low - vwap) / (open - (high + low) / 2)), 11), 7)) * -1)
    "alpha066": {
        "expression": "(rank(decay_linear(delta(vwap, 4), 7)) + ts_rank(decay_linear((low * 0.96633 + low * 0.03367 - vwap) / (open - (high + low) / 2), 11), 7)) * -1",
        "description": "Alpha#66: Vwap deviation decay rank",
        "category": "volume",
    },
    # Alpha#68: ((Ts_Rank(correlation(rank(high), rank(adv15), 9), 14) < rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 2))) * -1)
    "alpha068": {
        "expression": "(ts_rank(correlation(rank(high), rank(ts_mean(volume, 15)), 9), 14) < rank(delta(close * 0.518371 + low * 0.481629, 2)) * 14) * -1",
        "description": "Alpha#68: Open delta correlation vs combo delta",
        "category": "volume",
    },
    # Alpha#71: max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3), Ts_Rank(adv180, 12), 18), 4), 16), Ts_Rank(decay_linear((rank(((low + open) - (vwap + vwap)))^2), 16), 4))
    "alpha071": {
        "expression": "max(ts_rank(decay_linear(correlation(ts_rank(close, 3), ts_rank(ts_mean(volume, 180), 12), 18), 4), 16), ts_rank(decay_linear(power(rank(low + open - vwap - vwap), 2), 16), 4))",
        "description": "Alpha#71: Max of two complex decay ranks",
        "category": "volume",
    },
    # Alpha#72: (rank(decay_linear(correlation(((high + low) / 2), adv40, 9), 10)) / rank(decay_linear(correlation(ts_rank(vwap, 4), ts_rank(volume, 19), 7), 3)))
    "alpha072": {
        "expression": "rank(decay_linear(correlation((high + low) / 2, ts_mean(volume, 40), 9), 10)) / rank(decay_linear(correlation(ts_rank(vwap, 4), ts_rank(volume, 19), 7), 3))",
        "description": "Alpha#72: HLV correlation decay rank ratio",
        "category": "volume",
    },
    # Alpha#73: (max(rank(decay_linear(delta(vwap, 5), 3)), ts_rank(decay_linear(((delta(((open * 0.147155) + (low * 0.852845)), 2) / (open * 0.147155 + low * 0.852845)) * -1), 3), 17)) * -1)
    "alpha073": {
        "expression": "max(rank(decay_linear(delta(vwap, 5), 3)), ts_rank(decay_linear(-1 * delta(open * 0.147155 + low * 0.852845, 2) / (open * 0.147155 + low * 0.852845), 3), 17)) * -1",
        "description": "Alpha#73: Vwap delta decay rank",
        "category": "momentum",
    },
    # Alpha#74: ((rank(correlation(close, sum(adv30, 37), 15)) < rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11))) * -1)
    "alpha074": {
        "expression": "if_else(rank(correlation(close, ts_mean(ts_mean(volume, 30), 37), 15)) < rank(correlation(rank(high * 0.0261661 + vwap * 0.973839), rank(volume), 11)), -1, 0)",
        "description": "Alpha#74: Close-volume correlation vs open-vwap rank",
        "category": "volume",
    },
    # Alpha#75: (rank(correlation(vwap, volume, 4)) < rank(correlation(rank(low), rank(adv50), 12)))
    "alpha075": {
        "expression": "if_else(rank(correlation(vwap, volume, 4)) < rank(correlation(rank(low), rank(ts_mean(volume, 50)), 12)), 1, 0)",
        "description": "Alpha#75: Vwap-volume correlation rank",
        "category": "volume",
    },
    # Alpha#77: min(rank(decay_linear(((((high + low) / 2) + high) - (vwap + high)), 20)), rank(decay_linear(correlation(((high + low) / 2), adv40, 3), 6)))
    "alpha077": {
        "expression": "min(rank(decay_linear((high + low) / 2 + high - vwap - high, 20)), rank(decay_linear(correlation((high + low) / 2, ts_mean(volume, 40), 3), 6)))",
        "description": "Alpha#77: Min of HL-vwap decay ranks",
        "category": "volume",
    },
    # Alpha#78: (rank(correlation(sum(((low * 0.352233) + (vwap * 0.647767)), 20), sum(adv40, 20), 7))^rank(correlation(rank(vwap), rank(volume), 6)))
    "alpha078": {
        "expression": "power(rank(correlation(ts_sum(low * 0.352233 + vwap * 0.647767, 20), ts_sum(ts_mean(volume, 40), 20), 7)), rank(correlation(rank(vwap), rank(volume), 6)))",
        "description": "Alpha#78: Low-vwap sum correlation rank power",
        "category": "volume",
    },
    # Alpha#81: ((rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 50), 8))^4)), 15))) < rank(correlation(rank(vwap), rank(volume), 5))) * -1)
    "alpha081": {
        "expression": "if_else(rank(log(ts_product(rank(power(rank(correlation(vwap, ts_sum(ts_mean(volume, 10), 50), 8)), 4)), 15))) < rank(correlation(rank(vwap), rank(volume), 5)), -1, 0)",
        "description": "Alpha#81: Vwap-volume product rank vs correlation",
        "category": "volume",
    },
    # Alpha#83: ((rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume))) / (((high - low) / (sum(close, 5) / 5)) / (vwap - close)))
    "alpha083": {
        "expression": "rank(delay((high - low) / (ts_sum(close, 5) / 5), 2)) * rank(rank(volume)) / ((high - low) / (ts_sum(close, 5) / 5) / (vwap - close))",
        "description": "Alpha#83: High-low range rank",
        "category": "volatility",
    },
    # Alpha#84: SignedPower(Ts_Rank((vwap - ts_max(vwap, 15)), 21), delta(close, 5))
    "alpha084": {
        "expression": "signed_power(ts_rank(vwap - ts_max(vwap, 15), 21), delta(close, 5))",
        "description": "Alpha#84: Vwap max deviation signed power",
        "category": "momentum",
    },
    # Alpha#85: (rank(correlation(((high * 0.876703) + (close * 0.123297)), adv30, 10))^rank(correlation(Ts_Rank(((high + low) / 2), 4), Ts_Rank(volume, 10), 7)))
    "alpha085": {
        "expression": "power(rank(correlation(high * 0.876703 + close * 0.123297, ts_mean(volume, 30), 10)), rank(correlation(ts_rank((high + low) / 2, 4), ts_rank(volume, 10), 7)))",
        "description": "Alpha#85: Open-close combo correlation rank",
        "category": "volume",
    },
    # Alpha#86: ((Ts_Rank(correlation(close, sum(adv20, 15), 6), 20) < rank(((open + close) - (vwap + open)))) * -1)
    "alpha086": {
        "expression": "(ts_rank(correlation(close, ts_mean(ts_mean(volume, 20), 15), 6), 20) < rank((open + close) - (vwap + open)) * 20) * -1",
        "description": "Alpha#86: Close-volume correlation rank ratio",
        "category": "volume",
    },
    # Alpha#88: min(rank(decay_linear(((rank(open) + rank(low)) - (rank(high) + rank(close))), 8)), Ts_Rank(decay_linear(correlation(Ts_Rank(close, 8), Ts_Rank(adv60, 21), 8), 7), 3))
    "alpha088": {
        "expression": "min(rank(decay_linear((rank(open) + rank(low)) - (rank(high) + rank(close)), 8)), ts_rank(decay_linear(correlation(ts_rank(close, 8), ts_rank(ts_mean(volume, 60), 21), 8), 7), 3))",
        "description": "Alpha#88: Min of rank decay and correlation rank",
        "category": "momentum",
    },
    # Alpha#92: min(Ts_Rank(decay_linear(((((high + low) / 2) + close) < (low + open)), 15), 19), Ts_Rank(decay_linear(correlation(rank(low), rank(adv30), 8), 7), 7))
    "alpha092": {
        "expression": "min(ts_rank(decay_linear(if_else((high + low) / 2 + close < low + open, 1, 0), 15), 19), ts_rank(decay_linear(correlation(rank(low), rank(ts_mean(volume, 30)), 8), 7), 7))",
        "description": "Alpha#92: Min of HLV rank decay",
        "category": "volume",
    },
    # Alpha#94: ((rank((vwap - ts_min(vwap, 12)))^Ts_Rank(correlation(Ts_Rank(vwap, 20), Ts_Rank(adv60, 4), 18), 3)) * -1)
    "alpha094": {
        "expression": "power(rank(vwap - ts_min(vwap, 12)), ts_rank(correlation(ts_rank(vwap, 20), ts_rank(ts_mean(volume, 60), 4), 18), 3)) * -1",
        "description": "Alpha#94: Vwap min rank ratio",
        "category": "volume",
    },
    # Alpha#95: (rank((open - ts_min(open, 12))) < Ts_Rank((rank(correlation(sma(((high + low) / 2), 19), sma(adv40, 19), 13))^5), 12))
    "alpha095": {
        "expression": "rank(open - ts_min(open, 12)) * 12 < ts_rank(power(rank(correlation(ts_mean((high + low) / 2, 19), ts_mean(ts_mean(volume, 40), 19), 13)), 5), 12)",
        "description": "Alpha#95: Open min rank vs OHLC-volume correlation",
        "category": "volume",
    },
    # Alpha#96: (max(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume), 4), 4), 8), Ts_Rank(decay_linear(ts_argmax(correlation(Ts_Rank(close, 7), Ts_Rank(adv60, 4), 4), 13), 14), 13)) * -1)
    "alpha096": {
        "expression": "max(ts_rank(decay_linear(correlation(rank(vwap), rank(volume), 4), 4), 8), ts_rank(decay_linear(ts_argmax(correlation(ts_rank(close, 7), ts_rank(ts_mean(volume, 60), 4), 4), 13), 14), 13)) * -1",
        "description": "Alpha#96: Max of close-volume and vwap-volume decay ranks",
        "category": "volume",
    },
    # Alpha#98: (rank(decay_linear(correlation(vwap, sma(adv5, 26), 5), 7)) - rank(decay_linear(ts_rank(ts_argmin(correlation(rank(open), rank(adv15), 21), 9), 7), 8)))
    "alpha098": {
        "expression": "rank(decay_linear(correlation(vwap, ts_mean(ts_mean(volume, 5), 26), 5), 7)) - rank(decay_linear(ts_rank(ts_argmin(correlation(rank(open), rank(ts_mean(volume, 15)), 21), 9), 7), 8))",
        "description": "Alpha#98: Vwap-volume correlation decay rank",
        "category": "volume",
    },
    # Alpha#99: ((rank(correlation(sum(((high + low) / 2), 20), sum(adv60, 20), 9)) < rank(correlation(low, volume, 6))) * -1)
    "alpha099": {
        "expression": "if_else(rank(correlation(ts_sum((high + low) / 2, 20), ts_sum(ts_mean(volume, 60), 20), 9)) < rank(correlation(low, volume, 6)), -1, 0)",
        "description": "Alpha#99: Close-volume rank covariance",
        "category": "volume",
    },
    # Alpha#101: ((close - open) / ((high - low) + .001))
    "alpha101": {
        "expression": "(close - open) / ((high - low) + 0.001)",
        "description": "Alpha#101: Close-open to high-low ratio",
        "category": "momentum",
    },
}


def get_alpha101_expression(name: str) -> str:
    """Get the expression for an Alpha101 factor."""
    if name not in ALPHA101_FACTORS:
        raise ValueError(f"Unknown Alpha101 factor: {name}")
    return ALPHA101_FACTORS[name]["expression"]


def list_alpha101_factors() -> list[str]:
    """List all available Alpha101 factors."""
    return sorted(ALPHA101_FACTORS.keys())
