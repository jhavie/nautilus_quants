# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Alpha191 Built-in Factors — converted from pandas to nautilus_quants expressions.

Reference: WorldQuant Alpha191 factor set (Chinese market origin).
Source: WorldQuant_alpha101_code/alphas/alphas191.py

Filtering:
  - OHLCV-only (no vwap, amount, cap, benchmark, industry, turnover)
  - No magic-number linear aggregation (3+ hardcoded coefficients)
  - No factors requiring unavailable operators (Sequence, Regbeta, Wma,
    Rowmax/Rowmin, Sumif)
  - No broken/placeholder factors (return 0)

Skipped factors:
  vwap:       007, 008, 012, 013, 016, 017, 026, 036, 039, 041, 044, 045,
              061, 064, 073, 074, 077, 087, 090, 092, 101, 108, 114, 119,
              120, 121, 124, 125, 130, 131, 138, 154, 156, 163, 170, 179
  amount:     070, 095, 132, 144
  benchmark:  075, 149, 181, 182
  Regbeta/Seq:021, 116, 147
  Wma:        027
  Rowmax/min: 165, 183
  broken/0:   030, 143, 190
  magic coeff:028, 055, 137, 159
  too complex:166, 172, 186 (multi-branch conditionals with intermediate vars)
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Alpha191 factor expressions (OHLCV-only subset)
# ---------------------------------------------------------------------------

ALPHA191_FACTORS = {
    # -----------------------------------------------------------------------
    # Alpha191#1: -1 * CORR(RANK(DELTA(LOG(VOLUME),1)), RANK((CLOSE-OPEN)/OPEN), 6)
    # -----------------------------------------------------------------------
    "alpha191_001": {
        "expression": "-1 * correlation(rank(delta(log(volume), 1)), rank((close - open) / open), 6)",
        "description": "Alpha191#1: Correlation between ranked volume change and ranked intraday return",
        "category": "volume",
    },
    # -----------------------------------------------------------------------
    # Alpha191#2: -1 * delta(((close-low)-(high-close))/(high-low), 1)
    # -----------------------------------------------------------------------
    "alpha191_002": {
        "expression": "-1 * delta(((close - low) - (high - close)) / (high - low), 1)",
        "description": "Alpha191#2: Change in Williams %R-like indicator",
        "category": "reversion",
    },
    # -----------------------------------------------------------------------
    # Alpha191#3: SUM(CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1))), 6)
    # -----------------------------------------------------------------------
    "alpha191_003": {
        "expression": "ts_sum(if_else(close == delay(close, 1), 0, if_else(close > delay(close, 1), close - min(low, delay(close, 1)), close - max(high, delay(close, 1)))), 6)",
        "description": "Alpha191#3: Cumulative directional price movement over 6 days",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#4: Conditional on mean+std vs short-term mean and volume ratio
    # -----------------------------------------------------------------------
    "alpha191_004": {
        "expression": "if_else(ts_mean(close, 8) + stddev(close, 8) < ts_mean(close, 2), -1, if_else(ts_mean(close, 2) < ts_mean(close, 8) - stddev(close, 8), 1, if_else(volume / ts_mean(volume, 20) >= 1, 1, -1)))",
        "description": "Alpha191#4: Mean-std breakout with volume confirmation",
        "category": "reversion",
    },
    # -----------------------------------------------------------------------
    # Alpha191#5: -1 * TSMAX(CORR(TSRANK(VOLUME,5), TSRANK(HIGH,5), 5), 3)
    # -----------------------------------------------------------------------
    "alpha191_005": {
        "expression": "-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)",
        "description": "Alpha191#5: Max rolling correlation of volume and high ranks",
        "category": "volume",
    },
    # -----------------------------------------------------------------------
    # Alpha191#6: -1 * RANK(SIGN(DELTA(OPEN*0.85 + HIGH*0.15, 4)))
    # -----------------------------------------------------------------------
    "alpha191_006": {
        "expression": "-1 * rank(sign(delta(open * 0.85 + high * 0.15, 4)))",
        "description": "Alpha191#6: Ranked sign of weighted open-high delta",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#9: SMA(((H+L)/2-(DELAY(H,1)+DELAY(L,1))/2)*(H-L)/V, 7, 2)
    # → ts_mean approximation
    # -----------------------------------------------------------------------
    "alpha191_009": {
        "expression": "ts_mean(((high + low) / 2 - (delay(high, 1) + delay(low, 1)) / 2) * (high - low) / volume, 7)",
        "description": "Alpha191#9: Midpoint change weighted by range and inverse volume",
        "category": "volume",
    },
    # -----------------------------------------------------------------------
    # Alpha191#10: RANK(MAX(((RET<0)?STD(RET,20):CLOSE)^2, 5))
    # -----------------------------------------------------------------------
    "alpha191_010": {
        "expression": "rank(ts_max(power(if_else(returns < 0, stddev(returns, 20), close), 2), 5))",
        "description": "Alpha191#10: Ranked max of conditional squared value",
        "category": "volatility",
    },
    # -----------------------------------------------------------------------
    # Alpha191#11: SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*VOLUME, 6)
    # -----------------------------------------------------------------------
    "alpha191_011": {
        "expression": "ts_sum(((close - low) - (high - close)) / (high - low) * volume, 6)",
        "description": "Alpha191#11: Accumulation/distribution over 6 days",
        "category": "volume",
    },
    # -----------------------------------------------------------------------
    # Alpha191#14: CLOSE - DELAY(CLOSE, 5)
    # -----------------------------------------------------------------------
    "alpha191_014": {
        "expression": "close - delay(close, 5)",
        "description": "Alpha191#14: 5-day price change",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#15: OPEN / DELAY(CLOSE, 1) - 1
    # -----------------------------------------------------------------------
    "alpha191_015": {
        "expression": "open / delay(close, 1) - 1",
        "description": "Alpha191#15: Overnight return",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#18: CLOSE / DELAY(CLOSE, 5)
    # -----------------------------------------------------------------------
    "alpha191_018": {
        "expression": "close / delay(close, 5)",
        "description": "Alpha191#18: 5-day price ratio",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#19: Conditional 5-day return normalization
    # -----------------------------------------------------------------------
    "alpha191_019": {
        "expression": "if_else(close < delay(close, 5), (close - delay(close, 5)) / delay(close, 5), if_else(close == delay(close, 5), 0, (close - delay(close, 5)) / close))",
        "description": "Alpha191#19: Asymmetric 5-day return",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#20: (CLOSE - DELAY(CLOSE, 6)) / DELAY(CLOSE, 6) * 100
    # -----------------------------------------------------------------------
    "alpha191_020": {
        "expression": "(close - delay(close, 6)) / delay(close, 6) * 100",
        "description": "Alpha191#20: 6-day percentage return",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#22: SMA((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6) - DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6),3), 12, 1)
    # -----------------------------------------------------------------------
    "alpha191_022": {
        "expression": "ts_mean((close - ts_mean(close, 6)) / ts_mean(close, 6) - delay((close - ts_mean(close, 6)) / ts_mean(close, 6), 3), 12)",
        "description": "Alpha191#22: Smoothed change in close deviation from 6-day mean",
        "category": "reversion",
    },
    # -----------------------------------------------------------------------
    # Alpha191#23: Conditional RSI-like
    # SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1) /
    #   (SMA(up,20,1) + SMA(down,20,1)) * 100
    # -----------------------------------------------------------------------
    "alpha191_023": {
        "expression": "ts_mean(if_else(close > delay(close, 1), stddev(close, 20), 0), 20) / (ts_mean(if_else(close > delay(close, 1), stddev(close, 20), 0), 20) + ts_mean(if_else(close <= delay(close, 1), stddev(close, 20), 0), 20)) * 100",
        "description": "Alpha191#23: Volatility-weighted RSI analog",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#24: SMA(CLOSE - DELAY(CLOSE, 5), 5, 1)
    # -----------------------------------------------------------------------
    "alpha191_024": {
        "expression": "ts_mean(close - delay(close, 5), 5)",
        "description": "Alpha191#24: Smoothed 5-day price change",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#25: (-1*RANK(DELTA(CLOSE,7)*(1-RANK(DECAYLINEAR(VOLUME/MEAN(VOLUME,20),9)))))*(1+RANK(SUM(RET,250)))
    # -----------------------------------------------------------------------
    "alpha191_025": {
        "expression": "-1 * rank(delta(close, 7) * (1 - rank(decay_linear(volume / ts_mean(volume, 20), 9)))) * (1 + rank(ts_sum(returns, 250)))",
        "description": "Alpha191#25: Volume-adjusted momentum reversal",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#29: (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*VOLUME
    # -----------------------------------------------------------------------
    "alpha191_029": {
        "expression": "(close - delay(close, 6)) / delay(close, 6) * volume",
        "description": "Alpha191#29: 6-day return scaled by volume",
        "category": "volume",
    },
    # -----------------------------------------------------------------------
    # Alpha191#31: (CLOSE - MEAN(CLOSE, 12)) / MEAN(CLOSE, 12) * 100
    # -----------------------------------------------------------------------
    "alpha191_031": {
        "expression": "(close - ts_mean(close, 12)) / ts_mean(close, 12) * 100",
        "description": "Alpha191#31: Percent deviation from 12-day mean",
        "category": "reversion",
    },
    # -----------------------------------------------------------------------
    # Alpha191#32: -1 * SUM(RANK(CORR(RANK(HIGH), RANK(VOLUME), 3)), 3)
    # -----------------------------------------------------------------------
    "alpha191_032": {
        "expression": "-1 * ts_sum(rank(correlation(rank(high), rank(volume), 3)), 3)",
        "description": "Alpha191#32: Cumulative high-volume correlation rank",
        "category": "volume",
    },
    # -----------------------------------------------------------------------
    # Alpha191#33: ((-1*TSMIN(LOW,5)+DELAY(TSMIN(LOW,5),5))*RANK((SUM(RET,240)-SUM(RET,20))/220))*TSRANK(VOLUME,5)
    # -----------------------------------------------------------------------
    "alpha191_033": {
        "expression": "((-1 * ts_min(low, 5) + delay(ts_min(low, 5), 5)) * rank((ts_sum(returns, 240) - ts_sum(returns, 20)) / 220)) * ts_rank(volume, 5)",
        "description": "Alpha191#33: Low support break with long-term momentum and volume rank",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#34: MEAN(CLOSE, 12) / CLOSE
    # -----------------------------------------------------------------------
    "alpha191_034": {
        "expression": "ts_mean(close, 12) / close",
        "description": "Alpha191#34: 12-day mean to close ratio",
        "category": "reversion",
    },
    # -----------------------------------------------------------------------
    # Alpha191#35: MIN(RANK(DECAYLINEAR(DELTA(OPEN,1),15)), RANK(DECAYLINEAR(CORR(VOLUME,OPEN,17),7))) * -1
    # Note: OPEN*0.65 + OPEN*0.35 = OPEN, so simplifies to CORR(VOLUME, OPEN, 17)
    # -----------------------------------------------------------------------
    "alpha191_035": {
        "expression": "min(rank(decay_linear(delta(open, 1), 15)), rank(decay_linear(correlation(volume, open, 17), 7))) * -1",
        "description": "Alpha191#35: Min of open momentum and volume-open correlation decays",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#37: -1 * RANK(SUM(OPEN,5)*SUM(RET,5) - DELAY(SUM(OPEN,5)*SUM(RET,5), 10))
    # -----------------------------------------------------------------------
    "alpha191_037": {
        "expression": "-1 * rank(ts_sum(open, 5) * ts_sum(returns, 5) - delay(ts_sum(open, 5) * ts_sum(returns, 5), 10))",
        "description": "Alpha191#37: Open-weighted returns momentum change",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#38: ((SUM(HIGH,20)/20 < HIGH) ? -1*DELTA(HIGH,2) : 0)
    # -----------------------------------------------------------------------
    "alpha191_038": {
        "expression": "if_else(ts_mean(high, 20) < high, -1 * delta(high, 2), 0)",
        "description": "Alpha191#38: High above mean reversal",
        "category": "reversion",
    },
    # -----------------------------------------------------------------------
    # Alpha191#40: SUM(up_vol,26)/SUM(down_vol,26)*100
    # -----------------------------------------------------------------------
    "alpha191_040": {
        "expression": "ts_sum(if_else(close > delay(close, 1), volume, 0), 26) / ts_sum(if_else(close <= delay(close, 1), volume, 0), 26) * 100",
        "description": "Alpha191#40: Up/down volume ratio over 26 days",
        "category": "volume",
    },
    # -----------------------------------------------------------------------
    # Alpha191#42: -1 * RANK(STD(HIGH, 10)) * CORR(HIGH, VOLUME, 10)
    # -----------------------------------------------------------------------
    "alpha191_042": {
        "expression": "-1 * rank(stddev(high, 10)) * correlation(high, volume, 10)",
        "description": "Alpha191#42: High volatility rank times high-volume correlation",
        "category": "volatility",
    },
    # -----------------------------------------------------------------------
    # Alpha191#43: SUM(CLOSE>D(C,1)?V:(C<D(C,1)?-V:0), 6)
    # -----------------------------------------------------------------------
    "alpha191_043": {
        "expression": "ts_sum(if_else(close > delay(close, 1), volume, if_else(close < delay(close, 1), -1 * volume, 0)), 6)",
        "description": "Alpha191#43: On-balance volume over 6 days",
        "category": "volume",
    },
    # -----------------------------------------------------------------------
    # Alpha191#46: (MEAN(C,3)+MEAN(C,6)+MEAN(C,12)+MEAN(C,24))/(4*CLOSE)
    # -----------------------------------------------------------------------
    "alpha191_046": {
        "expression": "(ts_mean(close, 3) + ts_mean(close, 6) + ts_mean(close, 12) + ts_mean(close, 24)) / (4 * close)",
        "description": "Alpha191#46: Multi-period moving average ratio",
        "category": "reversion",
    },
    # -----------------------------------------------------------------------
    # Alpha191#47: SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100, 9, 1)
    # -----------------------------------------------------------------------
    "alpha191_047": {
        "expression": "ts_mean((ts_max(high, 6) - close) / (ts_max(high, 6) - ts_min(low, 6)) * 100, 9)",
        "description": "Alpha191#47: Williams %R smoothed over 9 days",
        "category": "reversion",
    },
    # -----------------------------------------------------------------------
    # Alpha191#48: (-1*(RANK(SIGN(C-D(C,1))+SIGN(D(C,1)-D(C,2))+SIGN(D(C,2)-D(C,3))))*SUM(V,5))/SUM(V,20)
    # -----------------------------------------------------------------------
    "alpha191_048": {
        "expression": "-1 * rank(sign(close - delay(close, 1)) + sign(delay(close, 1) - delay(close, 2)) + sign(delay(close, 2) - delay(close, 3))) * ts_sum(volume, 5) / ts_sum(volume, 20)",
        "description": "Alpha191#48: Consecutive direction rank with volume ratio",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#49: DI-like ratio over 12 days
    # -----------------------------------------------------------------------
    "alpha191_049": {
        "expression": "ts_sum(if_else(high + low > delay(high, 1) + delay(low, 1), 0, max(abs(high - delay(high, 1)), abs(low - delay(low, 1)))), 12) / (ts_sum(if_else(high + low > delay(high, 1) + delay(low, 1), 0, max(abs(high - delay(high, 1)), abs(low - delay(low, 1)))), 12) + ts_sum(if_else(high + low <= delay(high, 1) + delay(low, 1), 0, max(abs(high - delay(high, 1)), abs(low - delay(low, 1)))), 12))",
        "description": "Alpha191#49: Directional indicator ratio (down component)",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#50: DI-like difference ratio over 12 days
    # -----------------------------------------------------------------------
    "alpha191_050": {
        "expression": "(ts_sum(if_else(high + low <= delay(high, 1) + delay(low, 1), 0, max(abs(high - delay(high, 1)), abs(low - delay(low, 1)))), 12) - ts_sum(if_else(high + low > delay(high, 1) + delay(low, 1), 0, max(abs(high - delay(high, 1)), abs(low - delay(low, 1)))), 12)) / (ts_sum(if_else(high + low <= delay(high, 1) + delay(low, 1), 0, max(abs(high - delay(high, 1)), abs(low - delay(low, 1)))), 12) + ts_sum(if_else(high + low > delay(high, 1) + delay(low, 1), 0, max(abs(high - delay(high, 1)), abs(low - delay(low, 1)))), 12))",
        "description": "Alpha191#50: Directional indicator difference (DX analog)",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#51: Up DI ratio over 12 days
    # -----------------------------------------------------------------------
    "alpha191_051": {
        "expression": "ts_sum(if_else(high + low <= delay(high, 1) + delay(low, 1), 0, max(abs(high - delay(high, 1)), abs(low - delay(low, 1)))), 12) / (ts_sum(if_else(high + low <= delay(high, 1) + delay(low, 1), 0, max(abs(high - delay(high, 1)), abs(low - delay(low, 1)))), 12) + ts_sum(if_else(high + low > delay(high, 1) + delay(low, 1), 0, max(abs(high - delay(high, 1)), abs(low - delay(low, 1)))), 12))",
        "description": "Alpha191#51: Directional indicator ratio (up component)",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#52: SUM(MAX(0, HIGH-DELAY(HLC/3,1)),26)/SUM(MAX(0,DELAY(HLC/3,1)-LOW),26)*100
    # -----------------------------------------------------------------------
    "alpha191_052": {
        "expression": "ts_sum(max(high - delay((high + low + close) / 3, 1), 0), 26) / ts_sum(max(delay((high + low + close) / 3, 1) - low, 0), 26) * 100",
        "description": "Alpha191#52: Pivotal high/low pressure ratio",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#53: COUNT(CLOSE>DELAY(CLOSE,1), 12)/12*100
    # -----------------------------------------------------------------------
    "alpha191_053": {
        "expression": "ts_sum(if_else(close > delay(close, 1), 1, 0), 12) / 12 * 100",
        "description": "Alpha191#53: Percentage of up days in 12-day window",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#54: -1 * RANK(STD(ABS(CLOSE-OPEN)) + (CLOSE-OPEN) + CORR(CLOSE,OPEN,10))
    # Note: STD here is computed over the full series in the reference; we use stddev(abs(close-open), 20)
    # -----------------------------------------------------------------------
    "alpha191_054": {
        "expression": "-1 * rank(stddev(abs(close - open), 20) + (close - open) + correlation(close, open, 10))",
        "description": "Alpha191#54: Ranked intraday spread with open-close correlation",
        "category": "volatility",
    },
    # -----------------------------------------------------------------------
    # Alpha191#56: RANK(OPEN-TSMIN(OPEN,12)) < RANK(RANK(CORR(SUM((H+L)/2,19), SUM(MEAN(V,40),19),13))^5)
    # -----------------------------------------------------------------------
    "alpha191_056": {
        "expression": "if_else(rank(open - ts_min(open, 12)) < rank(power(rank(correlation(ts_sum((high + low) / 2, 19), ts_sum(ts_mean(volume, 40), 19), 13)), 5)), 1, 0)",
        "description": "Alpha191#56: Open breakout vs volume-price correlation rank",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#57: SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100, 3, 1)
    # -----------------------------------------------------------------------
    "alpha191_057": {
        "expression": "ts_mean((close - ts_min(low, 9)) / (ts_max(high, 9) - ts_min(low, 9)) * 100, 3)",
        "description": "Alpha191#57: Stochastic %K smoothed",
        "category": "reversion",
    },
    # -----------------------------------------------------------------------
    # Alpha191#58: COUNT(CLOSE>DELAY(CLOSE,1), 20)/20*100
    # -----------------------------------------------------------------------
    "alpha191_058": {
        "expression": "ts_sum(if_else(close > delay(close, 1), 1, 0), 20) / 20 * 100",
        "description": "Alpha191#58: Percentage of up days in 20-day window",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#59: SUM(cond, 20) — same pattern as alpha003 but 20-day window
    # -----------------------------------------------------------------------
    "alpha191_059": {
        "expression": "ts_sum(if_else(close == delay(close, 1), 0, if_else(close > delay(close, 1), close - min(low, delay(close, 1)), close - max(high, delay(close, 1)))), 20)",
        "description": "Alpha191#59: Cumulative directional price movement over 20 days",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#60: SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*VOLUME, 20)
    # -----------------------------------------------------------------------
    "alpha191_060": {
        "expression": "ts_sum(((close - low) - (high - close)) / (high - low) * volume, 20)",
        "description": "Alpha191#60: Accumulation/distribution over 20 days",
        "category": "volume",
    },
    # -----------------------------------------------------------------------
    # Alpha191#62: -1 * CORR(HIGH, RANK(VOLUME), 5)
    # -----------------------------------------------------------------------
    "alpha191_062": {
        "expression": "-1 * correlation(high, rank(volume), 5)",
        "description": "Alpha191#62: High-volume rank correlation (negated)",
        "category": "volume",
    },
    # -----------------------------------------------------------------------
    # Alpha191#63: SMA(MAX(CLOSE-DELAY(CLOSE,1),0),6,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),6,1)*100
    # RSI-like (6 period)
    # -----------------------------------------------------------------------
    "alpha191_063": {
        "expression": "ts_mean(max(close - delay(close, 1), 0), 6) / ts_mean(abs(close - delay(close, 1)), 6) * 100",
        "description": "Alpha191#63: RSI analog (6-day)",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#65: MEAN(CLOSE, 6) / CLOSE
    # -----------------------------------------------------------------------
    "alpha191_065": {
        "expression": "ts_mean(close, 6) / close",
        "description": "Alpha191#65: 6-day mean to close ratio",
        "category": "reversion",
    },
    # -----------------------------------------------------------------------
    # Alpha191#66: (CLOSE - MEAN(CLOSE, 6)) / MEAN(CLOSE, 6) * 100
    # -----------------------------------------------------------------------
    "alpha191_066": {
        "expression": "(close - ts_mean(close, 6)) / ts_mean(close, 6) * 100",
        "description": "Alpha191#66: Percent deviation from 6-day mean",
        "category": "reversion",
    },
    # -----------------------------------------------------------------------
    # Alpha191#67: SMA(MAX(CLOSE-DELAY(CLOSE,1),0),24,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),24,1)*100
    # RSI-like (24 period)
    # -----------------------------------------------------------------------
    "alpha191_067": {
        "expression": "ts_mean(max(close - delay(close, 1), 0), 24) / ts_mean(abs(close - delay(close, 1)), 24) * 100",
        "description": "Alpha191#67: RSI analog (24-day)",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#68: SMA(((H+L)/2-(D(H,1)+D(L,1))/2)*(H-L)/V, 15, 2)
    # -----------------------------------------------------------------------
    "alpha191_068": {
        "expression": "ts_mean(((high + low) / 2 - (delay(high, 1) + delay(low, 1)) / 2) * (high - low) / volume, 15)",
        "description": "Alpha191#68: Midpoint change weighted by range/volume (15-day)",
        "category": "volume",
    },
    # -----------------------------------------------------------------------
    # Alpha191#69: DTM/DBM directional indicator
    # -----------------------------------------------------------------------
    "alpha191_069": {
        "expression": "if_else(ts_sum(if_else(open <= delay(open, 1), 0, max(high - open, open - delay(open, 1))), 20) > ts_sum(if_else(open >= delay(open, 1), 0, max(open - low, open - delay(open, 1))), 20), (ts_sum(if_else(open <= delay(open, 1), 0, max(high - open, open - delay(open, 1))), 20) - ts_sum(if_else(open >= delay(open, 1), 0, max(open - low, open - delay(open, 1))), 20)) / ts_sum(if_else(open <= delay(open, 1), 0, max(high - open, open - delay(open, 1))), 20), if_else(ts_sum(if_else(open <= delay(open, 1), 0, max(high - open, open - delay(open, 1))), 20) == ts_sum(if_else(open >= delay(open, 1), 0, max(open - low, open - delay(open, 1))), 20), 0, (ts_sum(if_else(open <= delay(open, 1), 0, max(high - open, open - delay(open, 1))), 20) - ts_sum(if_else(open >= delay(open, 1), 0, max(open - low, open - delay(open, 1))), 20)) / ts_sum(if_else(open >= delay(open, 1), 0, max(open - low, open - delay(open, 1))), 20)))",
        "description": "Alpha191#69: DTM/DBM directional trend indicator",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#71: (CLOSE - MEAN(CLOSE, 24)) / MEAN(CLOSE, 24) * 100
    # -----------------------------------------------------------------------
    "alpha191_071": {
        "expression": "(close - ts_mean(close, 24)) / ts_mean(close, 24) * 100",
        "description": "Alpha191#71: Percent deviation from 24-day mean",
        "category": "reversion",
    },
    # -----------------------------------------------------------------------
    # Alpha191#72: SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100, 15, 1)
    # -----------------------------------------------------------------------
    "alpha191_072": {
        "expression": "ts_mean((ts_max(high, 6) - close) / (ts_max(high, 6) - ts_min(low, 6)) * 100, 15)",
        "description": "Alpha191#72: Williams %R smoothed over 15 days",
        "category": "reversion",
    },
    # -----------------------------------------------------------------------
    # Alpha191#76: STD(ABS(CLOSE/DELAY(CLOSE,1)-1)/VOLUME,20)/MEAN(ABS(CLOSE/DELAY(CLOSE,1)-1)/VOLUME,20)
    # -----------------------------------------------------------------------
    "alpha191_076": {
        "expression": "stddev(abs(close / delay(close, 1) - 1) / volume, 20) / ts_mean(abs(close / delay(close, 1) - 1) / volume, 20)",
        "description": "Alpha191#76: Coefficient of variation of return/volume",
        "category": "volatility",
    },
    # -----------------------------------------------------------------------
    # Alpha191#78: ((H+L+C)/3 - MA(HLC/3,12)) / (0.015*MEAN(ABS(C-MEAN(HLC/3,12)),12))
    # CCI indicator
    # -----------------------------------------------------------------------
    "alpha191_078": {
        "expression": "((high + low + close) / 3 - ts_mean((high + low + close) / 3, 12)) / (0.015 * ts_mean(abs(close - ts_mean((high + low + close) / 3, 12)), 12))",
        "description": "Alpha191#78: CCI (Commodity Channel Index, 12-day)",
        "category": "reversion",
    },
    # -----------------------------------------------------------------------
    # Alpha191#79: SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100
    # RSI-like (12 period)
    # -----------------------------------------------------------------------
    "alpha191_079": {
        "expression": "ts_mean(max(close - delay(close, 1), 0), 12) / ts_mean(abs(close - delay(close, 1)), 12) * 100",
        "description": "Alpha191#79: RSI analog (12-day)",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#80: (VOLUME - DELAY(VOLUME, 5)) / DELAY(VOLUME, 5) * 100
    # -----------------------------------------------------------------------
    "alpha191_080": {
        "expression": "(volume - delay(volume, 5)) / delay(volume, 5) * 100",
        "description": "Alpha191#80: 5-day volume percentage change",
        "category": "volume",
    },
    # -----------------------------------------------------------------------
    # Alpha191#81: SMA(VOLUME, 21, 2) → ts_mean(volume, 21)
    # -----------------------------------------------------------------------
    "alpha191_081": {
        "expression": "ts_mean(volume, 21)",
        "description": "Alpha191#81: 21-day volume moving average",
        "category": "volume",
    },
    # -----------------------------------------------------------------------
    # Alpha191#82: SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100, 20, 1)
    # -----------------------------------------------------------------------
    "alpha191_082": {
        "expression": "ts_mean((ts_max(high, 6) - close) / (ts_max(high, 6) - ts_min(low, 6)) * 100, 20)",
        "description": "Alpha191#82: Williams %R smoothed over 20 days",
        "category": "reversion",
    },
    # -----------------------------------------------------------------------
    # Alpha191#83: -1 * RANK(COV(RANK(HIGH), RANK(VOLUME), 5))
    # -----------------------------------------------------------------------
    "alpha191_083": {
        "expression": "-1 * rank(covariance(rank(high), rank(volume), 5))",
        "description": "Alpha191#83: Ranked high-volume covariance (negated)",
        "category": "volume",
    },
    # -----------------------------------------------------------------------
    # Alpha191#84: SUM(C>D(C,1)?V:(C<D(C,1)?-V:0), 20)
    # Note: in source code cond2/cond3 appear swapped — we follow the formula comment
    # -----------------------------------------------------------------------
    "alpha191_084": {
        "expression": "ts_sum(if_else(close > delay(close, 1), volume, if_else(close < delay(close, 1), -1 * volume, 0)), 20)",
        "description": "Alpha191#84: On-balance volume over 20 days",
        "category": "volume",
    },
    # -----------------------------------------------------------------------
    # Alpha191#85: TSRANK(VOLUME/MEAN(VOLUME,20),20)*TSRANK(-1*DELTA(CLOSE,7),8)
    # -----------------------------------------------------------------------
    "alpha191_085": {
        "expression": "ts_rank(volume / ts_mean(volume, 20), 20) * ts_rank(-1 * delta(close, 7), 8)",
        "description": "Alpha191#85: Volume ratio rank times close momentum rank",
        "category": "volume",
    },
    # -----------------------------------------------------------------------
    # Alpha191#86: Conditional on slope of close changes
    # -----------------------------------------------------------------------
    "alpha191_086": {
        "expression": "if_else((delay(close, 20) - delay(close, 10)) / 10 - (delay(close, 10) - close) / 10 > 0.25, -1, if_else((delay(close, 20) - delay(close, 10)) / 10 - (delay(close, 10) - close) / 10 < 0, 1, -1 * delta(close, 1)))",
        "description": "Alpha191#86: Acceleration-based momentum signal",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#88: (CLOSE - DELAY(CLOSE, 20)) / DELAY(CLOSE, 20) * 100
    # -----------------------------------------------------------------------
    "alpha191_088": {
        "expression": "(close - delay(close, 20)) / delay(close, 20) * 100",
        "description": "Alpha191#88: 20-day percentage return",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#89: 2*(SMA(CLOSE,13,2)-SMA(CLOSE,27,2)-SMA(SMA(CLOSE,13,2)-SMA(CLOSE,27,2),10,2))
    # MACD-like using ts_mean
    # -----------------------------------------------------------------------
    "alpha191_089": {
        "expression": "2 * (ts_mean(close, 13) - ts_mean(close, 27) - ts_mean(ts_mean(close, 13) - ts_mean(close, 27), 10))",
        "description": "Alpha191#89: MACD-like oscillator",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#91: (RANK(CLOSE-MAX(CLOSE,5))*RANK(CORR(MEAN(VOLUME,40),LOW,5)))*-1
    # -----------------------------------------------------------------------
    "alpha191_091": {
        "expression": "rank(close - ts_max(close, 5)) * rank(correlation(ts_mean(volume, 40), low, 5)) * -1",
        "description": "Alpha191#91: Close-high rank times volume-low correlation rank",
        "category": "reversion",
    },
    # -----------------------------------------------------------------------
    # Alpha191#93: SUM(OPEN>=DELAY(OPEN,1)?0:MAX(OPEN-LOW, OPEN-DELAY(OPEN,1)), 20)
    # -----------------------------------------------------------------------
    "alpha191_093": {
        "expression": "ts_sum(if_else(open >= delay(open, 1), 0, max(open - low, open - delay(open, 1))), 20)",
        "description": "Alpha191#93: Cumulative downward open pressure",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#94: SUM(C>D(C,1)?V:(C<D(C,1)?-V:0), 30)
    # -----------------------------------------------------------------------
    "alpha191_094": {
        "expression": "ts_sum(if_else(close > delay(close, 1), volume, if_else(close < delay(close, 1), -1 * volume, 0)), 30)",
        "description": "Alpha191#94: On-balance volume over 30 days",
        "category": "volume",
    },
    # -----------------------------------------------------------------------
    # Alpha191#96: SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1)
    # Stochastic %D
    # -----------------------------------------------------------------------
    "alpha191_096": {
        "expression": "ts_mean(ts_mean((close - ts_min(low, 9)) / (ts_max(high, 9) - ts_min(low, 9)) * 100, 3), 3)",
        "description": "Alpha191#96: Stochastic %D (double smoothed %K)",
        "category": "reversion",
    },
    # -----------------------------------------------------------------------
    # Alpha191#97: STD(VOLUME, 10)
    # -----------------------------------------------------------------------
    "alpha191_097": {
        "expression": "stddev(volume, 10)",
        "description": "Alpha191#97: 10-day volume standard deviation",
        "category": "volume",
    },
    # -----------------------------------------------------------------------
    # Alpha191#98: Conditional mean change threshold
    # -----------------------------------------------------------------------
    "alpha191_098": {
        "expression": "if_else(delta(ts_sum(close, 100) / 100, 100) / delay(close, 100) <= 0.05, -1 * (close - ts_min(close, 100)), -1 * delta(close, 3))",
        "description": "Alpha191#98: Mean change threshold (100-day)",
        "category": "reversion",
    },
    # -----------------------------------------------------------------------
    # Alpha191#99: -1 * RANK(COV(RANK(CLOSE), RANK(VOLUME), 5))
    # -----------------------------------------------------------------------
    "alpha191_099": {
        "expression": "-1 * rank(covariance(rank(close), rank(volume), 5))",
        "description": "Alpha191#99: Ranked close-volume covariance (negated)",
        "category": "volume",
    },
    # -----------------------------------------------------------------------
    # Alpha191#100: STD(VOLUME, 20)
    # -----------------------------------------------------------------------
    "alpha191_100": {
        "expression": "stddev(volume, 20)",
        "description": "Alpha191#100: 20-day volume standard deviation",
        "category": "volume",
    },
    # -----------------------------------------------------------------------
    # Alpha191#102: SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1)/SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1)*100
    # Volume RSI
    # -----------------------------------------------------------------------
    "alpha191_102": {
        "expression": "ts_mean(max(volume - delay(volume, 1), 0), 6) / ts_mean(abs(volume - delay(volume, 1)), 6) * 100",
        "description": "Alpha191#102: Volume RSI (6-day)",
        "category": "volume",
    },
    # -----------------------------------------------------------------------
    # Alpha191#103: ((20-LOWDAY(LOW,20))/20)*100
    # -----------------------------------------------------------------------
    "alpha191_103": {
        "expression": "(20 - ts_argmin(low, 20)) / 20 * 100",
        "description": "Alpha191#103: Days since 20-day low (normalized)",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#104: -1*(DELTA(CORR(HIGH,VOLUME,5),5)*RANK(STD(CLOSE,20)))
    # -----------------------------------------------------------------------
    "alpha191_104": {
        "expression": "-1 * delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))",
        "description": "Alpha191#104: Change in high-volume correlation times volatility rank",
        "category": "volume",
    },
    # -----------------------------------------------------------------------
    # Alpha191#105: -1 * CORR(RANK(OPEN), RANK(VOLUME), 10)
    # -----------------------------------------------------------------------
    "alpha191_105": {
        "expression": "-1 * correlation(rank(open), rank(volume), 10)",
        "description": "Alpha191#105: Ranked open-volume correlation (negated)",
        "category": "volume",
    },
    # -----------------------------------------------------------------------
    # Alpha191#106: CLOSE - DELAY(CLOSE, 20)
    # -----------------------------------------------------------------------
    "alpha191_106": {
        "expression": "close - delay(close, 20)",
        "description": "Alpha191#106: 20-day price change",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#107: ((-1*RANK(OPEN-DELAY(HIGH,1)))*RANK(OPEN-DELAY(CLOSE,1)))*RANK(OPEN-DELAY(LOW,1))
    # -----------------------------------------------------------------------
    "alpha191_107": {
        "expression": "-1 * rank(open - delay(high, 1)) * rank(open - delay(close, 1)) * rank(open - delay(low, 1))",
        "description": "Alpha191#107: Open gap rank triple product",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#109: SMA(HIGH-LOW,10,2)/SMA(SMA(HIGH-LOW,10,2),10,2)
    # -----------------------------------------------------------------------
    "alpha191_109": {
        "expression": "ts_mean(high - low, 10) / ts_mean(ts_mean(high - low, 10), 10)",
        "description": "Alpha191#109: Range ratio (short vs double-smoothed)",
        "category": "volatility",
    },
    # -----------------------------------------------------------------------
    # Alpha191#110: SUM(MAX(0,HIGH-DELAY(CLOSE,1)),20)/SUM(MAX(0,DELAY(CLOSE,1)-LOW),20)*100
    # -----------------------------------------------------------------------
    "alpha191_110": {
        "expression": "ts_sum(max(high - delay(close, 1), 0), 20) / ts_sum(max(delay(close, 1) - low, 0), 20) * 100",
        "description": "Alpha191#110: True range up/down ratio (20-day)",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#111: SMA(V*((C-L)-(H-C))/(H-L),11,2)-SMA(V*((C-L)-(H-C))/(H-L),4,2)
    # -----------------------------------------------------------------------
    "alpha191_111": {
        "expression": "ts_mean(volume * ((close - low) - (high - close)) / (high - low), 11) - ts_mean(volume * ((close - low) - (high - close)) / (high - low), 4)",
        "description": "Alpha191#111: AD oscillator (11-day vs 4-day)",
        "category": "volume",
    },
    # -----------------------------------------------------------------------
    # Alpha191#112: (SUM(up_move,12)-SUM(down_move,12))/(SUM(up_move,12)+SUM(down_move,12))*100
    # -----------------------------------------------------------------------
    "alpha191_112": {
        "expression": "(ts_sum(if_else(close - delay(close, 1) > 0, close - delay(close, 1), 0), 12) - ts_sum(if_else(close - delay(close, 1) < 0, abs(close - delay(close, 1)), 0), 12)) / (ts_sum(if_else(close - delay(close, 1) > 0, close - delay(close, 1), 0), 12) + ts_sum(if_else(close - delay(close, 1) < 0, abs(close - delay(close, 1)), 0), 12)) * 100",
        "description": "Alpha191#112: Directional movement balance (12-day)",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#113: -1*(RANK(SUM(DELAY(CLOSE,5),20)/20)*CORR(CLOSE,VOLUME,2))*RANK(CORR(SUM(CLOSE,5),SUM(CLOSE,20),2))
    # -----------------------------------------------------------------------
    "alpha191_113": {
        "expression": "-1 * rank(ts_sum(delay(close, 5), 20) / 20) * correlation(close, volume, 2) * rank(correlation(ts_sum(close, 5), ts_sum(close, 20), 2))",
        "description": "Alpha191#113: Delayed close rank times price-volume short correlations",
        "category": "volume",
    },
    # -----------------------------------------------------------------------
    # Alpha191#115: RANK(CORR(HIGH*0.9+CLOSE*0.1, MEAN(V,30),10))^RANK(CORR(TSRANK((H+L)/2,4),TSRANK(V,10),7))
    # -----------------------------------------------------------------------
    "alpha191_115": {
        "expression": "power(rank(correlation(high * 0.9 + close * 0.1, ts_mean(volume, 30), 10)), rank(correlation(ts_rank((high + low) / 2, 4), ts_rank(volume, 10), 7)))",
        "description": "Alpha191#115: Price-volume correlation power rank",
        "category": "volume",
    },
    # -----------------------------------------------------------------------
    # Alpha191#117: TSRANK(VOLUME,32)*(1-TSRANK((CLOSE+HIGH)-LOW,16))*(1-TSRANK(RET,32))
    # -----------------------------------------------------------------------
    "alpha191_117": {
        "expression": "ts_rank(volume, 32) * (1 - ts_rank(close + high - low, 16)) * (1 - ts_rank(returns, 32))",
        "description": "Alpha191#117: Volume rank times price range and return rank interaction",
        "category": "volume",
    },
    # -----------------------------------------------------------------------
    # Alpha191#118: SUM(HIGH-OPEN, 20) / SUM(OPEN-LOW, 20) * 100
    # -----------------------------------------------------------------------
    "alpha191_118": {
        "expression": "ts_sum(high - open, 20) / ts_sum(open - low, 20) * 100",
        "description": "Alpha191#118: Upper shadow to lower shadow ratio (20-day)",
        "category": "volatility",
    },
    # -----------------------------------------------------------------------
    # Alpha191#122: (SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)-DELAY(...))/DELAY(...)
    # Triple-smoothed log close momentum
    # -----------------------------------------------------------------------
    "alpha191_122": {
        "expression": "(ts_mean(ts_mean(ts_mean(log(close), 13), 13), 13) - delay(ts_mean(ts_mean(ts_mean(log(close), 13), 13), 13), 1)) / delay(ts_mean(ts_mean(ts_mean(log(close), 13), 13), 13), 1)",
        "description": "Alpha191#122: Triple-smoothed log close momentum",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#123: (RANK(CORR(SUM((H+L)/2,20),SUM(MEAN(V,60),20),9)) < RANK(CORR(LOW,V,6))) * -1
    # -----------------------------------------------------------------------
    "alpha191_123": {
        "expression": "if_else(rank(correlation(ts_sum((high + low) / 2, 20), ts_sum(ts_mean(volume, 60), 20), 9)) < rank(correlation(low, volume, 6)), -1, 0)",
        "description": "Alpha191#123: Volume-price midpoint vs low-volume correlation rank comparison",
        "category": "volume",
    },
    # -----------------------------------------------------------------------
    # Alpha191#126: (CLOSE + HIGH + LOW) / 3
    # Typical price
    # -----------------------------------------------------------------------
    "alpha191_126": {
        "expression": "(close + high + low) / 3",
        "description": "Alpha191#126: Typical price",
        "category": "reversion",
    },
    # -----------------------------------------------------------------------
    # Alpha191#127: MEAN((100*(CLOSE-TSMAX(CLOSE,12))/TSMAX(CLOSE,12))^2, 12)^(1/2)
    # -----------------------------------------------------------------------
    "alpha191_127": {
        "expression": "sqrt(ts_mean(power(100 * (close - ts_max(close, 12)) / ts_max(close, 12), 2), 12))",
        "description": "Alpha191#127: RMS deviation from 12-day high",
        "category": "volatility",
    },
    # -----------------------------------------------------------------------
    # Alpha191#128: 100-(100/(1+SUM(up_tp_vol,14)/SUM(down_tp_vol,14)))
    # Money Flow Index variant
    # -----------------------------------------------------------------------
    "alpha191_128": {
        "expression": "100 - 100 / (1 + ts_sum(if_else((high + low + close) / 3 > delay((high + low + close) / 3, 1), (high + low + close) / 3 * volume, 0), 14) / ts_sum(if_else((high + low + close) / 3 <= delay((high + low + close) / 3, 1), (high + low + close) / 3 * volume, 0), 14))",
        "description": "Alpha191#128: Money Flow Index (14-day)",
        "category": "volume",
    },
    # -----------------------------------------------------------------------
    # Alpha191#129: SUM((CLOSE-DELAY(CLOSE,1)<0 ? ABS(CLOSE-DELAY(CLOSE,1)) : 0), 12)
    # -----------------------------------------------------------------------
    "alpha191_129": {
        "expression": "ts_sum(if_else(close - delay(close, 1) < 0, abs(close - delay(close, 1)), 0), 12)",
        "description": "Alpha191#129: Cumulative negative price change (12-day)",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#133: (20-HIGHDAY(HIGH,20))/20*100 - (20-LOWDAY(LOW,20))/20*100
    # -----------------------------------------------------------------------
    "alpha191_133": {
        "expression": "(20 - ts_argmax(high, 20)) / 20 * 100 - (20 - ts_argmin(low, 20)) / 20 * 100",
        "description": "Alpha191#133: High day vs low day position difference",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#134: (CLOSE-DELAY(CLOSE,12))/DELAY(CLOSE,12)*VOLUME
    # -----------------------------------------------------------------------
    "alpha191_134": {
        "expression": "(close - delay(close, 12)) / delay(close, 12) * volume",
        "description": "Alpha191#134: 12-day return scaled by volume",
        "category": "volume",
    },
    # -----------------------------------------------------------------------
    # Alpha191#135: SMA(DELAY(CLOSE/DELAY(CLOSE,20),1), 20, 1)
    # -----------------------------------------------------------------------
    "alpha191_135": {
        "expression": "ts_mean(delay(close / delay(close, 20), 1), 20)",
        "description": "Alpha191#135: Smoothed lagged 20-day return",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#136: (-1*RANK(DELTA(RET,3)))*CORR(OPEN,VOLUME,10)
    # -----------------------------------------------------------------------
    "alpha191_136": {
        "expression": "-1 * rank(delta(returns, 3)) * correlation(open, volume, 10)",
        "description": "Alpha191#136: Return acceleration rank times open-volume correlation",
        "category": "volume",
    },
    # -----------------------------------------------------------------------
    # Alpha191#139: -1 * CORR(OPEN, VOLUME, 10)
    # -----------------------------------------------------------------------
    "alpha191_139": {
        "expression": "-1 * correlation(open, volume, 10)",
        "description": "Alpha191#139: Open-volume correlation (negated)",
        "category": "volume",
    },
    # -----------------------------------------------------------------------
    # Alpha191#140: MIN(RANK(DECAYLINEAR((RANK(OPEN)+RANK(LOW))-(RANK(HIGH)+RANK(CLOSE)),8)),
    #               TSRANK(DECAYLINEAR(CORR(TSRANK(CLOSE,8),TSRANK(MEAN(V,60),20),8),7),3))
    # -----------------------------------------------------------------------
    "alpha191_140": {
        "expression": "min(rank(decay_linear(rank(open) + rank(low) - rank(high) - rank(close), 8)), ts_rank(decay_linear(correlation(ts_rank(close, 8), ts_rank(ts_mean(volume, 60), 20), 8), 7), 3))",
        "description": "Alpha191#140: Min of rank-spread decay and close-volume correlation decay",
        "category": "reversion",
    },
    # -----------------------------------------------------------------------
    # Alpha191#141: RANK(CORR(RANK(HIGH), RANK(MEAN(VOLUME,15)), 9)) * -1
    # -----------------------------------------------------------------------
    "alpha191_141": {
        "expression": "rank(correlation(rank(high), rank(ts_mean(volume, 15)), 9)) * -1",
        "description": "Alpha191#141: High rank vs average volume rank correlation",
        "category": "volume",
    },
    # -----------------------------------------------------------------------
    # Alpha191#142: -1*RANK(TSRANK(CLOSE,10))*RANK(DELTA(DELTA(CLOSE,1),1))*RANK(TSRANK(VOLUME/MEAN(VOLUME,20),5))
    # -----------------------------------------------------------------------
    "alpha191_142": {
        "expression": "-1 * rank(ts_rank(close, 10)) * rank(delta(delta(close, 1), 1)) * rank(ts_rank(volume / ts_mean(volume, 20), 5))",
        "description": "Alpha191#142: Close rank times acceleration times volume rank",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#145: (MEAN(VOLUME,9)-MEAN(VOLUME,26))/MEAN(VOLUME,12)*100
    # -----------------------------------------------------------------------
    "alpha191_145": {
        "expression": "(ts_mean(volume, 9) - ts_mean(volume, 26)) / ts_mean(volume, 12) * 100",
        "description": "Alpha191#145: Volume oscillator (9/26/12)",
        "category": "volume",
    },
    # -----------------------------------------------------------------------
    # Alpha191#148: (RANK(CORR(OPEN,SUM(MEAN(V,60),9),6)) < RANK(OPEN-TSMIN(OPEN,14)))*-1
    # -----------------------------------------------------------------------
    "alpha191_148": {
        "expression": "if_else(rank(correlation(open, ts_sum(ts_mean(volume, 60), 9), 6)) < rank(open - ts_min(open, 14)), -1, 0)",
        "description": "Alpha191#148: Open-volume correlation vs open breakout rank comparison",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#150: (CLOSE + HIGH + LOW) / 3 * VOLUME
    # -----------------------------------------------------------------------
    "alpha191_150": {
        "expression": "(close + high + low) / 3 * volume",
        "description": "Alpha191#150: Typical price times volume (money flow)",
        "category": "volume",
    },
    # -----------------------------------------------------------------------
    # Alpha191#151: SMA(CLOSE - DELAY(CLOSE, 20), 20, 1)
    # -----------------------------------------------------------------------
    "alpha191_151": {
        "expression": "ts_mean(close - delay(close, 20), 20)",
        "description": "Alpha191#151: Smoothed 20-day price change",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#152: SMA(MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),12)-
    #               MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),26), 9, 1)
    # -----------------------------------------------------------------------
    "alpha191_152": {
        "expression": "ts_mean(ts_mean(delay(ts_mean(delay(close / delay(close, 9), 1), 9), 1), 12) - ts_mean(delay(ts_mean(delay(close / delay(close, 9), 1), 9), 1), 26), 9)",
        "description": "Alpha191#152: Double-smoothed return MACD signal",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#153: (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/4
    # -----------------------------------------------------------------------
    "alpha191_153": {
        "expression": "(ts_mean(close, 3) + ts_mean(close, 6) + ts_mean(close, 12) + ts_mean(close, 24)) / 4",
        "description": "Alpha191#153: Multi-period moving average composite",
        "category": "reversion",
    },
    # -----------------------------------------------------------------------
    # Alpha191#155: SMA(VOLUME,13,2)-SMA(VOLUME,27,2)-SMA(SMA(VOLUME,13,2)-SMA(VOLUME,27,2),10,2)
    # Volume MACD
    # -----------------------------------------------------------------------
    "alpha191_155": {
        "expression": "ts_mean(volume, 13) - ts_mean(volume, 27) - ts_mean(ts_mean(volume, 13) - ts_mean(volume, 27), 10)",
        "description": "Alpha191#155: Volume MACD histogram",
        "category": "volume",
    },
    # -----------------------------------------------------------------------
    # Alpha191#157: MIN(PROD(RANK(RANK(LOG(SUM(TSMIN(RANK(RANK(-1*RANK(DELTA(CLOSE-1,5)))),2),1)))),1),5)
    #             + TSRANK(DELAY(-1*RET,6),5)
    # Using ts_min for outer MIN(PROD(...),5) and ts_product for PROD
    # -----------------------------------------------------------------------
    "alpha191_157": {
        "expression": "ts_min(ts_product(rank(rank(log(ts_sum(ts_min(rank(rank(-1 * rank(delta(close - 1, 5)))), 2), 1)))), 1), 5) + ts_rank(delay(-1 * returns, 6), 5)",
        "description": "Alpha191#157: Complex nested rank of delta with lagged return rank",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#158: ((HIGH-SMA(CLOSE,15,2))-(LOW-SMA(CLOSE,15,2)))/CLOSE
    # Simplifies to (HIGH-LOW)/CLOSE
    # -----------------------------------------------------------------------
    "alpha191_158": {
        "expression": "(high - low) / close",
        "description": "Alpha191#158: Range as percentage of close",
        "category": "volatility",
    },
    # -----------------------------------------------------------------------
    # Alpha191#160: SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0), 20, 1)
    # -----------------------------------------------------------------------
    "alpha191_160": {
        "expression": "ts_mean(if_else(close <= delay(close, 1), stddev(close, 20), 0), 20)",
        "description": "Alpha191#160: Smoothed conditional downside volatility",
        "category": "volatility",
    },
    # -----------------------------------------------------------------------
    # Alpha191#161: MEAN(MAX(MAX(HIGH-LOW, ABS(DELAY(CLOSE,1)-HIGH)), ABS(DELAY(CLOSE,1)-LOW)), 12)
    # Average True Range (12-day)
    # -----------------------------------------------------------------------
    "alpha191_161": {
        "expression": "ts_mean(max(max(high - low, abs(delay(close, 1) - high)), abs(delay(close, 1) - low)), 12)",
        "description": "Alpha191#161: Average True Range (12-day)",
        "category": "volatility",
    },
    # -----------------------------------------------------------------------
    # Alpha191#167: SUM(CLOSE>DELAY(CLOSE,1) ? CLOSE-DELAY(CLOSE,1) : 0, 12)
    # -----------------------------------------------------------------------
    "alpha191_167": {
        "expression": "ts_sum(if_else(close > delay(close, 1), close - delay(close, 1), 0), 12)",
        "description": "Alpha191#167: Cumulative positive price change (12-day)",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#168: -1 * VOLUME / MEAN(VOLUME, 20)
    # -----------------------------------------------------------------------
    "alpha191_168": {
        "expression": "-1 * volume / ts_mean(volume, 20)",
        "description": "Alpha191#168: Negated volume ratio",
        "category": "volume",
    },
    # -----------------------------------------------------------------------
    # Alpha191#169: SMA(MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),12)-
    #               MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),26), 10, 1)
    # -----------------------------------------------------------------------
    "alpha191_169": {
        "expression": "ts_mean(ts_mean(delay(ts_mean(close - delay(close, 1), 9), 1), 12) - ts_mean(delay(ts_mean(close - delay(close, 1), 9), 1), 26), 10)",
        "description": "Alpha191#169: Smoothed momentum MACD signal",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#171: (-1*(LOW-CLOSE)*(OPEN^5)) / ((CLOSE-HIGH)*(CLOSE^5))
    # -----------------------------------------------------------------------
    "alpha191_171": {
        "expression": "-1 * (low - close) * power(open, 5) / ((close - high) * power(close, 5))",
        "description": "Alpha191#171: Power-weighted intraday range ratio",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#173: 3*SMA(CLOSE,13,2)-2*SMA(SMA(CLOSE,13,2),13,2)+SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)
    # TRIX-like
    # -----------------------------------------------------------------------
    "alpha191_173": {
        "expression": "3 * ts_mean(close, 13) - 2 * ts_mean(ts_mean(close, 13), 13) + ts_mean(ts_mean(ts_mean(log(close), 13), 13), 13)",
        "description": "Alpha191#173: Triple-smoothed price indicator (TRIX-like)",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#174: SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0), 20, 1)
    # -----------------------------------------------------------------------
    "alpha191_174": {
        "expression": "ts_mean(if_else(close > delay(close, 1), stddev(close, 20), 0), 20)",
        "description": "Alpha191#174: Smoothed conditional upside volatility",
        "category": "volatility",
    },
    # -----------------------------------------------------------------------
    # Alpha191#175: MEAN(MAX(MAX(HIGH-LOW, ABS(DELAY(CLOSE,1)-HIGH)), ABS(DELAY(CLOSE,1)-LOW)), 6)
    # Average True Range (6-day)
    # -----------------------------------------------------------------------
    "alpha191_175": {
        "expression": "ts_mean(max(max(high - low, abs(delay(close, 1) - high)), abs(delay(close, 1) - low)), 6)",
        "description": "Alpha191#175: Average True Range (6-day)",
        "category": "volatility",
    },
    # -----------------------------------------------------------------------
    # Alpha191#176: CORR(RANK((CLOSE-TSMIN(LOW,12))/(TSMAX(HIGH,12)-TSMIN(LOW,12))), RANK(VOLUME), 6)
    # -----------------------------------------------------------------------
    "alpha191_176": {
        "expression": "correlation(rank((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12))), rank(volume), 6)",
        "description": "Alpha191#176: Stochastic position rank vs volume rank correlation",
        "category": "volume",
    },
    # -----------------------------------------------------------------------
    # Alpha191#177: ((20-HIGHDAY(HIGH,20))/20)*100
    # -----------------------------------------------------------------------
    "alpha191_177": {
        "expression": "(20 - ts_argmax(high, 20)) / 20 * 100",
        "description": "Alpha191#177: Days since 20-day high (normalized)",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#178: (CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*VOLUME
    # -----------------------------------------------------------------------
    "alpha191_178": {
        "expression": "(close - delay(close, 1)) / delay(close, 1) * volume",
        "description": "Alpha191#178: Daily return scaled by volume",
        "category": "volume",
    },
    # -----------------------------------------------------------------------
    # Alpha191#180: Volume conditional momentum
    # -----------------------------------------------------------------------
    "alpha191_180": {
        "expression": "if_else(ts_mean(volume, 20) < volume, -1 * ts_rank(abs(delta(close, 7)), 60) * sign(delta(close, 7)), -1 * volume)",
        "description": "Alpha191#180: Volume-conditional momentum reversal",
        "category": "volume",
    },
    # -----------------------------------------------------------------------
    # Alpha191#184: RANK(CORR(DELAY(OPEN-CLOSE,1), CLOSE, 200)) + RANK(OPEN-CLOSE)
    # -----------------------------------------------------------------------
    "alpha191_184": {
        "expression": "rank(correlation(delay(open - close, 1), close, 200)) + rank(open - close)",
        "description": "Alpha191#184: Long-term open-close correlation rank plus current rank",
        "category": "reversion",
    },
    # -----------------------------------------------------------------------
    # Alpha191#185: RANK(-1 * ((1 - OPEN/CLOSE)^2))
    # -----------------------------------------------------------------------
    "alpha191_185": {
        "expression": "rank(-1 * power(1 - open / close, 2))",
        "description": "Alpha191#185: Ranked squared intraday return",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#187: SUM(OPEN<=DELAY(OPEN,1)?0:MAX(HIGH-OPEN, OPEN-DELAY(OPEN,1)), 20)
    # -----------------------------------------------------------------------
    "alpha191_187": {
        "expression": "ts_sum(if_else(open <= delay(open, 1), 0, max(high - open, open - delay(open, 1))), 20)",
        "description": "Alpha191#187: Cumulative upward open pressure",
        "category": "momentum",
    },
    # -----------------------------------------------------------------------
    # Alpha191#188: ((HIGH-LOW-SMA(HIGH-LOW,11,2))/SMA(HIGH-LOW,11,2))*100
    # -----------------------------------------------------------------------
    "alpha191_188": {
        "expression": "(high - low - ts_mean(high - low, 11)) / ts_mean(high - low, 11) * 100",
        "description": "Alpha191#188: Range deviation from smoothed range",
        "category": "volatility",
    },
    # -----------------------------------------------------------------------
    # Alpha191#189: MEAN(ABS(CLOSE-MEAN(CLOSE,6)), 6)
    # -----------------------------------------------------------------------
    "alpha191_189": {
        "expression": "ts_mean(abs(close - ts_mean(close, 6)), 6)",
        "description": "Alpha191#189: Mean absolute deviation from 6-day mean",
        "category": "volatility",
    },
    # -----------------------------------------------------------------------
    # Alpha191#191: (CORR(MEAN(VOLUME,20), LOW, 5) + (HIGH+LOW)/2) - CLOSE
    # -----------------------------------------------------------------------
    "alpha191_191": {
        "expression": "correlation(ts_mean(volume, 20), low, 5) + (high + low) / 2 - close",
        "description": "Alpha191#191: Volume-low correlation plus midpoint minus close",
        "category": "volume",
    },
}


def get_alpha191_expression(name: str) -> str:
    """Get the expression string for an Alpha191 factor.

    Parameters
    ----------
    name : str
        Factor name, e.g. ``"alpha191_001"``.

    Returns
    -------
    str
        The factor expression string.

    Raises
    ------
    ValueError
        If *name* is not a known Alpha191 factor.
    """
    if name not in ALPHA191_FACTORS:
        raise ValueError(f"Unknown Alpha191 factor: {name}")
    return ALPHA191_FACTORS[name]["expression"]


def list_alpha191_factors() -> list[str]:
    """List all available Alpha191 factor names, sorted.

    Returns
    -------
    list[str]
        Sorted list of factor names like ``["alpha191_001", ...]``.
    """
    return sorted(ALPHA191_FACTORS.keys())
