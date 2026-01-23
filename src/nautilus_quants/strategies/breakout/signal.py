"""Price-Volume Breakout Signal Calculator.

Entry conditions (Long):
1. Current close breaks above highest close of last N bars
2. Current volume breaks above highest volume of last N bars
3. Current close is above SMA

Exit conditions (Long):
1. Stop loss triggered (handled by strategy)
2. N consecutive bars close below SMA
"""


class PriceVolumeBreakoutSignal:
    """Price-volume breakout signal calculator."""

    def __init__(
        self,
        breakout_period: int = 30,
        sma_period: int = 30,
        exit_bars_below_sma: int = 5,
    ):
        """Initialize the signal calculator.

        Args:
            breakout_period: Number of bars to look back for breakout detection.
            sma_period: Period for SMA calculation (used externally).
            exit_bars_below_sma: Number of consecutive bars below SMA to trigger exit.
        """
        self.breakout_period = breakout_period
        self.sma_period = sma_period
        self.exit_bars_below_sma = exit_bars_below_sma

    def check_long(
        self,
        current_close: float,
        current_volume: float,
        recent_closes: list[float],
        recent_volumes: list[float],
        sma_value: float,
    ) -> bool:
        """Check if long entry conditions are met.

        Args:
            current_close: Current bar's close price.
            current_volume: Current bar's volume.
            recent_closes: List of recent close prices (excluding current).
            recent_volumes: List of recent volumes (excluding current).
            sma_value: Current SMA value.

        Returns:
            True if all long entry conditions are met.
        """
        if (
            len(recent_closes) < self.breakout_period
            or len(recent_volumes) < self.breakout_period
        ):
            return False

        highest_close = max(recent_closes[-self.breakout_period :])
        highest_volume = max(recent_volumes[-self.breakout_period :])

        price_breakout = current_close > highest_close
        volume_breakout = current_volume > highest_volume
        above_sma = current_close > sma_value

        return price_breakout and volume_breakout and above_sma

    def check_short(
        self,
        current_close: float,
        current_volume: float,
        recent_closes: list[float],
        recent_volumes: list[float],
        sma_value: float,
    ) -> bool:
        """Check if short entry conditions are met.

        Args:
            current_close: Current bar's close price.
            current_volume: Current bar's volume.
            recent_closes: List of recent close prices (excluding current).
            recent_volumes: List of recent volumes (excluding current).
            sma_value: Current SMA value.

        Returns:
            True if all short entry conditions are met.
        """
        if (
            len(recent_closes) < self.breakout_period
            or len(recent_volumes) < self.breakout_period
        ):
            return False

        lowest_close = min(recent_closes[-self.breakout_period :])
        highest_volume = max(recent_volumes[-self.breakout_period :])

        price_breakdown = current_close < lowest_close
        volume_breakout = current_volume > highest_volume
        below_sma = current_close < sma_value

        return price_breakdown and volume_breakout and below_sma

    def check_exit_long(
        self,
        recent_closes: list[float],
        sma_value: float,
        exit_bars: int,
    ) -> bool:
        """Check if long exit conditions are met.

        Args:
            recent_closes: List of recent close prices.
            sma_value: Current SMA value.
            exit_bars: Number of consecutive bars below SMA to trigger exit.

        Returns:
            True if exit conditions are met.
        """
        if len(recent_closes) < exit_bars:
            return False

        return all(c < sma_value for c in recent_closes[-exit_bars:])

    def check_exit_short(
        self,
        recent_closes: list[float],
        sma_value: float,
        exit_bars: int,
    ) -> bool:
        """Check if short exit conditions are met.

        Args:
            recent_closes: List of recent close prices.
            sma_value: Current SMA value.
            exit_bars: Number of consecutive bars above SMA to trigger exit.

        Returns:
            True if exit conditions are met.
        """
        if len(recent_closes) < exit_bars:
            return False

        return all(c > sma_value for c in recent_closes[-exit_bars:])
