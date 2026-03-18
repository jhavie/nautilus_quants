"""Unit tests for bar_spec parsing."""

import pytest
from nautilus_trader.model.enums import BarAggregation, PriceType

from nautilus_quants.utils.bar_spec import format_bar_spec, parse_bar_spec


class TestParseBarSpec:
    """Tests for parse_bar_spec function."""

    def test_parse_simplified_minute(self) -> None:
        """Test parsing simplified minute formats."""
        result = parse_bar_spec("1m")
        assert result.step == 1
        assert result.aggregation == BarAggregation.MINUTE
        assert result.price_type == PriceType.LAST

        result = parse_bar_spec("5m")
        assert result.step == 5
        assert result.aggregation == BarAggregation.MINUTE

        result = parse_bar_spec("15m")
        assert result.step == 15
        assert result.aggregation == BarAggregation.MINUTE

        result = parse_bar_spec("30m")
        assert result.step == 30
        assert result.aggregation == BarAggregation.MINUTE

    def test_parse_simplified_hour(self) -> None:
        """Test parsing simplified hour formats."""
        result = parse_bar_spec("1h")
        assert result.step == 1
        assert result.aggregation == BarAggregation.HOUR
        assert result.price_type == PriceType.LAST

        result = parse_bar_spec("4h")
        assert result.step == 4
        assert result.aggregation == BarAggregation.HOUR

    def test_parse_simplified_day(self) -> None:
        """Test parsing simplified day formats."""
        result = parse_bar_spec("1d")
        assert result.step == 1
        assert result.aggregation == BarAggregation.DAY
        assert result.price_type == PriceType.LAST

    def test_parse_native_format(self) -> None:
        """Test parsing native nautilus format."""
        result = parse_bar_spec("1-HOUR-LAST")
        assert result.step == 1
        assert result.aggregation == BarAggregation.HOUR
        assert result.price_type == PriceType.LAST

        result = parse_bar_spec("15-MINUTE-LAST")
        assert result.step == 15
        assert result.aggregation == BarAggregation.MINUTE
        assert result.price_type == PriceType.LAST

        result = parse_bar_spec("1-DAY-LAST")
        assert result.step == 1
        assert result.aggregation == BarAggregation.DAY
        assert result.price_type == PriceType.LAST

    def test_parse_case_insensitive(self) -> None:
        """Test parsing is case insensitive."""
        result1 = parse_bar_spec("1H")
        result2 = parse_bar_spec("1h")
        assert result1.step == result2.step
        assert result1.aggregation == result2.aggregation

        result3 = parse_bar_spec("1-hour-last")
        result4 = parse_bar_spec("1-HOUR-LAST")
        assert result3.step == result4.step
        assert result3.aggregation == result4.aggregation

    def test_parse_invalid_format_raises(self) -> None:
        """Test invalid formats raise ValueError."""
        with pytest.raises(ValueError, match="Invalid bar_spec format"):
            parse_bar_spec("1x")

        with pytest.raises(ValueError, match="Invalid bar_spec format"):
            parse_bar_spec("hourly")

        with pytest.raises(ValueError, match="Invalid bar_spec format"):
            parse_bar_spec("")

    def test_parse_invalid_step_raises(self) -> None:
        """Test invalid step values raise ValueError."""
        with pytest.raises(ValueError):
            parse_bar_spec("0h")

        with pytest.raises(ValueError):
            parse_bar_spec("abch")


class TestFormatBarSpec:
    """Tests for format_bar_spec function."""

    def test_format_external_hour(self) -> None:
        """Test formatting hour spec as EXTERNAL."""
        result = format_bar_spec("1h")
        assert result == "1-HOUR-LAST-EXTERNAL"

        result = format_bar_spec("4h")
        assert result == "4-HOUR-LAST-EXTERNAL"

    def test_format_internal_hour(self) -> None:
        """Test formatting hour spec as INTERNAL."""
        result = format_bar_spec("1h", internal=True)
        assert result == "1-HOUR-LAST-INTERNAL"

        result = format_bar_spec("4h", internal=True)
        assert result == "4-HOUR-LAST-INTERNAL"

    def test_format_external_minute(self) -> None:
        """Test formatting minute spec as EXTERNAL."""
        result = format_bar_spec("1m")
        assert result == "1-MINUTE-LAST-EXTERNAL"

        result = format_bar_spec("15m")
        assert result == "15-MINUTE-LAST-EXTERNAL"

    def test_format_internal_minute(self) -> None:
        """Test formatting minute spec as INTERNAL."""
        result = format_bar_spec("1m", internal=True)
        assert result == "1-MINUTE-LAST-INTERNAL"

    def test_format_day(self) -> None:
        """Test formatting day spec."""
        result = format_bar_spec("1d")
        assert result == "1-DAY-LAST-EXTERNAL"

        result = format_bar_spec("1d", internal=True)
        assert result == "1-DAY-LAST-INTERNAL"

    def test_format_case_insensitive(self) -> None:
        """Test format is case insensitive on input."""
        result1 = format_bar_spec("1H")
        result2 = format_bar_spec("1h")
        assert result1 == result2 == "1-HOUR-LAST-EXTERNAL"
