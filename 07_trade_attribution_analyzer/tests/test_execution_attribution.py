"""Tests for execution attribution (TCA) module."""

import numpy as np
import pandas as pd
import pytest

from core.execution_attribution import (
    compute_execution_cost,
    compute_vwap_benchmark,
    implementation_shortfall,
    summarize_execution_costs,
)
from core.models import Fill, Order, Side, OrderType, FillStatus


class TestComputeExecutionCost:
    def test_buy_at_worse_price(self):
        """Buy order filled above decision price -> positive cost."""
        fill = Fill(
            fill_time=pd.Timestamp("2024-01-01 10:00"),
            symbol="AAPL",
            fill_qty=100,
            fill_price=105.0,
            status=FillStatus.FULLY_FILLED,
            fees=10.0,
        )
        order = Order(
            order_time=pd.Timestamp("2024-01-01 09:30"),
            symbol="AAPL",
            side=Side.BUY,
            order_qty=100,
            order_type=OrderType.MARKET,
        )
        result = compute_execution_cost(
            fill=fill,
            order=order,
            decision_price=100.0,
            arrival_price=100.0,
            benchmark_price=100.0,
        )
        assert result.slippage_bps > 0
        assert result.total_cost_bps > 0
        assert result.total_cost_dollar > 0

    def test_sell_at_worse_price(self):
        """Sell order filled below decision price -> positive cost."""
        fill = Fill(
            fill_time=pd.Timestamp("2024-01-01 10:00"),
            symbol="AAPL",
            fill_qty=100,
            fill_price=95.0,
            status=FillStatus.FULLY_FILLED,
            fees=10.0,
        )
        order = Order(
            order_time=pd.Timestamp("2024-01-01 09:30"),
            symbol="AAPL",
            side=Side.SELL,
            order_qty=100,
            order_type=OrderType.MARKET,
        )
        result = compute_execution_cost(
            fill=fill,
            order=order,
            decision_price=100.0,
            arrival_price=100.0,
            benchmark_price=100.0,
        )
        assert result.slippage_bps > 0
        assert result.total_cost_bps > 0

    def test_delay_and_impact_separation(self):
        """Verify delay and impact are computed correctly."""
        fill = Fill(
            fill_time=pd.Timestamp("2024-01-01 10:00"),
            symbol="AAPL",
            fill_qty=100,
            fill_price=102.0,
            status=FillStatus.FULLY_FILLED,
        )
        order = Order(
            order_time=pd.Timestamp("2024-01-01 09:30"),
            symbol="AAPL",
            side=Side.BUY,
            order_qty=100,
        )
        result = compute_execution_cost(
            fill=fill,
            order=order,
            decision_price=100.0,
            arrival_price=101.0,
            benchmark_price=100.0,
        )
        # delay = (101 - 100) / 100 * 10000 = 100 bps
        assert np.isclose(result.delay_bps, 100.0)
        # impact = (102 - 101) / 101 * 10000 ≈ 99.01 bps
        assert np.isclose(result.impact_bps, (102.0 - 101.0) / 101.0 * 10000)


class TestImplementationShortfall:
    def test_single_trade(self):
        fill = Fill(
            fill_time=pd.Timestamp("2024-01-01 10:00"),
            symbol="AAPL",
            fill_qty=100,
            fill_price=102.0,
            status=FillStatus.FULLY_FILLED,
            fees=5.0,
        )
        order = Order(
            order_time=pd.Timestamp("2024-01-01 09:30"),
            symbol="AAPL",
            side=Side.BUY,
            order_qty=100,
        )
        result = implementation_shortfall(
            fills=[fill],
            orders=[order],
            decision_prices={"AAPL": 100.0},
        )
        expected_dollar = 100 * (102 - 100) + 5.0
        expected_bps = expected_dollar / (100 * 100) * 10000
        assert np.isclose(result["shortfall_dollar"], expected_dollar)
        assert np.isclose(result["shortfall_bps"], expected_bps)

    def test_empty_fills(self):
        result = implementation_shortfall([], [], {})
        assert result["shortfall_dollar"] == 0.0
        assert result["shortfall_bps"] == 0.0


class TestSummarizeExecutionCosts:
    def test_summary_statistics(self):
        fill = Fill(
            fill_time=pd.Timestamp("2024-01-01"),
            symbol="A",
            fill_qty=100,
            fill_price=101.0,
            status=FillStatus.FULLY_FILLED,
        )
        order = Order(
            order_time=pd.Timestamp("2024-01-01"),
            symbol="A",
            side=Side.BUY,
            order_qty=100,
        )
        bd1 = compute_execution_cost(fill, order, 100.0, 100.0, 100.0)
        bd2 = compute_execution_cost(fill, order, 100.0, 100.0, 100.0)
        summary = summarize_execution_costs([bd1, bd2])
        assert summary.loc["total_cost_bps", "mean"] > 0
        assert summary.loc["total_cost_bps", "count"] == 2


class TestVWAPBenchmark:
    def test_simple_vwap(self):
        trades = pd.DataFrame({
            "datetime": pd.to_datetime([
                "2024-01-01 09:31", "2024-01-01 09:32", "2024-01-01 09:33"
            ]),
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "price": [100.0, 101.0, 102.0],
            "size": [100, 100, 100],
        })
        vwap = compute_vwap_benchmark(trades, "AAPL", pd.Timestamp("2024-01-01"), 150, Side.BUY)
        expected = (100 * 100 + 50 * 101) / 150
        assert np.isclose(vwap, expected)

    def test_full_day_participation(self):
        trades = pd.DataFrame({
            "datetime": pd.to_datetime(["2024-01-01 09:31"]),
            "symbol": ["AAPL"],
            "price": [100.0],
            "size": [100],
        })
        vwap = compute_vwap_benchmark(trades, "AAPL", pd.Timestamp("2024-01-01"), 50, Side.BUY)
        assert vwap == 100.0
