"""Tests for opportunity cost module."""

import numpy as np
import pandas as pd
import pytest

from core.opportunity_cost import compute_opportunity_cost, summarize_opportunity_costs, total_friction
from core.models import Order, Side, OrderType


class TestComputeOpportunityCost:
    def test_fully_filled_zero_cost(self):
        order = Order(
            order_time=pd.Timestamp("2024-01-01"),
            symbol="AAPL",
            side=Side.BUY,
            order_qty=100,
            order_type=OrderType.MARKET,
        )
        result = compute_opportunity_cost(
            order=order,
            filled_qty=100,
            decision_price=100.0,
            evaluation_price=105.0,
        )
        assert result.opportunity_cost_dollar == 0.0
        assert result.opportunity_cost_bps == 0.0

    def test_buy_opportunity_cost_positive(self):
        """Buy order unfilled, price rises -> positive opportunity cost."""
        order = Order(
            order_time=pd.Timestamp("2024-01-01"),
            symbol="AAPL",
            side=Side.BUY,
            order_qty=100,
        )
        result = compute_opportunity_cost(
            order=order,
            filled_qty=0,
            decision_price=100.0,
            evaluation_price=105.0,
        )
        assert result.opportunity_cost_dollar == 100 * (105 - 100)
        assert result.opportunity_cost_bps == (105 - 100) / 100 * 10000

    def test_sell_opportunity_cost_positive(self):
        """Sell order unfilled, price falls -> positive opportunity cost."""
        order = Order(
            order_time=pd.Timestamp("2024-01-01"),
            symbol="AAPL",
            side=Side.SELL,
            order_qty=100,
        )
        result = compute_opportunity_cost(
            order=order,
            filled_qty=0,
            decision_price=100.0,
            evaluation_price=95.0,
        )
        assert result.opportunity_cost_dollar == 100 * (95 - 100) * -1  # side sign
        assert result.opportunity_cost_dollar > 0

    def test_partial_fill(self):
        order = Order(
            order_time=pd.Timestamp("2024-01-01"),
            symbol="AAPL",
            side=Side.BUY,
            order_qty=100,
        )
        result = compute_opportunity_cost(
            order=order,
            filled_qty=60,
            decision_price=100.0,
            evaluation_price=105.0,
        )
        assert result.unfilled_qty == 40
        assert result.opportunity_cost_dollar == 40 * (105 - 100)


class TestSummarizeOpportunityCosts:
    def test_overall_summary(self):
        order = Order(
            order_time=pd.Timestamp("2024-01-01"),
            symbol="AAPL",
            side=Side.BUY,
            order_qty=100,
        )
        r1 = compute_opportunity_cost(order, 0, 100.0, 105.0, reason="cancelled")
        r2 = compute_opportunity_cost(order, 0, 100.0, 103.0, reason="liquidity")
        summary = summarize_opportunity_costs([r1, r2])
        assert summary.loc["opportunity_cost_dollar", "sum"] == 100 * 5 + 100 * 3
        assert summary.loc["opportunity_cost_dollar", "count"] == 2

    def test_group_by_reason(self):
        order = Order(
            order_time=pd.Timestamp("2024-01-01"),
            symbol="AAPL",
            side=Side.BUY,
            order_qty=100,
        )
        r1 = compute_opportunity_cost(order, 0, 100.0, 105.0, reason="cancelled")
        r2 = compute_opportunity_cost(order, 0, 100.0, 103.0, reason="liquidity")
        r3 = compute_opportunity_cost(order, 0, 100.0, 104.0, reason="cancelled")
        grouped = summarize_opportunity_costs([r1, r2, r3], by_reason=True)
        assert len(grouped) == 2
        cancelled_sum = grouped[grouped["reason"] == "cancelled"][("opportunity_cost_dollar", "sum")].iloc[0]
        assert cancelled_sum == 100 * 5 + 100 * 4


class TestTotalFriction:
    def test_adds_correctly(self):
        result = total_friction(
            execution_cost_dollar=500.0,
            opportunity_cost_dollar=300.0,
            decision_value=100_000.0,
        )
        assert result["total_friction_dollar"] == 800.0
        assert result["total_friction_bps"] == 80.0
