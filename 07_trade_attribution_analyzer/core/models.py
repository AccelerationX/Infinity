"""Core data models for trade attribution analysis."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd


class Side(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    VWAP = "VWAP"


class FillStatus(Enum):
    FULLY_FILLED = "FULLY_FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    UNFILLED = "UNFILLED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True)
class Signal:
    """Strategy signal at a point in time."""

    signal_time: datetime
    symbol: str
    target_weight: Optional[float] = None
    target_shares: Optional[float] = None
    signal_price: Optional[float] = None  # price used to generate signal
    horizon_days: int = 1  # intended holding horizon

    def __post_init__(self):
        if self.target_weight is None and self.target_shares is None:
            raise ValueError("Either target_weight or target_shares must be provided.")


@dataclass(frozen=True)
class Order:
    """Trading order derived from a signal."""

    order_time: datetime
    symbol: str
    side: Side
    order_qty: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    signal_ref: Optional[str] = None  # reference to originating signal


@dataclass(frozen=True)
class Fill:
    """Actual execution result of an order."""

    fill_time: datetime
    symbol: str
    fill_qty: float
    fill_price: float
    status: FillStatus
    order_ref: Optional[str] = None
    fees: float = 0.0  # explicit fees for this fill

    @property
    def fill_value(self) -> float:
        return self.fill_qty * self.fill_price

    @property
    def avg_fill_price(self) -> float:
        return self.fill_price if self.fill_qty > 0 else 0.0


@dataclass
class Position:
    """End-of-day position snapshot."""

    date: datetime
    symbol: str
    shares: float
    market_price: float

    @property
    def market_value(self) -> float:
        return self.shares * self.market_price


@dataclass
class MarketBar:
    """Daily market data bar for a symbol."""

    date: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None

    def get_price(self, price_type: str = "close") -> float:
        if price_type == "open":
            return self.open
        if price_type == "close":
            return self.close
        if price_type == "vwap":
            return self.vwap if self.vwap is not None else self.close
        if price_type == "high":
            return self.high
        if price_type == "low":
            return self.low
        raise ValueError(f"Unknown price_type: {price_type}")


@dataclass
class AttributionResult:
    """Container for five-dimensional return decomposition."""

    period: str
    total_return: float
    benchmark_return: float
    selection_alpha: float
    allocation_effect: float
    interaction_effect: float
    execution_cost: float
    beta_return: float
    residual: float

    def validate(self, tol: float = 1e-9) -> bool:
        """Verify that components sum to total_return."""
        components = (
            self.benchmark_return
            + self.selection_alpha
            + self.allocation_effect
            + self.interaction_effect
            + self.execution_cost
            + self.beta_return
            + self.residual
        )
        return abs(components - self.total_return) < tol


@dataclass
class ExecutionCostBreakdown:
    """Detailed execution cost analysis for a single trade or period."""

    symbol: str
    side: Side
    quantity: float
    decision_price: float
    arrival_price: float
    fill_price: float
    benchmark_price: float

    # Component costs in basis points
    delay_bps: float = 0.0
    impact_bps: float = 0.0
    slippage_bps: float = 0.0
    fees_bps: float = 0.0
    total_cost_bps: float = 0.0

    # Dollar costs
    delay_dollar: float = 0.0
    impact_dollar: float = 0.0
    slippage_dollar: float = 0.0
    fees_dollar: float = 0.0
    total_cost_dollar: float = 0.0

    def compute(self):
        """Calculate all derived cost fields."""
        sign = 1.0 if self.side == Side.BUY else -1.0
        self.delay_bps = sign * (self.arrival_price - self.decision_price) / self.decision_price * 10_000
        self.impact_bps = sign * (self.fill_price - self.arrival_price) / self.arrival_price * 10_000
        self.slippage_bps = sign * (self.fill_price - self.benchmark_price) / self.benchmark_price * 10_000
        self.total_cost_bps = self.delay_bps + self.impact_bps + self.fees_bps

        gross_value = abs(self.quantity) * self.benchmark_price
        self.delay_dollar = gross_value * self.delay_bps / 10_000
        self.impact_dollar = gross_value * self.impact_bps / 10_000
        self.slippage_dollar = gross_value * self.slippage_bps / 10_000
        self.total_cost_dollar = gross_value * self.total_cost_bps / 10_000


@dataclass
class OpportunityCostResult:
    """Opportunity cost of unfilled or cancelled orders."""

    symbol: str
    unfilled_qty: float
    decision_price: float
    evaluation_price: float
    horizon_days: int

    # Cost in dollar and bps
    opportunity_cost_dollar: float = 0.0
    opportunity_cost_bps: float = 0.0
    reason: str = ""  # e.g., "liquidity", "cancelled", "superseded"

    def compute(self, side: Side):
        """Calculate opportunity cost based on side."""
        sign = 1.0 if side == Side.BUY else -1.0
        price_change = self.evaluation_price - self.decision_price
        self.opportunity_cost_dollar = sign * self.unfilled_qty * price_change
        gross_value = abs(self.unfilled_qty) * self.decision_price
        if gross_value > 0:
            self.opportunity_cost_bps = sign * price_change / self.decision_price * 10_000
