"""Execution attribution and Transaction Cost Analysis (TCA).

Provides rigorous decomposition of execution costs into:
  - Delay cost (from decision to arrival)
  - Market impact (from arrival to fill)
  - Slippage (fill vs benchmark price)
  - Fees (explicit transaction costs)

Reference:
  - Perold, A.F. (1988). "The Implementation Shortfall"
  - Kissell & Glantz (2003). "Optimal Trading Strategies"
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.models import ExecutionCostBreakdown, Fill, Order, Side


def compute_execution_cost(
    fill: Fill,
    order: Order,
    decision_price: float,
    arrival_price: float,
    benchmark_price: float,
) -> ExecutionCostBreakdown:
    """Compute full TCA breakdown for a single fill.

    Args:
        fill: The executed fill.
        order: The originating order.
        decision_price: Price at signal generation / decision time.
        arrival_price: Price when order arrived at market.
        benchmark_price: Benchmark price for slippage calculation (e.g., VWAP, close).

    Returns:
        ExecutionCostBreakdown with all fields computed.
    """
    result = ExecutionCostBreakdown(
        symbol=fill.symbol,
        side=order.side,
        quantity=fill.fill_qty,
        decision_price=decision_price,
        arrival_price=arrival_price,
        fill_price=fill.fill_price,
        benchmark_price=benchmark_price,
        fees_dollar=fill.fees,
    )

    # Fees in bps
    gross_value = abs(fill.fill_qty) * benchmark_price
    if gross_value > 0:
        result.fees_bps = fill.fees / gross_value * 10_000

    result.compute()
    return result


def compute_vwap_benchmark(
    trades: pd.DataFrame,
    symbol: str,
    date: pd.Timestamp,
    quantity: float,
    side: Side,
) -> float:
    """Compute the volume-weighted average price (VWAP) benchmark.

    For a realistic benchmark, this function should be fed with intraday tick
    or minute-bar data.  Here we provide the formula assuming `trades` has
    columns ['datetime', 'symbol', 'price', 'size'].

    Args:
        trades: Intraday trade records.
        symbol: Target symbol.
        date: Trading date.
        quantity: Target quantity (used to determine participation window).
        side: BUY or SELL.

    Returns:
        VWAP price for the target quantity.
    """
    day_trades = trades[
        (trades["symbol"] == symbol) & (trades["datetime"].dt.date == date.date())
    ].copy()

    if day_trades.empty:
        raise ValueError(f"No trade data found for {symbol} on {date.date()}")

    day_trades = day_trades.sort_values("datetime")
    total_volume = day_trades["size"].sum()

    if quantity >= total_volume:
        return float(np.average(day_trades["price"], weights=day_trades["size"]))

    # Walk forward until we fill the target quantity
    cumsum = 0.0
    vwap_num = 0.0
    for _, row in day_trades.iterrows():
        available = row["size"]
        take = min(available, quantity - cumsum)
        vwap_num += take * row["price"]
        cumsum += take
        if cumsum >= quantity:
            break

    return vwap_num / cumsum if cumsum > 0 else day_trades["price"].iloc[0]


def summarize_execution_costs(
    breakdowns: List[ExecutionCostBreakdown],
    by_symbol: bool = False,
) -> pd.DataFrame:
    """Summarize a list of ExecutionCostBreakdown objects.

    Args:
        breakdowns: List of individual trade breakdowns.
        by_symbol: If True, group by symbol; otherwise return overall summary.

    Returns:
        DataFrame with aggregated metrics.
    """
    if not breakdowns:
        return pd.DataFrame()

    df = pd.DataFrame([b.__dict__ for b in breakdowns])

    numeric_cols = [
        "delay_bps",
        "impact_bps",
        "slippage_bps",
        "fees_bps",
        "total_cost_bps",
        "delay_dollar",
        "impact_dollar",
        "slippage_dollar",
        "fees_dollar",
        "total_cost_dollar",
    ]

    if by_symbol:
        grouped = df.groupby("symbol")[numeric_cols].agg(["mean", "sum", "count"])
        return grouped.reset_index()

    summary = {}
    for col in numeric_cols:
        summary[col] = {
            "mean": df[col].mean(),
            "sum": df[col].sum(),
            "count": len(df),
            "median": df[col].median(),
            "std": df[col].std(),
        }

    return pd.DataFrame(summary).T


def implementation_shortfall(
    fills: List[Fill],
    orders: List[Order],
    decision_prices: Dict[str, float],
) -> Dict[str, float]:
    """Perold (1988) implementation shortfall at the portfolio level.

    Implementation Shortfall = Paper Return - Actual Return
                             = Cost of all frictions from decision to execution

    Args:
        fills: All executed fills in the period.
        orders: All orders in the period (for sign and quantity).
        decision_prices: Mapping from symbol to decision price.

    Returns:
        Dictionary with shortfall in dollar and bps.
    """
    if not fills:
        return {"shortfall_dollar": 0.0, "shortfall_bps": 0.0}

    total_shortfall = 0.0
    total_decision_value = 0.0

    order_map = {o.symbol: o for o in orders}

    for fill in fills:
        order = order_map.get(fill.symbol)
        if order is None:
            continue
        decision_price = decision_prices.get(fill.symbol)
        if decision_price is None or decision_price <= 0:
            continue

        sign = 1.0 if order.side == Side.BUY else -1.0
        shortfall = sign * fill.fill_qty * (fill.fill_price - decision_price)
        total_shortfall += shortfall + fill.fees
        total_decision_value += abs(fill.fill_qty) * decision_price

    shortfall_bps = (
        total_shortfall / total_decision_value * 10_000 if total_decision_value > 0 else 0.0
    )

    return {
        "shortfall_dollar": total_shortfall,
        "shortfall_bps": shortfall_bps,
    }
