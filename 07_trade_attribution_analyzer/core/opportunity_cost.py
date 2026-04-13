"""Opportunity cost calculation for unfilled or cancelled orders.

Quantifies the profit/loss that was foregone due to orders that did not
receive full (or any) execution.  This is a critical but often overlooked
component of total execution friction.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.models import OpportunityCostResult, Order, Side


def compute_opportunity_cost(
    order: Order,
    filled_qty: float,
    decision_price: float,
    evaluation_price: float,
    horizon_days: int = 1,
    reason: str = "",
) -> OpportunityCostResult:
    """Compute opportunity cost for a partially or fully unfilled order.

    Args:
        order: The original order.
        filled_qty: Quantity that was actually filled.
        decision_price: Price at decision time (signal generation).
        evaluation_price: Price at evaluation horizon (e.g., close of T+1, T+5).
        horizon_days: Number of days in the evaluation window.
        reason: Why the order was unfilled (e.g., 'cancelled', 'liquidity').

    Returns:
        OpportunityCostResult with dollar and bps costs.
    """
    unfilled_qty = order.order_qty - filled_qty
    if unfilled_qty <= 0:
        return OpportunityCostResult(
            symbol=order.symbol,
            unfilled_qty=0.0,
            decision_price=decision_price,
            evaluation_price=evaluation_price,
            horizon_days=horizon_days,
            opportunity_cost_dollar=0.0,
            opportunity_cost_bps=0.0,
            reason=reason,
        )

    result = OpportunityCostResult(
        symbol=order.symbol,
        unfilled_qty=unfilled_qty,
        decision_price=decision_price,
        evaluation_price=evaluation_price,
        horizon_days=horizon_days,
        reason=reason,
    )
    result.compute(order.side)
    return result


def summarize_opportunity_costs(
    results: List[OpportunityCostResult],
    by_reason: bool = False,
    by_symbol: bool = False,
) -> pd.DataFrame:
    """Aggregate opportunity cost results.

    Args:
        results: List of OpportunityCostResult objects.
        by_reason: If True, group by reason category.
        by_symbol: If True, group by symbol.

    Returns:
        DataFrame with aggregated opportunity cost statistics.
    """
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame([r.__dict__ for r in results])

    group_cols = []
    if by_reason:
        group_cols.append("reason")
    if by_symbol:
        group_cols.append("symbol")

    numeric_cols = ["opportunity_cost_dollar", "opportunity_cost_bps", "unfilled_qty"]

    if group_cols:
        grouped = df.groupby(group_cols)[numeric_cols].agg(["sum", "mean", "count"])
        return grouped.reset_index()

    summary = {}
    for col in numeric_cols:
        summary[col] = {
            "sum": df[col].sum(),
            "mean": df[col].mean(),
            "median": df[col].median(),
            "count": len(df),
            "std": df[col].std(),
        }

    return pd.DataFrame(summary).T


def total_friction(
    execution_cost_dollar: float,
    opportunity_cost_dollar: float,
    decision_value: float,
) -> Dict[str, float]:
    """Total friction combining explicit execution cost and opportunity cost.

    Args:
        execution_cost_dollar: Total explicit execution cost (negative = cost).
        opportunity_cost_dollar: Total opportunity cost (negative = cost).
        decision_value: Total notional value at decision time.

    Returns:
        Dictionary with total friction in dollar and bps.
    """
    total_dollar = execution_cost_dollar + opportunity_cost_dollar
    total_bps = total_dollar / decision_value * 10_000 if decision_value > 0 else 0.0
    return {
        "execution_cost_dollar": execution_cost_dollar,
        "opportunity_cost_dollar": opportunity_cost_dollar,
        "total_friction_dollar": total_dollar,
        "total_friction_bps": total_bps,
    }
