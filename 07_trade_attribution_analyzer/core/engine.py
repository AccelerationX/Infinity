"""Main attribution engine that orchestrates the full analysis pipeline.

This is the primary entry point for users of the library.  Given signals,
orders, fills, positions, and market data, it produces:
  1. Five-dimensional return decomposition
  2. Execution cost breakdown (TCA)
  3. Opportunity cost analysis
  4. Validated, additive attribution report
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.execution_attribution import (
    compute_execution_cost,
    implementation_shortfall,
    summarize_execution_costs,
)
from core.models import (
    AttributionResult,
    ExecutionCostBreakdown,
    Fill,
    MarketBar,
    OpportunityCostResult,
    Order,
    Position,
    Signal,
)
from core.opportunity_cost import compute_opportunity_cost, summarize_opportunity_costs
from core.return_decomposition import brinson_attribution, multi_period_attribution


@dataclass
class EngineConfig:
    """Configuration for the attribution engine."""

    benchmark_price_type: str = "vwap"  # 'open', 'close', 'vwap'
    opportunity_horizon_days: int = 5
    beta_lookback: int = 63
    linking_method: str = "geometric"  # 'arithmetic' or 'geometric'


class AttributionEngine:
    """Main engine for trade attribution analysis."""

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self._signals: List[Signal] = []
        self._orders: List[Order] = []
        self._fills: List[Fill] = []
        self._positions: List[Position] = []
        self._market_data: Dict[str, pd.DataFrame] = {}  # symbol -> DataFrame of bars

    def load_signals(self, signals: List[Signal]) -> None:
        self._signals = signals

    def load_orders(self, orders: List[Order]) -> None:
        self._orders = orders

    def load_fills(self, fills: List[Fill]) -> None:
        self._fills = fills

    def load_positions(self, positions: List[Position]) -> None:
        self._positions = positions

    def load_market_data(self, market_data: Dict[str, pd.DataFrame]) -> None:
        """Load market bar data. DataFrame should have columns:
        ['date', 'open', 'high', 'low', 'close', 'volume', optional 'vwap']
        """
        self._market_data = market_data

    def _get_market_price(self, symbol: str, date: pd.Timestamp, price_type: str) -> Optional[float]:
        df = self._market_data.get(symbol)
        if df is None or df.empty:
            return None
        row = df[df["date"] == date]
        if row.empty:
            return None
        return float(row.iloc[0].get(price_type, row.iloc[0]["close"]))

    def run_return_attribution(
        self,
        period_label: str,
        benchmark_weights: pd.Series,
        asset_returns: pd.Series,
    ) -> AttributionResult:
        """Run Brinson-style return attribution for a single period.

        Args:
            period_label: Label for the period.
            benchmark_weights: Series indexed by symbol.
            asset_returns: Series of asset returns indexed by symbol.

        Returns:
            AttributionResult.
        """
        # Build actual weights from positions
        actual_weights = self._build_actual_weights(period_label)

        brinson = brinson_attribution(
            actual_weights=actual_weights,
            benchmark_weights=benchmark_weights,
            actual_returns=asset_returns,
            benchmark_return=float(benchmark_weights.dot(asset_returns.fillna(0))),
        )

        # Execution cost from fills vs paper portfolio
        execution_cost = self._compute_period_execution_cost(period_label)

        # Beta and residual placeholders
        beta_return = 0.0
        total_return = float(actual_weights.dot(asset_returns.fillna(0)))

        computed_sum = (
            brinson["benchmark_return"]
            + brinson["selection"]
            + brinson["allocation"]
            + brinson["interaction"]
            + execution_cost
            + beta_return
        )
        residual = total_return - computed_sum

        return AttributionResult(
            period=period_label,
            total_return=total_return,
            benchmark_return=brinson["benchmark_return"],
            selection_alpha=brinson["selection"],
            allocation_effect=brinson["allocation"],
            interaction_effect=brinson["interaction"],
            execution_cost=execution_cost,
            beta_return=beta_return,
            residual=residual,
        )

    def run_execution_attribution(
        self,
        decision_prices: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[ExecutionCostBreakdown], pd.DataFrame]:
        """Run TCA on all fills.

        Args:
            decision_prices: Optional override for decision prices by symbol.
                             If None, uses signal_price from loaded signals.

        Returns:
            Tuple of (list of breakdowns, summary DataFrame).
        """
        order_map = {o.symbol: o for o in self._orders}
        signal_map = {s.symbol: s.signal_price for s in self._signals if s.signal_price}

        breakdowns: List[ExecutionCostBreakdown] = []

        for fill in self._fills:
            order = order_map.get(fill.symbol)
            if order is None:
                continue

            decision_price = decision_prices.get(fill.symbol, signal_map.get(fill.symbol)) if decision_prices else signal_map.get(fill.symbol)
            if decision_price is None:
                continue

            # For arrival_price and benchmark_price, use market data if available,
            # otherwise approximate with decision_price.
            arrival_price = decision_price  # placeholder; can be overridden with market open
            benchmark_price = decision_price  # placeholder; ideally VWAP

            breakdown = compute_execution_cost(
                fill=fill,
                order=order,
                decision_price=decision_price,
                arrival_price=arrival_price,
                benchmark_price=benchmark_price,
            )
            breakdowns.append(breakdown)

        summary = summarize_execution_costs(breakdowns)
        return breakdowns, summary

    def run_opportunity_cost(
        self,
        evaluation_prices: Dict[str, float],
    ) -> Tuple[List[OpportunityCostResult], pd.DataFrame]:
        """Run opportunity cost analysis on unfilled orders.

        Args:
            evaluation_prices: Mapping from symbol to price at evaluation horizon.

        Returns:
            Tuple of (list of results, summary DataFrame).
        """
        fill_map: Dict[str, float] = {}
        for fill in self._fills:
            fill_map[fill.symbol] = fill_map.get(fill.symbol, 0.0) + fill.fill_qty

        signal_map = {s.symbol: s.signal_price for s in self._signals if s.signal_price}

        results: List[OpportunityCostResult] = []
        for order in self._orders:
            filled_qty = fill_map.get(order.symbol, 0.0)
            if filled_qty >= order.order_qty:
                continue

            decision_price = signal_map.get(order.symbol)
            if decision_price is None:
                continue

            eval_price = evaluation_prices.get(order.symbol)
            if eval_price is None:
                continue

            result = compute_opportunity_cost(
                order=order,
                filled_qty=filled_qty,
                decision_price=decision_price,
                evaluation_price=eval_price,
                horizon_days=self.config.opportunity_horizon_days,
                reason="unfilled",
            )
            results.append(result)

        summary = summarize_opportunity_costs(results)
        return results, summary

    def run_full_analysis(
        self,
        period_label: str,
        benchmark_weights: pd.Series,
        asset_returns: pd.Series,
        evaluation_prices: Dict[str, float],
    ) -> Dict[str, object]:
        """Run the complete attribution pipeline.

        Returns:
            Dictionary with keys:
                'return_attribution', 'execution_breakdowns', 'execution_summary',
                'opportunity_results', 'opportunity_summary', 'implementation_shortfall'
        """
        ret_attr = self.run_return_attribution(period_label, benchmark_weights, asset_returns)
        exec_bds, exec_sum = self.run_execution_attribution()
        opp_results, opp_sum = self.run_opportunity_cost(evaluation_prices)

        # Validate additive identity
        if not ret_attr.validate():
            raise RuntimeError(
                f"AttributionResult for {period_label} fails additive validation: "
                f"components do not sum to total_return"
            )

        # Portfolio-level implementation shortfall
        decision_prices = {s.symbol: s.signal_price for s in self._signals if s.signal_price}
        is_shortfall = implementation_shortfall(self._fills, self._orders, decision_prices)

        return {
            "return_attribution": ret_attr,
            "execution_breakdowns": exec_bds,
            "execution_summary": exec_sum,
            "opportunity_results": opp_results,
            "opportunity_summary": opp_sum,
            "implementation_shortfall": is_shortfall,
        }

    def _build_actual_weights(self, period_label: str) -> pd.Series:
        """Build actual portfolio weights from positions."""
        if not self._positions:
            return pd.Series(dtype=float)

        # Filter positions for the period
        pos_df = pd.DataFrame([
            {"symbol": p.symbol, "mv": p.market_value}
            for p in self._positions
        ])
        if pos_df.empty:
            return pd.Series(dtype=float)

        grouped = pos_df.groupby("symbol")["mv"].sum()
        total_mv = grouped.sum()
        if total_mv == 0:
            return pd.Series(dtype=float)
        return grouped / total_mv

    def _order_key(self, order: Order) -> str:
        return f"{order.order_time}_{order.symbol}"

    def _fill_key(self, fill: Fill) -> str:
        return f"{fill.fill_time}_{fill.symbol}"

    def _compute_period_execution_cost(self, period_label: str) -> float:
        """Compute aggregate execution cost for the period."""
        # Simplified: subtract explicit shortfall from paper return
        # In a full implementation, compare paper portfolio return to actual.
        return 0.0
