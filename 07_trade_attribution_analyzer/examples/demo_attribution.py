"""
Demo: End-to-end trade attribution on synthetic data.

This script demonstrates the full pipeline of the Trade Attribution Analyzer,
including return decomposition, execution cost analysis, and opportunity cost
calculation.  It uses synthetic data so it can be run standalone.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from core.engine import AttributionEngine, EngineConfig
from core.models import Signal, Order, Fill, Position, Side, OrderType, FillStatus


def main():
    print("=" * 60)
    print("Trade Attribution Analyzer - Demo")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Setup synthetic strategy data
    # ------------------------------------------------------------------
    config = EngineConfig(
        benchmark_price_type="close",
        opportunity_horizon_days=5,
    )
    engine = AttributionEngine(config)

    # Strategy signals generated at market close on 2024-01-01
    signals = [
        Signal(
            signal_time=pd.Timestamp("2024-01-01 16:00"),
            symbol="AAPL",
            target_weight=0.40,
            signal_price=150.0,
        ),
        Signal(
            signal_time=pd.Timestamp("2024-01-01 16:00"),
            symbol="MSFT",
            target_weight=0.35,
            signal_price=250.0,
        ),
        Signal(
            signal_time=pd.Timestamp("2024-01-01 16:00"),
            symbol="TSLA",
            target_weight=0.25,
            signal_price=200.0,
        ),
    ]

    # Orders sent next morning
    orders = [
        Order(pd.Timestamp("2024-01-02 09:30"), "AAPL", Side.BUY, 100, OrderType.MARKET),
        Order(pd.Timestamp("2024-01-02 09:30"), "MSFT", Side.BUY, 50, OrderType.MARKET),
        Order(pd.Timestamp("2024-01-02 09:30"), "TSLA", Side.BUY, 40, OrderType.MARKET),
    ]

    # Actual fills (AAPL and MSFT filled with minor slippage, TSLA cancelled)
    fills = [
        Fill(pd.Timestamp("2024-01-02 10:05"), "AAPL", 100, 150.30, FillStatus.FULLY_FILLED, fees=10.0),
        Fill(pd.Timestamp("2024-01-02 10:05"), "MSFT", 50, 250.50, FillStatus.FULLY_FILLED, fees=8.0),
        # TSLA: cancelled -> zero fill qty, no Fill record
    ]

    # End-of-day positions
    positions = [
        Position(pd.Timestamp("2024-01-02"), "AAPL", 100, 150.30),
        Position(pd.Timestamp("2024-01-02"), "MSFT", 50, 250.50),
        Position(pd.Timestamp("2024-01-02"), "TSLA", 0, 202.0),
    ]

    # Market data bars
    market_data = {
        "AAPL": pd.DataFrame({
            "date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-09")],
            "open": [149.0, 150.0, 153.0],
            "high": [151.0, 151.0, 154.0],
            "low": [148.0, 149.0, 152.0],
            "close": [150.0, 150.30, 153.0],
            "volume": [1e6, 1.2e6, 1.1e6],
        }),
        "MSFT": pd.DataFrame({
            "date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-09")],
            "open": [249.0, 250.0, 252.0],
            "high": [251.0, 251.0, 253.0],
            "low": [248.0, 249.0, 251.0],
            "close": [250.0, 250.50, 252.0],
            "volume": [800_000, 900_000, 850_000],
        }),
        "TSLA": pd.DataFrame({
            "date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-09")],
            "open": [199.0, 200.0, 205.0],
            "high": [201.0, 202.0, 206.0],
            "low": [198.0, 199.0, 204.0],
            "close": [200.0, 202.0, 205.0],
            "volume": [2e6, 2.2e6, 2.1e6],
        }),
    }

    engine.load_signals(signals)
    engine.load_orders(orders)
    engine.load_fills(fills)
    engine.load_positions(positions)
    engine.load_market_data(market_data)

    # ------------------------------------------------------------------
    # 2. Run attribution
    # ------------------------------------------------------------------
    benchmark_weights = pd.Series({"AAPL": 0.33, "MSFT": 0.33, "TSLA": 0.34})
    asset_returns = pd.Series({"AAPL": 0.0020, "MSFT": 0.0020, "TSLA": 0.0100})
    eval_prices = {"AAPL": 153.0, "MSFT": 252.0, "TSLA": 205.0}  # T+5 prices

    result = engine.run_full_analysis(
        period_label="2024-01-02",
        benchmark_weights=benchmark_weights,
        asset_returns=asset_returns,
        evaluation_prices=eval_prices,
    )

    # ------------------------------------------------------------------
    # 3. Print results
    # ------------------------------------------------------------------
    ret = result["return_attribution"]
    print("\n[Return Attribution]")
    print(f"  Period           : {ret.period}")
    print(f"  Total Return     : {ret.total_return:.4%}")
    print(f"  Benchmark Return : {ret.benchmark_return:.4%}")
    print(f"  Selection Alpha  : {ret.selection_alpha:.4%}")
    print(f"  Allocation       : {ret.allocation_effect:.4%}")
    print(f"  Interaction      : {ret.interaction_effect:.4%}")
    print(f"  Execution Cost   : {ret.execution_cost:.4%}")
    print(f"  Beta Return      : {ret.beta_return:.4%}")
    print(f"  Residual         : {ret.residual:.4%}")
    print(f"  Validates        : {ret.validate()}")

    print("\n[Execution Cost Summary]")
    exec_summary = result["execution_summary"]
    if not exec_summary.empty:
        print(exec_summary)
    else:
        print("  No execution data.")

    print("\n[Implementation Shortfall]")
    isf = result["implementation_shortfall"]
    print(f"  Shortfall ($) : {isf['shortfall_dollar']:.2f}")
    print(f"  Shortfall (bps): {isf['shortfall_bps']:.2f}")

    print("\n[Opportunity Cost Summary]")
    opp_summary = result["opportunity_summary"]
    if not opp_summary.empty:
        print(opp_summary)
    else:
        print("  No opportunity cost data.")

    print("\n" + "=" * 60)
    print("Demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
