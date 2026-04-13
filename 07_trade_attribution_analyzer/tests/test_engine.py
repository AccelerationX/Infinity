"""End-to-end tests for the attribution engine."""

import numpy as np
import pandas as pd

from core.engine import AttributionEngine, EngineConfig
from core.models import Signal, Order, Fill, Position, Side, OrderType, FillStatus


def test_full_analysis_pipeline():
    """Run the complete engine pipeline on synthetic data."""
    config = EngineConfig(benchmark_price_type="close", opportunity_horizon_days=1)
    engine = AttributionEngine(config)

    # Signals
    signals = [
        Signal(signal_time=pd.Timestamp("2024-01-01"), symbol="AAPL", target_weight=0.5, signal_price=100.0),
        Signal(signal_time=pd.Timestamp("2024-01-01"), symbol="MSFT", target_weight=0.5, signal_price=200.0),
    ]

    # Orders
    orders = [
        Order(order_time=pd.Timestamp("2024-01-02"), symbol="AAPL", side=Side.BUY, order_qty=50),
        Order(order_time=pd.Timestamp("2024-01-02"), symbol="MSFT", side=Side.BUY, order_qty=25),
    ]

    # Fills (AAPL filled with slippage, MSFT fully filled)
    fills = [
        Fill(fill_time=pd.Timestamp("2024-01-02 10:00"), symbol="AAPL", fill_qty=50, fill_price=100.5, status=FillStatus.FULLY_FILLED, fees=5.0),
        Fill(fill_time=pd.Timestamp("2024-01-02 10:00"), symbol="MSFT", fill_qty=25, fill_price=200.0, status=FillStatus.FULLY_FILLED, fees=2.0),
    ]

    # Positions
    positions = [
        Position(date=pd.Timestamp("2024-01-02"), symbol="AAPL", shares=50, market_price=100.5),
        Position(date=pd.Timestamp("2024-01-02"), symbol="MSFT", shares=25, market_price=200.0),
    ]

    # Market data
    market_data = {
        "AAPL": pd.DataFrame({
            "date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")],
            "open": [99.0, 100.0],
            "high": [101.0, 101.0],
            "low": [98.0, 99.0],
            "close": [100.0, 100.5],
            "volume": [1e6, 1e6],
        }),
        "MSFT": pd.DataFrame({
            "date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")],
            "open": [199.0, 200.0],
            "high": [201.0, 201.0],
            "low": [198.0, 199.0],
            "close": [200.0, 200.0],
            "volume": [1e6, 1e6],
        }),
    }

    engine.load_signals(signals)
    engine.load_orders(orders)
    engine.load_fills(fills)
    engine.load_positions(positions)
    engine.load_market_data(market_data)

    benchmark_weights = pd.Series({"AAPL": 0.5, "MSFT": 0.5})
    asset_returns = pd.Series({"AAPL": 0.005, "MSFT": 0.0})
    eval_prices = {"AAPL": 101.0, "MSFT": 201.0}

    result = engine.run_full_analysis("2024-01-02", benchmark_weights, asset_returns, eval_prices)

    ret_attr = result["return_attribution"]
    assert ret_attr.validate()

    exec_bds = result["execution_breakdowns"]
    assert len(exec_bds) == 2
    aapl_cost = next(b for b in exec_bds if b.symbol == "AAPL")
    assert aapl_cost.total_cost_bps > 0

    is_shortfall = result["implementation_shortfall"]
    assert is_shortfall["shortfall_dollar"] > 0

    opp_results = result["opportunity_results"]
    # Both orders fully filled -> zero opportunity cost
    assert len(opp_results) == 0
