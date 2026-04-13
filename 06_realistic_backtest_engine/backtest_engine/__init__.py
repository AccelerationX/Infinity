"""
Realistic Backtest Engine - Anti Look-ahead Bias

A rigorous backtesting framework that prevents look-ahead bias and 
provides realistic execution simulation.

Core Principles:
1. Point-in-Time Data: Only use data available at decision time
2. Walk-forward Validation: Out-of-sample testing with rolling windows
3. Realistic Execution: Slippage, latency, market impact simulation
4. Attribution Analysis: Backtest vs Live trading gap analysis

This is NOT a toy backtester. It follows institutional-grade practices.
"""

__version__ = "1.0.0"
__author__ = "Research Team"

from .engine import BacktestEngine
from .config import BacktestConfig

__all__ = [
    "BacktestEngine",
    "BacktestConfig",
]
