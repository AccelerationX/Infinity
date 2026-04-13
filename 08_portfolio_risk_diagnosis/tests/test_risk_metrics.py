"""Tests for risk metrics module."""

import numpy as np
import pandas as pd
import pytest

from core.models import RiskSnapshot
from core.risk_metrics import compute_risk_metrics


class TestRiskMetrics:
    def test_var_consistency(self):
        """Parametric and historical VaR should be roughly consistent for normal data."""
        np.random.seed(42)
        n = 5000
        symbols = ["A", "B"]
        returns = pd.DataFrame(
            np.random.multivariate_normal([0.001, 0.001], [[0.001, 0.0005], [0.0005, 0.001]], size=n),
            columns=symbols,
        )
        snapshot = RiskSnapshot(
            timestamp=pd.Timestamp("2024-01-01"),
            weights=pd.Series({"A": 0.5, "B": 0.5}),
        )
        metrics = compute_risk_metrics(snapshot, returns)
        # For large normal sample, parametric and historical should be close
        assert abs(metrics.var_parametric_95 - metrics.var_historical_95) < 0.01

    def test_cvar_more_extreme_than_var(self):
        """CVaR should be more extreme (more negative) than VaR at same confidence."""
        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.normal(0.0, 0.02, size=(1000, 2)),
            columns=["A", "B"],
        )
        snapshot = RiskSnapshot(
            timestamp=pd.Timestamp("2024-01-01"),
            weights=pd.Series({"A": 0.5, "B": 0.5}),
        )
        metrics = compute_risk_metrics(snapshot, returns)
        assert metrics.cvar_parametric_95 <= metrics.var_parametric_95
        assert metrics.cvar_historical_95 <= metrics.var_historical_95

    def test_beta_and_tracking_error(self):
        """Beta and TE should be calculated correctly against a benchmark."""
        np.random.seed(42)
        n = 252
        market = np.random.normal(0.0, 0.01, size=n)
        # Asset A has beta=1.2, asset B has beta=0.8
        a_returns = 1.2 * market + np.random.normal(0.0, 0.005, size=n)
        b_returns = 0.8 * market + np.random.normal(0.0, 0.005, size=n)
        returns = pd.DataFrame({"A": a_returns, "B": b_returns})
        benchmark = pd.Series(market, index=returns.index)

        snapshot = RiskSnapshot(
            timestamp=pd.Timestamp("2024-01-01"),
            weights=pd.Series({"A": 0.5, "B": 0.5}),
        )
        metrics = compute_risk_metrics(snapshot, returns, benchmark_returns=benchmark)
        # Portfolio beta should be close to 1.0 (average of 1.2 and 0.8)
        assert 0.8 <= metrics.beta <= 1.2
        assert metrics.tracking_error >= 0

    def test_max_drawdown(self):
        """MDD should capture the largest peak-to-trough decline."""
        # Construct a series with a known drawdown
        r = pd.Series([0.1, -0.05, -0.05, 0.2, -0.1, -0.1])
        returns = pd.DataFrame({"A": r})
        snapshot = RiskSnapshot(
            timestamp=pd.Timestamp("2024-01-01"),
            weights=pd.Series({"A": 1.0}),
        )
        metrics = compute_risk_metrics(snapshot, returns)
        # NAV: 1.1, 1.045, 0.99275, 1.1913, 1.07217, 0.96495
        # Peak at 1.1913, trough at 0.96495 -> drawdown ~ -0.19
        assert metrics.max_drawdown < -0.15

    def test_sharpe_and_sortino(self):
        """Sharpe and Sortino should be positive for positive-mean returns."""
        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.normal(0.001, 0.01, size=(252, 2)),
            columns=["A", "B"],
        )
        snapshot = RiskSnapshot(
            timestamp=pd.Timestamp("2024-01-01"),
            weights=pd.Series({"A": 0.5, "B": 0.5}),
        )
        metrics = compute_risk_metrics(snapshot, returns, risk_free_rate=0.0)
        assert metrics.sharpe_ratio > 0
        assert metrics.sortino_ratio > 0
        # Sortino should generally be >= Sharpe for symmetric returns
        assert metrics.sortino_ratio >= metrics.sharpe_ratio
