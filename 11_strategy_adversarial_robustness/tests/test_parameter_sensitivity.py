"""Tests for parameter sensitivity module."""

import numpy as np
import pandas as pd

from core.models import StrategyResult
from core.parameter_sensitivity import (
    grid_search_sensitivity,
    morris_screening,
    sobol_sample,
    stability_region_volume,
)


class DummyStrategy:
    def run(self, prices: pd.DataFrame, **params) -> StrategyResult:
        alpha = params.get("alpha", 0.5)
        beta = params.get("beta", 0.5)
        # Simple deterministic metric for testing
        returns = pd.Series([0.01 * alpha - 0.005 * beta] * len(prices))
        # Add tiny noise to avoid std=0
        returns = returns + np.random.normal(0, 1e-6, size=len(returns))
        nav = (1 + returns).cumprod().iloc[-1]
        return StrategyResult(
            final_nav=nav,
            total_return=nav - 1,
            sharpe_ratio=float(returns.mean() / returns.std(ddof=1)),
            max_drawdown=0.0,
            returns=returns,
        )


class TestGridSearch:
    def test_monotonic_response(self):
        prices = pd.DataFrame({"A": [100.0] * 100})
        np.random.seed(42)
        strategy = DummyStrategy()
        result = grid_search_sensitivity(
            strategy, prices, "alpha", [0.1, 0.5, 1.0], {"beta": 0.5}
        )
        # Higher alpha -> higher Sharpe
        assert result.metric_values[0] < result.metric_values[1] < result.metric_values[2]


class TestSobolSample:
    def test_shape_and_bounds(self):
        bounds = {"alpha": (0.0, 1.0), "beta": (0.0, 1.0)}
        samples = sobol_sample(bounds, n_samples=32, seed=42)
        assert samples.shape == (32, 2)
        assert np.all(samples.values >= 0.0)
        assert np.all(samples.values <= 1.0)


class TestStabilityRegion:
    def test_fraction_calculation(self):
        samples = pd.DataFrame({"alpha": [0.1, 0.5, 0.9]})
        metrics = np.array([0.8, 1.0, 1.2])
        frac = stability_region_volume(samples, metrics, baseline_metric=1.0, threshold_ratio=0.9)
        assert frac == 2 / 3


class TestMorrisScreening:
    def test_identifies_important_parameter(self):
        prices = pd.DataFrame({"A": [100.0] * 100})
        strategy = DummyStrategy()
        bounds = {"alpha": (0.0, 1.0), "beta": (0.0, 1.0)}
        results = morris_screening(
            strategy, prices, bounds, {"alpha": 0.5, "beta": 0.5}, n_trajectories=5, seed=42
        )
        assert "alpha" in results
        assert "beta" in results
        # DummyStrategy has deterministic linear response; mu should be non-zero
        assert abs(results["alpha"].morris_mu) > 1e-6
