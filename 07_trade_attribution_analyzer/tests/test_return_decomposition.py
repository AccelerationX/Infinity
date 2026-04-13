"""Tests for return decomposition module.

Core invariant: all attribution components must sum exactly to total_return.
"""

import numpy as np
import pandas as pd
import pytest

from core.return_decomposition import brinson_attribution, holdings_based_attribution, multi_period_attribution
from core.models import AttributionResult


class TestBrinsonAttribution:
    def test_additive_identity(self):
        """Brinson components + benchmark return must equal actual portfolio return."""
        actual_w = pd.Series({"A": 0.5, "B": 0.5})
        bench_w = pd.Series({"A": 0.6, "B": 0.4})
        returns = pd.Series({"A": 0.10, "B": 0.02})
        bench_ret = float(bench_w.dot(returns))
        actual_ret = float(actual_w.reindex(returns.index).fillna(0.0).dot(returns))

        result = brinson_attribution(actual_w, bench_w, returns, bench_ret)

        total = result["benchmark_return"] + result["selection"] + result["allocation"] + result["interaction"]
        assert np.isclose(total, actual_ret)

    def test_zero_active_weights(self):
        """If actual == benchmark, selection+allocation+interaction should be zero."""
        w = pd.Series({"A": 0.5, "B": 0.5})
        returns = pd.Series({"A": 0.10, "B": 0.02})
        bench_ret = float(w.dot(returns))

        result = brinson_attribution(w, w, returns, bench_ret)
        assert np.isclose(result["selection"], 0.0)
        assert np.isclose(result["allocation"], 0.0)
        assert np.isclose(result["interaction"], 0.0)

    def test_missing_assets(self):
        """Should handle mismatched indices gracefully."""
        actual_w = pd.Series({"A": 1.0})
        bench_w = pd.Series({"A": 0.5, "B": 0.5})
        returns = pd.Series({"A": 0.10, "B": 0.02})
        bench_ret = float(bench_w.dot(returns))
        actual_ret = float(actual_w.reindex(returns.index).fillna(0.0).dot(returns))

        result = brinson_attribution(actual_w, bench_w, returns, bench_ret)
        total = result["benchmark_return"] + result["selection"] + result["allocation"] + result["interaction"]
        assert np.isclose(total, actual_ret)


class TestHoldingsBasedAttribution:
    def test_perfect_execution(self):
        """If paper == actual, execution_cost should be zero."""
        paper = pd.DataFrame({"weight": [0.5, 0.5], "return": [0.10, 0.02]}, index=["A", "B"])
        actual = pd.DataFrame({"weight": [0.5, 0.5], "return": [0.10, 0.02]}, index=["A", "B"])
        bench_w = pd.Series({"A": 0.6, "B": 0.4})
        market_ret = pd.Series([0.01] * 100)

        result = holdings_based_attribution(paper, actual, bench_w, market_ret, "2024Q1")
        assert np.isclose(result.execution_cost, 0.0)
        assert result.validate()

    def test_execution_cost_positive(self):
        """If actual underperforms paper due to execution, execution_cost > 0."""
        paper = pd.DataFrame({"weight": [0.5, 0.5], "return": [0.10, 0.02]}, index=["A", "B"])
        # Actual has worse returns (e.g., due to slippage)
        actual = pd.DataFrame({"weight": [0.5, 0.5], "return": [0.08, 0.01]}, index=["A", "B"])
        bench_w = pd.Series({"A": 0.6, "B": 0.4})
        market_ret = pd.Series([0.01] * 100)

        result = holdings_based_attribution(paper, actual, bench_w, market_ret, "2024Q1")
        paper_ret = float(np.sum(paper["weight"] * paper["return"]))
        actual_ret = float(np.sum(actual["weight"] * actual["return"]))
        assert np.isclose(result.execution_cost, paper_ret - actual_ret)
        assert result.validate()

    def test_additive_validation(self):
        """Every generated AttributionResult must pass validate()."""
        paper = pd.DataFrame({
            "weight": np.random.dirichlet(np.ones(5)),
            "return": np.random.normal(0.01, 0.05, 5),
        }, index=[f"S{i}" for i in range(5)])
        actual = pd.DataFrame({
            "weight": np.random.dirichlet(np.ones(5)),
            "return": paper["return"].values + np.random.normal(0, 0.005, 5),
        }, index=[f"S{i}" for i in range(5)])
        bench_w = pd.Series(np.random.dirichlet(np.ones(5)), index=[f"S{i}" for i in range(5)])
        market_ret = pd.Series([0.01] * 100)

        result = holdings_based_attribution(paper, actual, bench_w, market_ret, "rand")
        assert result.validate()


class TestMultiPeriodAttribution:
    def test_arithmetic_linking(self):
        """Arithmetic linking should sum components."""
        results = [
            AttributionResult("Q1", 0.10, 0.05, 0.02, 0.01, 0.01, -0.005, 0.005, -0.005),
            AttributionResult("Q2", 0.08, 0.04, 0.015, 0.01, 0.005, -0.003, 0.003, 0.002),
        ]
        agg = multi_period_attribution(results, linking_method="arithmetic")
        assert np.isclose(agg.total_return, 0.18)
        assert np.isclose(agg.benchmark_return, 0.09)
        assert np.isclose(agg.selection_alpha, 0.035)
        assert agg.validate()

    def test_geometric_linking(self):
        """Geometric linking should compound total_return."""
        results = [
            AttributionResult("Q1", 0.10, 0.05, 0.02, 0.01, 0.01, -0.005, 0.005, -0.005),
            AttributionResult("Q2", 0.08, 0.04, 0.015, 0.01, 0.005, -0.003, 0.003, 0.002),
        ]
        agg = multi_period_attribution(results, linking_method="geometric")
        expected_total = (1.10) * (1.08) - 1
        assert np.isclose(agg.total_return, expected_total)
        assert agg.validate()

    def test_empty_results_raises(self):
        with pytest.raises(ValueError):
            multi_period_attribution([])
