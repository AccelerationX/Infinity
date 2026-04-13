"""Tests for experiment comparison module."""

import numpy as np
import pandas as pd

from core.experiment_compare import bootstrap_significance, build_comparison_table, param_sensitivity_grid


class TestBuildComparisonTable:
    def test_basic(self):
        runs = [{"run_id": "r1"}, {"run_id": "r2"}]
        params = [
            {"run_id": "r1", "param_name": "alpha", "param_value": 0.1},
            {"run_id": "r2", "param_name": "alpha", "param_value": 0.5},
        ]
        metrics = [
            {"run_id": "r1", "metric_name": "sharpe", "metric_value": 1.0},
            {"run_id": "r2", "metric_name": "sharpe", "metric_value": 1.5},
        ]
        df = build_comparison_table(runs, params, metrics)
        assert "param.alpha" in df.columns
        assert "metric.sharpe" in df.columns
        assert len(df) == 2
        assert df.loc[df["run_id"] == "r1", "metric.sharpe"].iloc[0] == 1.0


class TestBootstrapSignificance:
    def test_significant_difference(self):
        a = np.random.normal(1.0, 0.1, size=100)
        b = np.random.normal(0.5, 0.1, size=100)
        result = bootstrap_significance(a, b, n_bootstrap=2_000, seed=42)
        assert result["significant"] is True
        assert result["mean_diff"] > 0

    def test_no_significant_difference(self):
        a = np.random.normal(0.0, 0.1, size=100)
        b = np.random.normal(0.0, 0.1, size=100)
        result = bootstrap_significance(a, b, n_bootstrap=2_000, seed=42)
        assert result["significant"] is False


class TestParamSensitivityGrid:
    def test_1d_grid(self):
        df = pd.DataFrame({
            "param.alpha": [0.1, 0.5, 1.0],
            "metric.sharpe": [1.0, 1.2, 1.5],
        })
        grid = param_sensitivity_grid(df, "param.alpha", metric="metric.sharpe")
        assert grid.iloc[0]["metric.sharpe"] == 1.0
