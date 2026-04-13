"""Tests for parameter search module."""

import numpy as np

from core.parameter_search import best_params, grid_search, random_search


def dummy_objective(params):
    alpha = params.get("alpha", 0.0)
    beta = params.get("beta", 0.0)
    return alpha * 2.0 - beta * 0.5


class TestGridSearch:
    def test_exhaustive(self):
        results = grid_search(dummy_objective, {"alpha": [0.0, 1.0], "beta": [0.0, 1.0]})
        assert len(results) == 4
        best = best_params(results)
        assert best["alpha"] == 1.0
        assert best["beta"] == 0.0


class TestRandomSearch:
    def test_samples_count(self):
        results = random_search(
            dummy_objective,
            {"alpha": (0.0, 1.0), "beta": (0.0, 1.0)},
            n_iter=20,
            seed=42,
        )
        assert len(results) == 20
        assert np.all(results["alpha"] >= 0.0) and np.all(results["alpha"] <= 1.0)

    def test_discrete_params(self):
        results = random_search(
            dummy_objective,
            {"alpha": (0.0, 10.0)},
            n_iter=10,
            seed=42,
            discrete_params=["alpha"],
        )
        assert all(isinstance(v, (int, np.integer)) for v in results["alpha"])


class TestBestParams:
    def test_maximize(self):
        df = grid_search(dummy_objective, {"alpha": [0.0, 1.0]})
        best = best_params(df)
        assert best["alpha"] == 1.0

    def test_minimize(self):
        df = grid_search(dummy_objective, {"alpha": [0.0, 1.0]})
        best = best_params(df, maximize=False)
        assert best["alpha"] == 0.0
