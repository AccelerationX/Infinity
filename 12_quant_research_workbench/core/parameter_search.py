"""Parameter search module.

Implements grid search and random search for hyperparameter optimization.
Provides a clean interface that can be integrated with external Bayesian
optimization libraries (Optuna, Ax, etc.).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd


def grid_search(
    objective: Callable[[Dict[str, Any]], float],
    param_grid: Dict[str, List[Any]],
) -> pd.DataFrame:
    """Exhaustive grid search over a discrete parameter space.

    Args:
        objective: Function that takes a parameter dict and returns a scalar metric.
        param_grid: Mapping from param_name -> list of values.

    Returns:
        DataFrame with columns for each parameter + 'metric'.
    """
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]

    results: List[Dict[str, Any]] = []
    import itertools
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        metric = objective(params)
        row = params.copy()
        row["metric"] = metric
        results.append(row)

    return pd.DataFrame(results)


def random_search(
    objective: Callable[[Dict[str, Any]], float],
    bounds: Dict[str, Tuple[float, float]],
    n_iter: int = 50,
    seed: Optional[int] = None,
    discrete_params: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Random search within continuous or bounded parameter spaces.

    Args:
        objective: Function that takes a parameter dict and returns a scalar metric.
        bounds: Mapping from param_name -> (min, max).
        n_iter: Number of random samples.
        seed: Random seed.
        discrete_params: List of parameter names that should be rounded to integers.

    Returns:
        DataFrame with columns for each parameter + 'metric'.
    """
    if seed is not None:
        np.random.seed(seed)

    keys = list(bounds.keys())
    lower = np.array([bounds[k][0] for k in keys])
    upper = np.array([bounds[k][1] for k in keys])
    discrete = set(discrete_params or [])

    results: List[Dict[str, Any]] = []
    for _ in range(n_iter):
        sample = np.random.rand(len(keys)) * (upper - lower) + lower
        params: Dict[str, Any] = {}
        for i, k in enumerate(keys):
            val = sample[i]
            if k in discrete:
                val = int(round(val))
            params[k] = val
        metric = objective(params)
        row = params.copy()
        row["metric"] = metric
        results.append(row)

    return pd.DataFrame(results)


def best_params(
    results_df: pd.DataFrame,
    metric_col: str = "metric",
    maximize: bool = True,
) -> Optional[Dict[str, Any]]:
    """Extract the best parameter combination from search results.

    Args:
        results_df: DataFrame output from grid_search or random_search.
        metric_col: Column name containing the metric.
        maximize: If True, maximize the metric; else minimize.

    Returns:
        Dictionary of best parameters, or None if results_df is empty.
    """
    if results_df.empty:
        return None
    if maximize:
        best_idx = results_df[metric_col].idxmax()
    else:
        best_idx = results_df[metric_col].idxmin()
    best_row = results_df.loc[best_idx].to_dict()
    best_row.pop(metric_col)
    return best_row


class BayesianSearchInterface:
    """Placeholder interface for integrating with external Bayesian optimization libraries.

    Usage pattern:
        def objective(params):
            metric = run_strategy(params)
            tracker.log_metrics(run_id, {"sharpe": metric})
            return metric

        interface = BayesianSearchInterface()
        interface.register_trial(params, metric)
        next_params = interface.suggest_next()  # delegated to Optuna/Ax
    """

    def __init__(self):
        self.history: List[Tuple[Dict[str, Any], float]] = []

    def register_trial(self, params: Dict[str, Any], metric: float) -> None:
        """Record a parameter-metric pair from a completed trial."""
        self.history.append((params, metric))

    def suggest_next(self) -> Dict[str, Any]:
        """Suggest next parameters.

        In a full implementation, this would delegate to Optuna/Ax/scikit-optimize.
        Here we provide a random fallback.
        """
        raise NotImplementedError(
            "BayesianSearchInterface.suggest_next() should be overridden "
            "when integrating with an actual Bayesian optimization library."
        )
