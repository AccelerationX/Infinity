"""Parameter sensitivity analysis module.

Implements grid search, Sobol sequence sampling, and Morris screening
for systematic assessment of strategy parameter robustness.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.models import SensitivityResult, Strategy, StrategyResult


def grid_search_sensitivity(
    strategy: Strategy,
    prices: pd.DataFrame,
    param_name: str,
    param_values: List[float],
    base_params: Dict[str, float],
) -> SensitivityResult:
    """Run a one-dimensional grid search over a single parameter.

    Args:
        strategy: Strategy to test.
        prices: Price data.
        param_name: Parameter to vary.
        param_values: Values to test.
        base_params: Baseline parameter dictionary.

    Returns:
        SensitivityResult with final NAV for each parameter value.
    """
    metric_values: List[float] = []
    for val in param_values:
        params = base_params.copy()
        params[param_name] = val
        result = strategy.run(prices, **params)
        metric_values.append(result.sharpe_ratio)

    return SensitivityResult(
        parameter_name=param_name,
        values_tested=param_values,
        metric_values=metric_values,
    )


def sobol_sample(
    bounds: Dict[str, Tuple[float, float]],
    n_samples: int,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate Sobol low-discrepancy samples within parameter bounds.

    Args:
        bounds: Mapping from param_name -> (min, max).
        n_samples: Number of samples.
        seed: Random seed.

    Returns:
        DataFrame of samples (n_samples x n_params).
    """
    try:
        from scipy.stats import qmc
    except ImportError:
        # Fallback to Latin Hypercube Sampling if scipy unavailable
        return _lhs_fallback(bounds, n_samples, seed)

    param_names = list(bounds.keys())
    n_dims = len(param_names)
    sampler = qmc.Sobol(d=n_dims, scramble=True, seed=seed)
    samples = sampler.random(n=n_samples)

    # Scale to bounds
    lower = np.array([bounds[p][0] for p in param_names])
    upper = np.array([bounds[p][1] for p in param_names])
    scaled = qmc.scale(samples, lower, upper)

    return pd.DataFrame(scaled, columns=param_names)


def _lhs_fallback(
    bounds: Dict[str, Tuple[float, float]],
    n_samples: int,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Latin Hypercube Sampling fallback when scipy is unavailable."""
    if seed is not None:
        np.random.seed(seed)

    param_names = list(bounds.keys())
    n_dims = len(param_names)
    samples = np.zeros((n_samples, n_dims))

    for d in range(n_dims):
        # Divide each dimension into n_samples equal-probability bins
        cut = np.linspace(0, 1, n_samples + 1)
        u = np.random.rand(n_samples)
        a = cut[:n_samples]
        b = cut[1:n_samples + 1]
        samples[:, d] = u * (b - a) + a
        np.random.shuffle(samples[:, d])

    # Scale to bounds
    lower = np.array([bounds[p][0] for p in param_names])
    upper = np.array([bounds[p][1] for p in param_names])
    scaled = samples * (upper - lower) + lower

    return pd.DataFrame(scaled, columns=param_names)


def morris_screening(
    strategy: Strategy,
    prices: pd.DataFrame,
    bounds: Dict[str, Tuple[float, float]],
    base_params: Dict[str, float],
    n_trajectories: int = 10,
    n_levels: int = 4,
    metric: str = "sharpe_ratio",
    seed: Optional[int] = None,
) -> Dict[str, SensitivityResult]:
    """Morris elementary effects screening for parameter sensitivity.

    Args:
        strategy: Strategy to test.
        prices: Price data.
        bounds: Parameter bounds.
        base_params: Baseline params (only params in bounds are screened).
        n_trajectories: Number of Morris trajectories.
        n_levels: Number of grid levels per parameter.
        metric: Metric to evaluate ("sharpe_ratio", "total_return", "max_drawdown").
        seed: Random seed.

    Returns:
        Dictionary mapping param_name -> SensitivityResult with Morris mu/sigma.
    """
    if seed is not None:
        np.random.seed(seed)

    param_names = list(bounds.keys())
    n_params = len(param_names)
    if n_params == 0:
        return {}

    delta = 1.0 / (n_levels - 1) if n_levels > 1 else 0.5
    lower = np.array([bounds[p][0] for p in param_names])
    upper = np.array([bounds[p][1] for p in param_names])

    elementary_effects: Dict[str, List[float]] = {p: [] for p in param_names}

    for _ in range(n_trajectories):
        # Random baseline point in [0,1]^p
        x_base = np.random.rand(n_params)
        # Ensure we can add delta without exceeding 1
        x_base = np.where(x_base > 1 - delta, x_base - delta, x_base)

        for i, p in enumerate(param_names):
            x_1 = x_base.copy()
            x_2 = x_base.copy()
            x_2[i] += delta

            # Map to actual parameter values
            params_1 = _map_params(x_1, param_names, lower, upper, base_params)
            params_2 = _map_params(x_2, param_names, lower, upper, base_params)

            y1 = _evaluate_metric(strategy, prices, params_1, metric)
            y2 = _evaluate_metric(strategy, prices, params_2, metric)

            # Elementary effect
            ee = (y2 - y1) / delta
            elementary_effects[p].append(ee)

    results: Dict[str, SensitivityResult] = {}
    for p in param_names:
        ees = elementary_effects[p]
        mu = float(np.mean(ees))
        sigma = float(np.std(ees, ddof=1)) if len(ees) > 1 else 0.0
        results[p] = SensitivityResult(
            parameter_name=p,
            values_tested=[],
            metric_values=[],
            morris_mu=mu,
            morris_sigma=sigma,
        )

    return results


def _map_params(
    x: np.ndarray,
    param_names: List[str],
    lower: np.ndarray,
    upper: np.ndarray,
    base_params: Dict[str, float],
) -> Dict[str, float]:
    """Map normalized x in [0,1] to actual parameter values."""
    params = base_params.copy()
    actual = x * (upper - lower) + lower
    for i, p in enumerate(param_names):
        params[p] = actual[i]
    return params


def _evaluate_metric(
    strategy: Strategy,
    prices: pd.DataFrame,
    params: Dict[str, float],
    metric: str,
) -> float:
    """Run strategy and extract metric."""
    result = strategy.run(prices, **params)
    if metric == "sharpe_ratio":
        return result.sharpe_ratio
    if metric == "total_return":
        return result.total_return
    if metric == "max_drawdown":
        return result.max_drawdown
    if metric == "final_nav":
        return result.final_nav
    if metric in result.metrics:
        return result.metrics[metric]
    raise ValueError(f"Unknown metric: {metric}")


def stability_region_volume(
    param_samples: pd.DataFrame,
    metric_values: np.ndarray,
    baseline_metric: float,
    threshold_ratio: float = 0.9,
) -> float:
    """Compute the fraction of parameter space where metric >= threshold_ratio * baseline.

    Args:
        param_samples: DataFrame of parameter samples.
        metric_values: Array of corresponding metric values.
        baseline_metric: Baseline metric value.
        threshold_ratio: Minimum acceptable ratio.

    Returns:
        Fraction of samples in the stable region.
    """
    threshold = baseline_metric * threshold_ratio
    stable_count = np.sum(metric_values >= threshold)
    return float(stable_count / len(metric_values)) if len(metric_values) > 0 else 0.0
