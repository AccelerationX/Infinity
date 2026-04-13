"""Experiment comparison and difference analysis module."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def build_comparison_table(
    runs: List[Dict[str, Any]],
    params: List[Dict[str, Any]],
    metrics: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Build a wide comparison table from runs, params, and metrics.

    Args:
        runs: List of run dicts with at least 'run_id'.
        params: List of param dicts with 'run_id', 'param_name', 'param_value'.
        metrics: List of metric dicts with 'run_id', 'metric_name', 'metric_value'.

    Returns:
        DataFrame with one row per run and columns for params/metrics.
    """
    # Pivot params
    params_df = pd.DataFrame(params)
    if not params_df.empty:
        params_wide = params_df.pivot_table(
            index="run_id", columns="param_name", values="param_value", aggfunc="first"
        )
        params_wide.columns = [f"param.{c}" for c in params_wide.columns]
    else:
        params_wide = pd.DataFrame()

    # Pivot metrics (take last step if multiple)
    metrics_df = pd.DataFrame(metrics)
    if not metrics_df.empty:
        metrics_wide = metrics_df.pivot_table(
            index="run_id", columns="metric_name", values="metric_value", aggfunc="last"
        )
        metrics_wide.columns = [f"metric.{c}" for c in metrics_wide.columns]
    else:
        metrics_wide = pd.DataFrame()

    runs_df = pd.DataFrame(runs).set_index("run_id")
    combined = runs_df.join(params_wide, how="left").join(metrics_wide, how="left")
    return combined.reset_index()


def bootstrap_significance(
    metric_values_a: np.ndarray,
    metric_values_b: np.ndarray,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Estimate bootstrap confidence interval for the difference between two metric distributions.

    Args:
        metric_values_a: Array of metric values for group A.
        metric_values_b: Array of metric values for group B.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level (e.g., 0.95).
        seed: Random seed.

    Returns:
        Dictionary with mean difference, CI lower/upper, and significance flag.
    """
    if seed is not None:
        np.random.seed(seed)

    a = np.asarray(metric_values_a)
    b = np.asarray(metric_values_b)

    diffs = []
    for _ in range(n_bootstrap):
        a_sample = np.random.choice(a, size=len(a), replace=True)
        b_sample = np.random.choice(b, size=len(b), replace=True)
        diffs.append(float(np.mean(a_sample) - np.mean(b_sample)))

    diffs = np.array(diffs)
    alpha = 1 - confidence
    ci_low = float(np.quantile(diffs, alpha / 2))
    ci_high = float(np.quantile(diffs, 1 - alpha / 2))
    mean_diff = float(np.mean(a) - np.mean(b))
    significant = not (ci_low <= 0 <= ci_high)

    return {
        "mean_diff": mean_diff,
        "ci_lower": ci_low,
        "ci_upper": ci_high,
        "significant": significant,
        "confidence": confidence,
    }


def param_sensitivity_grid(
    comparison_df: pd.DataFrame,
    param_x: str,
    param_y: Optional[str] = None,
    metric: str = "metric.sharpe_ratio",
) -> pd.DataFrame:
    """Build a sensitivity grid for one or two parameters vs a metric.

    Args:
        comparison_df: Output of build_comparison_table.
        param_x: Column name for x-axis parameter.
        param_y: Optional column name for y-axis parameter.
        metric: Column name for metric to evaluate.

    Returns:
        Pivoted DataFrame suitable for heatmap plotting.
    """
    cols = [param_x, metric]
    if param_y:
        cols.append(param_y)
    df = comparison_df[cols].dropna()

    if param_y:
        pivot = df.pivot_table(index=param_y, columns=param_x, values=metric, aggfunc="mean")
    else:
        pivot = df.groupby(param_x)[metric].mean().reset_index()
    return pivot
