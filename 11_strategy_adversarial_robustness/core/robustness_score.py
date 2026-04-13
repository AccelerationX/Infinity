"""Robustness scoring module.

Computes a composite robustness score from adversarial survival,
parameter stability, stress test performance, and walkforward consistency.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from core.models import (
    AdversarialTestResult,
    RobustnessScore,
    SensitivityResult,
    StrategyResult,
)


def compute_adversarial_survival_score(
    results: List[AdversarialTestResult],
    nav_threshold: float = 0.5,
) -> float:
    """Fraction of adversarial scenarios where final NAV stays above threshold."""
    if not results:
        return 0.0
    survivals = [1.0 if r.survival else 0.0 for r in results]
    return float(np.mean(survivals))


def compute_parameter_stability_score(
    sensitivity_results: List[SensitivityResult],
    sobol_samples: Optional[np.ndarray] = None,
    sobol_metrics: Optional[np.ndarray] = None,
    baseline_metric: Optional[float] = None,
    threshold_ratio: float = 0.9,
) -> float:
    """Parameter stability score from Sobol samples or Morris results.

    If Sobol samples are provided, computes the volume fraction of the
    stable region. Otherwise, uses the inverse coefficient of variation
    from grid-search results as a proxy.
    """
    if sobol_samples is not None and sobol_metrics is not None and baseline_metric is not None:
        stable_count = np.sum(sobol_metrics >= baseline_metric * threshold_ratio)
        return float(stable_count / len(sobol_metrics)) if len(sobol_metrics) > 0 else 0.0

    # Fallback: average normalized stability across 1D sensitivity curves
    if not sensitivity_results:
        return 0.0

    scores = []
    for sr in sensitivity_results:
        if len(sr.metric_values) < 2:
            continue
        vals = np.array(sr.metric_values)
        # Stability = 1 - normalized standard deviation
        stability = 1.0 - (np.std(vals, ddof=1) / (np.mean(np.abs(vals)) + 1e-9))
        scores.append(max(min(stability, 1.0), 0.0))

    return float(np.mean(scores)) if scores else 0.0


def compute_stress_test_score(
    adversarial_results: List[AdversarialTestResult],
    baseline_sharpe: float,
) -> float:
    """Worst-case Sharpe under adversarial stress relative to baseline."""
    if not adversarial_results or baseline_sharpe == 0:
        return 0.0
    worst_sharpe = min(r.strategy_result.sharpe_ratio for r in adversarial_results)
    ratio = worst_sharpe / baseline_sharpe
    # Normalize: ratio can be negative; map to [0,1]
    score = (ratio + 1.0) / 2.0
    return float(max(min(score, 1.0), 0.0))


def compute_walkforward_consistency_score(
    rolling_results: List[StrategyResult],
) -> float:
    """Inverse of normalized cross-sectional standard deviation of returns."""
    if len(rolling_results) < 2:
        return 0.0
    metrics = np.array([r.sharpe_ratio for r in rolling_results])
    mean_metric = np.mean(np.abs(metrics))
    if mean_metric < 1e-9:
        return 0.0
    cv = np.std(metrics, ddof=1) / mean_metric
    # Score = exp(-cv) so that low variation -> score near 1
    return float(np.exp(-cv))


def compute_robustness_score(
    adversarial_results: List[AdversarialTestResult],
    sensitivity_results: List[SensitivityResult],
    baseline_result: StrategyResult,
    rolling_results: Optional[List[StrategyResult]] = None,
    sobol_samples: Optional[np.ndarray] = None,
    sobol_metrics: Optional[np.ndarray] = None,
    weights: Optional[Dict[str, float]] = None,
) -> RobustnessScore:
    """Compute the composite robustness score.

    Args:
        adversarial_results: Results from adversarial path testing.
        sensitivity_results: Parameter sensitivity results.
        baseline_result: Baseline strategy result on normal data.
        rolling_results: Optional rolling-window backtest results.
        sobol_samples: Optional Sobol parameter samples.
        sobol_metrics: Optional metrics at Sobol samples.
        weights: Optional custom weights for sub-scores.

    Returns:
        RobustnessScore with overall score and sub-scores.
    """
    if weights is None:
        weights = {
            "adversarial": 0.30,
            "stability": 0.25,
            "stress": 0.25,
            "walkforward": 0.20,
        }

    aas = compute_adversarial_survival_score(adversarial_results)
    pss = compute_parameter_stability_score(
        sensitivity_results,
        sobol_samples=sobol_samples,
        sobol_metrics=sobol_metrics,
        baseline_metric=baseline_result.sharpe_ratio,
    )
    sts = compute_stress_test_score(adversarial_results, baseline_result.sharpe_ratio)
    wfc = compute_walkforward_consistency_score(rolling_results or [])

    overall = (
        weights["adversarial"] * aas
        + weights["stability"] * pss
        + weights["stress"] * sts
        + weights["walkforward"] * wfc
    )

    if overall >= 0.85:
        rating = "Excellent"
    elif overall >= 0.70:
        rating = "Good"
    elif overall >= 0.55:
        rating = "Fair"
    else:
        rating = "Fragile"

    return RobustnessScore(
        overall_score=overall,
        adversarial_survival_score=aas,
        parameter_stability_score=pss,
        stress_test_score=sts,
        walkforward_consistency_score=wfc,
        weights=weights,
        rating=rating,
    )
