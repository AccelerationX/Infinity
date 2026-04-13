"""Tests for robustness scoring module."""

import numpy as np
import pandas as pd

from core.models import AdversarialTestResult, StrategyResult
from core.robustness_score import (
    compute_adversarial_survival_score,
    compute_parameter_stability_score,
    compute_robustness_score,
    compute_stress_test_score,
    compute_walkforward_consistency_score,
)


def _make_result(sharpe: float, nav: float = 1.0) -> StrategyResult:
    return StrategyResult(
        final_nav=nav,
        total_return=nav - 1,
        sharpe_ratio=sharpe,
        max_drawdown=0.0,
        returns=pd.Series([0.0]),
    )


class TestAdversarialSurvivalScore:
    def test_all_survive(self):
        results = [
            AdversarialTestResult("p1", "TREND", 0.5, _make_result(1.0, 1.2), True),
            AdversarialTestResult("p2", "VOL", 0.5, _make_result(1.0, 1.1), True),
        ]
        assert compute_adversarial_survival_score(results) == 1.0

    def test_half_survive(self):
        results = [
            AdversarialTestResult("p1", "TREND", 0.5, _make_result(1.0, 0.3), False),
            AdversarialTestResult("p2", "VOL", 0.5, _make_result(1.0, 1.1), True),
        ]
        assert compute_adversarial_survival_score(results) == 0.5


class TestStressTestScore:
    def test_worse_than_baseline(self):
        adv = [
            AdversarialTestResult("p1", "TREND", 0.5, _make_result(0.5), True),
        ]
        baseline = _make_result(1.0)
        score = compute_stress_test_score(adv, baseline.sharpe_ratio)
        assert 0.0 < score < 1.0

    def test_empty_results(self):
        assert compute_stress_test_score([], 1.0) == 0.0


class TestWalkforwardConsistency:
    def test_consistent_scores_high(self):
        results = [_make_result(1.0), _make_result(1.05), _make_result(0.98)]
        score = compute_walkforward_consistency_score(results)
        assert score > 0.8

    def test_inconsistent_scores_low(self):
        results = [_make_result(2.0), _make_result(-1.0), _make_result(0.5)]
        score = compute_walkforward_consistency_score(results)
        assert score < 0.8


class TestParameterStability:
    def test_sobol_stability(self):
        samples = pd.DataFrame({"a": [0.0, 0.5, 1.0]})
        metrics = np.array([0.9, 1.0, 1.1])
        score = compute_parameter_stability_score(
            [], sobol_samples=samples.values, sobol_metrics=metrics, baseline_metric=1.0
        )
        assert score == 1.0  # All above 0.9 threshold


class TestCompositeScore:
    def test_excellent_rating(self):
        adv = [AdversarialTestResult("p1", "TREND", 0.5, _make_result(1.0, 1.5), True)]
        baseline = _make_result(1.5)
        from core.parameter_sensitivity import SensitivityResult
        sens = [SensitivityResult("x", [1, 2], [1.4, 1.45, 1.55])]
        score = compute_robustness_score(adv, sens, baseline, rolling_results=[baseline, baseline])
        assert score.rating in ("Excellent", "Good")

    def test_fragile_rating(self):
        adv = [AdversarialTestResult("p1", "TREND", 0.5, _make_result(-1.0, 0.3), False)]
        baseline = _make_result(1.0)
        score = compute_robustness_score(adv, [], baseline)
        assert score.rating == "Fragile"
