"""Main robustness testing engine.

Orchestrates adversarial market testing, parameter sensitivity analysis,
failure mode diagnosis, and composite robustness scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.adversarial_market import generate_adversarial_library
from core.failure_modes import run_all_diagnoses
from core.models import (
    AdversarialTestResult,
    RobustnessReport,
    Strategy,
    StrategyResult,
)
from core.parameter_sensitivity import grid_search_sensitivity, morris_screening, sobol_sample, stability_region_volume
from core.robustness_score import compute_robustness_score


@dataclass
class EngineConfig:
    """Configuration for the robustness testing engine."""

    adversarial_nav_threshold: float = 0.5
    param_stability_threshold: float = 0.9
    n_sobol_samples: int = 64
    n_morris_trajectories: int = 10
    seed: int = 42


class RobustnessEngine:
    """Main engine for adversarial robustness testing."""

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()

    def run_adversarial_tests(
        self,
        strategy: Strategy,
        base_params: Dict[str, float],
        paths: Optional[List] = None,
    ) -> List[AdversarialTestResult]:
        """Run strategy on adversarial path library."""
        if paths is None:
            paths = generate_adversarial_library(base_seed=self.config.seed)

        results: List[AdversarialTestResult] = []
        for path in paths:
            result = strategy.run(path.prices, **base_params)
            survival = result.final_nav >= self.config.adversarial_nav_threshold
            results.append(AdversarialTestResult(
                path_name=path.name,
                scenario_type=path.scenario_type,
                intensity=path.intensity,
                strategy_result=result,
                survival=survival,
            ))
        return results

    def run_parameter_sensitivity(
        self,
        strategy: Strategy,
        prices: pd.DataFrame,
        param_name: str,
        param_values: List[float],
        base_params: Dict[str, float],
    ) -> object:
        """Run 1D grid search sensitivity."""
        return grid_search_sensitivity(strategy, prices, param_name, param_values, base_params)

    def run_morris_screening(
        self,
        strategy: Strategy,
        prices: pd.DataFrame,
        bounds: Dict[str, Tuple[float, float]],
        base_params: Dict[str, float],
    ) -> Dict[str, object]:
        """Run Morris elementary effects screening."""
        return morris_screening(
            strategy=strategy,
            prices=prices,
            bounds=bounds,
            base_params=base_params,
            n_trajectories=self.config.n_morris_trajectories,
            seed=self.config.seed,
        )

    def run_sobol_stability(
        self,
        strategy: Strategy,
        prices: pd.DataFrame,
        bounds: Dict[str, Tuple[float, float]],
        base_params: Dict[str, float],
        baseline_metric: float,
    ) -> Tuple[pd.DataFrame, np.ndarray, float]:
        """Run Sobol sampling and compute stability region volume."""
        samples = sobol_sample(bounds, self.config.n_sobol_samples, seed=self.config.seed)
        metrics = []
        for _, row in samples.iterrows():
            params = base_params.copy()
            params.update(row.to_dict())
            result = strategy.run(prices, **params)
            metrics.append(result.sharpe_ratio)
        metrics_arr = np.array(metrics)
        stable_frac = stability_region_volume(
            samples, metrics_arr, baseline_metric, self.config.param_stability_threshold
        )
        return samples, metrics_arr, stable_frac

    def run_failure_diagnosis(
        self,
        baseline_result: StrategyResult,
        benchmark_returns: Optional[pd.Series] = None,
        factor_returns: Optional[pd.Series] = None,
        avg_daily_volumes: Optional[pd.Series] = None,
        insample_result: Optional[StrategyResult] = None,
        outsample_result: Optional[StrategyResult] = None,
    ) -> List:
        """Run full failure mode diagnosis suite."""
        return run_all_diagnoses(
            strategy_result=baseline_result,
            benchmark_returns=benchmark_returns,
            factor_returns=factor_returns,
            avg_daily_volumes=avg_daily_volumes,
            insample_result=insample_result,
            outsample_result=outsample_result,
        )

    def run_full_analysis(
        self,
        strategy: Strategy,
        strategy_name: str,
        base_params: Dict[str, float],
        baseline_prices: pd.DataFrame,
        param_sensitivity_config: Optional[Dict[str, List[float]]] = None,
        param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        benchmark_returns: Optional[pd.Series] = None,
        factor_returns: Optional[pd.Series] = None,
        avg_daily_volumes: Optional[pd.Series] = None,
        insample_result: Optional[StrategyResult] = None,
        outsample_result: Optional[StrategyResult] = None,
    ) -> RobustnessReport:
        """Run the complete robustness testing pipeline.

        Returns:
            RobustnessReport with adversarial results, sensitivity, diagnoses, and score.
        """
        # Baseline
        baseline_result = strategy.run(baseline_prices, **base_params)

        # 1. Adversarial tests
        adv_results = self.run_adversarial_tests(strategy, base_params)

        # 2. Parameter sensitivity (grid search)
        sensitivity_results: List = []
        if param_sensitivity_config:
            for param_name, param_values in param_sensitivity_config.items():
                sensitivity_results.append(
                    self.run_parameter_sensitivity(
                        strategy, baseline_prices, param_name, param_values, base_params
                    )
                )

        # 3. Sobol stability
        sobol_samples = None
        sobol_metrics = None
        if param_bounds:
            sobol_samples, sobol_metrics, _ = self.run_sobol_stability(
                strategy, baseline_prices, param_bounds, base_params, baseline_result.sharpe_ratio
            )

        # 4. Morris screening
        if param_bounds:
            morris_results = self.run_morris_screening(
                strategy, baseline_prices, param_bounds, base_params
            )
            for p, res in morris_results.items():
                sensitivity_results.append(res)

        # 5. Failure diagnosis
        diagnoses = self.run_failure_diagnosis(
            baseline_result=baseline_result,
            benchmark_returns=benchmark_returns,
            factor_returns=factor_returns,
            avg_daily_volumes=avg_daily_volumes,
            insample_result=insample_result,
            outsample_result=outsample_result,
        )

        # 6. Robustness score
        score = compute_robustness_score(
            adversarial_results=adv_results,
            sensitivity_results=sensitivity_results,
            baseline_result=baseline_result,
            sobol_samples=sobol_samples.values if sobol_samples is not None else None,
            sobol_metrics=sobol_metrics,
        )

        return RobustnessReport(
            strategy_name=strategy_name,
            timestamp=datetime.now(),
            adversarial_results=adv_results,
            sensitivity_results=sensitivity_results,
            failure_diagnoses=diagnoses,
            robustness_score=score,
        )
