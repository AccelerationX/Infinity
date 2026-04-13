"""Core data models for adversarial robustness testing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional, Protocol

import numpy as np
import pandas as pd


class FailureMode(Enum):
    REGIME_BREAKDOWN = "REGIME_BREAKDOWN"
    CROWDING_CRASH = "CROWDING_CRASH"
    LIQUIDITY_TRAP = "LIQUIDITY_TRAP"
    FACTOR_DECAY = "FACTOR_DECAY"
    DATA_SNOOPING = "DATA_SNOOPING"


class Strategy(Protocol):
    """Protocol for a strategy that can be tested by the robustness engine."""

    def run(self, prices: pd.DataFrame, **kwargs) -> StrategyResult:
        ...


@dataclass
class StrategyResult:
    """Result of running a strategy on a price path."""

    final_nav: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    returns: pd.Series
    positions: Optional[pd.DataFrame] = None
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class AdversarialPath:
    """A synthetic adversarial price path with metadata."""

    name: str
    prices: pd.DataFrame  # index=dates, columns=symbols
    scenario_type: str
    intensity: float  # 0.0 = none, 1.0 = extreme
    seed: int


@dataclass
class AdversarialTestResult:
    """Result of testing a strategy on an adversarial path."""

    path_name: str
    scenario_type: str
    intensity: float
    strategy_result: StrategyResult
    survival: bool  # final_nav > threshold


@dataclass
class SensitivityResult:
    """Parameter sensitivity analysis result."""

    parameter_name: str
    values_tested: List[float]
    metric_values: List[float]
    morris_mu: Optional[float] = None
    morris_sigma: Optional[float] = None


@dataclass
class FailureDiagnosis:
    """Diagnosis of a specific failure mode for a strategy."""

    mode: FailureMode
    severity_score: float  # 0.0 - 1.0
    indicators: Dict[str, float]
    description: str


@dataclass
class RobustnessScore:
    """Composite robustness score with sub-scores."""

    overall_score: float
    adversarial_survival_score: float
    parameter_stability_score: float
    stress_test_score: float
    walkforward_consistency_score: float
    weights: Dict[str, float]
    rating: str  # Excellent / Good / Fair / Fragile


@dataclass
class RobustnessReport:
    """Complete robustness testing report."""

    strategy_name: str
    timestamp: datetime
    adversarial_results: List[AdversarialTestResult]
    sensitivity_results: List[SensitivityResult]
    failure_diagnoses: List[FailureDiagnosis]
    robustness_score: RobustnessScore
