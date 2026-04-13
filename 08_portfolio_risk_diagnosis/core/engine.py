"""Main risk diagnosis engine that orchestrates the full analysis pipeline.

This is the primary entry point for users of the library.  Given a portfolio
snapshot, historical returns, a factor model, and alert rules, it produces:
  1. Risk decomposition
  2. Standard risk metrics
  3. Stress test results
  4. Risk alerts
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from core.models import (
    FactorModel,
    RiskAlert,
    RiskDecomposition,
    RiskMetrics,
    RiskReport,
    RiskSnapshot,
    StressResult,
)
from core.risk_alerts import AlertRule, evaluate_all_alerts
from core.risk_decomposition import decompose_risk
from core.risk_metrics import compute_risk_metrics
from core.stress_testing import run_scenario_library


@dataclass
class EngineConfig:
    """Configuration for the risk diagnosis engine."""

    risk_free_rate: float = 0.0
    confidence_levels: Optional[List[float]] = None
    use_dynamic_alerts: bool = False

    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.95, 0.99]


class RiskDiagnosisEngine:
    """Main engine for portfolio risk diagnosis."""

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self._returns_history: Optional[pd.DataFrame] = None
        self._benchmark_returns: Optional[pd.Series] = None
        self._factor_model: Optional[FactorModel] = None
        self._factor_groups: Optional[Dict[str, List[str]]] = None
        self._alert_rules: List[AlertRule] = []

    def load_returns_history(self, returns: pd.DataFrame) -> None:
        """Load historical asset returns (dates x symbols)."""
        self._returns_history = returns

    def load_benchmark_returns(self, benchmark: pd.Series) -> None:
        """Load benchmark returns aligned with returns_history."""
        self._benchmark_returns = benchmark

    def load_factor_model(self, factor_model: FactorModel) -> None:
        """Load multi-factor model for risk decomposition."""
        self._factor_model = factor_model

    def load_factor_groups(self, groups: Dict[str, List[str]]) -> None:
        """Load factor group mapping."""
        self._factor_groups = groups

    def load_alert_rules(self, rules: List[AlertRule]) -> None:
        """Load alert rule configurations."""
        self._alert_rules = rules

    def run(
        self,
        snapshot: RiskSnapshot,
    ) -> RiskReport:
        """Run the complete risk diagnosis pipeline on a portfolio snapshot.

        Args:
            snapshot: Current portfolio weights and timestamp.

        Returns:
            RiskReport with decomposition, metrics, stress results, and alerts.
        """
        decomposition: Optional[RiskDecomposition] = None
        metrics: Optional[RiskMetrics] = None
        stress_results: List[StressResult] = []
        alerts: List[RiskAlert] = []

        # 1. Risk decomposition
        if self._factor_model is not None:
            decomposition = decompose_risk(
                snapshot=snapshot,
                factor_model=self._factor_model,
                factor_groups=self._factor_groups,
            )
            if not decomposition.validate():
                raise RuntimeError(
                    f"Risk decomposition failed additive validation at {snapshot.timestamp}"
                )

        # 2. Risk metrics
        if self._returns_history is not None:
            metrics = compute_risk_metrics(
                snapshot=snapshot,
                returns_history=self._returns_history,
                benchmark_returns=self._benchmark_returns,
                risk_free_rate=self.config.risk_free_rate,
                confidence_levels=self.config.confidence_levels,
            )

        # 3. Stress testing
        if self._factor_model is not None:
            stress_results = run_scenario_library(
                snapshot=snapshot,
                factor_model=self._factor_model,
            )

        # 4. Alerts
        if self._alert_rules and metrics is not None:
            metric_values = {
                "volatility": metrics.portfolio_volatility,
                "var_95": metrics.var_parametric_95,
                "cvar_95": metrics.cvar_parametric_95,
                "max_drawdown": metrics.max_drawdown,
                "beta": metrics.beta,
                "tracking_error": metrics.tracking_error,
            }
            alerts = evaluate_all_alerts(
                timestamp=snapshot.timestamp,
                metrics=metric_values,
                rules=self._alert_rules,
            )

        return RiskReport(
            timestamp=snapshot.timestamp,
            snapshot=snapshot,
            decomposition=decomposition,
            metrics=metrics,
            stress_results=stress_results,
            alerts=alerts,
        )
