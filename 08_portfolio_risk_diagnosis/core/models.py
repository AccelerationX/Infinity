"""Core data models for portfolio risk diagnosis."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class AlertLevel(Enum):
    NORMAL = "NORMAL"
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class VaRMethod(Enum):
    PARAMETRIC = "PARAMETRIC"
    HISTORICAL = "HISTORICAL"


@dataclass
class RiskSnapshot:
    """A point-in-time snapshot of portfolio holdings."""

    timestamp: datetime
    weights: pd.Series  # indexed by symbol
    prices: Optional[pd.Series] = None  # indexed by symbol
    market_values: Optional[pd.Series] = None  # indexed by symbol

    @property
    def total_value(self) -> float:
        if self.market_values is not None:
            return float(self.market_values.sum())
        return 1.0

    @property
    def symbols(self) -> List[str]:
        return list(self.weights.index)


@dataclass
class FactorModel:
    """Multi-factor model specification."""

    factor_names: List[str]
    exposures: pd.DataFrame  # index=symbols, columns=factor_names
    factor_cov: pd.DataFrame  # index/columns=factor_names
    idio_var: pd.Series  # index=symbols

    def validate(self) -> bool:
        if list(self.exposures.columns) != self.factor_names:
            return False
        if list(self.factor_cov.index) != self.factor_names:
            return False
        if list(self.factor_cov.columns) != self.factor_names:
            return False
        return True


@dataclass
class RiskDecomposition:
    """Risk decomposition result for a single portfolio snapshot."""

    timestamp: datetime
    total_volatility: float
    factor_risk: float
    idio_risk: float
    group_contributions: Dict[str, float]  # group_name -> ARC in volatility units
    asset_mrc: pd.Series  # Marginal Risk Contribution per asset
    asset_arc: pd.Series  # Absolute Risk Contribution per asset
    effective_n: float

    def validate(self, tol: float = 1e-9) -> bool:
        """Verify that ARCs sum to total volatility."""
        arc_sum = float(self.asset_arc.sum())
        group_sum = sum(self.group_contributions.values()) + self.idio_risk
        return (
            abs(arc_sum - self.total_volatility) < tol
            and abs(group_sum - self.total_volatility) < tol
        )


@dataclass
class RiskMetrics:
    """Standard risk metrics for a portfolio."""

    timestamp: datetime
    portfolio_mean_return: float
    portfolio_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    beta: float
    tracking_error: float
    var_parametric_95: float
    var_parametric_99: float
    var_historical_95: float
    var_historical_99: float
    cvar_parametric_95: float
    cvar_parametric_99: float
    cvar_historical_95: float
    cvar_historical_99: float
    max_drawdown: float


@dataclass
class StressScenario:
    """A predefined or custom stress scenario."""

    name: str
    factor_shocks: Dict[str, float]  # factor_name -> shock in decimal
    idio_shock_scale: float = 0.0  # multiplier for idiosyncratic shocks
    correlation_override: Optional[pd.DataFrame] = None


@dataclass
class StressResult:
    """Result of running a stress scenario on a portfolio."""

    scenario_name: str
    timestamp: datetime
    expected_pnl: float  # portfolio return under scenario
    factor_pnl: Dict[str, float]  # contribution by factor
    idio_pnl: float
    stressed_volatility: Optional[float] = None


@dataclass
class RiskAlert:
    """A single risk alert triggered by the system."""

    timestamp: datetime
    metric_name: str
    metric_value: float
    threshold_value: float
    level: AlertLevel
    message: str


@dataclass
class RiskReport:
    """Complete risk diagnosis report for a portfolio snapshot."""

    timestamp: datetime
    snapshot: RiskSnapshot
    decomposition: Optional[RiskDecomposition] = None
    metrics: Optional[RiskMetrics] = None
    stress_results: List[StressResult] = field(default_factory=list)
    alerts: List[RiskAlert] = field(default_factory=list)
