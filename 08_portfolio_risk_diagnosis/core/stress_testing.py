"""Stress testing and scenario analysis module.

Provides linear approximate stress P&L and Monte-Carlo-based
stressed volatility estimates under predefined or custom scenarios.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.models import FactorModel, RiskSnapshot, StressResult, StressScenario


def run_stress_scenario(
    snapshot: RiskSnapshot,
    factor_model: FactorModel,
    scenario: StressScenario,
) -> StressResult:
    """Run a single stress scenario using linear factor approximation.

    Args:
        snapshot: Portfolio weights.
        factor_model: Multi-factor model.
        scenario: Stress scenario specification.

    Returns:
        StressResult with expected P&L and factor contributions.
    """
    w = snapshot.weights.reindex(factor_model.exposures.index).fillna(0.0).values
    B = factor_model.exposures.values
    Sigma_f = factor_model.factor_cov.values
    factor_names = factor_model.factor_names

    # Portfolio factor exposures (k x 1)
    port_exp = B.T @ w

    # Factor P&L
    factor_pnl: Dict[str, float] = {}
    total_factor_pnl = 0.0
    for i, fname in enumerate(factor_names):
        shock = scenario.factor_shocks.get(fname, 0.0)
        contrib = port_exp[i] * shock
        factor_pnl[fname] = float(contrib)
        total_factor_pnl += contrib

    # Idiosyncratic P&L (approximate as zero mean, but scaled stress)
    idio_pnl = 0.0
    if scenario.idio_shock_scale != 0.0:
        D_diag = factor_model.idio_var.reindex(factor_model.exposures.index).fillna(0.0).values
        # Approximate idio shock as a proportional draw from idio vol
        idio_vol_per_asset = np.sqrt(D_diag)
        idio_pnl = float(np.sum(w * idio_vol_per_asset * scenario.idio_shock_scale))

    expected_pnl = total_factor_pnl + idio_pnl

    return StressResult(
        scenario_name=scenario.name,
        timestamp=snapshot.timestamp,
        expected_pnl=expected_pnl,
        factor_pnl=factor_pnl,
        idio_pnl=idio_pnl,
    )


def run_scenario_library(
    snapshot: RiskSnapshot,
    factor_model: FactorModel,
    library: Optional[List[StressScenario]] = None,
) -> List[StressResult]:
    """Run a library of predefined stress scenarios."""
    if library is None:
        library = _default_scenario_library()
    return [run_stress_scenario(snapshot, factor_model, s) for s in library]


def monte_carlo_stress(
    snapshot: RiskSnapshot,
    factor_model: FactorModel,
    scenario: StressScenario,
    n_sims: int = 10_000,
    seed: Optional[int] = None,
) -> StressResult:
    """Monte Carlo stress test by shifting factor distribution mean.

    Args:
        snapshot: Portfolio weights.
        factor_model: Multi-factor model.
        scenario: Stress scenario (used for factor mean shifts).
        n_sims: Number of simulations.
        seed: Random seed for reproducibility.

    Returns:
        StressResult enriched with stressed_volatility from simulations.
    """
    if seed is not None:
        np.random.seed(seed)

    w = snapshot.weights.reindex(factor_model.exposures.index).fillna(0.0).values
    B = factor_model.exposures.values
    Sigma_f = factor_model.factor_cov.values
    D_diag = factor_model.idio_var.reindex(factor_model.exposures.index).fillna(0.0).values

    k = len(factor_model.factor_names)
    factor_means = np.array([scenario.factor_shocks.get(f, 0.0) for f in factor_model.factor_names])

    # Simulate factor returns
    factor_sims = np.random.multivariate_normal(factor_means, Sigma_f, size=n_sims)
    idio_sims = np.random.normal(0.0, np.sqrt(D_diag), size=(n_sims, len(w)))

    asset_returns = factor_sims @ B.T + idio_sims
    portfolio_returns = asset_returns @ w

    linear_result = run_stress_scenario(snapshot, factor_model, scenario)
    linear_result.stressed_volatility = float(portfolio_returns.std(ddof=1))
    return linear_result


def _default_scenario_library() -> List[StressScenario]:
    """Return a set of institutionally relevant stress scenarios."""
    return [
        StressScenario(
            name="Market Crash",
            factor_shocks={"MARKET": -0.20},
        ),
        StressScenario(
            name="Liquidity Crisis",
            factor_shocks={"LIQUIDITY": -0.50, "VOLATILITY": 0.30},
        ),
        StressScenario(
            name="Style Reversal",
            factor_shocks={"VALUE": 0.15, "MOMENTUM": -0.15},
        ),
        StressScenario(
            name="Interest Rate Rise",
            factor_shocks={"RATE": 0.01},
        ),
    ]
