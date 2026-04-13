"""Tests for stress testing module."""

import numpy as np
import pandas as pd

from core.models import FactorModel, RiskSnapshot, StressScenario
from core.stress_testing import run_stress_scenario, monte_carlo_stress


class TestStressTesting:
    def test_market_crash_linear(self):
        """A pure market exposure portfolio should lose ~20% in market crash scenario."""
        snapshot = RiskSnapshot(
            timestamp=pd.Timestamp("2024-01-01"),
            weights=pd.Series({"A": 1.0}),
        )
        factor_model = FactorModel(
            factor_names=["MKT"],
            exposures=pd.DataFrame({"MKT": [1.0]}, index=["A"]),
            factor_cov=pd.DataFrame([[0.04]], index=["MKT"], columns=["MKT"]),
            idio_var=pd.Series({"A": 0.0}),
        )
        scenario = StressScenario(
            name="Market Crash",
            factor_shocks={"MKT": -0.20},
        )
        result = run_stress_scenario(snapshot, factor_model, scenario)
        assert np.isclose(result.expected_pnl, -0.20)

    def test_factor_contributions(self):
        """Factor contributions should sum to total P&L minus idio."""
        snapshot = RiskSnapshot(
            timestamp=pd.Timestamp("2024-01-01"),
            weights=pd.Series({"A": 0.5, "B": 0.5}),
        )
        factor_model = FactorModel(
            factor_names=["MKT", "VALUE"],
            exposures=pd.DataFrame({"MKT": [1.0, 0.8], "VALUE": [0.5, -0.2]}, index=["A", "B"]),
            factor_cov=pd.DataFrame(
                [[0.04, 0.0], [0.0, 0.02]],
                index=["MKT", "VALUE"],
                columns=["MKT", "VALUE"],
            ),
            idio_var=pd.Series({"A": 0.01, "B": 0.01}),
        )
        scenario = StressScenario(
            name="Mixed Shock",
            factor_shocks={"MKT": -0.10, "VALUE": 0.05},
            idio_shock_scale=0.02,
        )
        result = run_stress_scenario(snapshot, factor_model, scenario)
        factor_sum = sum(result.factor_pnl.values())
        assert np.isclose(result.expected_pnl, factor_sum + result.idio_pnl)

    def test_monte_carlo_stressed_vol(self):
        """Monte Carlo stress should return a stressed volatility."""
        snapshot = RiskSnapshot(
            timestamp=pd.Timestamp("2024-01-01"),
            weights=pd.Series({"A": 0.5, "B": 0.5}),
        )
        factor_model = FactorModel(
            factor_names=["MKT"],
            exposures=pd.DataFrame({"MKT": [1.0, 1.0]}, index=["A", "B"]),
            factor_cov=pd.DataFrame([[0.04]], index=["MKT"], columns=["MKT"]),
            idio_var=pd.Series({"A": 0.01, "B": 0.01}),
        )
        scenario = StressScenario(
            name="Market Crash",
            factor_shocks={"MKT": -0.20},
        )
        result = monte_carlo_stress(snapshot, factor_model, scenario, n_sims=5_000, seed=42)
        assert result.stressed_volatility is not None
        assert result.stressed_volatility > 0

    def test_monotonicity_of_shock(self):
        """Larger shock should produce larger loss (monotonicity)."""
        snapshot = RiskSnapshot(
            timestamp=pd.Timestamp("2024-01-01"),
            weights=pd.Series({"A": 1.0}),
        )
        factor_model = FactorModel(
            factor_names=["MKT"],
            exposures=pd.DataFrame({"MKT": [1.0]}, index=["A"]),
            factor_cov=pd.DataFrame([[0.04]], index=["MKT"], columns=["MKT"]),
            idio_var=pd.Series({"A": 0.0}),
        )
        s1 = run_stress_scenario(snapshot, factor_model, StressScenario("s1", {"MKT": -0.10}))
        s2 = run_stress_scenario(snapshot, factor_model, StressScenario("s2", {"MKT": -0.20}))
        assert s2.expected_pnl < s1.expected_pnl
