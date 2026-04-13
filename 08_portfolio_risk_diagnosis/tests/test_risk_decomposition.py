"""Tests for risk decomposition module.

Core invariant: ARCs must sum to total volatility.
"""

import numpy as np
import pandas as pd
import pytest

from core.models import FactorModel, RiskSnapshot
from core.risk_decomposition import decompose_risk


class TestRiskDecomposition:
    def test_additive_identity(self):
        """ARC sum must equal total volatility."""
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
        result = decompose_risk(snapshot, factor_model)
        assert result.validate()
        assert np.isclose(result.asset_arc.sum(), result.total_volatility)

    def test_single_asset_concentration(self):
        """Single asset portfolio: factor + idio risk."""
        snapshot = RiskSnapshot(
            timestamp=pd.Timestamp("2024-01-01"),
            weights=pd.Series({"A": 1.0}),
        )
        factor_model = FactorModel(
            factor_names=["MKT"],
            exposures=pd.DataFrame({"MKT": [1.2]}, index=["A"]),
            factor_cov=pd.DataFrame([[0.04]], index=["MKT"], columns=["MKT"]),
            idio_var=pd.Series({"A": 0.0}),
        )
        result = decompose_risk(snapshot, factor_model)
        expected_vol = np.sqrt(1.2 ** 2 * 0.04)
        assert np.isclose(result.total_volatility, expected_vol)
        assert result.validate()

    def test_group_decomposition(self):
        """Multiple factors grouped into market and style."""
        snapshot = RiskSnapshot(
            timestamp=pd.Timestamp("2024-01-01"),
            weights=pd.Series({"A": 0.6, "B": 0.4}),
        )
        factor_model = FactorModel(
            factor_names=["MKT", "VALUE"],
            exposures=pd.DataFrame({"MKT": [1.0, 1.0], "VALUE": [0.5, -0.3]}, index=["A", "B"]),
            factor_cov=pd.DataFrame(
                [[0.04, 0.01], [0.01, 0.02]],
                index=["MKT", "VALUE"],
                columns=["MKT", "VALUE"],
            ),
            idio_var=pd.Series({"A": 0.01, "B": 0.01}),
        )
        groups = {"MARKET": ["MKT"], "STYLE": ["VALUE"]}
        result = decompose_risk(snapshot, factor_model, factor_groups=groups)
        assert result.validate()
        group_sum = sum(result.group_contributions.values()) + result.idio_risk
        assert np.isclose(group_sum, result.total_volatility)
        assert "MARKET" in result.group_contributions
        assert "STYLE" in result.group_contributions

    def test_zero_weights(self):
        """Zero weights -> zero volatility."""
        snapshot = RiskSnapshot(
            timestamp=pd.Timestamp("2024-01-01"),
            weights=pd.Series({"A": 0.0, "B": 0.0}),
        )
        factor_model = FactorModel(
            factor_names=["MKT"],
            exposures=pd.DataFrame({"MKT": [1.0, 1.0]}, index=["A", "B"]),
            factor_cov=pd.DataFrame([[0.04]], index=["MKT"], columns=["MKT"]),
            idio_var=pd.Series({"A": 0.01, "B": 0.01}),
        )
        result = decompose_risk(snapshot, factor_model)
        assert result.total_volatility == 0.0
        assert result.validate()

    def test_effective_n_diversified(self):
        """Equal weights among N assets -> effective_n ≈ N."""
        n = 10
        weights = pd.Series({f"S{i}": 1.0 / n for i in range(n)})
        snapshot = RiskSnapshot(
            timestamp=pd.Timestamp("2024-01-01"),
            weights=weights,
        )
        factor_model = FactorModel(
            factor_names=["MKT"],
            exposures=pd.DataFrame({"MKT": [1.0] * n}, index=[f"S{i}" for i in range(n)]),
            factor_cov=pd.DataFrame([[0.04]], index=["MKT"], columns=["MKT"]),
            idio_var=pd.Series({f"S{i}": 0.01 for i in range(n)}),
        )
        result = decompose_risk(snapshot, factor_model)
        assert np.isclose(result.effective_n, n, rtol=0.01)

    def test_effective_n_concentrated(self):
        """Single asset -> effective_n = 1."""
        snapshot = RiskSnapshot(
            timestamp=pd.Timestamp("2024-01-01"),
            weights=pd.Series({"A": 1.0}),
        )
        factor_model = FactorModel(
            factor_names=["MKT"],
            exposures=pd.DataFrame({"MKT": [1.0]}, index=["A"]),
            factor_cov=pd.DataFrame([[0.04]], index=["MKT"], columns=["MKT"]),
            idio_var=pd.Series({"A": 0.01}),
        )
        result = decompose_risk(snapshot, factor_model)
        assert np.isclose(result.effective_n, 1.0)
