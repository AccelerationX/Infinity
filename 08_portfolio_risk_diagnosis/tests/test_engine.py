"""End-to-end tests for the risk diagnosis engine."""

import numpy as np
import pandas as pd

from core.engine import EngineConfig, RiskDiagnosisEngine
from core.models import FactorModel, RiskSnapshot
from core.risk_alerts import AlertRule


def test_engine_full_pipeline():
    """Run the complete engine pipeline on synthetic data."""
    np.random.seed(42)
    n = 252
    symbols = ["AAPL", "MSFT", "TSLA"]
    returns = pd.DataFrame(
        np.random.multivariate_normal(
            [0.001, 0.001, 0.001],
            [[0.0004, 0.0002, 0.0001],
             [0.0002, 0.0004, 0.0001],
             [0.0001, 0.0001, 0.0009]],
            size=n,
        ),
        columns=symbols,
    )
    benchmark = pd.Series(np.random.normal(0.001, 0.015, size=n), index=returns.index)

    factor_model = FactorModel(
        factor_names=["MKT", "TECH"],
        exposures=pd.DataFrame(
            {"MKT": [1.0, 1.0, 1.2], "TECH": [0.5, 0.5, 1.0]},
            index=symbols,
        ),
        factor_cov=pd.DataFrame(
            [[0.0002, 0.00005], [0.00005, 0.0003]],
            index=["MKT", "TECH"],
            columns=["MKT", "TECH"],
        ),
        idio_var=pd.Series({"AAPL": 0.0002, "MSFT": 0.0002, "TSLA": 0.0004}),
    )

    snapshot = RiskSnapshot(
        timestamp=pd.Timestamp("2024-01-01"),
        weights=pd.Series({"AAPL": 0.4, "MSFT": 0.35, "TSLA": 0.25}),
    )

    rules = [
        AlertRule("volatility", info_threshold=0.15, direction="above"),
        AlertRule("max_drawdown", warning_threshold=-0.10, direction="below"),
    ]

    config = EngineConfig(risk_free_rate=0.03)
    engine = RiskDiagnosisEngine(config)
    engine.load_returns_history(returns)
    engine.load_benchmark_returns(benchmark)
    engine.load_factor_model(factor_model)
    engine.load_factor_groups({"MARKET": ["MKT"], "SECTOR": ["TECH"]})
    engine.load_alert_rules(rules)

    report = engine.run(snapshot)

    assert report.decomposition is not None
    assert report.decomposition.validate()
    assert report.metrics is not None
    assert len(report.stress_results) > 0
    assert isinstance(report.alerts, list)

    # Verify stress results contain expected scenarios
    scenario_names = {s.scenario_name for s in report.stress_results}
    assert "Market Crash" in scenario_names
