"""
Demo: Portfolio Risk Diagnosis Engine on synthetic data.

Demonstrates full pipeline: risk decomposition, metrics, stress testing,
and intelligent alerting.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from core.engine import EngineConfig, RiskDiagnosisEngine
from core.models import FactorModel, RiskSnapshot
from core.risk_alerts import AlertRule


def main():
    print("=" * 60)
    print("Portfolio Risk Diagnosis Engine - Demo")
    print("=" * 60)

    np.random.seed(42)
    n = 252
    symbols = ["AAPL", "MSFT", "TSLA", "NVDA"]

    # Synthetic daily returns
    returns = pd.DataFrame(
        np.random.multivariate_normal(
            mean=[0.0008] * 4,
            cov=[[0.0004, 0.0002, 0.0001, 0.0001],
                 [0.0002, 0.0004, 0.0001, 0.0001],
                 [0.0001, 0.0001, 0.0010, 0.0003],
                 [0.0001, 0.0001, 0.0003, 0.0008]],
            size=n,
        ),
        columns=symbols,
    )
    benchmark = pd.Series(np.random.normal(0.0005, 0.015, size=n), index=returns.index)

    # Two-factor model: Market + Tech
    factor_model = FactorModel(
        factor_names=["MKT", "TECH"],
        exposures=pd.DataFrame(
            {"MKT": [1.0, 1.0, 1.3, 1.2], "TECH": [0.4, 0.5, 1.0, 1.0]},
            index=symbols,
        ),
        factor_cov=pd.DataFrame(
            [[0.0002, 0.00005], [0.00005, 0.0004]],
            index=["MKT", "TECH"],
            columns=["MKT", "TECH"],
        ),
        idio_var=pd.Series({"AAPL": 0.0002, "MSFT": 0.0002, "TSLA": 0.0005, "NVDA": 0.0003}),
    )

    # Portfolio snapshot: concentrated in tech
    snapshot = RiskSnapshot(
        timestamp=pd.Timestamp("2024-01-01"),
        weights=pd.Series({"AAPL": 0.20, "MSFT": 0.20, "TSLA": 0.35, "NVDA": 0.25}),
    )

    # Alert rules
    rules = [
        AlertRule("volatility", info_threshold=0.18, warning_threshold=0.22, critical_threshold=0.28, direction="above"),
        AlertRule("var_95", warning_threshold=-0.03, critical_threshold=-0.05, direction="below"),
        AlertRule("max_drawdown", warning_threshold=-0.15, critical_threshold=-0.25, direction="below"),
    ]

    config = EngineConfig(risk_free_rate=0.04)
    engine = RiskDiagnosisEngine(config)
    engine.load_returns_history(returns)
    engine.load_benchmark_returns(benchmark)
    engine.load_factor_model(factor_model)
    engine.load_factor_groups({"MARKET": ["MKT"], "SECTOR": ["TECH"]})
    engine.load_alert_rules(rules)

    # Custom stress scenarios matching our factor names
    from core.stress_testing import run_stress_scenario, StressScenario
    custom_scenarios = [
        StressScenario("Market Crash", {"MKT": -0.20}),
        StressScenario("Tech Sector Crash", {"MKT": -0.10, "TECH": -0.25}),
        StressScenario("Rate Rise + Tech Sell-off", {"MKT": -0.05, "TECH": -0.10}),
    ]

    report = engine.run(snapshot)

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    dec = report.decomposition
    if dec:
        print("\n[Risk Decomposition]")
        print(f"  Total Volatility : {dec.total_volatility:.4%}")
        print(f"  Effective N      : {dec.effective_n:.2f}")
        for group, arc in dec.group_contributions.items():
            pct = arc / dec.total_volatility * 100 if dec.total_volatility > 0 else 0
            print(f"  {group:12s} ARC : {arc:.4%} ({pct:.1f}%)")
        print(f"  Idiosyncratic    : {dec.idio_risk:.4%}")
        print(f"  Validates        : {dec.validate()}")

    met = report.metrics
    if met:
        print("\n[Risk Metrics]")
        print(f"  Mean Return      : {met.portfolio_mean_return:.4%}")
        print(f"  Volatility       : {met.portfolio_volatility:.4%}")
        print(f"  Sharpe Ratio     : {met.sharpe_ratio:.3f}")
        print(f"  Sortino Ratio    : {met.sortino_ratio:.3f}")
        print(f"  Beta             : {met.beta:.3f}")
        print(f"  Tracking Error   : {met.tracking_error:.4%}")
        print(f"  VaR (95%)        : {met.var_parametric_95:.4%}")
        print(f"  CVaR (95%)       : {met.cvar_parametric_95:.4%}")
        print(f"  Max Drawdown     : {met.max_drawdown:.4%}")

    print("\n[Stress Test Results]")
    for scenario in custom_scenarios:
        sr = run_stress_scenario(snapshot, factor_model, scenario)
        print(f"  {sr.scenario_name:25s}: P&L = {sr.expected_pnl:+.4%}")

    print("\n[Risk Alerts]")
    if report.alerts:
        for alert in report.alerts:
            print(f"  [{alert.level.value}] {alert.metric_name} = {alert.metric_value:.4f} | {alert.message}")
    else:
        print("  No alerts triggered.")

    print("\n" + "=" * 60)
    print("Demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
