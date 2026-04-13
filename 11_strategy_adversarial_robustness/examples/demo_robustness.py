"""
Demo: Strategy Adversarial Robustness Testing Platform.

Demonstrates adversarial market simulation, parameter sensitivity analysis,
failure mode diagnosis, and composite robustness scoring on a synthetic
momentum strategy.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from core.engine import EngineConfig, RobustnessEngine
from core.models import StrategyResult


class SimpleMomentumStrategy:
    """A toy momentum strategy for demonstration."""

    def run(self, prices: pd.DataFrame, lookback: float = 10, top_n: float = 1, **kwargs) -> StrategyResult:
        lookback = max(2, int(round(lookback)))
        top_n = max(1, int(round(top_n)))

        rets = prices.pct_change().dropna()
        if len(rets) < lookback:
            return StrategyResult(
                final_nav=1.0, total_return=0.0, sharpe_ratio=0.0, max_drawdown=0.0, returns=pd.Series()
            )

        mom = rets.rolling(lookback).mean().dropna()
        picks = mom.idxmax(axis=1)
        strategy_rets = pd.Series(index=picks.index, dtype=float)

        for t in strategy_rets.index:
            asset = picks.loc[t]
            try:
                next_t = rets.index[rets.index.get_loc(t) + 1]
                strategy_rets.loc[t] = rets.loc[next_t, asset]
            except IndexError:
                strategy_rets.loc[t] = 0.0

        strategy_rets = strategy_rets.dropna()
        if len(strategy_rets) == 0 or strategy_rets.std(ddof=1) == 0:
            nav = 1.0 + strategy_rets.sum()
            return StrategyResult(
                final_nav=nav, total_return=nav - 1, sharpe_ratio=0.0, max_drawdown=0.0, returns=strategy_rets
            )

        nav = (1 + strategy_rets).cumprod()
        mdd = float((nav / nav.cummax() - 1).min())
        sharpe = float(strategy_rets.mean() / strategy_rets.std(ddof=1))
        return StrategyResult(
            final_nav=float(nav.iloc[-1]),
            total_return=float(nav.iloc[-1] - 1),
            sharpe_ratio=sharpe,
            max_drawdown=mdd,
            returns=strategy_rets,
        )


def main():
    print("=" * 60)
    print("Strategy Adversarial Robustness Platform - Demo")
    print("=" * 60)

    # Construct a price path with an upward trend then a crash
    prices = pd.DataFrame({
        "A": [100.0] * 30 + [110.0] * 30 + [120.0] * 30 + [90.0] * 30,
        "B": [100.0] * 120,
    })
    # Add slight noise
    prices = prices + pd.DataFrame({
        "A": [i * 0.01 for i in range(120)],
        "B": [i * 0.005 for i in range(120)],
    })

    strategy = SimpleMomentumStrategy()
    engine = RobustnessEngine(EngineConfig(seed=42))

    report = engine.run_full_analysis(
        strategy=strategy,
        strategy_name="simple_momentum",
        base_params={"lookback": 5, "top_n": 1},
        baseline_prices=prices,
        param_sensitivity_config={"lookback": [3, 5, 10, 15]},
        param_bounds={"lookback": (2.0, 20.0), "top_n": (1.0, 2.0)},
    )

    print(f"\nStrategy: {report.strategy_name}")
    print(f"Timestamp: {report.timestamp}")

    print("\n[Adversarial Test Results]")
    for r in report.adversarial_results:
        status = "SURVIVED" if r.survival else "FAILED"
        print(
            f"  {r.scenario_type:20s} (int={r.intensity:.1f}): "
            f"NAV={r.strategy_result.final_nav:.3f}, Sharpe={r.strategy_result.sharpe_ratio:.2f} [{status}]"
        )

    print("\n[Parameter Sensitivity]")
    for s in report.sensitivity_results:
        if s.values_tested:
            print(f"  Grid {s.parameter_name}: {s.values_tested} -> Sharpe={s.metric_values}")
        else:
            print(f"  Morris {s.parameter_name}: mu={s.morris_mu:.3f}, sigma={s.morris_sigma:.3f}")

    print("\n[Failure Mode Diagnosis]")
    for d in report.failure_diagnoses:
        print(f"  {d.mode.value:20s}: severity={d.severity_score:.2f} | {d.description}")

    score = report.robustness_score
    print("\n[Robustness Score]")
    print(f"  Overall : {score.overall_score:.3f} ({score.rating})")
    print(f"  Adversarial Survival : {score.adversarial_survival_score:.3f}")
    print(f"  Parameter Stability  : {score.parameter_stability_score:.3f}")
    print(f"  Stress Test          : {score.stress_test_score:.3f}")
    print(f"  Walkforward Consistency : {score.walkforward_consistency_score:.3f}")

    print("\n" + "=" * 60)
    print("Demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
