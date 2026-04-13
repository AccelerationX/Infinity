"""End-to-end tests for the robustness engine."""

import pandas as pd

from core.engine import EngineConfig, RobustnessEngine
from core.models import StrategyResult


class DummyMomentumStrategy:
    """A simple momentum strategy for testing."""

    def run(self, prices: pd.DataFrame, lookback: float = 10, top_n: float = 1, **kwargs) -> StrategyResult:
        lookback = max(2, int(round(lookback)))
        top_n = max(1, int(round(top_n)))
        rets = prices.pct_change().dropna()
        if len(rets) < lookback:
            return StrategyResult(
                final_nav=1.0, total_return=0.0, sharpe_ratio=0.0, max_drawdown=0.0, returns=pd.Series()
            )
        mom = rets.rolling(lookback).mean().dropna()
        # Pick the asset with highest momentum each period
        picks = mom.idxmax(axis=1)
        strategy_rets = pd.Series(index=picks.index, dtype=float)
        for t in strategy_rets.index:
            asset = picks.loc[t]
            # Assume we can trade at next day's return
            try:
                next_t = rets.index[rets.index.get_loc(t) + 1]
                strategy_rets.loc[t] = rets.loc[next_t, asset]
            except IndexError:
                strategy_rets.loc[t] = 0.0

        strategy_rets = strategy_rets.dropna()
        if len(strategy_rets) == 0 or strategy_rets.std(ddof=1) == 0:
            nav = 1.0 + strategy_rets.sum()
            return StrategyResult(final_nav=nav, total_return=nav - 1, sharpe_ratio=0.0, max_drawdown=0.0, returns=strategy_rets)

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


def test_engine_full_pipeline():
    prices = pd.DataFrame({
        "A": [100.0] * 20 + [110.0] * 20 + [120.0] * 20 + [100.0] * 20,
        "B": [100.0] * 80,
    })
    # Pad with slight noise to avoid std=0
    prices = prices + pd.DataFrame({
        "A": [i * 0.01 for i in range(80)],
        "B": [i * 0.005 for i in range(80)],
    })

    strategy = DummyMomentumStrategy()
    engine = RobustnessEngine(EngineConfig(seed=42))

    report = engine.run_full_analysis(
        strategy=strategy,
        strategy_name="dummy_momentum",
        base_params={"lookback": 5, "top_n": 1},
        baseline_prices=prices,
        param_sensitivity_config={"lookback": [3, 5, 10]},
        param_bounds={"lookback": (2.0, 15.0), "top_n": (1.0, 2.0)},
    )

    assert report.strategy_name == "dummy_momentum"
    assert len(report.adversarial_results) > 0
    assert len(report.sensitivity_results) > 0
    assert report.robustness_score is not None
    assert report.robustness_score.overall_score >= 0.0
    assert report.robustness_score.overall_score <= 1.0
