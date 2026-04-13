"""Tests for failure mode diagnosis module."""

import numpy as np
import pandas as pd

from core.failure_modes import (
    diagnose_crowding_crash,
    diagnose_data_snooping,
    diagnose_factor_decay,
    diagnose_liquidity_trap,
    diagnose_regime_breakdown,
)
from core.models import FailureMode, StrategyResult


def _make_result(returns: list) -> StrategyResult:
    s = pd.Series(returns)
    nav = (1 + s).cumprod()
    return StrategyResult(
        final_nav=float(nav.iloc[-1]),
        total_return=float(nav.iloc[-1] - 1),
        sharpe_ratio=float(s.mean() / s.std(ddof=1)) if s.std() > 0 else 0.0,
        max_drawdown=float((nav / nav.cummax() - 1).min()),
        returns=s,
    )


class TestRegimeBreakdown:
    def test_sharpe_degradation_detected(self):
        # First 40 days good, then terrible
        rets = [0.01] * 40 + [-0.03] * 40
        result = _make_result(rets)
        diag = diagnose_regime_breakdown(result, window=20)
        assert diag.mode == FailureMode.REGIME_BREAKDOWN
        assert diag.severity_score > 0.5


class TestCrowdingCrash:
    def test_high_turnover_flagged(self):
        result = _make_result([0.001] * 100)
        result.positions = pd.DataFrame(
            {"A": [0.5, 0.0, 0.5, 0.0] * 25, "B": [0.5, 1.0, 0.5, 1.0] * 25}
        )
        diag = diagnose_crowding_crash(result)
        assert diag.mode == FailureMode.CROWDING_CRASH
        assert diag.severity_score > 0.0


class TestLiquidityTrap:
    def test_high_participation_flagged(self):
        result = _make_result([0.001] * 100)
        result.positions = pd.DataFrame({"A": [1_000_000] * 100})
        volumes = pd.Series({"A": 5_000_000})
        diag = diagnose_liquidity_trap(result, volumes)
        assert diag.mode == FailureMode.LIQUIDITY_TRAP
        assert diag.severity_score > 0.0

    def test_missing_data_returns_zero(self):
        result = _make_result([0.001] * 100)
        diag = diagnose_liquidity_trap(result)
        assert diag.severity_score == 0.0


class TestFactorDecay:
    def test_correlation_drop_detected(self):
        rets = [0.01] * 50 + [-0.01] * 50
        result = _make_result(rets)
        factor = pd.Series([0.01] * 50 + [-0.01] * 50)  # perfectly aligned then same drop -> no decay
        diag = diagnose_factor_decay(result, factor, window=20)
        # With perfect correlation throughout, severity should be low
        assert diag.severity_score < 0.5


class TestDataSnooping:
    def test_large_decay_detected(self):
        insample = _make_result([0.02] * 100)
        outsample = _make_result([-0.01] * 100)
        diag = diagnose_data_snooping(insample, outsample)
        assert diag.mode == FailureMode.DATA_SNOOPING
        assert diag.severity_score > 0.5
