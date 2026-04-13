"""Failure mode classification and diagnosis module.

Identifies and quantifies typical strategy failure patterns:
  - Regime Breakdown
  - Crowding Crash
  - Liquidity Trap
  - Factor Decay
  - Data Snooping
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.models import FailureDiagnosis, FailureMode, StrategyResult


def diagnose_regime_breakdown(
    strategy_result: StrategyResult,
    benchmark_returns: Optional[pd.Series] = None,
    window: int = 21,
) -> FailureDiagnosis:
    """Diagnose regime breakdown by analyzing rolling Sharpe degradation."""
    rets = strategy_result.returns.dropna()
    if len(rets) < window * 2:
        return FailureDiagnosis(
            mode=FailureMode.REGIME_BREAKDOWN,
            severity_score=0.0,
            indicators={},
            description="Insufficient data for regime breakdown diagnosis.",
        )

    rolling_sharpe = rets.rolling(window).mean() / rets.rolling(window).std()
    sharpe_drop = float(rolling_sharpe.iloc[window:].min() - rolling_sharpe.iloc[:window].mean())

    # Severity: normalize by typical Sharpe range (~2.0)
    severity = min(max(-sharpe_drop / 2.0, 0.0), 1.0)

    indicators = {
        "rolling_sharpe_early_mean": float(rolling_sharpe.iloc[:window].mean()),
        "rolling_sharpe_min": float(rolling_sharpe.min()),
        "sharpe_drop": sharpe_drop,
    }

    return FailureDiagnosis(
        mode=FailureMode.REGIME_BREAKDOWN,
        severity_score=severity,
        indicators=indicators,
        description=(
            f"Rolling Sharpe dropped from {indicators['rolling_sharpe_early_mean']:.2f} "
            f"to {indicators['rolling_sharpe_min']:.2f}."
        ),
    )


def diagnose_crowding_crash(
    strategy_result: StrategyResult,
    turnover_threshold: float = 0.5,  # annualized turnover threshold
) -> FailureDiagnosis:
    """Diagnose crowding crash risk from turnover and concentration."""
    rets = strategy_result.returns.dropna()
    if len(rets) < 2:
        return FailureDiagnosis(
            mode=FailureMode.CROWDING_CRASH,
            severity_score=0.0,
            indicators={},
            description="Insufficient data.",
        )

    # Approximate turnover from position changes if available
    turnover = 0.0
    if strategy_result.positions is not None:
        pos = strategy_result.positions.fillna(0.0)
        if len(pos) > 1:
            turnover = float(np.mean(np.sum(np.abs(pos.diff().dropna()), axis=1)) / 2.0)

    # Concentration: average max position weight
    concentration = 0.0
    if strategy_result.positions is not None:
        pos = strategy_result.positions.fillna(0.0)
        max_w = pos.max(axis=1)
        concentration = float(max_w.mean())

    # Severity score
    t_score = min(turnover / turnover_threshold, 1.0)
    c_score = min(concentration / 0.30, 1.0)
    severity = 0.5 * t_score + 0.5 * c_score

    indicators = {
        "approx_turnover": turnover,
        "avg_max_position_weight": concentration,
    }

    return FailureDiagnosis(
        mode=FailureMode.CROWDING_CRASH,
        severity_score=severity,
        indicators=indicators,
        description=(
            f"Approx turnover={turnover:.2%}, avg max weight={concentration:.2%}. "
            f"Crowding severity={severity:.2f}."
        ),
    )


def diagnose_liquidity_trap(
    strategy_result: StrategyResult,
    avg_daily_volumes: Optional[pd.Series] = None,
) -> FailureDiagnosis:
    """Diagnose liquidity trap from position size vs market volume."""
    if strategy_result.positions is None or avg_daily_volumes is None:
        return FailureDiagnosis(
            mode=FailureMode.LIQUIDITY_TRAP,
            severity_score=0.0,
            indicators={},
            description="Position or volume data unavailable.",
        )

    pos = strategy_result.positions.fillna(0.0)
    # Assume positions are share counts; normalize to participation rate
    adv = avg_daily_volumes.reindex(pos.columns).fillna(1e9)
    participation = pos.div(adv, axis=1)
    max_participation = float(participation.max().max())

    # Severity: > 10% ADV is severe, > 1% is moderate
    severity = min(max((max_participation - 0.01) / 0.09, 0.0), 1.0)

    indicators = {"max_participation_rate": max_participation}

    return FailureDiagnosis(
        mode=FailureMode.LIQUIDITY_TRAP,
        severity_score=severity,
        indicators=indicators,
        description=f"Max participation rate = {max_participation:.2%} of ADV.",
    )


def diagnose_factor_decay(
    strategy_result: StrategyResult,
    factor_returns: Optional[pd.Series] = None,
    window: int = 21,
) -> FailureDiagnosis:
    """Diagnose factor decay from rolling correlation between strategy and factor returns."""
    rets = strategy_result.returns.dropna()
    if factor_returns is None or len(rets) < window * 2:
        return FailureDiagnosis(
            mode=FailureMode.FACTOR_DECAY,
            severity_score=0.0,
            indicators={},
            description="Factor return data unavailable or insufficient.",
        )

    aligned = pd.concat([rets, factor_returns], axis=1).dropna()
    if len(aligned) < window * 2:
        return FailureDiagnosis(
            mode=FailureMode.FACTOR_DECAY,
            severity_score=0.0,
            indicators={},
            description="Aligned data insufficient.",
        )

    rolling_corr = aligned.iloc[:, 0].rolling(window).corr(aligned.iloc[:, 1])
    early_corr = float(rolling_corr.iloc[window:2*window].mean())
    late_corr = float(rolling_corr.iloc[-window:].mean())
    if pd.isna(early_corr) or pd.isna(late_corr):
        return FailureDiagnosis(
            mode=FailureMode.FACTOR_DECAY,
            severity_score=0.0,
            indicators={"early_factor_corr": np.nan, "late_factor_corr": np.nan, "corr_drop": np.nan},
            description="Insufficient data for reliable rolling correlation.",
        )
    corr_drop = early_corr - late_corr

    severity = min(max(corr_drop / 0.5, 0.0), 1.0)

    indicators = {
        "early_factor_corr": early_corr,
        "late_factor_corr": late_corr,
        "corr_drop": corr_drop,
    }

    return FailureDiagnosis(
        mode=FailureMode.FACTOR_DECAY,
        severity_score=severity,
        indicators=indicators,
        description=(
            f"Factor correlation dropped from {early_corr:.2f} to {late_corr:.2f}."
        ),
    )


def diagnose_data_snooping(
    insample_result: StrategyResult,
    outsample_result: StrategyResult,
) -> FailureDiagnosis:
    """Diagnose data snooping from in-sample vs out-of-sample Sharpe decay."""
    ins_sharpe = insample_result.sharpe_ratio
    out_sharpe = outsample_result.sharpe_ratio

    if ins_sharpe == 0:
        severity = 0.0
    else:
        ratio = out_sharpe / ins_sharpe
        severity = min(max(1.0 - ratio, 0.0), 1.0)

    indicators = {
        "in_sample_sharpe": ins_sharpe,
        "out_of_sample_sharpe": out_sharpe,
        "sharpe_ratio": ratio if ins_sharpe != 0 else 0.0,
    }

    return FailureDiagnosis(
        mode=FailureMode.DATA_SNOOPING,
        severity_score=severity,
        indicators=indicators,
        description=(
            f"IS Sharpe={ins_sharpe:.2f}, OOS Sharpe={out_sharpe:.2f}. "
            f"Decay severity={severity:.2f}."
        ),
    )


def run_all_diagnoses(
    strategy_result: StrategyResult,
    benchmark_returns: Optional[pd.Series] = None,
    factor_returns: Optional[pd.Series] = None,
    avg_daily_volumes: Optional[pd.Series] = None,
    insample_result: Optional[StrategyResult] = None,
    outsample_result: Optional[StrategyResult] = None,
) -> List[FailureDiagnosis]:
    """Run the full suite of failure mode diagnoses."""
    diagnoses: List[FailureDiagnosis] = []
    diagnoses.append(diagnose_regime_breakdown(strategy_result, benchmark_returns))
    diagnoses.append(diagnose_crowding_crash(strategy_result))
    diagnoses.append(diagnose_liquidity_trap(strategy_result, avg_daily_volumes))
    diagnoses.append(diagnose_factor_decay(strategy_result, factor_returns))
    if insample_result is not None and outsample_result is not None:
        diagnoses.append(diagnose_data_snooping(insample_result, outsample_result))
    return diagnoses
