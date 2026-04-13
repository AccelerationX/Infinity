"""Standard risk metrics calculation.

Includes VaR, CVaR, Max Drawdown, Beta, Tracking Error, Sharpe, Sortino.
Implements both parametric and historical simulation methods where applicable.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from core.models import RiskMetrics, RiskSnapshot


def compute_risk_metrics(
    snapshot: RiskSnapshot,
    returns_history: pd.DataFrame,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
    confidence_levels: Optional[list] = None,
) -> RiskMetrics:
    """Compute comprehensive risk metrics for a portfolio.

    Args:
        snapshot: Portfolio weights snapshot.
        returns_history: DataFrame of historical asset returns (index=dates, columns=symbols).
        benchmark_returns: Optional Series of benchmark returns aligned with returns_history.
        risk_free_rate: Annualized risk-free rate (default 0).
        confidence_levels: List of confidence levels for VaR/CVaR (default [0.95, 0.99]).

    Returns:
        RiskMetrics dataclass with all standard metrics.
    """
    if confidence_levels is None:
        confidence_levels = [0.95, 0.99]

    # Align weights with returns columns
    w = snapshot.weights.reindex(returns_history.columns).fillna(0.0).values
    portfolio_returns = returns_history @ w

    mean_ret = float(portfolio_returns.mean())
    vol = float(portfolio_returns.std(ddof=1))

    # Sharpe ratio (daily scale, assuming returns are daily)
    sharpe = (mean_ret - risk_free_rate / 252) / vol if vol > 1e-15 else 0.0

    # Sortino ratio
    downside = portfolio_returns - risk_free_rate / 252
    downside_dev = float(np.sqrt(np.mean(np.minimum(downside, 0.0) ** 2)))
    sortino = (mean_ret - risk_free_rate / 252) / downside_dev if downside_dev > 1e-15 else 0.0

    # Beta and tracking error
    beta = 0.0
    te = 0.0
    if benchmark_returns is not None:
        aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        if len(aligned) >= 2:
            cov = aligned.cov().iloc[0, 1]
            bench_var = aligned.iloc[:, 1].var(ddof=1)
            beta = float(cov / bench_var) if bench_var > 1e-15 else 0.0
            active_returns = aligned.iloc[:, 0] - aligned.iloc[:, 1]
            te = float(active_returns.std(ddof=1))

    # VaR and CVaR
    var_p95 = _var_parametric(mean_ret, vol, 0.95)
    var_p99 = _var_parametric(mean_ret, vol, 0.99)
    var_h95 = _var_historical(portfolio_returns, 0.95)
    var_h99 = _var_historical(portfolio_returns, 0.99)

    cvar_p95 = _cvar_parametric(mean_ret, vol, 0.95)
    cvar_p99 = _var_parametric(mean_ret, vol, 0.99)
    cvar_h95 = _cvar_historical(portfolio_returns, 0.95)
    cvar_h99 = _cvar_historical(portfolio_returns, 0.99)

    # Max drawdown
    mdd = _max_drawdown(portfolio_returns)

    return RiskMetrics(
        timestamp=snapshot.timestamp,
        portfolio_mean_return=mean_ret,
        portfolio_volatility=vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        beta=beta,
        tracking_error=te,
        var_parametric_95=var_p95,
        var_parametric_99=var_p99,
        var_historical_95=var_h95,
        var_historical_99=var_h99,
        cvar_parametric_95=cvar_p95,
        cvar_parametric_99=cvar_p99,
        cvar_historical_95=cvar_h95,
        cvar_historical_99=cvar_h99,
        max_drawdown=mdd,
    )


def _var_parametric(mean: float, vol: float, confidence: float) -> float:
    """Parametric VaR assuming normal distribution."""
    z = -1.0 * _norm_z(confidence)
    return mean + z * vol


def _var_historical(returns: pd.Series, confidence: float) -> float:
    """Historical simulation VaR."""
    if len(returns) == 0:
        return 0.0
    return float(np.percentile(returns.dropna(), (1 - confidence) * 100))


def _cvar_parametric(mean: float, vol: float, confidence: float) -> float:
    """Parametric CVaR (Expected Shortfall) assuming normal distribution."""
    alpha = 1 - confidence
    z = -1.0 * _norm_z(confidence)
    phi_z = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * z * z)
    return mean - vol * (phi_z / alpha)


def _cvar_historical(returns: pd.Series, confidence: float) -> float:
    """Historical simulation CVaR."""
    clean = returns.dropna()
    if len(clean) == 0:
        return 0.0
    var = np.percentile(clean, (1 - confidence) * 100)
    tail = clean[clean <= var]
    if len(tail) == 0:
        return var
    return float(tail.mean())


def _norm_z(confidence: float) -> float:
    """Standard normal quantile (two-sided).  e.g., 0.95 -> 1.645."""
    # Hardcoded common values to avoid scipy dependency
    known = {
        0.50: 0.0,
        0.75: 0.6745,
        0.90: 1.2816,
        0.95: 1.6449,
        0.975: 1.9600,
        0.99: 2.3263,
        0.995: 2.5758,
        0.999: 3.0902,
    }
    if confidence in known:
        return known[confidence]
    # Rational approximation for inverse normal CDF (Abramowitz & Stegun, 26.2.23)
    p = confidence
    if p > 0.5:
        t = np.sqrt(-2.0 * np.log(1.0 - p))
    else:
        t = np.sqrt(-2.0 * np.log(p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    num = c0 + c1 * t + c2 * t * t
    den = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t
    z = t - num / den
    if p < 0.5:
        z = -z
    return float(z)


def _max_drawdown(returns: pd.Series) -> float:
    """Compute maximum drawdown from a return series."""
    clean = returns.dropna()
    if len(clean) == 0:
        return 0.0
    nav = (1 + clean).cumprod()
    running_max = nav.cummax()
    drawdown = nav / running_max - 1.0
    return float(drawdown.min())
