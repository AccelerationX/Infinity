"""Adversarial market simulation module.

Generates synthetic price paths that are adversarial to a given strategy
by introducing trend reversals, volatility jumps, liquidity shocks,
and correlation breakdowns.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from core.models import AdversarialPath


def generate_gbm_paths(
    n_steps: int,
    symbols: List[str],
    mu: np.ndarray,
    sigma: np.ndarray,
    corr: np.ndarray,
    dt: float = 1 / 252,
    start_prices: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate correlated GBM price paths.

    Args:
        n_steps: Number of time steps.
        symbols: Asset symbols.
        mu: Annualized drift vector (n_assets,).
        sigma: Annualized volatility vector (n_assets,).
        corr: Correlation matrix (n_assets, n_assets).
        dt: Time step in years.
        start_prices: Starting prices (n_assets,). Defaults to 100.
        seed: Random seed.

    Returns:
        DataFrame of prices with dates as index and symbols as columns.
    """
    if seed is not None:
        np.random.seed(seed)

    n_assets = len(symbols)
    if start_prices is None:
        start_prices = np.full(n_assets, 100.0)

    # Cholesky decomposition of correlation matrix
    L = np.linalg.cholesky(corr)

    # Generate correlated Brownian motions (n_steps-1 to leave room for start price)
    dW = np.random.standard_normal((n_steps - 1, n_assets)) @ L.T * np.sqrt(dt)

    # GBM: dS/S = mu dt + sigma dW
    log_returns = (mu - 0.5 * sigma ** 2) * dt + sigma * dW
    prices = start_prices * np.exp(np.cumsum(log_returns, axis=0))

    # Prepend starting prices as first row
    prices = np.vstack([start_prices, prices])
    dates = pd.date_range(start="2024-01-01", periods=n_steps, freq="B")
    return pd.DataFrame(prices, index=dates, columns=symbols)


def make_trend_reversal(
    base_prices: pd.DataFrame,
    reversal_start_idx: int,
    intensity: float = 1.0,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Inject a trend reversal into an existing price path.

    Intensity=1.0 means the drift after reversal is approximately
    the negative of the pre-reversal drift.
    """
    if seed is not None:
        np.random.seed(seed)

    prices = base_prices.copy()
    n_steps = len(prices)
    symbols = list(prices.columns)
    n_assets = len(symbols)

    # Estimate pre-reversal daily log-return drift
    pre = prices.iloc[:reversal_start_idx]
    log_pre = np.log(pre)
    daily_mu = log_pre.diff().mean().values  # (n_assets,)

    # Post-reversal drift: reverse and amplify
    post_mu = -daily_mu * intensity
    post_sigma = log_pre.diff().std().fillna(0.02).values

    # Regenerate post-reversal segment
    post_steps = n_steps - reversal_start_idx
    start_prices = prices.iloc[reversal_start_idx - 1].values

    dW = np.random.standard_normal((post_steps, n_assets)) * np.sqrt(1 / 252)
    log_returns = post_mu + post_sigma * dW
    post_prices = start_prices * np.exp(np.cumsum(log_returns, axis=0))

    prices.iloc[reversal_start_idx:] = post_prices
    return prices


def make_volatility_jump(
    base_prices: pd.DataFrame,
    jump_start_idx: int,
    intensity: float = 1.0,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Inject a volatility jump: sigma increases by (1 + intensity)."""
    if seed is not None:
        np.random.seed(seed)

    prices = base_prices.copy()
    n_steps = len(prices)
    symbols = list(prices.columns)
    n_assets = len(symbols)

    log_returns = np.log(prices).diff().dropna()
    base_sigma = log_returns.std().values
    jump_sigma = base_sigma * (1.0 + intensity)

    post_steps = n_steps - jump_start_idx
    start_prices = prices.iloc[jump_start_idx - 1].values
    mu = log_returns.mean().values

    dW = np.random.standard_normal((post_steps, n_assets))
    post_log_returns = mu + jump_sigma * dW
    post_prices = start_prices * np.exp(np.cumsum(post_log_returns, axis=0))

    prices.iloc[jump_start_idx:] = post_prices
    return prices


def make_correlation_crash(
    base_prices: pd.DataFrame,
    crash_start_idx: int,
    intensity: float = 1.0,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Inject a correlation crash: all assets become highly correlated."""
    if seed is not None:
        np.random.seed(seed)

    prices = base_prices.copy()
    n_steps = len(prices)
    n_assets = len(prices.columns)

    log_returns = np.log(prices).diff().dropna()
    mu = log_returns.mean().values
    sigma = log_returns.std().values

    # Target correlation: lambda * 1 + (1-lambda) * I, then interpolate
    lam = intensity * 0.9  # max crash to 0.9 correlation
    if n_assets > 1:
        # Ensure positive definite: cap lambda
        lam = min(lam, 0.99 / (n_assets - 1))
    target_corr = lam * np.ones((n_assets, n_assets)) + (1 - lam) * np.eye(n_assets)

    post_steps = n_steps - crash_start_idx
    start_prices = prices.iloc[crash_start_idx - 1].values

    L = np.linalg.cholesky(target_corr)
    dW = np.random.standard_normal((post_steps, n_assets)) @ L.T
    post_log_returns = mu + sigma * dW
    post_prices = start_prices * np.exp(np.cumsum(post_log_returns, axis=0))

    prices.iloc[crash_start_idx:] = post_prices
    return prices


def generate_adversarial_library(
    n_steps: int = 252,
    symbols: Optional[List[str]] = None,
    base_seed: int = 42,
) -> List[AdversarialPath]:
    """Generate a library of adversarial price paths with varying intensity."""
    if symbols is None:
        symbols = ["A", "B", "C"]

    n_assets = len(symbols)
    mu = np.full(n_assets, 0.08 / 252)  # ~8% annual drift
    sigma = np.full(n_assets, 0.20 / np.sqrt(252))  # ~20% annual vol
    corr = np.eye(n_assets) * 0.5 + 0.5  # 0.5 correlation off-diagonal
    np.fill_diagonal(corr, 1.0)

    base_prices = generate_gbm_paths(n_steps, symbols, mu, sigma, corr, seed=base_seed)

    paths: List[AdversarialPath] = []
    intensities = [0.3, 0.6, 1.0]
    mid = n_steps // 2

    for i, intensity in enumerate(intensities):
        paths.append(AdversarialPath(
            name=f"trend_reversal_{int(intensity*100)}",
            prices=make_trend_reversal(base_prices, mid, intensity, seed=base_seed + i + 1),
            scenario_type="TREND_REVERSAL",
            intensity=intensity,
            seed=base_seed + i + 1,
        ))
        paths.append(AdversarialPath(
            name=f"vol_jump_{int(intensity*100)}",
            prices=make_volatility_jump(base_prices, mid, intensity, seed=base_seed + i + 10),
            scenario_type="VOLATILITY_JUMP",
            intensity=intensity,
            seed=base_seed + i + 10,
        ))
        paths.append(AdversarialPath(
            name=f"corr_crash_{int(intensity*100)}",
            prices=make_correlation_crash(base_prices, mid, intensity, seed=base_seed + i + 20),
            scenario_type="CORRELATION_CRASH",
            intensity=intensity,
            seed=base_seed + i + 20,
        ))

    return paths
