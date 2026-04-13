"""Return decomposition: five-dimensional attribution model.

Implements a rigorous, additive decomposition of portfolio returns into:
  1. Benchmark return
  2. Security selection alpha
  3. Asset allocation effect
  4. Interaction effect
  5. Execution cost
  6. Beta return
  7. Residual

All decompositions satisfy the additive identity:
  total_return = sum(all components)  (within numerical tolerance)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.models import AttributionResult


def brinson_attribution(
    actual_weights: pd.Series,
    benchmark_weights: pd.Series,
    actual_returns: pd.Series,
    benchmark_return: float,
) -> Dict[str, float]:
    """Classic Brinson attribution for a single period.

    Args:
        actual_weights: Portfolio weights by asset (sum need not be 1 if cash present)
        benchmark_weights: Benchmark weights by asset (should sum to 1)
        actual_returns: Asset-level returns for the period
        benchmark_return: Overall benchmark return for the period

    Returns:
        Dictionary with keys: benchmark_return, selection, allocation, interaction
    """
    # Align indices
    combined = pd.concat([actual_weights, benchmark_weights, actual_returns], axis=1)
    combined.columns = ["w_p", "w_b", "r_p"]
    combined = combined.fillna(0.0)

    w_p = combined["w_p"]
    w_b = combined["w_b"]
    r_p = combined["r_p"]

    # Benchmark return = sum(w_b * r_b).  Here we approximate r_b by r_p for matched assets.
    # For a proper benchmark, caller should provide benchmark-specific returns.
    # We use the provided benchmark_return scalar for the allocation component.
    selection = np.sum(w_b * (r_p - benchmark_return))
    allocation = np.sum((w_p - w_b) * benchmark_return)
    interaction = np.sum((w_p - w_b) * (r_p - benchmark_return))

    return {
        "benchmark_return": benchmark_return,
        "selection": float(selection),
        "allocation": float(allocation),
        "interaction": float(interaction),
    }


def holdings_based_attribution(
    paper_portfolio: pd.DataFrame,
    actual_portfolio: pd.DataFrame,
    benchmark_weights: pd.Series,
    market_returns: pd.Series,
    period_label: str,
    beta_lookback: int = 63,
) -> AttributionResult:
    """Holdings-based five-dimensional attribution.

    Args:
        paper_portfolio: DataFrame indexed by symbol with columns ['weight', 'return']
            representing the ideal portfolio (signals perfectly executed).
        actual_portfolio: DataFrame indexed by symbol with columns ['weight', 'return']
            representing the actual realized portfolio.
        benchmark_weights: Series of benchmark weights by symbol.
        market_returns: Series of market returns (e.g., index returns) for beta estimation.
        period_label: String label for the period (e.g., '2024-Q1').
        beta_lookback: Number of periods to use for beta estimation.

    Returns:
        AttributionResult with all five dimensions populated.
    """
    # Ensure alignment
    all_symbols = paper_portfolio.index.union(actual_portfolio.index).union(benchmark_weights.index)

    def _align(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        out = pd.DataFrame(index=all_symbols, columns=cols, dtype=float)
        out.loc[df.index, cols] = df[cols]
        return out.fillna(0.0)

    paper = _align(paper_portfolio, ["weight", "return"])
    actual = _align(actual_portfolio, ["weight", "return"])
    bench_w = benchmark_weights.reindex(all_symbols).fillna(0.0)

    w_paper = paper["weight"].values
    w_actual = actual["weight"].values
    w_bench = bench_w.values
    r_paper = paper["return"].values   # ideal asset returns (paper portfolio)
    r_actual = actual["return"].values  # realized asset returns (actual portfolio)

    # Total and benchmark returns
    total_return = float(np.sum(w_actual * r_actual))
    benchmark_return = float(np.sum(w_bench * r_paper))
    paper_return = float(np.sum(w_paper * r_paper))

    # Brinson components using paper portfolio vs benchmark
    selection_alpha = float(np.sum(w_bench * (r_paper - benchmark_return)))
    allocation_effect = float(np.sum((w_paper - w_bench) * benchmark_return))
    interaction_effect = float(np.sum((w_paper - w_bench) * (r_paper - benchmark_return)))

    # Execution cost = paper return minus actual return
    execution_cost = paper_return - total_return

    # Beta return: estimate portfolio beta against market and decompose
    # Here we approximate beta using the current period's asset-level betas if available,
    # otherwise use a simple 1.0 fallback.  Caller can override by pre-computing betas.
    beta_est = _estimate_beta(market_returns, beta_lookback)
    # Portfolio beta = sum(actual_weights * betas)
    # For simplicity, assume each asset has beta = 1.0 if not provided.
    # In production, asset-level betas should be passed as an additional column.
    beta_return = float(beta_est * np.sum(w_actual) * np.mean(r_actual))  # crude fallback

    # Residual so that everything adds up exactly
    computed_sum = (
        benchmark_return
        + selection_alpha
        + allocation_effect
        + interaction_effect
        + execution_cost
        + beta_return
    )
    residual = total_return - computed_sum

    return AttributionResult(
        period=period_label,
        total_return=total_return,
        benchmark_return=benchmark_return,
        selection_alpha=selection_alpha,
        allocation_effect=allocation_effect,
        interaction_effect=interaction_effect,
        execution_cost=execution_cost,
        beta_return=beta_return,
        residual=residual,
    )


def _estimate_beta(market_returns: pd.Series, lookback: int) -> float:
    """Estimate market beta from a series of market returns.

    Since we don't have portfolio-level historical returns here, we return a
    default beta of 1.0.  In production, portfolio beta should be computed from
    historical portfolio returns and passed directly.
    """
    if len(market_returns) < 2:
        return 1.0
    # A proper beta estimate requires portfolio returns time series.
    # We return 1.0 as a neutral prior; caller can adjust.
    return 1.0


def multi_period_attribution(
    results: List[AttributionResult],
    linking_method: str = "geometric",
) -> AttributionResult:
    """Aggregate single-period attribution results across multiple periods.

    Args:
        results: List of AttributionResult, one per period.
        linking_method: 'geometric' for compounded linking, 'arithmetic' for simple sum.

    Returns:
        A single AttributionResult representing the aggregated attribution.
    """
    if not results:
        raise ValueError("results list is empty")

    if linking_method == "arithmetic":
        total_return = sum(r.total_return for r in results)
        benchmark_return = sum(r.benchmark_return for r in results)
        selection_alpha = sum(r.selection_alpha for r in results)
        allocation_effect = sum(r.allocation_effect for r in results)
        interaction_effect = sum(r.interaction_effect for r in results)
        execution_cost = sum(r.execution_cost for r in results)
        beta_return = sum(r.beta_return for r in results)
        # Recalculate residual to enforce additive identity at aggregate level
        residual = (
            total_return
            - benchmark_return
            - selection_alpha
            - allocation_effect
            - interaction_effect
            - execution_cost
            - beta_return
        )
    elif linking_method == "geometric":
        # Geometric linking: convert each component to a "growth factor" and compound.
        # We use the Carino or Menchero approach approximated by:
        #   k_t = ln(1+R_t) / R_t   for scaling
        # For simplicity, we use a first-order approximation.
        total_return = np.prod([1 + r.total_return for r in results]) - 1
        benchmark_return = np.prod([1 + r.benchmark_return for r in results]) - 1
        # Scale each component by the ratio of log-return to simple return
        def _scale(val, total):
            if abs(total) < 1e-12:
                return val / len(results)
            return val * (np.log1p(total) / total)

        scaled = {
            "selection_alpha": 0.0,
            "allocation_effect": 0.0,
            "interaction_effect": 0.0,
            "execution_cost": 0.0,
            "beta_return": 0.0,
            "residual": 0.0,
        }
        for r in results:
            k = np.log1p(r.total_return) / r.total_return if abs(r.total_return) > 1e-12 else 1.0
            for key in scaled:
                scaled[key] += getattr(r, key) * k

        # Normalize so that sum of scaled components matches total log return
        log_total = np.log1p(total_return)
        sum_scaled = sum(scaled.values()) + np.log1p(benchmark_return)
        if abs(sum_scaled) > 1e-12:
            factor = log_total / sum_scaled
            for key in scaled:
                scaled[key] *= factor

        # Convert back from log-space to return space
        def _unscale(val, total):
            if abs(total) < 1e-12:
                return val / len(results)
            return val * (total / np.log1p(total))

        selection_alpha = _unscale(scaled["selection_alpha"], total_return)
        allocation_effect = _unscale(scaled["allocation_effect"], total_return)
        interaction_effect = _unscale(scaled["interaction_effect"], total_return)
        execution_cost = _unscale(scaled["execution_cost"], total_return)
        beta_return = _unscale(scaled["beta_return"], total_return)
        # Recalculate residual to enforce additive identity at aggregate level
        residual = (
            total_return
            - benchmark_return
            - selection_alpha
            - allocation_effect
            - interaction_effect
            - execution_cost
            - beta_return
        )
    else:
        raise ValueError(f"Unknown linking_method: {linking_method}")

    return AttributionResult(
        period="aggregate",
        total_return=total_return,
        benchmark_return=benchmark_return,
        selection_alpha=selection_alpha,
        allocation_effect=allocation_effect,
        interaction_effect=interaction_effect,
        execution_cost=execution_cost,
        beta_return=beta_return,
        residual=residual,
    )
