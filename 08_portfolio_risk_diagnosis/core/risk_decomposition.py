"""Risk decomposition based on multi-factor models.

Implements marginal and absolute risk contribution (MRC / ARC)
and decomposes total portfolio volatility into factor groups
and idiosyncratic components.

Reference:
  - Litterman, R. (1996). "Hot Spots and Hedges."
  - Meucci, A. (2005). Risk and Asset Allocation.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.models import FactorModel, RiskDecomposition, RiskSnapshot


def decompose_risk(
    snapshot: RiskSnapshot,
    factor_model: FactorModel,
    factor_groups: Optional[Dict[str, List[str]]] = None,
) -> RiskDecomposition:
    """Decompose portfolio volatility into factor and idiosyncratic contributions.

    Args:
        snapshot: Portfolio weights and timestamp.
        factor_model: Multi-factor model with exposures, factor cov, idio var.
        factor_groups: Optional mapping from group_name -> list of factor names.
                       If None, each factor is treated as its own group.

    Returns:
        RiskDecomposition with MRC, ARC, and group-level contributions.
    """
    if not factor_model.validate():
        raise ValueError("FactorModel validation failed")

    w = snapshot.weights.reindex(factor_model.exposures.index).fillna(0.0).values
    B = factor_model.exposures.values
    Sigma_f = factor_model.factor_cov.values
    D_diag = factor_model.idio_var.reindex(factor_model.exposures.index).fillna(0.0).values
    D = np.diag(D_diag)

    # Full covariance matrix: Sigma = B @ Sigma_f @ B.T + D
    Sigma = B @ Sigma_f @ B.T + D

    # Portfolio volatility
    sigma_p = float(np.sqrt(max(w @ Sigma @ w, 0.0)))
    if sigma_p < 1e-15:
        # Degenerate case: zero volatility
        zeros = pd.Series(0.0, index=factor_model.exposures.index)
        return RiskDecomposition(
            timestamp=snapshot.timestamp,
            total_volatility=0.0,
            factor_risk=0.0,
            idio_risk=0.0,
            group_contributions={},
            asset_mrc=zeros,
            asset_arc=zeros,
            effective_n=_effective_n(w),
        )

    # Marginal Risk Contribution per asset: MRC = (Sigma @ w) / sigma_p
    mrc_values = (Sigma @ w) / sigma_p
    arc_values = w * mrc_values

    # Factor risk and idiosyncratic risk in variance units
    var_factor = float(w @ B @ Sigma_f @ B.T @ w)
    var_idio = float(w @ D @ w)
    factor_risk_vol = np.sqrt(max(var_factor, 0.0))
    idio_risk_vol = np.sqrt(max(var_idio, 0.0))

    # Group-level absolute contributions in volatility units
    # We decompose each asset's ARC into factor-group contributions and
    # idiosyncratic contribution. This ensures exact additivity.
    #
    # For asset i: ARC_i = w_i * (Σw)_i / sigma_p
    # (Σw)_i = sum_k B_ik * (Σ_f B'w)_k + D_ii * w_i
    # Factor k contribution to ARC_i = w_i * B_ik * (Σ_f B'w)_k / sigma_p
    # Idio contribution to ARC_i = w_i * D_ii * w_i / sigma_p = w_i^2 * D_ii / sigma_p
    group_contributions: Dict[str, float] = {}

    if factor_groups is None:
        factor_groups = {name: [name] for name in factor_model.factor_names}

    k = len(factor_model.factor_names)
    factor_idx = {name: i for i, name in enumerate(factor_model.factor_names)}

    # Precompute Σ_f @ B'w (k x 1)
    sf_btw = Sigma_f @ (B.T @ w)

    # Initialize group contributions
    for group_name in factor_groups:
        group_contributions[group_name] = 0.0

    idio_arc = 0.0
    for i, sym in enumerate(factor_model.exposures.index):
        wi = w[i]
        # Idiosyncratic contribution from asset i
        idio_arc += (wi ** 2) * D_diag[i] / sigma_p
        # Factor contributions from asset i
        for group_name, factors in factor_groups.items():
            group_factor_contrib = 0.0
            for f in factors:
                idx = factor_idx.get(f)
                if idx is not None:
                    group_factor_contrib += wi * B[i, idx] * sf_btw[idx] / sigma_p
            group_contributions[group_name] += group_factor_contrib

    asset_mrc = pd.Series(mrc_values, index=factor_model.exposures.index)
    asset_arc = pd.Series(arc_values, index=factor_model.exposures.index)

    return RiskDecomposition(
        timestamp=snapshot.timestamp,
        total_volatility=sigma_p,
        factor_risk=factor_risk_vol,
        idio_risk=idio_arc,
        group_contributions=group_contributions,
        asset_mrc=asset_mrc,
        asset_arc=asset_arc,
        effective_n=_effective_n(w),
    )


def _effective_n(weights: np.ndarray) -> float:
    """Compute effective number of assets (inverse HHI)."""
    w = np.asarray(weights)
    if np.sum(np.abs(w)) < 1e-15:
        return 0.0
    # Normalize to sum to 1 for concentration calculation
    w_norm = w / np.sum(w)
    hhi = float(np.sum(w_norm ** 2))
    return 1.0 / hhi if hhi > 1e-15 else 0.0
