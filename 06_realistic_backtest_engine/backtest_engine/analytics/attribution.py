"""
Backtest vs Live Trading Attribution Analysis

Identifies and quantifies sources of discrepancy between backtest and live performance.

Key question: Why did the live results differ from the backtest?

Sources of discrepancy:
1. Data differences (revised data, survivorship bias)
2. Execution differences (slippage, market impact)
3. Latency (signal delays)
4. Capacity constraints (strategy saturation)
5. Market regime changes
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats


@dataclass
class PerformancePeriod:
    """Performance over a specific period"""
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    returns: pd.Series  # Daily returns
    positions: Optional[pd.DataFrame] = None
    trades: Optional[pd.DataFrame] = None


class AttributionAnalyzer:
    """
    Analyzes the gap between backtest and live performance.
    """
    
    def __init__(
        self,
        backtest: PerformancePeriod,
        live: PerformancePeriod,
    ):
        """
        Args:
            backtest: Backtest performance period
            live: Live trading performance period
        """
        self.backtest = backtest
        self.live = live
        
        # Align dates
        common_start = max(backtest.start_date, live.start_date)
        common_end = min(backtest.end_date, live.end_date)
        
        self.backtest_returns = backtest.returns[
            (backtest.returns.index >= common_start) &
            (backtest.returns.index <= common_end)
        ]
        
        self.live_returns = live.returns[
            (live.returns.index >= common_start) &
            (live.returns.index <= common_end)
        ]
    
    def calculate_gap_metrics(self) -> Dict:
        """
        Calculate basic gap metrics.
        
        Returns:
            Dictionary with gap statistics
        """
        # Ensure same length
        min_len = min(len(self.backtest_returns), len(self.live_returns))
        bt_rets = self.backtest_returns.iloc[:min_len]
        live_rets = self.live_returns.iloc[:min_len]
        
        # Calculate metrics
        bt_total = (1 + bt_rets).prod() - 1
        live_total = (1 + live_rets).prod() - 1
        gap = bt_total - live_total
        
        bt_sharpe = bt_rets.mean() / bt_rets.std() * np.sqrt(252)
        live_sharpe = live_rets.mean() / live_rets.std() * np.sqrt(252)
        
        bt_maxdd = self._calculate_max_drawdown(bt_rets)
        live_maxdd = self._calculate_max_drawdown(live_rets)
        
        # Correlation
        correlation = bt_rets.corr(live_rets)
        
        # Statistical test
        t_stat, p_value = stats.ttest_ind(bt_rets, live_rets)
        
        return {
            'backtest_return': bt_total,
            'live_return': live_total,
            'return_gap': gap,
            'gap_bps': gap * 10000,
            'backtest_sharpe': bt_sharpe,
            'live_sharpe': live_sharpe,
            'sharpe_gap': bt_sharpe - live_sharpe,
            'backtest_maxdd': bt_maxdd,
            'live_maxdd': live_maxdd,
            'correlation': correlation,
            'statistical_significance': p_value < 0.05,
            'p_value': p_value,
        }
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def execution_attribution(self) -> Dict:
        """
        Attribute gap to execution differences.
        
        Components:
        - Slippage
        - Commission
        - Market impact
        - Latency
        """
        if self.backtest.trades is None or self.live.trades is None:
            return {'error': 'Trade data not available'}
        
        bt_trades = self.backtest.trades
        live_trades = self.live.trades
        
        # Calculate average slippage
        bt_slippage = bt_trades.get('slippage', pd.Series(0)).mean()
        live_slippage = live_trades.get('slippage', pd.Series(0)).mean()
        slippage_diff = live_slippage - bt_slippage
        
        # Commission
        bt_commission = bt_trades.get('commission', pd.Series(0)).sum()
        live_commission = live_trades.get('commission', pd.Series(0)).sum()
        
        # Market impact
        bt_impact = bt_trades.get('market_impact', pd.Series(0)).mean()
        live_impact = live_trades.get('market_impact', pd.Series(0)).mean()
        impact_diff = live_impact - bt_impact
        
        return {
            'backtest_slippage_bps': bt_slippage * 10000,
            'live_slippage_bps': live_slippage * 10000,
            'slippage_attribution_bps': slippage_diff * 10000,
            
            'backtest_commission': bt_commission,
            'live_commission': live_commission,
            'commission_attribution': live_commission - bt_commission,
            
            'backtest_impact_bps': bt_impact * 10000,
            'live_impact_bps': live_impact * 10000,
            'impact_attribution_bps': impact_diff * 10000,
        }
    
    def timing_attribution(self) -> Dict:
        """
        Attribute gap to timing differences.
        
        Measures how much of the gap is due to:
        - Entry timing
        - Exit timing
        - Holding period differences
        """
        if self.backtest.positions is None or self.live.positions is None:
            return {'error': 'Position data not available'}
        
        # Calculate correlation of position changes
        bt_pos = self.backtest.positions.set_index('date')['position']
        live_pos = self.live.positions.set_index('date')['position']
        
        # Align
        common_dates = bt_pos.index.intersection(live_pos.index)
        bt_aligned = bt_pos.loc[common_dates]
        live_aligned = live_pos.loc[common_dates]
        
        position_correlation = bt_aligned.corr(live_aligned)
        
        # Calculate timing lag
        # Find lag that maximizes correlation
        max_lag = 5
        best_correlation = position_correlation
        best_lag = 0
        
        for lag in range(1, max_lag + 1):
            corr = bt_aligned.iloc[lag:].corr(live_aligned.iloc[:-lag])
            if corr > best_correlation:
                best_correlation = corr
                best_lag = lag
        
        return {
            'position_correlation': position_correlation,
            'best_lag_days': best_lag,
            'best_lag_correlation': best_correlation,
            'timing_cost_estimate_bps': (1 - best_correlation) * 100,
        }
    
    def full_attribution_report(self) -> Dict:
        """
        Generate full attribution report.
        
        Breaks down the gap into explainable components.
        """
        gap_metrics = self.calculate_gap_metrics()
        execution_attr = self.execution_attribution()
        timing_attr = self.timing_attribution()
        
        # Estimate unexplained portion
        total_gap_bps = gap_metrics['gap_bps']
        
        explained_bps = 0
        if 'slippage_attribution_bps' in execution_attr:
            explained_bps += execution_attr['slippage_attribution_bps']
        if 'impact_attribution_bps' in execution_attr:
            explained_bps += execution_attr['impact_attribution_bps']
        if 'timing_cost_estimate_bps' in timing_attr:
            explained_bps += timing_attr['timing_cost_estimate_bps']
        
        unexplained_bps = total_gap_bps - explained_bps
        
        report = {
            'summary': {
                'backtest_return': gap_metrics['backtest_return'],
                'live_return': gap_metrics['live_return'],
                'total_gap_bps': total_gap_bps,
                'explained_bps': explained_bps,
                'unexplained_bps': unexplained_bps,
                'explanation_ratio': explained_bps / total_gap_bps if total_gap_bps != 0 else 0,
            },
            'gap_metrics': gap_metrics,
            'execution_attribution': execution_attr,
            'timing_attribution': timing_attr,
        }
        
        return report


class DataSnoopingAttribution:
    """
    Attributes performance degradation to data snooping.
    
    Data snooping occurs when:
    - Strategy was overfit to historical data
    - Multiple strategies tested, best selected
    - Look-ahead bias in data
    """
    
    def __init__(
        self,
        num_strategies_tested: int,
        backtest_length_years: float,
        live_length_years: float,
    ):
        self.num_strategies = num_strategies_tested
        self.bt_length = backtest_length_years
        self.live_length = live_length_years
    
    def deflated_sharpe(
        self,
        observed_sharpe: float,
        skewness: float = 0,
        kurtosis: float = 3,
    ) -> float:
        """
        Calculate deflated Sharpe ratio.
        
        Adjusts for multiple testing bias.
        """
        from scipy.stats import norm
        
        # Variance of Sharpe ratio
        sr_var = (1 + 0.5 * observed_sharpe**2 - skewness * observed_sharpe +
                  (kurtosis - 3) / 4 * observed_sharpe**2) / (self.bt_length * 252)
        
        # Multiple testing adjustment
        if self.num_strategies > 1:
            # Probability of exceeding observed Sharpe by chance
            prob_exceed = 1 - (1 - norm.cdf(-observed_sharpe, 0, np.sqrt(sr_var))) ** self.num_strategies
            deflated_sr = norm.ppf(1 - prob_exceed, 0, np.sqrt(sr_var))
        else:
            deflated_sr = observed_sharpe
        
        return deflated_sr
    
    def probability_of_backtest_overfitting(
        self,
        backtest_returns: pd.Series,
        live_returns: pd.Series,
    ) -> float:
        """
        Estimate probability of backtest overfitting (PBO).
        
        Uses the method from Bailey et al. (2017).
        """
        # Simplified version - full implementation requires CSCV
        bt_sharpe = backtest_returns.mean() / backtest_returns.std() * np.sqrt(252)
        live_sharpe = live_returns.mean() / live_returns.std() * np.sqrt(252)
        
        # If live Sharpe is much lower than backtest, likely overfit
        ratio = live_sharpe / bt_sharpe if bt_sharpe > 0 else 0
        
        # Estimate PBO
        if ratio < 0.5:
            pbo = 0.8
        elif ratio < 0.8:
            pbo = 0.5
        elif ratio < 1.0:
            pbo = 0.3
        else:
            pbo = 0.1
        
        return pbo


class OutOfSampleValidator:
    """
    Validates if out-of-sample performance is consistent with backtest.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
    
    def validate(
        self,
        backtest_returns: pd.Series,
        oos_returns: pd.Series,
    ) -> Dict:
        """
        Validate OOS performance.
        
        Returns:
            Validation results including whether OOS is in expected range
        """
        bt_mean = backtest_returns.mean()
        bt_std = backtest_returns.std()
        
        oos_mean = oos_returns.mean()
        oos_std = oos_returns.std()
        
        # Predicted OOS range (assuming same distribution)
        n_oos = len(oos_returns)
        se_mean = bt_std / np.sqrt(n_oos)
        
        from scipy.stats import t
        t_critical = t.ppf((1 + self.confidence_level) / 2, n_oos - 1)
        
        lower_bound = bt_mean - t_critical * se_mean
        upper_bound = bt_mean + t_critical * se_mean
        
        within_bounds = lower_bound <= oos_mean <= upper_bound
        
        return {
            'backtest_mean': bt_mean,
            'oos_mean': oos_mean,
            'predicted_range': (lower_bound, upper_bound),
            'within_bounds': within_bounds,
            'oos_std': oos_std,
            'backtest_std': bt_std,
            'std_ratio': oos_std / bt_std if bt_std > 0 else float('inf'),
        }
