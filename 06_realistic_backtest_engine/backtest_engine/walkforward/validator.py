"""
Walk-Forward Analysis

The gold standard for out-of-sample testing.

Key concept: Train on past data, test on future data, then roll forward.
This prevents overfitting and provides realistic performance estimates.

Structure:
[Train Period] [Test Period] -> Roll -> [Train Period] [Test Period] -> ...

With purging and embargo to prevent leakage between train and test.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Callable, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class WFWindow:
    """A single walk-forward window"""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    
    def __repr__(self):
        return (f"WFWindow(train={self.train_start.date()} to {self.train_end.date()}, "
                f"test={self.test_start.date()} to {self.test_end.date()})")


class WalkForwardValidator:
    """
    Walk-forward cross-validation for time series.
    
    Unlike k-fold CV, this respects temporal ordering and prevents
    look-ahead bias in model validation.
    """
    
    def __init__(
        self,
        train_window: int,
        test_window: int,
        step_size: int,
        purge_length: int = 0,
        embargo_length: int = 0,
    ):
        """
        Args:
            train_window: Number of periods in training window
            test_window: Number of periods in testing window
            step_size: How many periods to roll forward each iteration
            purge_length: Number of periods to purge between train and test
            embargo_length: Number of periods to embargo after test
        """
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.purge_length = purge_length
        self.embargo_length = embargo_length
    
    def generate_windows(
        self,
        dates: pd.DatetimeIndex,
    ) -> List[WFWindow]:
        """
        Generate walk-forward windows.
        
        Args:
            dates: All available dates
            
        Returns:
            List of WFWindow objects
        """
        windows = []
        n = len(dates)
        
        # Start after we have enough training data
        start_idx = self.train_window + self.purge_length
        
        while start_idx + self.test_window + self.embargo_length <= n:
            # Training period
            train_start_idx = start_idx - self.train_window
            train_end_idx = start_idx - self.purge_length
            
            # Testing period
            test_start_idx = start_idx
            test_end_idx = start_idx + self.test_window
            
            # Embargo check
            if test_end_idx + self.embargo_length > n:
                break
            
            window = WFWindow(
                train_start=dates[train_start_idx],
                train_end=dates[train_end_idx - 1],
                test_start=dates[test_start_idx],
                test_end=dates[test_end_idx - 1],
            )
            
            windows.append(window)
            
            # Roll forward
            start_idx += self.step_size
        
        return windows
    
    def validate(
        self,
        data: pd.DataFrame,
        strategy_factory: Callable,
        metrics_fn: Callable,
    ) -> Dict:
        """
        Run walk-forward validation.
        
        Args:
            data: Full dataset with 'date' column
            strategy_factory: Function that creates strategy given train data
            metrics_fn: Function to compute performance metrics
            
        Returns:
            Dictionary with results for each window and aggregated results
        """
        dates = pd.DatetimeIndex(sorted(data['date'].unique()))
        windows = self.generate_windows(dates)
        
        print(f"Walk-Forward Validation: {len(windows)} windows")
        
        results = []
        
        for i, window in enumerate(windows):
            print(f"\nWindow {i+1}/{len(windows)}: {window}")
            
            # Split data
            train_mask = (
                (data['date'] >= window.train_start) &
                (data['date'] <= window.train_end)
            )
            test_mask = (
                (data['date'] >= window.test_start) &
                (data['date'] <= window.test_end)
            )
            
            train_data = data[train_mask]
            test_data = data[test_mask]
            
            # Train strategy
            strategy = strategy_factory(train_data)
            
            # Test strategy
            predictions = strategy.predict(test_data)
            
            # Compute metrics
            metrics = metrics_fn(test_data, predictions)
            metrics['window'] = i + 1
            metrics['train_period'] = (window.train_start, window.train_end)
            metrics['test_period'] = (window.test_start, window.test_end)
            
            results.append(metrics)
        
        # Aggregate results
        aggregated = self._aggregate_results(results)
        
        return {
            'window_results': results,
            'aggregated': aggregated,
            'num_windows': len(windows),
        }
    
    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggregate results across windows"""
        if not results:
            return {}
        
        # Extract numeric metrics
        numeric_metrics = {}
        for key in results[0].keys():
            if isinstance(results[0][key], (int, float)):
                values = [r[key] for r in results if key in r]
                if values:
                    numeric_metrics[key] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'sharpe': np.mean(values) / np.std(values) if np.std(values) > 0 else 0,
                    }
        
        return numeric_metrics


class PurgedKFold:
    """
    Purged k-fold cross-validation for time series.
    
    From "Advances in Financial Machine Learning" by Marcos Lopez de Prado.
    
    Between train and test sets, we purge observations that are too close
    in time to prevent information leakage.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        purge_length: int = 10,
        embargo_length: int = 5,
    ):
        self.n_splits = n_splits
        self.purge_length = purge_length
        self.embargo_length = embargo_length
    
    def split(self, dates: pd.DatetimeIndex):
        """
        Generate train/test splits with purging.
        
        Yields:
            (train_indices, test_indices) tuples
        """
        n = len(dates)
        fold_size = n // self.n_splits
        
        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = min((i + 1) * fold_size, n)
            
            # Purge boundaries
            train_end = max(0, test_start - self.purge_length)
            train_start = 0
            
            # Embargo after test
            test_end_with_embargo = min(test_end + self.embargo_length, n)
            
            train_indices = list(range(train_start, train_end))
            test_indices = list(range(test_start, test_end))
            
            yield train_indices, test_indices


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation
    
    From Lopez de Prado (2018).
    
    Generates multiple backtest paths to estimate the distribution
    of strategy performance, not just a single backtest.
    
    This is the most rigorous form of backtest validation.
    """
    
    def __init__(
        self,
        n_splits: int = 10,
        n_test_splits: int = 2,
        purge_length: int = 10,
    ):
        """
        Args:
            n_splits: Total number of splits
            n_test_splits: Number of splits to use for testing
            purge_length: Number of periods to purge
        """
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge_length = purge_length
    
    def generate_paths(self, dates: pd.DatetimeIndex) -> List[Dict]:
        """
        Generate multiple backtest paths.
        
        Returns:
            List of path configurations
        """
        from itertools import combinations
        
        # Generate all combinations of test splits
        split_indices = range(self.n_splits)
        test_combinations = list(combinations(split_indices, self.n_test_splits))
        
        paths = []
        for test_splits in test_combinations:
            path = self._create_path(dates, test_splits)
            paths.append(path)
        
        return paths
    
    def _create_path(
        self,
        dates: pd.DatetimeIndex,
        test_splits: Tuple[int, ...],
    ) -> Dict:
        """Create a single backtest path"""
        n = len(dates)
        fold_size = n // self.n_splits
        
        # Determine test periods
        test_indices = set()
        for split_idx in test_splits:
            start = split_idx * fold_size
            end = min((split_idx + 1) * fold_size, n)
            test_indices.update(range(start, end))
        
        # Determine train periods (everything else, with purging)
        train_indices = set()
        for i in range(n):
            if i not in test_indices:
                # Check if close to any test index
                is_purged = False
                for test_idx in test_indices:
                    if abs(i - test_idx) <= self.purge_length:
                        is_purged = True
                        break
                
                if not is_purged:
                    train_indices.add(i)
        
        return {
            'train_indices': sorted(list(train_indices)),
            'test_indices': sorted(list(test_indices)),
            'test_splits': test_splits,
        }


class BacktestRobustnessAnalyzer:
    """
    Analyzes the robustness of backtest results.
    
    Uses multiple validation techniques to assess if results are
    likely to hold out of sample.
    """
    
    def __init__(self):
        self.metrics = {}
    
    def analyze_walk_forward(self, wf_results: Dict) -> Dict:
        """
        Analyze walk-forward results for consistency.
        
        Look for:
        1. Consistent performance across windows
        2. No degradation over time
        3. Reasonable variance
        """
        window_results = wf_results['window_results']
        aggregated = wf_results['aggregated']
        
        analysis = {
            'num_windows': len(window_results),
            'sharpe_ratio': aggregated.get('sharpe', 0),
            'consistency_score': 0,
            'warnings': [],
        }
        
        # Check for consistency
        if 'returns' in aggregated:
            returns_by_window = [r.get('returns', 0) for r in window_results]
            
            # Warning if too many negative windows
            negative_ratio = sum(1 for r in returns_by_window if r < 0) / len(returns_by_window)
            if negative_ratio > 0.3:
                analysis['warnings'].append(
                    f"High negative window ratio: {negative_ratio:.1%}"
                )
            
            # Warning if deteriorating trend
            if len(returns_by_window) > 3:
                first_half = np.mean(returns_by_window[:len(returns_by_window)//2])
                second_half = np.mean(returns_by_window[len(returns_by_window)//2:])
                if second_half < first_half * 0.5:
                    analysis['warnings'].append(
                        f"Performance deteriorating: {first_half:.2%} -> {second_half:.2%}"
                    )
        
        # Consistency score
        if analysis['sharpe_ratio'] > 1.0 and not analysis['warnings']:
            analysis['consistency_score'] = 5
        elif analysis['sharpe_ratio'] > 0.5:
            analysis['consistency_score'] = 3
        else:
            analysis['consistency_score'] = 1
        
        return analysis
    
    def deflated_sharpe_ratio(
        self,
        sharpe_ratio: float,
        num_trials: int,
        skewness: float = 0,
        kurtosis: float = 3,
    ) -> float:
        """
        Deflated Sharpe Ratio from Lopez de Prado (2019).
        
        Adjusts Sharpe ratio for multiple trials (data snooping).
        
        Args:
            sharpe_ratio: Observed Sharpe ratio
            num_trials: Number of strategy variations tested
            skewness: Return skewness
            kurtosis: Return kurtosis
            
        Returns:
            Deflated Sharpe ratio (lower than nominal)
        """
        import scipy.stats as stats
        
        # Standard error of Sharpe ratio
        sr_var = (1 + 0.5 * sharpe_ratio**2 - skewness * sharpe_ratio +
                  (kurtosis - 3) / 4 * sharpe_ratio**2)
        
        # Multiple testing adjustment
        if num_trials > 1:
            # Bonferroni-style adjustment
            adjusted_sr = sharpe_ratio * np.sqrt(1 / num_trials)
        else:
            adjusted_sr = sharpe_ratio
        
        return adjusted_sr
