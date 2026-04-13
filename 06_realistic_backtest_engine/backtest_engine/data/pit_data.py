"""
Point-in-Time (PIT) Data Management

CRITICAL for preventing look-ahead bias:
- Only data available at time T can be used for decisions at T
- No use of revised/finalized data that wasn't available at T
- Proper handling of as-of dates for fundamental data
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings


class PointInTimeData:
    """
    Point-in-Time Data Container
    
    Ensures that backtests only use information that was actually
    available at the time of the decision.
    """
    
    def __init__(
        self,
        price_data: pd.DataFrame,
        availability_delay: str = "1d",
        corporate_actions: Optional[pd.DataFrame] = None,
    ):
        """
        Args:
            price_data: DataFrame with columns [date, symbol, open, high, low, close, volume]
            availability_delay: Delay between market close and data availability
            corporate_actions: DataFrame with [date, symbol, action, ratio]
        """
        self.price_data = price_data.copy()
        self.availability_delay = availability_delay
        self.corporate_actions = corporate_actions
        
        # Ensure date column is datetime
        self.price_data['date'] = pd.to_datetime(self.price_data['date'])
        self.price_data = self.price_data.sort_values(['symbol', 'date'])
        
        # Track what data is available at each point in time
        self._build_availability_index()
    
    def _build_availability_index(self):
        """
        Build index of when each data point becomes available.
        
        This is the core mechanism for preventing look-ahead bias.
        """
        # Add availability timestamp
        delay = self._parse_delay(self.availability_delay)
        self.price_data['available_at'] = self.price_data['date'] + delay
        
    def _parse_delay(self, delay_str: str) -> timedelta:
        """Parse delay string to timedelta"""
        if delay_str.endswith('d'):
            return timedelta(days=int(delay_str[:-1]))
        elif delay_str.endswith('h'):
            return timedelta(hours=int(delay_str[:-1]))
        else:
            return timedelta(days=1)
    
    def get_data_as_of(
        self,
        as_of_date: datetime,
        symbols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Get data that is available as of a specific date.
        
        This is the KEY function for preventing look-ahead bias.
        Only returns data where available_at <= as_of_date.
        
        Args:
            as_of_date: The point in time we want data for
            symbols: List of symbols to filter (None = all)
            
        Returns:
            DataFrame with data available at as_of_date
        """
        mask = self.price_data['available_at'] <= as_of_date
        
        if symbols is not None:
            mask &= self.price_data['symbol'].isin(symbols)
        
        data = self.price_data[mask].copy()
        
        # Get most recent data for each symbol
        data = data.sort_values('date').groupby('symbol').last().reset_index()
        
        return data
    
    def get_historical_data(
        self,
        symbol: str,
        end_date: datetime,
        lookback_periods: int,
    ) -> pd.DataFrame:
        """
        Get historical price data up to a point in time.
        
        Args:
            symbol: Stock symbol
            end_date: End date (inclusive, PIT)
            lookback_periods: Number of periods to look back
            
        Returns:
            DataFrame with historical data
        """
        # Get all data available as of end_date
        data = self.get_data_as_of(end_date, [symbol])
        
        if data.empty:
            return pd.DataFrame()
        
        # Get symbol-specific data
        symbol_data = self.price_data[
            (self.price_data['symbol'] == symbol) &
            (self.price_data['available_at'] <= end_date)
        ].sort_values('date')
        
        # Return last N periods
        return symbol_data.tail(lookback_periods)
    
    def get_universe_as_of(
        self,
        as_of_date: datetime,
        min_price: float = 1.0,
        min_volume: float = 1e6,
    ) -> List[str]:
        """
        Get tradable universe as of a specific date.
        
        Filters out stocks that:
        - Are delisted
        - Have insufficient liquidity
        - Don't meet price requirements
        
        Args:
            as_of_date: Point in time
            min_price: Minimum stock price
            min_volume: Minimum average daily volume
            
        Returns:
            List of symbols in universe
        """
        data = self.get_data_as_of(as_of_date)
        
        if data.empty:
            return []
        
        # Filter by price and volume
        mask = (data['close'] >= min_price) & (data['volume'] >= min_volume)
        universe = data[mask]['symbol'].tolist()
        
        return universe


class CorporateActionHandler:
    """
    Handles corporate actions (splits, dividends) in a PIT manner.
    
    CRITICAL: Must use announcement dates, not ex-dates or pay dates,
    to avoid look-ahead bias.
    """
    
    def __init__(self, actions_data: pd.DataFrame):
        """
        Args:
            actions_data: DataFrame with columns:
                - date: Announcement date (when info became available)
                - symbol: Stock symbol
                - action_type: 'split' or 'dividend'
                - ratio: Split ratio or dividend amount
                - ex_date: Ex-date (for reference, not for PIT)
        """
        self.actions = actions_data.copy()
        self.actions['date'] = pd.to_datetime(self.actions['date'])
        self.actions = self.actions.sort_values('date')
    
    def get_adjustment_factor(
        self,
        symbol: str,
        as_of_date: datetime,
        price_date: datetime,
    ) -> float:
        """
        Get cumulative adjustment factor between price_date and as_of_date.
        
        This adjusts historical prices to be comparable as of as_of_date.
        
        Args:
            symbol: Stock symbol
            as_of_date: Current point in time
            price_date: Date of the price to adjust
            
        Returns:
            Adjustment factor (multiply price by this)
        """
        # Get all actions announced between price_date and as_of_date
        mask = (
            (self.actions['symbol'] == symbol) &
            (self.actions['date'] > price_date) &
            (self.actions['date'] <= as_of_date)
        )
        
        actions = self.actions[mask]
        
        if actions.empty:
            return 1.0
        
        # Calculate cumulative adjustment
        factor = 1.0
        for _, action in actions.iterrows():
            if action['action_type'] == 'split':
                factor *= action['ratio']
            elif action['action_type'] == 'dividend':
                # For dividends, we need the stock price at that time
                # This is a simplified version
                pass
        
        return factor
    
    def adjust_price(
        self,
        price: float,
        symbol: str,
        price_date: datetime,
        as_of_date: datetime,
    ) -> float:
        """Adjust a historical price to be comparable as of as_of_date"""
        factor = self.get_adjustment_factor(symbol, as_of_date, price_date)
        return price * factor


class SurvivorshipBiasFreeData:
    """
    Data that includes delisted stocks to prevent survivorship bias.
    
    Survivorship bias is a major issue in backtesting:
    - Only testing on stocks that survived gives inflated returns
    - Must include delisted stocks for realistic results
    """
    
    def __init__(
        self,
        active_stocks: pd.DataFrame,
        delisted_stocks: Optional[pd.DataFrame] = None,
    ):
        self.active = active_stocks
        self.delisted = delisted_stocks or pd.DataFrame()
        
        # Combine
        self.all_stocks = pd.concat([self.active, self.delisted])
        
    def get_full_universe(self, date: datetime) -> pd.DataFrame:
        """
        Get all stocks that existed as of date, including delisted ones.
        
        This is essential for realistic backtests.
        """
        # Stocks that were listed before date
        mask = self.all_stocks['listing_date'] <= date
        
        # And either still active or delisted after date
        delisted_mask = (
            (self.all_stocks['delisted_date'].isna()) |
            (self.all_stocks['delisted_date'] > date)
        )
        
        return self.all_stocks[mask & delisted_mask]


class DataSnoopingDetector:
    """
    Detects potential data snooping / look-ahead bias in backtests.
    
    This is a safety check to ensure the backtest is realistic.
    """
    
    @staticmethod
    def check_for_lookahead_bias(trades: pd.DataFrame, data: PointInTimeData) -> Dict:
        """
        Check if any trades used future information.
        
        Args:
            trades: DataFrame with [date, symbol, action]
            data: PointInTimeData instance
            
        Returns:
            Dictionary with check results
        """
        issues = []
        
        for _, trade in trades.iterrows():
            trade_date = pd.to_datetime(trade['date'])
            symbol = trade['symbol']
            
            # Check if we had data for this symbol at trade time
            available_data = data.get_data_as_of(trade_date, [symbol])
            
            if available_data.empty:
                issues.append({
                    'date': trade_date,
                    'symbol': symbol,
                    'issue': 'No data available at trade time',
                })
            elif available_data['available_at'].iloc[0] > trade_date:
                issues.append({
                    'date': trade_date,
                    'symbol': symbol,
                    'issue': 'Data not yet available at trade time',
                })
        
        return {
            'clean': len(issues) == 0,
            'num_issues': len(issues),
            'issues': issues,
        }
    
    @staticmethod
    def check_for_survivorship_bias(
        returns: pd.Series,
        backtest_universe: set,
        full_universe: set,
    ) -> Dict:
        """
        Check if backtest only included surviving stocks.
        
        Args:
            returns: Strategy returns
            backtest_universe: Set of symbols in backtest
            full_universe: Set of all possible symbols (including delisted)
            
        Returns:
            Dictionary with check results
        """
        missing_stocks = full_universe - backtest_universe
        
        # If we excluded >20% of stocks, flag as potential survivorship bias
        bias_ratio = len(missing_stocks) / len(full_universe) if full_universe else 0
        
        return {
            'clean': bias_ratio < 0.2,
            'bias_ratio': bias_ratio,
            'num_missing': len(missing_stocks),
            'missing_sample': list(missing_stocks)[:10],
        }
