"""
Realistic Backtest Engine

Main engine that orchestrates:
1. Point-in-Time data access
2. Walk-forward validation
3. Realistic execution simulation
4. Performance attribution
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
import warnings

from .config import BacktestConfig
from .data.pit_data import PointInTimeData
from .execution.simulator import ExecutionSimulator, Order, OrderSide, OrderType
from .walkforward.validator import WalkForwardValidator
from .analytics.attribution import PerformancePeriod, AttributionAnalyzer


class BacktestEngine:
    """
    Main backtest engine.
    
    Provides institutional-grade backtesting with:
    - Point-in-time data (no look-ahead bias)
    - Walk-forward validation
    - Realistic execution simulation
    - Comprehensive attribution analysis
    """
    
    def __init__(self, config: BacktestConfig):
        """
        Initialize backtest engine.
        
        Args:
            config: Backtest configuration
        """
        self.config = config
        config.validate()
        
        # Components
        self.data: Optional[PointInTimeData] = None
        self.execution_simulator = ExecutionSimulator(
            commission_rate=config.execution.commission_rate,
            min_commission=config.execution.min_commission,
        )
        
        # State
        self.current_date: Optional[datetime] = None
        self.positions: Dict[str, float] = {}
        self.cash: float = config.initial_capital
        self.equity_curve: List[Dict] = []
        self.trades: List[Dict] = []
        
    def load_data(self, data: pd.DataFrame):
        """
        Load point-in-time data.
        
        Args:
            data: DataFrame with price data
        """
        self.data = PointInTimeData(
            data,
            availability_delay=self.config.data.data_availability_delay,
        )
        print(f"Loaded data: {len(data)} rows")
    
    def run_backtest(
        self,
        strategy: Callable,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict:
        """
        Run single backtest.
        
        Args:
            strategy: Strategy function that generates signals
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            Backtest results dictionary
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Get date range
        all_dates = self.data.price_data['date'].unique()
        if start_date:
            all_dates = [d for d in all_dates if d >= start_date]
        if end_date:
            all_dates = [d for d in all_dates if d <= end_date]
        
        print(f"Running backtest from {all_dates[0]} to {all_dates[-1]}")
        print(f"Total trading days: {len(all_dates)}")
        
        # Reset state
        self.positions = {}
        self.cash = self.config.initial_capital
        self.equity_curve = []
        self.trades = []
        
        # Main loop
        for i, date in enumerate(all_dates):
            self.current_date = date
            
            # Get available data as of this date
            available_data = self.data.get_data_as_of(date)
            
            if available_data.empty:
                continue
            
            # Get current portfolio value
            portfolio_value = self._calculate_portfolio_value(available_data)
            
            # Record equity
            self.equity_curve.append({
                'date': date,
                'equity': portfolio_value,
                'cash': self.cash,
            })
            
            # Generate signals from strategy
            try:
                signals = strategy(
                    date=date,
                    data=available_data,
                    current_positions=self.positions,
                    current_equity=portfolio_value,
                )
            except Exception as e:
                warnings.warn(f"Strategy error on {date}: {e}")
                signals = []
            
            # Execute orders
            if signals:
                self._execute_signals(signals, available_data, date)
            
            # Progress
            if (i + 1) % 63 == 0:  # Quarterly
                print(f"  Progress: {date.date()} - Equity: ${portfolio_value:,.2f}")
        
        # Calculate performance metrics
        results = self._calculate_performance()
        
        print(f"\nBacktest complete!")
        print(f"  Final equity: ${results['final_equity']:,.2f}")
        print(f"  Total return: {results['total_return']:.2%}")
        print(f"  Sharpe ratio: {results['sharpe_ratio']:.2f}")
        print(f"  Max drawdown: {results['max_drawdown']:.2%}")
        print(f"  Number of trades: {results['num_trades']}")
        
        return results
    
    def run_walk_forward(
        self,
        strategy_factory: Callable,
        metrics_fn: Optional[Callable] = None,
    ) -> Dict:
        """
        Run walk-forward validation.
        
        Args:
            strategy_factory: Function that creates strategy given training data
            metrics_fn: Function to compute performance metrics
            
        Returns:
            Walk-forward results
        """
        if not self.config.walkforward.enabled:
            raise ValueError("Walk-forward not enabled in config")
        
        if self.data is None:
            raise ValueError("No data loaded")
        
        # Create validator
        validator = WalkForwardValidator(
            train_window=self.config.walkforward.train_window,
            test_window=self.config.walkforward.test_window,
            step_size=self.config.walkforward.step_size,
            purge_length=self.config.walkforward.purge_length,
            embargo_length=self.config.walkforward.embargo_length,
        )
        
        # Get all dates
        dates = pd.DatetimeIndex(self.data.price_data['date'].unique())
        
        # Generate windows
        windows = validator.generate_windows(dates)
        
        print(f"Walk-forward validation: {len(windows)} windows")
        
        results = []
        
        for i, window in enumerate(windows):
            print(f"\n{'='*60}")
            print(f"Window {i+1}/{len(windows)}")
            print(f"  Train: {window.train_start.date()} to {window.train_end.date()}")
            print(f"  Test:  {window.test_start.date()} to {window.test_end.date()}")
            
            # Train strategy
            strategy = strategy_factory(window)
            
            # Run backtest on test period
            bt_result = self.run_backtest(
                strategy,
                start_date=window.test_start,
                end_date=window.test_end,
            )
            
            bt_result['window'] = i + 1
            bt_result['train_period'] = (window.train_start, window.train_end)
            bt_result['test_period'] = (window.test_start, window.test_end)
            
            results.append(bt_result)
        
        # Aggregate results
        aggregated = self._aggregate_walk_forward_results(results)
        
        return {
            'window_results': results,
            'aggregated': aggregated,
            'num_windows': len(windows),
        }
    
    def _execute_signals(
        self,
        signals: List[Dict],
        market_data: pd.DataFrame,
        date: datetime,
    ):
        """
        Execute trading signals.
        
        Args:
            signals: List of signal dictionaries
            market_data: Current market data
            date: Current date
        """
        for signal in signals:
            symbol = signal['symbol']
            side = OrderSide.BUY if signal['side'] == 'buy' else OrderSide.SELL
            quantity = signal['quantity']
            
            # Check risk limits
            if not self._check_risk_limits(symbol, quantity, market_data):
                continue
            
            # Create order
            order = Order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=OrderType.MARKET,
                timestamp=date,
            )
            
            # Execute
            if symbol in market_data['symbol'].values:
                symbol_data = market_data[market_data['symbol'] == symbol].iloc[0]
                fill = self.execution_simulator.execute_order(
                    order, symbol_data, date
                )
                
                # Update positions
                if side == OrderSide.BUY:
                    self.positions[symbol] = self.positions.get(symbol, 0) + fill.quantity
                    self.cash -= fill.quantity * fill.fill_price + fill.commission
                else:
                    self.positions[symbol] = self.positions.get(symbol, 0) - fill.quantity
                    self.cash += fill.quantity * fill.fill_price - fill.commission
                
                # Record trade
                self.trades.append({
                    'date': date,
                    'symbol': symbol,
                    'side': side.name,
                    'quantity': fill.quantity,
                    'fill_price': fill.fill_price,
                    'slippage': fill.slippage,
                    'commission': fill.commission,
                })
    
    def _check_risk_limits(
        self,
        symbol: str,
        quantity: float,
        market_data: pd.DataFrame,
    ) -> bool:
        """Check if trade violates risk limits"""
        # Position limit
        new_position = self.positions.get(symbol, 0) + quantity
        portfolio_value = self._calculate_portfolio_value(market_data)
        
        if symbol in market_data['symbol'].values:
            price = market_data[market_data['symbol'] == symbol]['close'].iloc[0]
            position_value = abs(new_position) * price
            
            if position_value > portfolio_value * self.config.risk.max_single_position_pct:
                return False
        
        return True
    
    def _calculate_portfolio_value(self, market_data: pd.DataFrame) -> float:
        """Calculate total portfolio value"""
        position_value = 0.0
        
        for symbol, quantity in self.positions.items():
            if quantity == 0:
                continue
            
            if symbol in market_data['symbol'].values:
                price = market_data[market_data['symbol'] == symbol]['close'].iloc[0]
                position_value += quantity * price
        
        return self.cash + position_value
    
    def _calculate_performance(self) -> Dict:
        """Calculate performance metrics"""
        if not self.equity_curve:
            return {}
        
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('date', inplace=True)
        
        # Returns
        equity_df['returns'] = equity_df['equity'].pct_change()
        
        # Total return
        total_return = (equity_df['equity'].iloc[-1] / self.config.initial_capital) - 1
        
        # Annualized return
        num_days = len(equity_df)
        annual_return = (1 + total_return) ** (252 / num_days) - 1
        
        # Volatility
        volatility = equity_df['returns'].std() * np.sqrt(252)
        
        # Sharpe ratio (assuming 0 risk-free rate for simplicity)
        sharpe = annual_return / volatility if volatility > 0 else 0
        
        # Max drawdown
        cumulative = (1 + equity_df['returns']).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'initial_equity': self.config.initial_capital,
            'final_equity': equity_df['equity'].iloc[-1],
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'num_trades': len(self.trades),
            'equity_curve': equity_df,
            'trades': pd.DataFrame(self.trades),
        }
    
    def _aggregate_walk_forward_results(self, results: List[Dict]) -> Dict:
        """Aggregate walk-forward results"""
        returns = [r['total_return'] for r in results]
        sharpes = [r['sharpe_ratio'] for r in results]
        
        return {
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'mean_sharpe': np.mean(sharpes),
            'std_sharpe': np.std(sharpes),
            'win_rate': sum(1 for r in returns if r > 0) / len(returns),
            'consistency': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
        }


def create_example_strategy():
    """Create an example moving average crossover strategy"""
    
    def strategy(date, data, current_positions, current_equity):
        """Simple MA crossover strategy"""
        signals = []
        
        for _, row in data.iterrows():
            symbol = row['symbol']
            
            # Get historical data for this symbol
            # In real implementation, would use proper historical lookup
            close = row['close']
            
            # Simple signal: buy if price > 50-day MA (simulated)
            # In real implementation, calculate actual MA
            ma20 = close * 0.95  # Placeholder
            
            if close > ma20 and current_positions.get(symbol, 0) <= 0:
                # Buy signal
                quantity = 100
                signals.append({
                    'symbol': symbol,
                    'side': 'buy',
                    'quantity': quantity,
                })
            elif close < ma20 and current_positions.get(symbol, 0) > 0:
                # Sell signal
                signals.append({
                    'symbol': symbol,
                    'side': 'sell',
                    'quantity': current_positions[symbol],
                })
        
        return signals
    
    return strategy
