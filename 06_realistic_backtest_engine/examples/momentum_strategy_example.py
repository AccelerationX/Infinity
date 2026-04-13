"""
Momentum Strategy Example with Realistic Backtest

Demonstrates:
1. Point-in-Time data usage
2. Walk-forward validation
3. Realistic execution simulation
4. Attribution analysis
"""
import sys
sys.path.insert(0, "D:\\ResearchProjects\\06_realistic_backtest_engine")

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backtest_engine import BacktestEngine, BacktestConfig
from backtest_engine.config import DataConfig, ExecutionConfig, WalkForwardConfig
from backtest_engine.data.pit_data import PointInTimeData


def generate_sample_data(
    symbols: list,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Generate synthetic price data for demonstration.
    
    In real usage, this would be replaced with actual market data.
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    data = []
    for symbol in symbols:
        # Generate random walk with momentum
        np.random.seed(hash(symbol) % 2**32)
        returns = np.random.normal(0.0005, 0.02, len(dates))
        
        # Add some momentum (trend)
        for i in range(20, len(returns)):
            if np.mean(returns[i-20:i-10]) > 0.001:
                returns[i] += 0.001  # Upward momentum
            elif np.mean(returns[i-20:i-10]) < -0.001:
                returns[i] -= 0.001  # Downward momentum
        
        prices = 100 * (1 + returns).cumprod()
        
        for i, date in enumerate(dates):
            data.append({
                'date': date,
                'symbol': symbol,
                'open': prices[i] * (1 + np.random.normal(0, 0.001)),
                'high': prices[i] * (1 + abs(np.random.normal(0, 0.01))),
                'low': prices[i] * (1 - abs(np.random.normal(0, 0.01))),
                'close': prices[i],
                'volume': np.random.randint(1000000, 10000000),
            })
    
    return pd.DataFrame(data)


def create_momentum_strategy(lookback: int = 20):
    """
    Create a simple momentum strategy.
    
    Buys stocks with positive momentum, sells when momentum turns negative.
    """
    
    def strategy(date, data, current_positions, current_equity):
        """
        Strategy function called each day.
        
        Args:
            date: Current date
            data: Available market data (PIT)
            current_positions: Current portfolio positions
            current_equity: Current portfolio value
            
        Returns:
            List of trade signals
        """
        signals = []
        
        # We need at least lookback days of data
        if len(data) < lookback:
            return signals
        
        # Check each stock
        for _, row in data.iterrows():
            symbol = row['symbol']
            current_price = row['close']
            
            # In real implementation, would get historical prices
            # For demo, use simplified momentum calculation
            # momentum = (current_price / price_lookback_days_ago) - 1
            
            # Simulate momentum with random logic for demonstration
            np.random.seed(hash(symbol) % 2**32 + date.day)
            momentum = np.random.normal(0, 0.02)
            
            current_pos = current_positions.get(symbol, 0)
            
            # Entry signal: positive momentum and no position
            if momentum > 0.01 and current_pos <= 0:
                # Calculate position size (fixed dollar amount)
                target_value = current_equity * 0.1  # 10% per position
                quantity = int(target_value / current_price)
                
                if quantity > 0:
                    signals.append({
                        'symbol': symbol,
                        'side': 'buy',
                        'quantity': quantity,
                    })
            
            # Exit signal: negative momentum and have position
            elif momentum < -0.01 and current_pos > 0:
                signals.append({
                    'symbol': symbol,
                    'side': 'sell',
                    'quantity': current_pos,  # Sell all
                })
        
        return signals
    
    return strategy


def run_simple_backtest():
    """Run a simple backtest demonstration"""
    print("="*70)
    print("MOMENTUM STRATEGY BACKTEST EXAMPLE")
    print("="*70)
    
    # Generate sample data
    print("\n[1/4] Generating sample data...")
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    price_data = generate_sample_data(
        symbols=symbols,
        start_date='2020-01-01',
        end_date='2023-12-31',
    )
    print(f"  Generated {len(price_data)} data points for {len(symbols)} symbols")
    
    # Create configuration
    print("\n[2/4] Configuring backtest engine...")
    config = BacktestConfig(
        initial_capital=1_000_000,
        data=DataConfig(
            data_availability_delay="1d",  # 1-day delay
        ),
        execution=ExecutionConfig(
            commission_rate=0.001,  # 10 bps
            base_slippage_bps=5.0,  # 5 bps slippage
        ),
    )
    
    # Create engine
    engine = BacktestEngine(config)
    engine.load_data(price_data)
    print("  Engine configured with realistic execution costs")
    
    # Create strategy
    strategy = create_momentum_strategy(lookback=20)
    
    # Run backtest
    print("\n[3/4] Running backtest...")
    results = engine.run_backtest(strategy)
    
    # Print results
    print("\n[4/4] Backtest Results:")
    print("-" * 50)
    print(f"Initial Capital: ${config.initial_capital:,.2f}")
    print(f"Final Equity:    ${results['final_equity']:,.2f}")
    print(f"Total Return:    {results['total_return']:.2%}")
    print(f"Annual Return:   {results['annual_return']:.2%}")
    print(f"Volatility:      {results['volatility']:.2%}")
    print(f"Sharpe Ratio:    {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:    {results['max_drawdown']:.2%}")
    print(f"Calmar Ratio:    {results['calmar_ratio']:.2f}")
    print(f"Number of Trades: {results['num_trades']}")
    
    return results


def run_walk_forward_demo():
    """Demonstrate walk-forward validation"""
    print("\n" + "="*70)
    print("WALK-FORWARD VALIDATION EXAMPLE")
    print("="*70)
    
    # Generate data
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    price_data = generate_sample_data(
        symbols=symbols,
        start_date='2018-01-01',
        end_date='2023-12-31',
    )
    
    # Config with walk-forward
    config = BacktestConfig(
        initial_capital=1_000_000,
        walk_forward=WalkForwardConfig(
            enabled=True,
            train_window=252,  # 1 year
            test_window=63,    # 3 months
            step_size=63,      # 3 months
        ),
    )
    
    engine = BacktestEngine(config)
    engine.load_data(price_data)
    
    # Strategy factory
    def strategy_factory(window):
        # Could optimize parameters on training data here
        return create_momentum_strategy(lookback=20)
    
    # Run walk-forward
    print("\nRunning walk-forward validation...")
    wf_results = engine.run_walk_forward(strategy_factory)
    
    # Print aggregated results
    agg = wf_results['aggregated']
    print("\nWalk-Forward Results:")
    print("-" * 50)
    print(f"Number of Windows: {wf_results['num_windows']}")
    print(f"Mean Return:       {agg['mean_return']:.2%}")
    print(f"Return Std:        {agg['std_return']:.2%}")
    print(f"Mean Sharpe:       {agg['mean_sharpe']:.2f}")
    print(f"Win Rate:          {agg['win_rate']:.1%}")
    print(f"Consistency:       {agg['consistency']:.2f}")
    
    return wf_results


def demonstrate_pit_data():
    """Demonstrate Point-in-Time data usage"""
    print("\n" + "="*70)
    print("POINT-IN-TIME DATA DEMONSTRATION")
    print("="*70)
    
    # Create sample data with availability delay
    dates = pd.date_range('2020-01-01', '2020-01-10', freq='D')
    
    data = []
    for i, date in enumerate(dates):
        data.append({
            'date': date,
            'symbol': 'AAPL',
            'close': 100 + i,
            'volume': 1000000,
        })
    
    df = pd.DataFrame(data)
    
    # Create PIT data with 1-day delay
    pit_data = PointInTimeData(df, availability_delay="1d")
    
    # Demonstrate what data is available at each point
    print("\nData Availability:")
    print("-" * 50)
    print(f"{'Date':<15} {'Latest Available Data':<25} {'Data Point Date'}")
    print("-" * 50)
    
    for date in dates[2:]:  # Start from 3rd day
        available = pit_data.get_data_as_of(date)
        if not available.empty:
            latest_date = available['date'].max()
            print(f"{date.strftime('%Y-%m-%d'):<15} "
                  f"{latest_date.strftime('%Y-%m-%d'):<25} "
                  f"{date.strftime('%Y-%m-%d')}")
    
    print("\nKey Insight: Data for date T is not available until T+1")
    print("This prevents look-ahead bias in backtests.")


def main():
    """Main function"""
    print("="*70)
    print("REALISTIC BACKTEST ENGINE - EXAMPLE SUITE")
    print("="*70)
    
    # Run demonstrations
    demonstrate_pit_data()
    
    print("\n" + "="*70)
    
    # Simple backtest
    results = run_simple_backtest()
    
    # Walk-forward (commented out for speed)
    # wf_results = run_walk_forward_demo()
    
    print("\n" + "="*70)
    print("EXAMPLE COMPLETE")
    print("="*70)
    print("\nKey Takeaways:")
    print("1. Point-in-Time data prevents look-ahead bias")
    print("2. Realistic execution costs are crucial")
    print("3. Walk-forward validation prevents overfitting")
    print("4. Attribution analysis explains discrepancies")


if __name__ == "__main__":
    main()
