"""
Configuration for Realistic Backtest Engine
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from enum import Enum


class DataFrequency(Enum):
    """Supported data frequencies"""
    TICK = "tick"
    MINUTE = "1min"
    MINUTE_5 = "5min"
    MINUTE_15 = "15min"
    MINUTE_30 = "30min"
    HOUR = "1h"
    DAILY = "1d"


class ExecutionModel(Enum):
    """Execution simulation models"""
    IDEAL = "ideal"           # No slippage, ideal fills
    SIMPLE = "simple"         # Fixed slippage
    REALISTIC = "realistic"   # Volume-based slippage with market impact
    LATENCY_AWARE = "latency" # Include latency simulation


@dataclass
class DataConfig:
    """
    Data configuration
    
    Key for preventing look-ahead bias:
    - Use point-in-time data only
    - Proper handling of corporate actions
    - Survivorship bias prevention
    """
    # Data source
    data_path: str = "./data"
    symbols: List[str] = field(default_factory=list)
    frequency: DataFrequency = DataFrequency.DAILY
    
    # Date range
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
    # Point-in-Time settings
    use_adjusted_prices: bool = False  # Use unadjusted to avoid future info
    handle_corporate_actions: bool = True
    
    # Survivorship bias prevention
    include_delisted: bool = True  # Critical for realistic backtests
    
    # Look-ahead prevention
    data_availability_delay: str = "1d"  # Delay between close and data availability
    

@dataclass
class ExecutionConfig:
    """
    Execution simulation configuration
    
    Realistic execution is crucial for reliable backtests.
    """
    model: ExecutionModel = ExecutionModel.REALISTIC
    
    # Commission structure
    commission_rate: float = 0.001  # 0.1% per trade
    min_commission: float = 1.0
    
    # Slippage (basis points)
    base_slippage_bps: float = 5.0  # 5 bps base slippage
    volatility_impact_factor: float = 0.1  # Additional slippage based on volatility
    volume_impact_factor: float = 0.01  # Market impact per unit of volume
    
    # Latency simulation
    latency_ms: float = 100.0  # Order latency in milliseconds
    latency_std_ms: float = 50.0  # Standard deviation of latency
    
    # Market impact model (Almgren-Chriss style)
    permanent_impact_coef: float = 0.1
    temporary_impact_coef: float = 0.05
    
    # Position sizing limits
    max_position_pct: float = 0.1  # Max 10% of portfolio in single position
    max_turnover_daily: float = 0.5  # Max 50% daily turnover


@dataclass
class WalkForwardConfig:
    """
    Walk-forward analysis configuration
    
    Walk-forward is the gold standard for out-of-sample testing.
    """
    enabled: bool = True
    
    # Window configuration
    train_window: int = 252 * 2  # 2 years training
    test_window: int = 63        # 3 months testing
    step_size: int = 63          # Roll forward 3 months each time
    
    # Validation
    min_train_samples: int = 252
    min_test_samples: int = 21
    
    # Purging and embargo
    purge_length: int = 5   # Days to purge between train and test
    embargo_length: int = 5 # Days to embargo after test


@dataclass
class RiskConfig:
    """Risk management configuration"""
    # Position limits
    max_gross_exposure: float = 2.0  # Max 200% gross exposure
    max_net_exposure: float = 1.5   # Max 150% net exposure
    
    # Drawdown controls
    max_drawdown_pct: float = 0.20  # Stop at 20% drawdown
    daily_loss_limit_pct: float = 0.05  # Stop if daily loss > 5%
    
    # Concentration limits
    max_single_position_pct: float = 0.10
    max_sector_exposure_pct: float = 0.30
    

@dataclass
class BacktestConfig:
    """Main backtest configuration"""
    # Sub-configs
    data: DataConfig = field(default_factory=DataConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    walkforward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    
    # General settings
    initial_capital: float = 1_000_000.0
    benchmark_symbol: str = "SPY"
    
    # Reporting
    save_trades: bool = True
    save_daily_pnl: bool = True
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    def validate(self):
        """Validate configuration"""
        assert self.initial_capital > 0, "Initial capital must be positive"
        assert self.execution.commission_rate >= 0, "Commission must be non-negative"
        assert self.walkforward.train_window > self.walkforward.test_window, \
            "Train window must be larger than test window"
        return True
