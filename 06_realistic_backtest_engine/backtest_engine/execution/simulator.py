"""
Realistic Execution Simulator

Simulates real-world execution conditions:
1. Slippage - difference between expected and actual fill price
2. Market impact - price movement caused by our order
3. Latency - delay between decision and execution
4. Liquidity constraints - can't fill large orders at best price

Based on academic models:
- Almgren-Chriss for market impact
- Kyle's lambda for liquidity
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    VWAP = "vwap"


class OrderSide(Enum):
    BUY = 1
    SELL = -1


@dataclass
class Order:
    """Trade order"""
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    timestamp: Optional[pd.Timestamp] = None


@dataclass
class Fill:
    """Execution fill"""
    symbol: str
    side: OrderSide
    quantity: float
    fill_price: float
    expected_price: float
    slippage: float
    commission: float
    timestamp: pd.Timestamp
    
    @property
    def market_impact(self) -> float:
        """Price impact of the trade"""
        return abs(self.fill_price - self.expected_price) / self.expected_price


class SlippageModel:
    """
    Base class for slippage models.
    
    Slippage is the difference between expected price and actual fill price.
    """
    
    def calculate_slippage(
        self,
        order: Order,
        market_data: pd.Series,
        **kwargs
    ) -> float:
        """
        Calculate slippage for an order.
        
        Returns:
            Slippage as a percentage (e.g., 0.001 = 10 bps)
        """
        raise NotImplementedError


class FixedSlippage(SlippageModel):
    """Fixed slippage model - constant bps per trade"""
    
    def __init__(self, basis_points: float = 5.0):
        self.basis_points = basis_points
    
    def calculate_slippage(self, order: Order, market_data: pd.Series, **kwargs) -> float:
        return self.basis_points / 10000.0


class VolumeBasedSlippage(SlippageModel):
    """
    Volume-based slippage model.
    
    Higher slippage for:
    - Large orders relative to volume
    - Low liquidity stocks
    - High volatility periods
    """
    
    def __init__(
        self,
        base_bps: float = 2.0,
        volume_impact_coef: float = 0.1,
        vol_impact_coef: float = 0.05,
    ):
        self.base_bps = base_bps
        self.volume_impact_coef = volume_impact_coef
        self.vol_impact_coef = vol_impact_coef
    
    def calculate_slippage(
        self,
        order: Order,
        market_data: pd.Series,
        **kwargs
    ) -> float:
        """
        Calculate slippage based on order size and market conditions.
        
        Formula: slippage = base + vol_impact * participation + vol_factor * volatility
        """
        base_slippage = self.base_bps / 10000.0
        
        # Volume impact (participation rate)
        daily_volume = market_data.get('volume', 1e6)
        participation_rate = abs(order.quantity) / daily_volume
        volume_impact = self.volume_impact_coef * participation_rate
        
        # Volatility impact
        volatility = market_data.get('volatility', 0.02)  # Default 2% daily vol
        vol_impact = self.vol_impact_coef * volatility * np.sqrt(participation_rate)
        
        total_slippage = base_slippage + volume_impact + vol_impact
        
        # Direction: buys slip up, sells slip down
        if order.side == OrderSide.BUY:
            return total_slippage
        else:
            return -total_slippage


class MarketImpactModel:
    """
    Market impact model based on Almgren-Chriss (2000).
    
    Decomposes impact into:
    - Temporary impact: transient, decays quickly
    - Permanent impact: lasting change in price
    """
    
    def __init__(
        self,
        temp_impact_coef: float = 0.05,
        perm_impact_coef: float = 0.01,
        decay_factor: float = 0.5,
    ):
        """
        Args:
            temp_impact_coef: Coefficient for temporary impact
            perm_impact_coef: Coefficient for permanent impact
            decay_factor: Decay rate of temporary impact
        """
        self.temp_impact_coef = temp_impact_coef
        self.perm_impact_coef = perm_impact_coef
        self.decay_factor = decay_factor
        
        # Track outstanding impact
        self.outstanding_impact: Dict[str, float] = {}
    
    def calculate_impact(
        self,
        order: Order,
        market_data: pd.Series,
        execution_time: float = 1.0,  # Hours to execute
    ) -> Tuple[float, float]:
        """
        Calculate market impact.
        
        Args:
            order: Order to execute
            market_data: Current market conditions
            execution_time: Time to complete execution (hours)
            
        Returns:
            (temporary_impact, permanent_impact) as percentages
        """
        quantity = abs(order.quantity)
        adv = market_data.get('adv', 1e6)  # Average daily volume
        volatility = market_data.get('volatility', 0.02)
        
        # Participation rate
        participation = quantity / adv
        
        # Almgren-Chriss formulas
        # Temporary impact (cost of urgency)
        temp_impact = (
            self.temp_impact_coef * volatility *
            (quantity / (adv * execution_time / 6.5)) ** 0.6  # 0.6 exponent from paper
        )
        
        # Permanent impact (information leakage)
        perm_impact = self.perm_impact_coef * volatility * np.sqrt(participation)
        
        # Apply decay to previous outstanding impact
        for symbol in self.outstanding_impact:
            self.outstanding_impact[symbol] *= self.decay_factor
        
        # Add new impact
        self.outstanding_impact[order.symbol] = (
            self.outstanding_impact.get(order.symbol, 0) + temp_impact
        )
        
        return temp_impact, perm_impact


class LatencySimulator:
    """
    Simulates execution latency.
    
    In reality, there's delay between:
    1. Signal generation
    2. Order submission
    3. Order arrival at exchange
    4. Fill confirmation
    
    This can significantly affect high-frequency strategies.
    """
    
    def __init__(
        self,
        base_latency_ms: float = 100.0,
        latency_std_ms: float = 50.0,
        min_latency_ms: float = 10.0,
    ):
        self.base_latency = base_latency_ms
        self.latency_std = latency_std_ms
        self.min_latency = min_latency_ms
    
    def simulate_latency(self) -> float:
        """Simulate latency in milliseconds"""
        latency = np.random.normal(self.base_latency, self.latency_std)
        return max(latency, self.min_latency)
    
    def get_fill_price(
        self,
        signal_price: float,
        market_data: pd.Series,
        side: OrderSide,
        latency_ms: float,
    ) -> float:
        """
        Get the actual fill price considering latency.
        
        Price may have moved during latency period.
        """
        # Estimate price drift during latency
        volatility = market_data.get('volatility', 0.02)  # Daily volatility
        vol_per_ms = volatility / (6.5 * 3600 * 1000)  # Per ms
        
        # Random price move
        drift = np.random.normal(0, vol_per_ms * np.sqrt(latency_ms))
        
        if side == OrderSide.BUY:
            # Buy at worse price (higher)
            fill_price = signal_price * (1 + abs(drift))
        else:
            # Sell at worse price (lower)
            fill_price = signal_price * (1 - abs(drift))
        
        return fill_price


class ExecutionSimulator:
    """
    Main execution simulator combining all models.
    """
    
    def __init__(
        self,
        slippage_model: Optional[SlippageModel] = None,
        impact_model: Optional[MarketImpactModel] = None,
        latency_model: Optional[LatencySimulator] = None,
        commission_rate: float = 0.001,
        min_commission: float = 1.0,
    ):
        """
        Args:
            slippage_model: Model for slippage
            impact_model: Model for market impact
            latency_model: Model for latency
            commission_rate: Commission as fraction of trade value
            min_commission: Minimum commission per trade
        """
        self.slippage_model = slippage_model or VolumeBasedSlippage()
        self.impact_model = impact_model or MarketImpactModel()
        self.latency_model = latency_model or LatencySimulator()
        self.commission_rate = commission_rate
        self.min_commission = min_commission
    
    def execute_order(
        self,
        order: Order,
        market_data: pd.Series,
        timestamp: pd.Timestamp,
    ) -> Fill:
        """
        Execute an order and return fill details.
        
        Args:
            order: Order to execute
            market_data: Current market data (OHLCV, etc.)
            timestamp: Execution timestamp
            
        Returns:
            Fill object with execution details
        """
        # Expected price (mid for market orders)
        if order.order_type == OrderType.MARKET:
            expected_price = (market_data['high'] + market_data['low']) / 2
        else:
            expected_price = order.limit_price or market_data['close']
        
        # Simulate latency
        latency_ms = self.latency_model.simulate_latency()
        
        # Get fill price with latency
        fill_price = self.latency_model.get_fill_price(
            expected_price, market_data, order.side, latency_ms
        )
        
        # Add slippage
        slippage = self.slippage_model.calculate_slippage(order, market_data)
        fill_price = fill_price * (1 + slippage)
        
        # Add market impact
        temp_impact, perm_impact = self.impact_model.calculate_impact(
            order, market_data
        )
        fill_price = fill_price * (1 + temp_impact + perm_impact)
        
        # Calculate commission
        trade_value = abs(order.quantity) * fill_price
        commission = max(trade_value * self.commission_rate, self.min_commission)
        
        # Round to tick size (assume $0.01 for stocks)
        fill_price = round(fill_price, 2)
        
        return Fill(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            fill_price=fill_price,
            expected_price=expected_price,
            slippage=slippage,
            commission=commission,
            timestamp=timestamp,
        )
    
    def execute_orders(
        self,
        orders: List[Order],
        market_data: pd.DataFrame,
        timestamp: pd.Timestamp,
    ) -> List[Fill]:
        """
        Execute multiple orders.
        
        Note: In reality, simultaneous orders may have cross-impact.
        This is a simplified version.
        """
        fills = []
        for order in orders:
            # Get market data for this symbol
            symbol_data = market_data[market_data['symbol'] == order.symbol]
            if len(symbol_data) > 0:
                fill = self.execute_order(order, symbol_data.iloc[0], timestamp)
                fills.append(fill)
        
        return fills


class VWAPExecutor:
    """
    VWAP (Volume Weighted Average Price) execution.
    
    Splits large orders into smaller slices executed over time
    to minimize market impact.
    """
    
    def __init__(
        self,
        num_slices: int = 10,
        participation_rate: float = 0.1,  # Max 10% of volume
    ):
        self.num_slices = num_slices
        self.participation_rate = participation_rate
    
    def create_slices(self, order: Order, market_data: pd.DataFrame) -> List[Order]:
        """
        Split order into VWAP slices.
        
        Args:
            order: Original order
            market_data: Historical intraday volume profile
            
        Returns:
            List of slice orders
        """
        quantity_per_slice = order.quantity / self.num_slices
        
        slices = []
        for i in range(self.num_slices):
            slice_order = Order(
                symbol=order.symbol,
                side=order.side,
                quantity=quantity_per_slice,
                order_type=OrderType.MARKET,
            )
            slices.append(slice_order)
        
        return slices


class ExecutionQualityAnalyzer:
    """
    Analyzes execution quality.
    
    Metrics:
    - Implementation shortfall
    - VWAP slippage
    - Market impact estimate
    """
    
    def analyze_fills(self, fills: List[Fill]) -> Dict:
        """
        Analyze a list of fills.
        
        Returns:
            Dictionary with quality metrics
        """
        if not fills:
            return {}
        
        total_slippage_bps = sum(f.slippage for f in fills) * 10000
        avg_slippage_bps = total_slippage_bps / len(fills)
        
        total_commission = sum(f.commission for f in fills)
        total_volume = sum(abs(f.quantity) * f.fill_price for f in fills)
        
        return {
            'num_trades': len(fills),
            'avg_slippage_bps': avg_slippage_bps,
            'total_commission': total_commission,
            'commission_bps': (total_commission / total_volume) * 10000,
            'total_market_impact_bps': sum(f.market_impact for f in fills) * 10000,
        }
