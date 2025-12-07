"""
Simulated Market Data Generator for Testing Trading Strategies

This module generates realistic price data that mimics real market behavior
including trends, mean reversion, volatility clusters, and random noise.
"""

import random
import math
import time
from dataclasses import dataclass
from typing import List, Tuple, Generator
from collections import deque


@dataclass
class SimulatedTick:
    """A single simulated price tick."""
    timestamp: float
    price: float
    volume: float
    side: str  # 'BUY' or 'SELL'


@dataclass 
class SimulatedOrderBook:
    """Simulated order book snapshot."""
    bids: List[Tuple[float, float]]  # (price, quantity)
    asks: List[Tuple[float, float]]
    mid_price: float
    spread: float


class MarketSimulator:
    """
    Generates realistic market data for strategy testing.
    
    Features:
    - Geometric Brownian Motion for price evolution
    - Mean reversion component
    - Volatility clustering (GARCH-like)
    - Realistic bid/ask spread
    - Volume patterns
    """
    
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        initial_price: float = 100000.0,
        volatility: float = 0.02,  # 2% daily volatility
        mean_reversion_strength: float = 0.1,
        trend_strength: float = 0.0,  # Positive = uptrend, negative = downtrend
        tick_interval: float = 0.1,  # seconds between ticks
        spread_bps: float = 5.0,  # spread in basis points
    ):
        self.symbol = symbol
        self.price = initial_price
        self.initial_price = initial_price
        self.volatility = volatility
        self.mean_reversion_strength = mean_reversion_strength
        self.trend_strength = trend_strength
        self.tick_interval = tick_interval
        self.spread_bps = spread_bps
        
        # State variables
        self.current_volatility = volatility
        self.tick_count = 0
        self.price_history = deque(maxlen=1000)
        self.trade_history = deque(maxlen=100)
        
        # Market regime (changes over time)
        self.regime = "normal"  # normal, trending_up, trending_down, volatile
        self.regime_duration = 0
        self.regime_length = random.randint(50, 200)
        
    def _update_regime(self):
        """Randomly change market regime to create varied conditions."""
        self.regime_duration += 1
        if self.regime_duration >= self.regime_length:
            self.regime_duration = 0
            self.regime_length = random.randint(50, 200)
            
            regimes = ["normal", "trending_up", "trending_down", "volatile", "mean_reverting"]
            weights = [0.3, 0.2, 0.2, 0.15, 0.15]
            self.regime = random.choices(regimes, weights=weights)[0]
    
    def _get_regime_params(self) -> Tuple[float, float, float]:
        """Get volatility, trend, and mean reversion based on current regime."""
        if self.regime == "normal":
            return self.volatility, 0.0, self.mean_reversion_strength
        elif self.regime == "trending_up":
            return self.volatility * 0.8, 0.001, self.mean_reversion_strength * 0.5
        elif self.regime == "trending_down":
            return self.volatility * 0.8, -0.001, self.mean_reversion_strength * 0.5
        elif self.regime == "volatile":
            return self.volatility * 2.0, 0.0, self.mean_reversion_strength * 0.3
        elif self.regime == "mean_reverting":
            return self.volatility * 0.6, 0.0, self.mean_reversion_strength * 2.0
        return self.volatility, 0.0, self.mean_reversion_strength
    
    def _generate_price_change(self) -> float:
        """Generate next price using GBM + mean reversion + regime effects."""
        self._update_regime()
        vol, trend, mr_strength = self._get_regime_params()
        
        # Random component (Geometric Brownian Motion)
        random_shock = random.gauss(0, 1) * vol * math.sqrt(self.tick_interval)
        
        # Trend component
        drift = trend * self.tick_interval
        
        # Mean reversion component (pull towards initial price)
        log_price_ratio = math.log(self.price / self.initial_price)
        mean_reversion = -mr_strength * log_price_ratio * self.tick_interval
        
        # Combine all components
        price_change_pct = drift + mean_reversion + random_shock
        
        return price_change_pct
    
    def next_tick(self) -> SimulatedTick:
        """Generate the next price tick."""
        self.tick_count += 1
        
        # Update price
        price_change = self._generate_price_change()
        self.price *= (1 + price_change)
        
        # Ensure price stays positive
        self.price = max(self.price, self.initial_price * 0.1)
        
        # Generate volume (higher during volatile periods)
        base_volume = random.uniform(0.01, 0.5)
        if self.regime == "volatile":
            base_volume *= 2
        
        # Determine trade side (slightly biased based on price movement)
        side = "BUY" if price_change > 0 else "SELL"
        if random.random() < 0.3:  # 30% random flip
            side = "SELL" if side == "BUY" else "BUY"
        
        tick = SimulatedTick(
            timestamp=time.time(),
            price=round(self.price, 2),
            volume=round(base_volume, 6),
            side=side
        )
        
        self.price_history.append(self.price)
        self.trade_history.append(tick)
        
        return tick
    
    def get_order_book(self, depth: int = 20) -> SimulatedOrderBook:
        """Generate a simulated order book around current price."""
        spread = self.price * (self.spread_bps / 10000)
        
        # Add some randomness to spread during volatile periods
        if self.regime == "volatile":
            spread *= random.uniform(1.0, 2.0)
        
        mid_price = self.price
        best_bid = mid_price - spread / 2
        best_ask = mid_price + spread / 2
        
        bids = []
        asks = []
        
        # Generate bid levels (descending price)
        for i in range(depth):
            price_offset = (i + 1) * spread * 0.5 * random.uniform(0.8, 1.2)
            price = round(best_bid - price_offset, 2)
            # Volume tends to be higher further from mid-price
            quantity = round(random.uniform(0.1, 2.0) * (1 + i * 0.1), 6)
            bids.append((price, quantity))
        
        # Generate ask levels (ascending price)
        for i in range(depth):
            price_offset = (i + 1) * spread * 0.5 * random.uniform(0.8, 1.2)
            price = round(best_ask + price_offset, 2)
            quantity = round(random.uniform(0.1, 2.0) * (1 + i * 0.1), 6)
            asks.append((price, quantity))
        
        return SimulatedOrderBook(
            bids=bids,
            asks=asks,
            mid_price=round(mid_price, 2),
            spread=round(spread, 2)
        )
    
    def get_recent_trades(self, count: int = 20) -> List[dict]:
        """Get recent trades in Binance-like format."""
        trades = list(self.trade_history)[-count:]
        return [
            {
                'p': str(t.price),
                'q': str(t.volume),
                'm': t.side == 'SELL',  # Binance format: True = seller is maker
                'E': int(t.timestamp * 1000)
            }
            for t in trades
        ]
    
    def generate_historical_data(self, periods: int = 500) -> List[float]:
        """Generate historical price data for backtesting."""
        prices = [self.initial_price]
        temp_price = self.initial_price
        
        for _ in range(periods - 1):
            # Simple random walk with mean reversion
            change = random.gauss(0, self.volatility * temp_price)
            mr = -0.01 * (temp_price - self.initial_price)
            temp_price += change + mr
            temp_price = max(temp_price, self.initial_price * 0.5)
            prices.append(round(temp_price, 2))
        
        return prices


class PaperTrader:
    """
    Simulated trading account for paper trading.
    Tracks positions, PnL, and trade history without real money.
    """
    
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.cash_balance = initial_balance
        self.positions = {}  # symbol -> quantity
        self.trade_history = []
        self.pnl_history = []
        
    def get_position(self, symbol: str) -> float:
        """Get current position for a symbol."""
        return self.positions.get(symbol, 0.0)
    
    def execute_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        strategy_name: str = "Manual"
    ) -> dict:
        """
        Execute a simulated trade.
        
        Returns:
            dict with trade details and result
        """
        side = side.upper()
        
        trade = {
            'timestamp': time.time(),
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'strategy': strategy_name,
            'status': 'FILLED',
            'message': ''
        }
        
        cost = quantity * price
        
        if side == 'BUY':
            if cost > self.cash_balance:
                trade['status'] = 'REJECTED'
                trade['message'] = f'Insufficient balance. Need ${cost:.2f}, have ${self.cash_balance:.2f}'
                return trade
            
            self.cash_balance -= cost
            self.positions[symbol] = self.positions.get(symbol, 0.0) + quantity
            trade['message'] = f'Bought {quantity} {symbol} @ ${price:.2f}'
            
        elif side == 'SELL':
            current_position = self.positions.get(symbol, 0.0)
            if quantity > current_position:
                # Allow short selling in simulation
                pass
            
            self.cash_balance += cost
            self.positions[symbol] = self.positions.get(symbol, 0.0) - quantity
            trade['message'] = f'Sold {quantity} {symbol} @ ${price:.2f}'
        
        self.trade_history.append(trade)
        return trade
    
    def get_portfolio_value(self, current_prices: dict) -> float:
        """Calculate total portfolio value."""
        total = self.cash_balance
        for symbol, qty in self.positions.items():
            if symbol in current_prices:
                total += qty * current_prices[symbol]
        return total
    
    def get_total_pnl(self, current_prices: dict) -> float:
        """Calculate total profit/loss."""
        return self.get_portfolio_value(current_prices) - self.initial_balance
    
    def get_trade_summary(self) -> dict:
        """Get summary of trading activity."""
        if not self.trade_history:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0
            }
        
        total = len(self.trade_history)
        buys = [t for t in self.trade_history if t['side'] == 'BUY']
        sells = [t for t in self.trade_history if t['side'] == 'SELL']
        
        return {
            'total_trades': total,
            'buy_trades': len(buys),
            'sell_trades': len(sells),
            'cash_balance': self.cash_balance,
            'positions': dict(self.positions)
        }
    
    def reset(self):
        """Reset the paper trading account."""
        self.cash_balance = self.initial_balance
        self.positions = {}
        self.trade_history = []
        self.pnl_history = []


# Pre-configured market scenarios for testing
MARKET_SCENARIOS = {
    'trending_bull': {
        'initial_price': 100000.0,
        'volatility': 0.015,
        'trend_strength': 0.0005,
        'mean_reversion_strength': 0.05,
        'description': 'Bullish trending market with moderate volatility'
    },
    'trending_bear': {
        'initial_price': 100000.0,
        'volatility': 0.02,
        'trend_strength': -0.0005,
        'mean_reversion_strength': 0.05,
        'description': 'Bearish trending market'
    },
    'sideways': {
        'initial_price': 100000.0,
        'volatility': 0.01,
        'trend_strength': 0.0,
        'mean_reversion_strength': 0.2,
        'description': 'Range-bound sideways market'
    },
    'volatile': {
        'initial_price': 100000.0,
        'volatility': 0.04,
        'trend_strength': 0.0,
        'mean_reversion_strength': 0.1,
        'description': 'High volatility choppy market'
    },
    'realistic': {
        'initial_price': 100000.0,
        'volatility': 0.02,
        'trend_strength': 0.0,
        'mean_reversion_strength': 0.1,
        'description': 'Realistic market with regime changes'
    }
}


def create_simulator(scenario: str = 'realistic', **kwargs) -> MarketSimulator:
    """Create a simulator with a predefined scenario."""
    if scenario in MARKET_SCENARIOS:
        params = MARKET_SCENARIOS[scenario].copy()
        params.pop('description', None)
        params.update(kwargs)
        return MarketSimulator(**params)
    return MarketSimulator(**kwargs)
