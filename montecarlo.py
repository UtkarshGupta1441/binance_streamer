"""
Monte Carlo Backtesting Module
==============================
Provides robust strategy testing through multiple randomized simulations.

This module allows you to:
- Run thousands of backtests with varying market conditions
- Calculate confidence intervals for strategy performance
- Assess strategy robustness across different scenarios
- Generate statistical reports on strategy viability
"""

import random
import math
import time
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy


@dataclass
class BacktestResult:
    """Results from a single backtest run."""
    run_id: int
    total_return: float  # Percentage return
    total_pnl: float  # Dollar PnL
    sharpe_ratio: float
    max_drawdown: float  # Maximum peak-to-trough decline
    win_rate: float  # Percentage of winning trades
    total_trades: int
    profit_factor: float  # Gross profit / gross loss
    avg_trade_pnl: float
    final_balance: float
    price_path: List[float] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    strategy_signals: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class MonteCarloResults:
    """Aggregated results from Monte Carlo simulation."""
    num_simulations: int
    strategy_name: str
    
    # Return statistics
    mean_return: float
    median_return: float
    std_return: float
    min_return: float
    max_return: float
    
    # Confidence intervals (95%)
    return_ci_lower: float
    return_ci_upper: float
    
    # Risk metrics
    mean_sharpe: float
    mean_max_drawdown: float
    worst_drawdown: float
    
    # Win rate statistics
    mean_win_rate: float
    
    # Probability metrics
    prob_profit: float  # Probability of positive return
    prob_loss_over_10pct: float  # Probability of losing > 10%
    prob_gain_over_10pct: float  # Probability of gaining > 10%
    
    # Value at Risk (VaR)
    var_95: float  # 95% VaR - worst expected loss at 95% confidence
    var_99: float  # 99% VaR
    
    # Expected Shortfall (CVaR)
    cvar_95: float  # Average loss in worst 5% of cases
    
    # Individual run results (for plotting)
    all_results: List[BacktestResult] = field(default_factory=list)
    
    # Distribution data
    return_percentiles: Dict[int, float] = field(default_factory=dict)


class PricePathGenerator:
    """Generates randomized price paths for Monte Carlo simulation."""
    
    def __init__(
        self,
        initial_price: float = 100000.0,
        num_periods: int = 500,
        base_volatility: float = 0.02,
        seed: Optional[int] = None
    ):
        self.initial_price = initial_price
        self.num_periods = num_periods
        self.base_volatility = base_volatility
        self.seed = seed
        
    def generate_gbm_path(
        self,
        drift: float = 0.0,
        volatility: Optional[float] = None
    ) -> List[float]:
        """
        Generate price path using Geometric Brownian Motion.
        
        dS = μS dt + σS dW
        
        Args:
            drift: Expected return per period (μ)
            volatility: Price volatility per period (σ)
        """
        vol = volatility or self.base_volatility
        prices = [self.initial_price]
        price = self.initial_price
        
        for _ in range(self.num_periods - 1):
            # GBM: S(t+1) = S(t) * exp((μ - σ²/2)dt + σ√dt * Z)
            dt = 1.0
            z = random.gauss(0, 1)
            log_return = (drift - 0.5 * vol**2) * dt + vol * math.sqrt(dt) * z
            price *= math.exp(log_return)
            prices.append(price)
            
        return prices
    
    def generate_mean_reverting_path(
        self,
        mean_price: Optional[float] = None,
        reversion_speed: float = 0.1,
        volatility: Optional[float] = None
    ) -> List[float]:
        """
        Generate price path using Ornstein-Uhlenbeck process.
        
        dS = θ(μ - S)dt + σdW
        
        Args:
            mean_price: Long-term mean price (μ)
            reversion_speed: Speed of reversion to mean (θ)
            volatility: Price volatility (σ)
        """
        mean = mean_price or self.initial_price
        vol = volatility or self.base_volatility
        prices = [self.initial_price]
        price = self.initial_price
        
        for _ in range(self.num_periods - 1):
            dw = random.gauss(0, 1)
            dp = reversion_speed * (mean - price) + vol * price * dw
            price += dp
            price = max(price, mean * 0.1)  # Floor at 10% of mean
            prices.append(price)
            
        return prices
    
    def generate_regime_switching_path(
        self,
        regime_probs: Dict[str, float] = None,
        regime_params: Dict[str, Dict] = None
    ) -> Tuple[List[float], List[str]]:
        """
        Generate price path with regime switching (Bull/Bear/Sideways).
        
        Returns:
            Tuple of (prices, regimes)
        """
        if regime_probs is None:
            regime_probs = {'bull': 0.35, 'bear': 0.25, 'sideways': 0.40}
        
        if regime_params is None:
            regime_params = {
                'bull': {'drift': 0.001, 'volatility': 0.015},
                'bear': {'drift': -0.001, 'volatility': 0.025},
                'sideways': {'drift': 0.0, 'volatility': 0.01}
            }
        
        regimes = list(regime_probs.keys())
        weights = list(regime_probs.values())
        
        prices = [self.initial_price]
        regime_history = []
        price = self.initial_price
        
        current_regime = random.choices(regimes, weights=weights)[0]
        regime_duration = 0
        regime_length = random.randint(30, 100)
        
        for _ in range(self.num_periods - 1):
            regime_duration += 1
            if regime_duration >= regime_length:
                current_regime = random.choices(regimes, weights=weights)[0]
                regime_duration = 0
                regime_length = random.randint(30, 100)
            
            params = regime_params[current_regime]
            z = random.gauss(0, 1)
            log_return = params['drift'] + params['volatility'] * z
            price *= math.exp(log_return)
            
            prices.append(price)
            regime_history.append(current_regime)
            
        return prices, regime_history
    
    def generate_jump_diffusion_path(
        self,
        drift: float = 0.0,
        volatility: Optional[float] = None,
        jump_intensity: float = 0.1,  # Probability of jump per period
        jump_mean: float = 0.0,
        jump_std: float = 0.05
    ) -> List[float]:
        """
        Generate price path with jump diffusion (Merton model).
        
        Includes random jumps to simulate flash crashes or rallies.
        """
        vol = volatility or self.base_volatility
        prices = [self.initial_price]
        price = self.initial_price
        
        for _ in range(self.num_periods - 1):
            # Normal diffusion
            z = random.gauss(0, 1)
            diffusion = (drift - 0.5 * vol**2) + vol * z
            
            # Jump component
            jump = 0
            if random.random() < jump_intensity:
                jump = random.gauss(jump_mean, jump_std)
            
            price *= math.exp(diffusion + jump)
            prices.append(price)
            
        return prices


class StrategyEvaluator:
    """Evaluates trading strategies on price paths."""
    
    def __init__(self, strategy_manager=None):
        """
        Args:
            strategy_manager: Rust StrategyManager instance (optional)
        """
        self.strategy_manager = strategy_manager
        self._python_strategies = {}
        
    def add_python_strategy(self, name: str, strategy_func: Callable):
        """
        Add a Python-based strategy for evaluation.
        
        Args:
            name: Strategy name
            strategy_func: Function(prices, idx) -> 'BUY' | 'SELL' | 'HOLD'
        """
        self._python_strategies[name] = strategy_func
    
    def evaluate_on_path(
        self,
        prices: List[float],
        initial_balance: float = 10000.0,
        position_size: float = 0.1,  # 10% of portfolio per trade
        use_rust_strategies: bool = True
    ) -> Dict[str, BacktestResult]:
        """
        Evaluate all strategies on a price path.
        
        Returns:
            Dict mapping strategy name to BacktestResult
        """
        results = {}
        
        # Evaluate Rust strategies if available
        if use_rust_strategies:
            rust_results = self._evaluate_rust_strategies(
                prices, initial_balance, position_size
            )
            results.update(rust_results)
        
        # Evaluate Python strategies
        for name, strategy_func in self._python_strategies.items():
            result = self._evaluate_python_strategy(
                name, strategy_func, prices, initial_balance, position_size
            )
            results[name] = result
            
        return results
    
    def _evaluate_rust_strategies(
        self,
        prices: List[float],
        initial_balance: float,
        position_size: float
    ) -> Dict[str, BacktestResult]:
        """Evaluate Rust-based strategies."""
        from binance_streamer import StrategyManager
        
        # Create fresh strategy manager for clean state
        sm = StrategyManager()
        
        strategy_states = {}
        strategy_equities = {}
        strategy_signals = {}
        strategy_trades = {}
        
        for price in prices:
            results = sm.update(price)
            
            for result in results:
                name = result.name
                signal = result.signal
                
                if name not in strategy_states:
                    strategy_states[name] = {
                        'balance': initial_balance,
                        'position': 0.0,
                        'entry_price': 0.0,
                        'trades': [],
                        'pnl_history': []
                    }
                    strategy_equities[name] = [initial_balance]
                    strategy_signals[name] = []
                    strategy_trades[name] = []
                
                state = strategy_states[name]
                strategy_signals[name].append(signal)
                
                # Execute signals
                trade_qty = (state['balance'] * position_size) / price
                
                if signal == 'BUY' and state['position'] <= 0:
                    # Close short if any, then go long
                    if state['position'] < 0:
                        close_pnl = (state['entry_price'] - price) * abs(state['position'])
                        state['balance'] += close_pnl
                        state['pnl_history'].append(close_pnl)
                        strategy_trades[name].append({'type': 'close_short', 'pnl': close_pnl})
                    
                    # Open long
                    state['position'] = trade_qty
                    state['entry_price'] = price
                    strategy_trades[name].append({'type': 'open_long', 'price': price, 'qty': trade_qty})
                    
                elif signal == 'SELL' and state['position'] >= 0:
                    # Close long if any, then go short
                    if state['position'] > 0:
                        close_pnl = (price - state['entry_price']) * state['position']
                        state['balance'] += close_pnl
                        state['pnl_history'].append(close_pnl)
                        strategy_trades[name].append({'type': 'close_long', 'pnl': close_pnl})
                    
                    # Open short
                    state['position'] = -trade_qty
                    state['entry_price'] = price
                    strategy_trades[name].append({'type': 'open_short', 'price': price, 'qty': trade_qty})
                
                # Calculate equity
                unrealized = 0
                if state['position'] > 0:
                    unrealized = (price - state['entry_price']) * state['position']
                elif state['position'] < 0:
                    unrealized = (state['entry_price'] - price) * abs(state['position'])
                
                equity = state['balance'] + unrealized
                strategy_equities[name].append(equity)
        
        # Build results
        results = {}
        for name, state in strategy_states.items():
            equity_curve = strategy_equities[name]
            pnl_history = state['pnl_history']
            
            # Calculate metrics
            final_balance = equity_curve[-1]
            total_return = (final_balance - initial_balance) / initial_balance * 100
            total_pnl = final_balance - initial_balance
            
            # Calculate max drawdown
            max_drawdown = self._calculate_max_drawdown(equity_curve)
            
            # Calculate Sharpe ratio (assuming risk-free rate = 0)
            returns = [(equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1] 
                      for i in range(1, len(equity_curve))]
            sharpe = self._calculate_sharpe(returns) if returns else 0
            
            # Win rate and profit factor
            wins = [p for p in pnl_history if p > 0]
            losses = [p for p in pnl_history if p < 0]
            win_rate = len(wins) / len(pnl_history) * 100 if pnl_history else 0
            
            gross_profit = sum(wins) if wins else 0
            gross_loss = abs(sum(losses)) if losses else 1
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            avg_trade_pnl = sum(pnl_history) / len(pnl_history) if pnl_history else 0
            
            results[name] = BacktestResult(
                run_id=0,
                total_return=total_return,
                total_pnl=total_pnl,
                sharpe_ratio=sharpe,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                total_trades=len(pnl_history),
                profit_factor=profit_factor,
                avg_trade_pnl=avg_trade_pnl,
                final_balance=final_balance,
                price_path=prices,
                equity_curve=equity_curve,
                strategy_signals={name: strategy_signals[name]}
            )
        
        return results
    
    def _evaluate_python_strategy(
        self,
        name: str,
        strategy_func: Callable,
        prices: List[float],
        initial_balance: float,
        position_size: float
    ) -> BacktestResult:
        """Evaluate a Python-based strategy."""
        balance = initial_balance
        position = 0.0
        entry_price = 0.0
        equity_curve = [initial_balance]
        pnl_history = []
        signals = []
        
        for i, price in enumerate(prices):
            signal = strategy_func(prices[:i+1], i)
            signals.append(signal)
            
            trade_qty = (balance * position_size) / price
            
            if signal == 'BUY' and position <= 0:
                if position < 0:
                    close_pnl = (entry_price - price) * abs(position)
                    balance += close_pnl
                    pnl_history.append(close_pnl)
                position = trade_qty
                entry_price = price
                
            elif signal == 'SELL' and position >= 0:
                if position > 0:
                    close_pnl = (price - entry_price) * position
                    balance += close_pnl
                    pnl_history.append(close_pnl)
                position = -trade_qty
                entry_price = price
            
            unrealized = 0
            if position > 0:
                unrealized = (price - entry_price) * position
            elif position < 0:
                unrealized = (entry_price - price) * abs(position)
            
            equity_curve.append(balance + unrealized)
        
        final_balance = equity_curve[-1]
        total_return = (final_balance - initial_balance) / initial_balance * 100
        
        returns = [(equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1] 
                  for i in range(1, len(equity_curve))]
        
        wins = [p for p in pnl_history if p > 0]
        losses = [p for p in pnl_history if p < 0]
        
        return BacktestResult(
            run_id=0,
            total_return=total_return,
            total_pnl=final_balance - initial_balance,
            sharpe_ratio=self._calculate_sharpe(returns),
            max_drawdown=self._calculate_max_drawdown(equity_curve),
            win_rate=len(wins) / len(pnl_history) * 100 if pnl_history else 0,
            total_trades=len(pnl_history),
            profit_factor=sum(wins) / abs(sum(losses)) if losses else 0,
            avg_trade_pnl=sum(pnl_history) / len(pnl_history) if pnl_history else 0,
            final_balance=final_balance,
            price_path=prices,
            equity_curve=equity_curve,
            strategy_signals={name: signals}
        )
    
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown percentage."""
        if not equity_curve:
            return 0.0
        
        peak = equity_curve[0]
        max_dd = 0.0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _calculate_sharpe(self, returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate annualized Sharpe ratio."""
        if not returns or len(returns) < 2:
            return 0.0
        
        excess_returns = [r - risk_free_rate for r in returns]
        mean_return = statistics.mean(excess_returns)
        std_return = statistics.stdev(excess_returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualize (assuming 252 trading days)
        sharpe = (mean_return / std_return) * math.sqrt(252)
        return sharpe


class MonteCarloBacktester:
    """
    Monte Carlo backtester for robust strategy evaluation.
    
    Runs multiple simulations with randomized market conditions
    to assess strategy performance distribution.
    """
    
    def __init__(
        self,
        num_simulations: int = 1000,
        num_periods: int = 500,
        initial_price: float = 100000.0,
        initial_balance: float = 10000.0,
        position_size: float = 0.1,
        seed: Optional[int] = None
    ):
        self.num_simulations = num_simulations
        self.num_periods = num_periods
        self.initial_price = initial_price
        self.initial_balance = initial_balance
        self.position_size = position_size
        self.seed = seed
        
        if seed is not None:
            random.seed(seed)
        
        self.path_generator = PricePathGenerator(
            initial_price=initial_price,
            num_periods=num_periods
        )
        self.evaluator = StrategyEvaluator()
        
    def run(
        self,
        path_type: str = 'regime_switching',
        volatility_range: Tuple[float, float] = (0.01, 0.04),
        drift_range: Tuple[float, float] = (-0.001, 0.001),
        use_rust_strategies: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, MonteCarloResults]:
        """
        Run Monte Carlo simulation.
        
        Args:
            path_type: 'gbm', 'mean_reverting', 'regime_switching', 'jump_diffusion'
            volatility_range: (min, max) volatility for randomization
            drift_range: (min, max) drift for randomization
            use_rust_strategies: Whether to use Rust-based strategies
            progress_callback: Optional callback(current, total) for progress updates
            
        Returns:
            Dict mapping strategy name to MonteCarloResults
        """
        all_strategy_results: Dict[str, List[BacktestResult]] = {}
        
        for i in range(self.num_simulations):
            # Randomize parameters
            vol = random.uniform(*volatility_range)
            drift = random.uniform(*drift_range)
            
            # Generate price path
            if path_type == 'gbm':
                prices = self.path_generator.generate_gbm_path(drift=drift, volatility=vol)
            elif path_type == 'mean_reverting':
                prices = self.path_generator.generate_mean_reverting_path(volatility=vol)
            elif path_type == 'regime_switching':
                prices, _ = self.path_generator.generate_regime_switching_path()
            elif path_type == 'jump_diffusion':
                prices = self.path_generator.generate_jump_diffusion_path(
                    drift=drift, volatility=vol
                )
            else:
                prices = self.path_generator.generate_gbm_path(drift=drift, volatility=vol)
            
            # Evaluate strategies
            results = self.evaluator.evaluate_on_path(
                prices,
                initial_balance=self.initial_balance,
                position_size=self.position_size,
                use_rust_strategies=use_rust_strategies
            )
            
            # Collect results by strategy
            for strategy_name, result in results.items():
                result.run_id = i
                if strategy_name not in all_strategy_results:
                    all_strategy_results[strategy_name] = []
                all_strategy_results[strategy_name].append(result)
            
            # Progress callback
            if progress_callback:
                progress_callback(i + 1, self.num_simulations)
        
        # Aggregate results
        aggregated = {}
        for strategy_name, results in all_strategy_results.items():
            aggregated[strategy_name] = self._aggregate_results(strategy_name, results)
        
        return aggregated
    
    def _aggregate_results(
        self,
        strategy_name: str,
        results: List[BacktestResult]
    ) -> MonteCarloResults:
        """Aggregate individual backtest results into Monte Carlo statistics."""
        returns = [r.total_return for r in results]
        sharpes = [r.sharpe_ratio for r in results]
        drawdowns = [r.max_drawdown for r in results]
        win_rates = [r.win_rate for r in results]
        
        # Sort returns for percentile calculations
        sorted_returns = sorted(returns)
        n = len(sorted_returns)
        
        # Calculate confidence interval (95%)
        ci_lower_idx = int(0.025 * n)
        ci_upper_idx = int(0.975 * n)
        
        # Calculate VaR (Value at Risk)
        var_95_idx = int(0.05 * n)
        var_99_idx = int(0.01 * n)
        
        # Calculate CVaR (Expected Shortfall)
        cvar_95_values = sorted_returns[:var_95_idx + 1]
        cvar_95 = statistics.mean(cvar_95_values) if cvar_95_values else 0
        
        # Probability calculations
        prob_profit = len([r for r in returns if r > 0]) / n * 100
        prob_loss_10 = len([r for r in returns if r < -10]) / n * 100
        prob_gain_10 = len([r for r in returns if r > 10]) / n * 100
        
        # Percentiles
        percentiles = {}
        for p in [5, 10, 25, 50, 75, 90, 95]:
            idx = int(p / 100 * n)
            percentiles[p] = sorted_returns[min(idx, n-1)]
        
        return MonteCarloResults(
            num_simulations=n,
            strategy_name=strategy_name,
            mean_return=statistics.mean(returns),
            median_return=statistics.median(returns),
            std_return=statistics.stdev(returns) if n > 1 else 0,
            min_return=min(returns),
            max_return=max(returns),
            return_ci_lower=sorted_returns[ci_lower_idx],
            return_ci_upper=sorted_returns[ci_upper_idx],
            mean_sharpe=statistics.mean(sharpes),
            mean_max_drawdown=statistics.mean(drawdowns),
            worst_drawdown=max(drawdowns),
            mean_win_rate=statistics.mean(win_rates),
            prob_profit=prob_profit,
            prob_loss_over_10pct=prob_loss_10,
            prob_gain_over_10pct=prob_gain_10,
            var_95=sorted_returns[var_95_idx],
            var_99=sorted_returns[var_99_idx],
            cvar_95=cvar_95,
            all_results=results,
            return_percentiles=percentiles
        )
    
    def generate_report(self, results: Dict[str, MonteCarloResults]) -> str:
        """Generate a text report of Monte Carlo results."""
        lines = []
        lines.append("=" * 80)
        lines.append("MONTE CARLO BACKTEST REPORT")
        lines.append(f"Simulations: {self.num_simulations} | Periods: {self.num_periods}")
        lines.append("=" * 80)
        
        for name, mc in results.items():
            lines.append(f"\n{'─' * 40}")
            lines.append(f"Strategy: {name}")
            lines.append(f"{'─' * 40}")
            
            lines.append(f"\n📊 RETURN STATISTICS")
            lines.append(f"  Mean Return:      {mc.mean_return:+.2f}%")
            lines.append(f"  Median Return:    {mc.median_return:+.2f}%")
            lines.append(f"  Std Deviation:    {mc.std_return:.2f}%")
            lines.append(f"  Min Return:       {mc.min_return:+.2f}%")
            lines.append(f"  Max Return:       {mc.max_return:+.2f}%")
            lines.append(f"  95% CI:           [{mc.return_ci_lower:+.2f}%, {mc.return_ci_upper:+.2f}%]")
            
            lines.append(f"\n📈 RISK METRICS")
            lines.append(f"  Mean Sharpe:      {mc.mean_sharpe:.3f}")
            lines.append(f"  Mean Max DD:      {mc.mean_max_drawdown:.2f}%")
            lines.append(f"  Worst Drawdown:   {mc.worst_drawdown:.2f}%")
            lines.append(f"  VaR (95%):        {mc.var_95:+.2f}%")
            lines.append(f"  VaR (99%):        {mc.var_99:+.2f}%")
            lines.append(f"  CVaR (95%):       {mc.cvar_95:+.2f}%")
            
            lines.append(f"\n🎯 PROBABILITIES")
            lines.append(f"  P(Profit):        {mc.prob_profit:.1f}%")
            lines.append(f"  P(Gain > 10%):    {mc.prob_gain_over_10pct:.1f}%")
            lines.append(f"  P(Loss > 10%):    {mc.prob_loss_over_10pct:.1f}%")
            lines.append(f"  Mean Win Rate:    {mc.mean_win_rate:.1f}%")
            
            lines.append(f"\n📉 RETURN PERCENTILES")
            for p, val in sorted(mc.return_percentiles.items()):
                lines.append(f"  {p}th percentile: {val:+.2f}%")
        
        lines.append("\n" + "=" * 80)
        
        return "\n".join(lines)


# Convenience function for quick backtesting
def run_monte_carlo(
    num_simulations: int = 100,
    num_periods: int = 500,
    path_type: str = 'regime_switching',
    initial_balance: float = 10000.0,
    verbose: bool = True
) -> Dict[str, MonteCarloResults]:
    """
    Quick Monte Carlo backtest with default settings.
    
    Args:
        num_simulations: Number of simulations to run
        num_periods: Number of price periods per simulation
        path_type: Type of price path generation
        initial_balance: Starting capital
        verbose: Whether to print progress
        
    Returns:
        Dict of strategy name to MonteCarloResults
    """
    backtester = MonteCarloBacktester(
        num_simulations=num_simulations,
        num_periods=num_periods,
        initial_balance=initial_balance
    )
    
    def progress(current, total):
        if verbose and current % 10 == 0:
            print(f"Progress: {current}/{total} simulations ({current/total*100:.1f}%)")
    
    results = backtester.run(
        path_type=path_type,
        progress_callback=progress if verbose else None
    )
    
    if verbose:
        print(backtester.generate_report(results))
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Running Monte Carlo Backtest...")
    results = run_monte_carlo(num_simulations=100, verbose=True)
