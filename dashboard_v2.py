"""
Trading Strategy Simulator v2
=============================
Fixed version with proper mode switching and no duplicate UI elements.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time

# ============================================================================
# LATENCY TRACKING UTILITIES
# ============================================================================
class LatencyTracker:
    """Tracks and prints operation latencies to the console."""
    
    def __init__(self):
        self.enabled = True
        self.tick_count = 0
        self.summary_interval = 10  # Print summary every N ticks
        self.latencies = {
            'simulator_tick': [],
            'strategy_update': [],
            'chart_render': [],
            'total_loop': []
        }
    
    def measure(self, operation_name):
        """Context manager to measure operation latency."""
        return LatencyContext(self, operation_name)
    
    def record(self, operation_name, latency_ms):
        """Record a latency measurement."""
        if operation_name in self.latencies:
            self.latencies[operation_name].append(latency_ms)
    
    def on_tick_complete(self):
        """Called when a tick completes. Prints summary every N ticks."""
        self.tick_count += 1
        
        # Only print summary every N ticks
        if self.tick_count % self.summary_interval == 0:
            self.print_summary()
            # Clear latencies after printing summary
            for key in self.latencies:
                self.latencies[key] = []
    
    def print_summary(self):
        """Print latency summary statistics for the last N ticks."""
        print("\n" + "="*70)
        print(f"📊 LATENCY SUMMARY (Ticks {self.tick_count - self.summary_interval + 1} - {self.tick_count})")
        print("="*70)
        
        for name, values in self.latencies.items():
            if values:
                avg = sum(values) / len(values)
                min_val = min(values)
                max_val = max(values)
                print(f"  {name:20s}: avg={avg:7.2f}ms | min={min_val:7.2f}ms | max={max_val:7.2f}ms")
            else:
                print(f"  {name:20s}: no data")
        
        print("="*70 + "\n")


class LatencyContext:
    """Context manager for measuring operation latency."""
    
    def __init__(self, tracker, operation_name):
        self.tracker = tracker
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        self.tracker.record(self.operation_name, elapsed_ms)
        return False


# ============================================================================
# PAGE CONFIG (Must be first)
# ============================================================================
st.set_page_config(
    page_title="Trading Strategy Simulator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# IMPORTS WITH LATENCY TRACKING (only on first run)
# ============================================================================
RUST_AVAILABLE = False
SIMULATOR_AVAILABLE = False

# Use a flag to track if we've already printed startup info
if 'startup_printed' not in st.session_state:
    st.session_state.startup_printed = True
    print("\n" + "="*70)
    print("🚀 TRADING STRATEGY SIMULATOR - Starting Up")
    print("="*70)
    
    # Measure Rust import latency
    rust_import_start = time.perf_counter()
    try:
        from binance_streamer import StrategyManager
        RUST_AVAILABLE = True
        rust_import_ms = (time.perf_counter() - rust_import_start) * 1000
        print(f"✅ Rust StrategyManager loaded in {rust_import_ms:.2f}ms")
    except ImportError as e:
        rust_import_ms = (time.perf_counter() - rust_import_start) * 1000
        print(f"❌ Rust import failed in {rust_import_ms:.2f}ms: {e}")
    
    # Measure Simulator import latency
    sim_import_start = time.perf_counter()
    try:
        from simulator import MarketSimulator, PaperTrader, MARKET_SCENARIOS, create_simulator
        SIMULATOR_AVAILABLE = True
        sim_import_ms = (time.perf_counter() - sim_import_start) * 1000
        print(f"✅ Simulator module loaded in {sim_import_ms:.2f}ms")
    except ImportError as e:
        sim_import_ms = (time.perf_counter() - sim_import_start) * 1000
        print(f"❌ Simulator import failed in {sim_import_ms:.2f}ms: {e}")
    
    print("="*70 + "\n")
else:
    # Imports without printing (subsequent runs)
    try:
        from binance_streamer import StrategyManager
        RUST_AVAILABLE = True
    except ImportError:
        pass
    
    try:
        from simulator import MarketSimulator, PaperTrader, MARKET_SCENARIOS, create_simulator
        SIMULATOR_AVAILABLE = True
    except ImportError:
        pass

# Monte Carlo import
MONTECARLO_AVAILABLE = False
try:
    from montecarlo import MonteCarloBacktester, run_monte_carlo
    MONTECARLO_AVAILABLE = True
except ImportError:
    pass

# Initialize latency tracker in session state (persists across reruns)
if 'latency_tracker' not in st.session_state:
    st.session_state.latency_tracker = LatencyTracker()

latency_tracker = st.session_state.latency_tracker

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.is_running = False
    st.session_state.simulator = None
    st.session_state.paper_trader = None
    st.session_state.strategy_manager = None
    st.session_state.price_history = []
    st.session_state.pnl_history = {'Trend_Follower': [], 'Mean_Reversion': [], 'Momentum_RSI': []}
    st.session_state.strategy_positions = {'Trend_Follower': 0, 'Mean_Reversion': 0, 'Momentum_RSI': 0}

# ============================================================================
# CSS
# ============================================================================
st.markdown("""
<style>
    div[data-testid="stMetric"] { background-color: #1e1e2e; padding: 10px; border-radius: 8px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 16px; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def reset_all():
    st.session_state.is_running = False
    st.session_state.simulator = None
    st.session_state.price_history = []
    st.session_state.pnl_history = {'Trend_Follower': [], 'Mean_Reversion': [], 'Momentum_RSI': []}
    st.session_state.strategy_positions = {'Trend_Follower': 0, 'Mean_Reversion': 0, 'Momentum_RSI': 0}
    st.session_state.paper_trader = None
    st.session_state.strategy_manager = None


# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.markdown("# 🎮 Control Panel")
st.sidebar.markdown("---")

# Mode Selection
st.sidebar.markdown("### 🔄 Mode")
app_mode = st.sidebar.radio(
    "Select Mode:",
    options=['live_sim', 'monte_carlo'],
    format_func=lambda x: {
        'live_sim': '📊 Live Simulation',
        'monte_carlo': '🎲 Monte Carlo Backtest'
    }.get(x, x),
    key='app_mode',
    help="Live Simulation: Watch strategies in real-time. Monte Carlo: Run thousands of backtests."
)

st.sidebar.markdown("---")

# Simulation Settings
st.sidebar.markdown("### 🎲 Simulation Settings")

if SIMULATOR_AVAILABLE:
    scenario = st.sidebar.selectbox(
        "Market Scenario:",
        options=['realistic', 'trending_bull', 'trending_bear', 'sideways', 'volatile'],
        format_func=lambda x: {
            'realistic': '🎯 Realistic (Random movements)',
            'trending_bull': '📈 Bull Market (Upward trend)',
            'trending_bear': '📉 Bear Market (Downward trend)',
            'sideways': '↔️ Sideways (Range-bound)',
            'volatile': '🌊 Volatile (High swings)'
        }.get(x, x),
        key='sim_scenario',
        help="Choose how the simulated market will behave"
    )
    
    speed = st.sidebar.slider(
        "Update Speed:", 1, 15, 5, 
        key='sim_speed',
        help="How fast prices update (higher = faster)"
    )
    
    initial_balance = st.sidebar.number_input(
        "Starting Balance ($):",
        min_value=1000.0,
        max_value=100000.0,
        value=10000.0,
        step=1000.0,
        help="Your initial paper trading balance"
    )
    
    st.sidebar.markdown("---")
    
    # Strategy Parameters Section
    st.sidebar.markdown("### ⚙️ Strategy Parameters")
    st.sidebar.caption("Adjust these to see how strategies perform differently")
    
    with st.sidebar.expander("📈 Trend Follower (EMA)", expanded=False):
        st.markdown("**EMA Crossover Strategy**")
        st.caption("Buys when short EMA crosses above long EMA")
        ema_short = st.slider("Short EMA Period:", 5, 30, 12, key="ema_short",
            help="Shorter period = more responsive to recent prices")
        ema_long = st.slider("Long EMA Period:", 20, 100, 26, key="ema_long",
            help="Longer period = smoother, less noise")
        if ema_short >= ema_long:
            st.warning("Short EMA should be less than Long EMA")
    
    with st.sidebar.expander("🔄 Mean Reversion (Bollinger)", expanded=False):
        st.markdown("**Bollinger Bands Strategy**")
        st.caption("Buys at lower band, sells at upper band")
        bb_period = st.slider("BB Period:", 10, 50, 20, key="bb_period",
            help="Period for calculating the moving average")
        bb_std = st.slider("BB Std Deviation:", 1.0, 3.0, 2.0, 0.1, key="bb_std",
            help="Higher = wider bands, fewer signals")
    
    with st.sidebar.expander("⚡ Momentum (RSI)", expanded=False):
        st.markdown("**RSI Strategy**")
        st.caption("Buys when oversold, sells when overbought")
        rsi_period = st.slider("RSI Period:", 7, 21, 14, key="rsi_period",
            help="Period for RSI calculation")
        rsi_oversold = st.slider("Oversold Level:", 20, 40, 30, key="rsi_oversold",
            help="Buy signal when RSI drops below this")
        rsi_overbought = st.slider("Overbought Level:", 60, 80, 70, key="rsi_overbought",
            help="Sell signal when RSI rises above this")
        if rsi_oversold >= rsi_overbought:
            st.warning("Oversold should be less than Overbought")
    
    st.sidebar.markdown("---")
    
    # Action Buttons
    st.sidebar.markdown("### 🚀 Actions")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("▶️ Start", type="primary", use_container_width=True, key='sim_start'):
            reset_all()
            st.session_state.simulator = create_simulator(scenario=scenario, tick_interval=1.0/speed)
            st.session_state.paper_trader = PaperTrader(initial_balance=initial_balance)
            if RUST_AVAILABLE:
                st.session_state.strategy_manager = StrategyManager()
            st.session_state.is_running = True
            st.rerun()
    with col2:
        if st.button("⏹️ Stop", use_container_width=True, key='sim_stop'):
            st.session_state.is_running = False
            st.rerun()
    
    if st.sidebar.button("🔄 Reset All Data", use_container_width=True, key='sim_reset'):
        reset_all()
        st.rerun()
else:
    st.sidebar.error("❌ Simulator not available! Please install dependencies.")

# Trade History in Sidebar
if st.session_state.paper_trader:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### � Your Trades")
    trades = st.session_state.paper_trader.trade_history
    if trades:
        for trade in reversed(trades[-5:]):
            emoji = "🟢" if trade['side'] == 'BUY' else "🔴"
            st.sidebar.caption(f"{emoji} {trade['side']} {trade['quantity']:.4f} @ ${trade['price']:,.2f}")
        st.sidebar.caption(f"Total: {len(trades)} trades")
    else:
        st.sidebar.caption("No trades yet - use BUY/SELL buttons!")

st.sidebar.markdown("---")
st.sidebar.caption("📊 Trading Strategy Simulator")
st.sidebar.caption("⚡ Powered by Rust + Python")


# ============================================================================
# MAIN CONTENT
# ============================================================================

# Check which mode we're in
if app_mode == 'monte_carlo':
    # =========================================================================
    # MONTE CARLO BACKTESTING MODE
    # =========================================================================
    st.markdown("# 🎲 Monte Carlo Backtesting")
    st.caption("Test strategy robustness across thousands of randomized market scenarios")
    
    if not MONTECARLO_AVAILABLE:
        st.error("❌ Monte Carlo module not available. Make sure `montecarlo.py` exists.")
    elif not RUST_AVAILABLE:
        st.error("❌ Rust strategies not available. Run `maturin develop` first.")
    else:
        # Monte Carlo Settings
        st.markdown("### ⚙️ Backtest Configuration")
        
        mc_col1, mc_col2, mc_col3 = st.columns(3)
        
        with mc_col1:
            num_simulations = st.number_input(
                "Number of Simulations:",
                min_value=10,
                max_value=5000,
                value=100,
                step=10,
                help="More simulations = more accurate statistics, but slower"
            )
        
        with mc_col2:
            num_periods = st.number_input(
                "Periods per Simulation:",
                min_value=100,
                max_value=2000,
                value=500,
                step=100,
                help="Number of price ticks per backtest"
            )
        
        with mc_col3:
            path_type = st.selectbox(
                "Price Path Model:",
                options=['regime_switching', 'gbm', 'mean_reverting', 'jump_diffusion'],
                format_func=lambda x: {
                    'regime_switching': '🔄 Regime Switching (Recommended)',
                    'gbm': '📈 Geometric Brownian Motion',
                    'mean_reverting': '🔁 Mean Reverting',
                    'jump_diffusion': '⚡ Jump Diffusion (Crashes/Rallies)'
                }.get(x, x),
                help="How price paths are generated"
            )
        
        mc_col4, mc_col5 = st.columns(2)
        
        with mc_col4:
            mc_initial_balance = st.number_input(
                "Initial Balance ($):",
                min_value=1000.0,
                max_value=100000.0,
                value=10000.0,
                step=1000.0,
                key='mc_balance'
            )
        
        with mc_col5:
            position_size = st.slider(
                "Position Size (% of portfolio):",
                min_value=5,
                max_value=50,
                value=10,
                help="How much of portfolio to use per trade"
            ) / 100
        
        st.markdown("---")
        
        # Run Button
        if st.button("🚀 Run Monte Carlo Backtest", type="primary", use_container_width=True):
            with st.spinner(f"Running {num_simulations} simulations..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(current, total):
                    progress_bar.progress(current / total)
                    status_text.text(f"Simulation {current}/{total} ({current/total*100:.0f}%)")
                
                # Run backtest
                backtester = MonteCarloBacktester(
                    num_simulations=num_simulations,
                    num_periods=num_periods,
                    initial_balance=mc_initial_balance,
                    position_size=position_size
                )
                
                mc_results = backtester.run(
                    path_type=path_type,
                    progress_callback=update_progress
                )
                
                # Store results in session state
                st.session_state.mc_results = mc_results
                st.session_state.mc_report = backtester.generate_report(mc_results)
                
                progress_bar.empty()
                status_text.empty()
        
        # Display Results
        if 'mc_results' in st.session_state and st.session_state.mc_results:
            mc_results = st.session_state.mc_results
            
            st.markdown("---")
            st.markdown("## 📊 Results")
            
            # Strategy comparison cards
            strat_cols = st.columns(len(mc_results))
            
            colors = {'Trend_Follower': '#2196F3', 'Mean_Reversion': '#4CAF50', 'Momentum_RSI': '#9C27B0'}
            
            best_strategy = max(mc_results.items(), key=lambda x: x[1].mean_return)
            
            for i, (name, mc) in enumerate(mc_results.items()):
                with strat_cols[i]:
                    is_best = name == best_strategy[0]
                    
                    if is_best:
                        st.success(f"🏆 **{name.replace('_', ' ')}**")
                    else:
                        st.info(f"**{name.replace('_', ' ')}**")
                    
                    st.metric(
                        "Mean Return",
                        f"{mc.mean_return:+.2f}%",
                        delta=f"±{mc.std_return:.2f}%"
                    )
                    st.metric(
                        "Probability of Profit",
                        f"{mc.prob_profit:.1f}%"
                    )
                    st.metric(
                        "Mean Sharpe Ratio",
                        f"{mc.mean_sharpe:.3f}"
                    )
                    st.metric(
                        "Worst Drawdown",
                        f"-{mc.worst_drawdown:.2f}%"
                    )
            
            st.markdown("---")
            
            # Detailed tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "📈 Return Distribution", 
                "📊 Risk Analysis",
                "📉 Equity Curves Sample",
                "📝 Full Report"
            ])
            
            with tab1:
                st.markdown("### Return Distribution by Strategy")
                
                # Histogram of returns
                fig = go.Figure()
                
                for name, mc in mc_results.items():
                    returns = [r.total_return for r in mc.all_results]
                    fig.add_trace(go.Histogram(
                        x=returns,
                        name=name.replace('_', ' '),
                        opacity=0.7,
                        marker_color=colors.get(name, '#888'),
                        nbinsx=30
                    ))
                
                fig.update_layout(
                    title="Distribution of Returns Across All Simulations",
                    xaxis_title="Return (%)",
                    yaxis_title="Frequency",
                    barmode='overlay',
                    height=400,
                    legend=dict(orientation='h', yanchor='bottom', y=1.02)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Percentile table
                st.markdown("### Return Percentiles")
                
                percentile_data = []
                for name, mc in mc_results.items():
                    row = {'Strategy': name.replace('_', ' ')}
                    for p, val in mc.return_percentiles.items():
                        row[f'{p}th'] = f"{val:+.2f}%"
                    percentile_data.append(row)
                
                st.dataframe(pd.DataFrame(percentile_data), use_container_width=True)
            
            with tab2:
                st.markdown("### Risk Metrics Comparison")
                
                # Risk metrics table
                risk_data = []
                for name, mc in mc_results.items():
                    risk_data.append({
                        'Strategy': name.replace('_', ' '),
                        'VaR (95%)': f"{mc.var_95:+.2f}%",
                        'VaR (99%)': f"{mc.var_99:+.2f}%",
                        'CVaR (95%)': f"{mc.cvar_95:+.2f}%",
                        'Mean Max DD': f"{mc.mean_max_drawdown:.2f}%",
                        'Worst DD': f"{mc.worst_drawdown:.2f}%",
                        'P(Loss>10%)': f"{mc.prob_loss_over_10pct:.1f}%"
                    })
                
                st.dataframe(pd.DataFrame(risk_data), use_container_width=True)
                
                st.markdown("""
                **📖 Glossary:**
                - **VaR (Value at Risk)**: Worst expected loss at given confidence level
                - **CVaR (Expected Shortfall)**: Average loss in worst cases
                - **Max DD (Drawdown)**: Largest peak-to-trough decline
                """)
                
                # Box plot of returns
                fig = go.Figure()
                
                for name, mc in mc_results.items():
                    returns = [r.total_return for r in mc.all_results]
                    fig.add_trace(go.Box(
                        y=returns,
                        name=name.replace('_', ' '),
                        marker_color=colors.get(name, '#888')
                    ))
                
                fig.update_layout(
                    title="Return Distribution (Box Plot)",
                    yaxis_title="Return (%)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.markdown("### Sample Equity Curves")
                st.caption("Showing 10 random simulation paths per strategy")
                
                for name, mc in mc_results.items():
                    st.markdown(f"#### {name.replace('_', ' ')}")
                    
                    fig = go.Figure()
                    
                    # Plot 10 random equity curves
                    sample_results = mc.all_results[:10] if len(mc.all_results) >= 10 else mc.all_results
                    
                    for j, result in enumerate(sample_results):
                        fig.add_trace(go.Scatter(
                            y=result.equity_curve,
                            mode='lines',
                            name=f'Run {result.run_id}',
                            line=dict(width=1),
                            opacity=0.6
                        ))
                    
                    fig.add_hline(
                        y=mc_initial_balance, 
                        line_dash="dash", 
                        line_color="white",
                        annotation_text="Initial Balance"
                    )
                    
                    fig.update_layout(
                        height=300,
                        showlegend=False,
                        yaxis_title="Portfolio Value ($)",
                        xaxis_title="Time (ticks)"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                st.markdown("### Full Statistical Report")
                st.code(st.session_state.mc_report, language='text')
                
                # Download button
                st.download_button(
                    label="📥 Download Report",
                    data=st.session_state.mc_report,
                    file_name="monte_carlo_report.txt",
                    mime="text/plain"
                )
        
        else:
            # No results yet - show explanation
            st.markdown("---")
            st.info("👆 Configure your backtest settings above and click **Run Monte Carlo Backtest** to start.")
            
            st.markdown("### 📚 What is Monte Carlo Backtesting?")
            
            mc_exp1, mc_exp2 = st.columns(2)
            
            with mc_exp1:
                st.markdown("""
                **Monte Carlo simulation** runs your trading strategies through 
                thousands of different randomized market scenarios to answer:
                
                - What's the **average expected return**?
                - What's the **probability of making money**?
                - What's the **worst-case scenario** (Value at Risk)?
                - Which strategy is most **robust** across conditions?
                """)
            
            with mc_exp2:
                st.markdown("""
                **Why is this useful?**
                
                A strategy might look great in one market condition but fail in others.
                Monte Carlo testing reveals:
                
                - ✅ Strategies that work across many scenarios
                - ❌ Strategies that only work in specific conditions
                - 📊 Statistical confidence in your results
                """)
            
            st.markdown("### 🔧 Price Path Models Explained")
            
            path_col1, path_col2 = st.columns(2)
            
            with path_col1:
                st.markdown("""
                **🔄 Regime Switching** (Recommended)
                - Randomly switches between bull, bear, and sideways markets
                - Most realistic for testing strategy adaptability
                
                **📈 Geometric Brownian Motion**
                - Classic random walk model
                - Constant volatility assumption
                """)
            
            with path_col2:
                st.markdown("""
                **🔁 Mean Reverting**
                - Price tends to return to a mean value
                - Tests mean reversion strategies
                
                **⚡ Jump Diffusion**
                - Includes random "jumps" (flash crashes/rallies)
                - Tests strategy resilience to shocks
                """)

else:
    # =========================================================================
    # LIVE SIMULATION MODE (Original)
    # =========================================================================
    st.markdown("# 📈 Trading Strategy Simulator")
    st.caption("Learn algorithmic trading by comparing strategies in real-time simulation")

    if st.session_state.is_running:
        st.success("🟢 **SIMULATION RUNNING** — Watch the strategies compete!")
    else:
        st.info("👈 **Getting Started:** Configure settings in the sidebar and click **Start** to begin")

    # ============================================================================
    # ACTIVE VIEW
    # ============================================================================
    if st.session_state.is_running:
        # Start tracking total loop time
        loop_start = time.perf_counter()
        
        current_price = 0
        mid_price = 0
        spread = 0
        bids_df = pd.DataFrame()
        asks_df = pd.DataFrame()
        regime = ""
        symbol = "BTCUSDT"
        
        if st.session_state.simulator:
            sim = st.session_state.simulator
            
            # Measure simulator tick latency
            with latency_tracker.measure('simulator_tick'):
                tick = sim.next_tick()
                ob = sim.get_order_book(20)
            
            current_price = tick.price
            mid_price = ob.mid_price
            spread = ob.spread
            bids_df = pd.DataFrame(ob.bids, columns=['price', 'quantity'])
            asks_df = pd.DataFrame(ob.asks, columns=['price', 'quantity'])
            regime = sim.regime.replace('_', ' ').title()
            symbol = sim.symbol
        
        if current_price > 0:
            st.session_state.price_history.append(current_price)
            if len(st.session_state.price_history) > 500:
                st.session_state.price_history = st.session_state.price_history[-500:]
        
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("💰 Price", f"${current_price:,.2f}")
        col2.metric("📊 Spread", f"${spread:,.4f}")
        col3.metric("📈 Regime", regime)
        col4.metric("⏱️ Ticks", len(st.session_state.price_history))
        
        if st.session_state.paper_trader:
            pt = st.session_state.paper_trader
            pnl = pt.get_total_pnl({symbol: current_price})
            bcol1, bcol2, bcol3 = st.columns(3)
            bcol1.metric("💵 Cash", f"${pt.cash_balance:,.2f}")
            bcol2.metric("📊 PnL", f"${pnl:,.2f}")
            bcol3.metric("🔢 Trades", pt.get_trade_summary().get('total_trades', 0))
        
        st.markdown("---")
        
        if RUST_AVAILABLE and st.session_state.strategy_manager and current_price > 0:
            st.markdown("### 🤖 Strategy Comparison")
            
            sm = st.session_state.strategy_manager
            
            # Measure strategy update latency (Rust computation)
            with latency_tracker.measure('strategy_update'):
                results = sm.update(current_price)
            
            for result in results:
                name = result.name
                signal = result.signal
                old_pos = st.session_state.strategy_positions.get(name, 0)
                if signal == "BUY" and old_pos <= 0:
                    st.session_state.strategy_positions[name] = 1
                elif signal == "SELL" and old_pos >= 0:
                    st.session_state.strategy_positions[name] = -1
            
            if len(st.session_state.price_history) > 1:
                price_change = st.session_state.price_history[-1] - st.session_state.price_history[-2]
                for name in st.session_state.strategy_positions:
                    pos = st.session_state.strategy_positions[name]
                    pnl_change = pos * price_change
                    current_pnl = st.session_state.pnl_history[name][-1] if st.session_state.pnl_history[name] else 0
                    st.session_state.pnl_history[name].append(current_pnl + pnl_change)
            
            best_strat = None
            best_pnl = float('-inf')
            strat_cols = st.columns(3)
            
            for i, result in enumerate(results):
                name = result.name
                signal = result.signal
                confidence = result.confidence
                pnl_list = st.session_state.pnl_history.get(name, [0])
                accumulated_pnl = pnl_list[-1] if pnl_list else 0
                position = st.session_state.strategy_positions.get(name, 0)
                
                if accumulated_pnl > best_pnl:
                    best_pnl = accumulated_pnl
                    best_strat = name
                
                pos_text = "LONG 📈" if position > 0 else "SHORT 📉" if position < 0 else "FLAT ➖"
                
                with strat_cols[i]:
                    if signal == "BUY":
                        st.success(f"**{name.replace('_', ' ')}**")
                    elif signal == "SELL":
                        st.error(f"**{name.replace('_', ' ')}**")
                    else:
                        st.info(f"**{name.replace('_', ' ')}**")
                    st.metric(f"Signal: {signal}", f"${accumulated_pnl:,.2f}", delta=pos_text)
                    st.progress(min(confidence, 1.0))
            
            if best_strat:
                st.success(f"🏆 **Leading:** {best_strat.replace('_', ' ')} (${best_pnl:,.2f})")
            
            if st.session_state.paper_trader:
                st.markdown("---")
                st.markdown("### 💼 Paper Trading")
                st.caption("Practice buying and selling without real money!")
                tcol1, tcol2, tcol3 = st.columns([1, 1, 1])
                with tcol1:
                    qty = st.number_input("Qty:", 0.001, 10.0, 0.01, 0.001, format="%.3f", key='trade_qty',
                        help="Amount of crypto to buy or sell")
                with tcol2:
                    if st.button("🟢 BUY", type="primary", use_container_width=True, key='buy_btn'):
                        result = st.session_state.paper_trader.execute_trade(symbol, "BUY", qty, current_price, best_strat or "Manual")
                        st.toast(f"✅ {result['message']}" if result['status'] == 'FILLED' else f"❌ {result['message']}")
                with tcol3:
                    if st.button("🔴 SELL", use_container_width=True, key='sell_btn'):
                        result = st.session_state.paper_trader.execute_trade(symbol, "SELL", qty, current_price, best_strat or "Manual")
                        st.toast(f"✅ {result['message']}" if result['status'] == 'FILLED' else f"❌ {result['message']}")
            
            st.markdown("---")
            tab1, tab2, tab3 = st.tabs(["📈 Price", "📊 PnL", "📕 Order Book"])
            
            # Measure chart rendering latency
            chart_start = time.perf_counter()
            
            with tab1:
                if len(st.session_state.price_history) > 5:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=st.session_state.price_history[-200:], mode='lines', line=dict(color='#26A69A', width=2)))
                    fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                if any(len(v) > 3 for v in st.session_state.pnl_history.values()):
                    fig = go.Figure()
                    colors = {'Trend_Follower': '#2196F3', 'Mean_Reversion': '#4CAF50', 'Momentum_RSI': '#9C27B0'}
                    for name, pnl_list in st.session_state.pnl_history.items():
                        if pnl_list:
                            fig.add_trace(go.Scatter(y=pnl_list[-200:], mode='lines', name=name.replace('_', ' '), line=dict(color=colors.get(name, '#888'), width=2)))
                    fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', legend=dict(orientation='h'))
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                if not bids_df.empty and not asks_df.empty:
                    bids_sorted = bids_df.sort_values('price', ascending=False).head(15)
                    asks_sorted = asks_df.sort_values('price').head(15)
                    bids_sorted['cum'] = bids_sorted['quantity'].cumsum()
                    asks_sorted['cum'] = asks_sorted['quantity'].cumsum()
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=bids_sorted['price'], y=bids_sorted['cum'], fill='tozeroy', name='Bids', line=dict(color='#26A69A')))
                    fig.add_trace(go.Scatter(x=asks_sorted['price'], y=asks_sorted['cum'], fill='tozeroy', name='Asks', line=dict(color='#EF5350')))
                    fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)
            
            chart_elapsed_ms = (time.perf_counter() - chart_start) * 1000
            latency_tracker.record('chart_render', chart_elapsed_ms)
        else:
            st.warning("⚠️ Run `maturin develop` to enable strategies.")
        
        # Calculate and print total loop latency
        total_loop_ms = (time.perf_counter() - loop_start) * 1000
        latency_tracker.record('total_loop', total_loop_ms)
        
        # Track tick completion (prints summary every 10 ticks)
        latency_tracker.on_tick_complete()
        
        time.sleep(0.1)
        st.rerun()

    else:
        st.markdown("---")
        
        # =========================================================================
        # COMPREHENSIVE WELCOME GUIDE FOR NOVICE USERS
        # =========================================================================
        st.markdown("## 📚 Welcome to the Trading Strategy Simulator!")
        st.markdown("*A safe place to learn algorithmic trading without risking real money*")
        
        st.markdown("---")
        
        # What is this app?
        st.markdown("### 🎯 What Does This App Do?")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            This simulator lets you **watch 3 different trading strategies compete** against each other 
            in a simulated cryptocurrency market. You can:
            
            - 🎓 **Learn** how algorithmic trading strategies work
            - 🧪 **Test** strategies in different market conditions (bull, bear, sideways)
            - 📊 **Compare** which strategy performs best
            - 💰 **Practice** paper trading without risking real money
            - ⚙️ **Experiment** with strategy parameters to see their effects
            """)
        with col2:
            st.success("💡 **No real money involved!** Everything is simulated for learning purposes.")
        
        st.markdown("---")
        
        # The Three Strategies Explained
        st.markdown("### 🧠 Understanding the 3 Strategies")
        st.caption("Each strategy uses different technical indicators to decide when to buy or sell")
        
        strat1, strat2, strat3 = st.columns(3)
        
        with strat1:
            st.markdown("#### 📈 Trend Follower")
            st.markdown("""
            **How it works:**  
            Uses two Exponential Moving Averages (EMAs) - a fast one and a slow one.
            
            **When it buys:** Fast EMA crosses above slow EMA (uptrend starting)
            
            **When it sells:** Fast EMA crosses below slow EMA (downtrend starting)
            
            **Best for:** Markets with clear trends
            
            *Think of it as "riding the wave" 🌊*
            """)
        
        with strat2:
            st.markdown("#### 🔄 Mean Reversion")
            st.markdown("""
            **How it works:**  
            Uses Bollinger Bands - a channel around the average price.
            
            **When it buys:** Price drops to lower band (oversold)
            
            **When it sells:** Price rises to upper band (overbought)
            
            **Best for:** Sideways/ranging markets
            
            *Think of it as "rubber band snapping back" 🔄*
            """)
        
        with strat3:
            st.markdown("#### ⚡ Momentum RSI")
            st.markdown("""
            **How it works:**  
            Uses RSI (Relative Strength Index) to measure momentum.
            
            **When it buys:** RSI below 30 (oversold condition)
            
            **When it sells:** RSI above 70 (overbought condition)
            
            **Best for:** Catching reversals
            
            *Think of it as "catching exhaustion points" ⚡*
            """)
        
        st.markdown("---")
        
        # Step by Step Guide
        st.markdown("### 🚀 How to Use This App (Step by Step)")
        
        st.markdown("""
        #### Step 1️⃣ — Choose a Market Scenario
        In the sidebar, select a **Market Scenario**:
        - **Realistic** - Random market movements (good for general testing)
        - **Bull Market** - Prices tend to go up (test trend-following strategies)
        - **Bear Market** - Prices tend to go down (test in downtrends)
        - **Sideways** - Prices stay in a range (test mean reversion)
        - **Volatile** - Big swings up and down (stress test strategies)
        
        #### Step 2️⃣ — Adjust Settings (Optional)
        - **Update Speed** - How fast the simulation runs
        - **Starting Balance** - How much paper money to start with
        - **Strategy Parameters** - Expand each strategy section to tweak settings
        
        #### Step 3️⃣ — Click "▶️ Start"
        The simulation will begin! Watch the:
        - **Price chart** updating in real-time
        - **Strategy cards** showing BUY/SELL/HOLD signals
        - **PnL (Profit & Loss)** for each strategy
        
        #### Step 4️⃣ — Try Paper Trading
        Use the **BUY** and **SELL** buttons to practice trading manually. Your trades appear in the sidebar under "Your Trades".
        
        #### Step 5️⃣ — Analyze Results
        - Look at which strategy has the highest PnL
        - Check the **📊 PnL tab** to see performance over time
        - The **🏆 Leading** indicator shows the current winner
        """)
        
        st.markdown("---")
        
        # Understanding the Display
        st.markdown("### 📊 Reading the Dashboard")
        
        read1, read2 = st.columns(2)
        
        with read1:
            st.markdown("""
            #### Strategy Cards Color Coding
            - 🟢 **Green** = Strategy says **BUY**
            - 🔴 **Red** = Strategy says **SELL**
            - 🔵 **Blue** = Strategy says **HOLD**
            
            #### Key Metrics
            - **Price** - Current simulated price
            - **Spread** - Difference between buy/sell prices
            - **Regime** - Current market type
            - **Ticks** - Number of price updates
            """)
        
        with read2:
            st.markdown("""
            #### Strategy Card Info
            - **Signal** - Current recommendation (BUY/SELL/HOLD)
            - **Dollar Amount** - Accumulated profit/loss
            - **Position** - LONG (bought), SHORT (sold), or FLAT
            - **Progress Bar** - Strategy confidence level
            
            #### Tabs
            - **📈 Price** - Live price chart
            - **📊 PnL** - Profit comparison graph
            - **📕 Order Book** - Buy/sell pressure visualization
            """)
        
        st.markdown("---")
        
        # Tips for Beginners
        st.markdown("### 💡 Tips for Beginners")
        
        tip1, tip2, tip3 = st.columns(3)
        
        with tip1:
            st.info("""
            **🎮 Experiment Freely**
            
            Try different scenarios to see which strategy works best in each market condition. There's no wrong answer - it's all about learning!
            """)
        
        with tip2:
            st.info("""
            **⚙️ Tweak Parameters**
            
            Adjust strategy parameters in the sidebar to see how they affect performance. Small changes can make big differences!
            """)
        
        with tip3:
            st.info("""
            **📈 No Strategy is Perfect**
            
            Different strategies work better in different conditions. The "best" strategy changes depending on the market!
            """)
        
        st.markdown("---")
        st.markdown("### 🎬 Ready to Start?")
        st.markdown("**👈 Use the sidebar on the left to configure your simulation and click Start!**")
