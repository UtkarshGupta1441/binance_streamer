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
# PAGE CONFIG (Must be first)
# ============================================================================
st.set_page_config(
    page_title="Trading Strategy Simulator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# IMPORTS
# ============================================================================
RUST_AVAILABLE = False
SIMULATOR_AVAILABLE = False

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
    current_price = 0
    mid_price = 0
    spread = 0
    bids_df = pd.DataFrame()
    asks_df = pd.DataFrame()
    regime = ""
    symbol = "BTCUSDT"
    
    if st.session_state.simulator:
        sim = st.session_state.simulator
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
    else:
        st.warning("⚠️ Run `maturin develop` to enable strategies.")
    
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
