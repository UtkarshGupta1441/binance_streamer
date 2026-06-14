# Trading Strategy Simulator

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://binancestreamer.streamlit.app/)

A comprehensive trading strategy simulation platform that allows you to compare different algorithmic trading strategies in a risk-free environment. Built with a Streamlit web interface and powered by high-performance Rust-based trading algorithms.

**🌐 Live Demo:** [https://binancestreamer.streamlit.app/](https://binancestreamer.streamlit.app/)

## 🎯 What Is This?

This is an **educational trading simulator** designed to help you:
- Learn how different trading strategies work
- Compare strategy performance under various market conditions
- Practice paper trading without risking real money
- Understand technical indicators like EMA, Bollinger Bands, and RSI

> **Note**: This is a simulation tool for educational purposes only. No real trading or connection to live markets is involved.

## ✨ Features

### Trading Strategies
Three professional-grade algorithmic strategies powered by Rust:

| Strategy | Indicator | How It Works |
|----------|-----------|--------------|
| **Trend Follower** | EMA Crossover | Buys when short-term EMA crosses above long-term EMA (uptrend), sells when it crosses below (downtrend) |
| **Mean Reversion** | Bollinger Bands | Buys when price drops below lower band (oversold), sells when it rises above upper band (overbought) |
| **Momentum RSI** | Relative Strength Index | Buys when RSI indicates oversold conditions, sells when overbought |

### Market Simulation
Five realistic market scenarios to test your strategies:

- 🎯 **Realistic** - Natural price movements with varied volatility
- 📈 **Trending Bull** - Upward trending market
- 📉 **Trending Bear** - Downward trending market  
- ➡️ **Sideways** - Range-bound, choppy market
- ⚡ **Volatile** - High volatility with large swings

### Strategy Parameter Tuning
Fine-tune each strategy's parameters to optimize performance:

- **EMA Crossover**: Adjust short/long period lengths
- **Bollinger Bands**: Configure period and standard deviation multiplier
- **RSI**: Set period, oversold, and overbought thresholds

### Paper Trading
- Track simulated trades with entry/exit prices
- Monitor P&L (Profit and Loss) in real-time
- View complete trade history
- Set custom starting balance

### 🎲 Monte Carlo Backtesting (NEW!)
Run thousands of backtests to statistically validate strategy performance:

- **Multiple Price Path Models**:
  - Regime Switching (bull/bear/sideways transitions)
  - Geometric Brownian Motion (classic random walk)
  - Mean Reverting (Ornstein-Uhlenbeck process)
  - Jump Diffusion (flash crashes/rallies)

- **Comprehensive Statistics**:
  - Mean, median, and standard deviation of returns
  - 95% confidence intervals
  - Value at Risk (VaR) at 95% and 99%
  - Conditional VaR (Expected Shortfall)
  - Probability of profit/loss
  - Sharpe ratio distribution

- **Visual Analysis**:
  - Return distribution histograms
  - Box plots comparing strategies
  - Sample equity curves
  - Downloadable reports

### Live Visualization
- Real-time price chart with strategy signals
- P&L comparison across all strategies
- Position tracking and trade markers
- Performance metrics dashboard

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Web Interface** | Streamlit |
| **Visualization** | Plotly |
| **Data Processing** | Pandas |
| **Trading Algorithms** | Rust |
| **Python-Rust Bridge** | PyO3 & Maturin |
| **Cryptography** | `rustls` (Pure Rust, Optimized for Cloud Deployment) |

## 📦 Installation

### Prerequisites

- [Python 3.8+](https://www.python.org/downloads/)
- [Rust](https://www.rust-lang.org/tools/install)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd binance_streamer
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate it
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (CMD)
.venv\Scripts\activate.bat

# macOS/Linux
source .venv/bin/activate
```

### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Build Rust Backend

```bash
maturin develop
```

You should see output confirming the package was installed successfully.

## 🚀 Quick Start

### Running the Simulator

```bash
streamlit run dashboard_v2.py
```

The dashboard will open in your browser at `http://localhost:8501`.

### ☁️ Cloud Deployment (Streamlit Community Cloud)

This app is heavily optimized for zero-configuration deployment on **Streamlit Community Cloud**.
The repository is pre-configured with:
- `packages.txt`: For required system-level dependencies.
- `requirements.txt`: Python package limits relaxed to ensure availability of pre-compiled wheels for newer Python versions.
- `.cargo/config.toml`: Enforces memory usage guardrails (`jobs=1`) preventing Out-Of-Memory (OOM) crashes during `maturin` builds in cloud containers.
- Pure Rust Cryptography (`rustls`): Bypasses system OpenSSL (`libssl-dev`) requirements entirely for flawless CI/CD.

### First Time Users - Getting Started

1. **Configure Settings** (Left Sidebar):
   - Set your starting balance (default: $10,000)
   - Choose a market scenario (start with "Realistic")
   
2. **Tune Strategy Parameters** (Optional):
   - Expand strategy sections to customize
   - Or use default values which work well
   
3. **Start Simulation**:
   - Click the green "▶️ Start Simulation" button
   - Watch the price chart update in real-time
   
4. **Monitor Performance**:
   - Compare P&L across strategies
   - View trade signals on the chart
   - Check trade history at the bottom

5. **Experiment**:
   - Try different market scenarios
   - Adjust strategy parameters
   - Reset and compare results

## 📁 Project Structure

```
binance_streamer/
├── dashboard_v2.py      # Main Streamlit application
├── simulator.py         # Market simulation engine
├── montecarlo.py        # Monte Carlo backtesting module (NEW!)
├── requirements.txt     # Python dependencies
├── Cargo.toml          # Rust project configuration
├── src/                # Rust source code
│   ├── lib.rs          # Python module definition
│   ├── strategy_manager.rs
│   ├── indicators/     # Technical indicators
│   │   ├── ema.rs      # Exponential Moving Average
│   │   ├── bollinger.rs # Bollinger Bands
│   │   └── rsi.rs      # Relative Strength Index
│   └── strategies/     # Trading strategies
│       ├── trend_follower.rs
│       ├── mean_reversion.rs
│       └── momentum.rs
└── data/               # Data storage
```

## 📊 Understanding the Dashboard

### Mode Selection
Switch between two modes in the sidebar:
- **📊 Live Simulation**: Watch strategies compete in real-time
- **🎲 Monte Carlo Backtest**: Run statistical analysis across thousands of simulations

### Main Display (Live Simulation)
- **Price Chart**: Shows simulated price with buy/sell signals
- **P&L Chart**: Compares profit/loss across strategies
- **Metrics Cards**: Current price, balance, and positions

### Sidebar Controls
- **Starting Balance**: Initial capital for simulation
- **Market Scenario**: Type of market to simulate
- **Strategy Parameters**: Fine-tune indicator settings
- **Start/Stop**: Control simulation

### Trade History Table
- Entry and exit timestamps
- Position direction (Long/Short)
- Entry and exit prices
- Profit/Loss per trade

### Monte Carlo Results (Backtest Mode)
- **Strategy Cards**: Mean return, probability of profit, Sharpe ratio, worst drawdown
- **Return Distribution Tab**: Histogram of returns across all simulations
- **Risk Analysis Tab**: VaR, CVaR, box plots, risk metrics table
- **Equity Curves Tab**: Sample paths showing strategy behavior
- **Full Report Tab**: Downloadable statistical summary

## 🎓 Learning Resources

### Strategy Concepts

**EMA Crossover (Trend Following)**
- Uses two moving averages of different lengths
- Shorter EMA reacts faster to price changes
- Crossovers signal trend changes

**Bollinger Bands (Mean Reversion)**
- Middle band = 20-period moving average
- Upper/Lower bands = ±2 standard deviations
- Price tends to revert to the mean

**RSI (Momentum)**
- Measures speed and magnitude of price changes
- Scale of 0-100
- <30 = Oversold, >70 = Overbought

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add new trading strategies
- Improve market simulation models
- Enhance visualizations
- Fix bugs or improve documentation

## ⚠️ Disclaimer

This software is for **educational purposes only**. It does not constitute financial advice and should not be used for real trading decisions. Past simulated performance does not guarantee future results.

## 📄 License

MIT License - feel free to use and modify for your own projects.
