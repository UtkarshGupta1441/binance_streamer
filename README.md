# Real-Time Binance Trading Dashboard

This project is a real-time cryptocurrency trading dashboard that streams live order book and trade data from Binance. It features a web-based interface built with Streamlit and a high-performance backend component written in Rust for efficient data processing.

## Features

- **Live Data Streaming**: Connects to Binance WebSocket streams for real-time order book depth and trade updates.
- **Dual Visualization Modes**:
    - **Depth Chart**: A dynamic, cumulative depth chart visualizing market liquidity for bids and asks.
    - **Order Book**: A classic, table-style view mimicking the Binance UI, showing price, amount, and total for both sides of the book.
- **Real-Time Metrics**: Displays key trading indicators such as mid-price, spread, and book imbalance.
- **Recent Trades**: Shows a live feed of the most recent market trades.
- **High-Performance Rust Core**: Utilizes a Rust backend (integrated via PyO3 and Maturin) for potential high-speed data processing tasks, demonstrating a powerful polyglot architecture.
- **Interactive UI**: Built with Streamlit for a responsive and easy-to-use web interface.

## Tech Stack

- **Frontend**: Streamlit
- **Data Manipulation**: Pandas
- **Charting**: Plotly
- **Backend Core**: Rust
- **Python-Rust Bridge**: PyO3 & Maturin
- **API & WebSocket**: `requests`, `websocket-client`

## Project Setup and Installation

Follow these steps to get the project running locally.

### Prerequisites

- [Python 3.8+](https://www.python.org/downloads/)
- [Rust](https://www.rust-lang.org/tools/install)

### 1. Clone the Repository

```bash
git clone <repository-url>
cd binance_streamer
```

### 2. Set Up a Python Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
# Create the virtual environment
python -m venv .venv

# Activate it
# On Windows (PowerShell/CMD)
.venv\Scripts\Activate.ps1
# On macOS/Linux
source .venv/bin/activate
```

### 3. Install Python Dependencies

Install all the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Build the Rust Backend

The Rust component needs to be compiled into a Python module. `maturin` handles this process seamlessly.

```bash
maturin develop
```

If successful, you will see output indicating that the package has been installed in your virtual environment.

## Usage

Once the setup is complete, you can run the Streamlit application.

```bash
streamlit run dashboard.py
```

This will start the web server and open the dashboard in your default web browser. You can then enter a trading symbol (e.g., `BTCUSDT`) and click "Start/Update Stream" to begin viewing live data.

## File Structure

- `dashboard.py`: The main Streamlit application file containing the UI and data visualization logic.
- `research.py`: A script intended for backend data processing, which is run as a subprocess by the dashboard.
- `src/`: This directory contains the Rust source code for the high-performance backend component.
  - `lib.rs`: The main Rust library file that defines the Python module using PyO3.
  - `*.rs`: Other Rust modules for specific functionalities.
- `Cargo.toml`: The manifest file for the Rust project, defining its dependencies and metadata.
- `requirements.txt`: A list of all Python packages required for the project.
- `data/`: Directory used for storing data files, such as the `trades.csv` log.
- `README.md`: This file.
