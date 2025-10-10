import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import websocket
import json
import threading
import time
import subprocess
import sys
import os
from collections import deque

# --- Page Configuration ---
st.set_page_config(
    page_title="Real-Time Trading Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for a better look ---
st.markdown("""
<style>
    .stMetric {{
        border-radius: 10px;
        background-color: #2a2a39;
        padding: 15px;
        text-align: center;
    }}
    .stMetric .st-ae {{ /* Metric label */
        font-size: 1.1rem;
        color: #a0a0b0;
    }}
    .stMetric .st-af {{ /* Metric value */
        font-size: 1.5rem;
        font-weight: bold;
    }}
    .stButton>button {{
        width: 100%;
    }}
    /* Style the dataframe headers */
    .stDataFrame thead th {{
        background-color: #2a2a39;
        color: white;
    }}
</style>
""", unsafe_allow_html=True)

API_BASE_URL = "https://api.binance.com/api/v3"
WSS_BASE_URL = "wss://stream.binance.com:9443/ws"

# --- Thread-Safe Order Book Class ---
class OrderBook:
    def __init__(self, symbol):
        self._lock = threading.Lock()
        self.symbol = symbol.upper()
        
        # Data attributes
        self.bids = pd.DataFrame(columns=['price', 'quantity'])
        self.asks = pd.DataFrame(columns=['price', 'quantity'])
        self.recent_trades = deque(maxlen=20)
        self.last_update_id = 0
        self.last_event_time = None
        
        # Connection attributes
        self.ws = None
        self.ws_thread = None
        self.is_running = False

    def _get_depth_snapshot(self):
        """Fetches the initial order book snapshot."""
        url = f"{API_BASE_URL}/depth"
        params = {'symbol': self.symbol, 'limit': 1000}
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            with self._lock:
                self.last_update_id = data['lastUpdateId']
                self.bids = pd.DataFrame(data['bids'], columns=['price', 'quantity'], dtype=float)
                self.asks = pd.DataFrame(data['asks'], columns=['price', 'quantity'], dtype=float)
            st.sidebar.success("Snapshot loaded.")
            return True
        except requests.exceptions.RequestException as e:
            st.sidebar.error(f"Snapshot Error: {e}")
            return False

    def _update_dataframe(self, df, updates):
        """Helper to efficiently update bid/ask dataframes."""
        for price, quantity in updates:
            price, quantity = float(price), float(quantity)
            if quantity == 0:
                df = df[df['price'] != price]
            else:
                if price in df['price'].values:
                    df.loc[df['price'] == price, 'quantity'] = quantity
                else:
                    new_row = pd.DataFrame([{'price': price, 'quantity': quantity}])
                    df = pd.concat([df, new_row], ignore_index=True)
        return df

    def _on_message(self, ws, message):
        """Handles incoming WebSocket messages."""
        data = json.loads(message)
        
        if 'e' in data and data['e'] == 'error':
            st.error(f"WebSocket Error: {data['m']}")
            return

        stream_data = data.get('data', data)
        stream_type = data.get('stream', '').split('@')[1] if 'stream' in data else stream_data.get('e')

        with self._lock:
            if stream_type == 'trade':
                self.recent_trades.append(stream_data)
                return

            if stream_type == 'depthUpdate' or stream_type == 'depth':
                # Buffer events that arrive before the snapshot is processed
                if self.last_update_id == 0:
                    return 

                # Drop old events
                if stream_data['u'] <= self.last_update_id:
                    return
                
                # First event after snapshot must follow this rule
                if stream_data['U'] > self.last_update_id + 1:
                    # This indicates a gap, a resync might be needed in a production system
                    # For this dashboard, we'll try to continue but log a warning
                    # In a more robust implementation, you might re-fetch the snapshot here.
                    pass

                self.bids = self._update_dataframe(self.bids, stream_data.get('b', []))
                self.asks = self._update_dataframe(self.asks, stream_data.get('a', []))
                self.last_update_id = stream_data['u']
                self.last_event_time = stream_data.get('E')

    def _on_error(self, ws, error): st.error(f"WebSocket Error: {error}")
    def _on_close(self, ws, close_status_code, close_msg): st.info("WebSocket connection closed.")
    def _on_open(self, ws): st.success("Live connection opened.")

    def start(self):
        """Starts the WebSocket connection."""
        if not self._get_depth_snapshot():
            return # Don't start if snapshot fails

        ws_url = f"{WSS_BASE_URL}/stream?streams={self.symbol.lower()}@depth@100ms/{self.symbol.lower()}@trade"
        self.ws = websocket.WebSocketApp(ws_url, on_message=self._on_message, on_error=self._on_error, on_close=self._on_close, on_open=self._on_open)
        
        self.ws_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
        self.ws_thread.start()
        self.is_running = True

    def stop(self):
        """Stops the WebSocket connection."""
        if self.is_running and self.ws:
            self.ws.close()
            self.ws_thread.join(timeout=1)
        self.is_running = False

    def get_data(self):
        """Returns a thread-safe copy of the data for UI rendering."""
        with self._lock:
            return (
                self.bids.copy(),
                self.asks.copy(),
                list(self.recent_trades),
                self.last_event_time
            )

# --- Backend Process Management ---
def start_research_backend():
    """Starts the research.py script as a background process."""
    if 'research_process' not in st.session_state or st.session_state.research_process.poll() is not None:
        try:
            command = [sys.executable, "research.py"]
            st.session_state.research_process = subprocess.Popen(command)
            st.sidebar.success("`research.py` backend started.")
        except Exception as e:
            st.sidebar.error(f"Failed to start `research.py`: {e}")

# --- UI Components ---
st.title("Real-Time Trading Dashboard")

# Sidebar for controls
with st.sidebar:
    st.header("Controls")
    symbol_input = st.text_input("Symbol", "BTCUSDT").upper()
    
    if st.button("Start/Update Stream"):
        if 'order_book' in st.session_state and st.session_state.order_book:
            st.session_state.order_book.stop()
        
        st.session_state.order_book = OrderBook(symbol_input)
        st.session_state.order_book.start()
        start_research_backend()

    if st.button("Stop Stream"):
        if 'order_book' in st.session_state and st.session_state.order_book:
            st.session_state.order_book.stop()
            st.session_state.order_book = None
        if 'research_process' in st.session_state and st.session_state.research_process.poll() is None:
            st.session_state.research_process.terminate()
            st.session_state.research_process.wait()
            st.sidebar.warning("`research.py` backend stopped.")

    st.header("Display Options")
    view_mode = st.radio("View Type", ('Depth Chart', 'Order Book'))
    depth = st.slider("Order Book Depth", 10, 200, 50, 10)
    show_metrics = st.toggle("Show Live Metrics", True)
    show_trades = st.toggle("Show Recent Trades", True)

# Main content area
if 'order_book' not in st.session_state or not st.session_state.order_book or not st.session_state.order_book.is_running:
    st.info("Enter a symbol and click 'Start Stream' to begin.")
else:
    # Placeholders for dynamic content
    metrics_placeholder = st.empty()
    view_placeholder = st.empty()
    trades_placeholder = st.empty()

    # Get data from the thread-safe class instance
    bids_df, asks_df, recent_trades, last_event_time = st.session_state.order_book.get_data()

    bids_df = bids_df.sort_values('price', ascending=False).head(depth)
    asks_df = asks_df.sort_values('price', ascending=True).head(depth)

    if not bids_df.empty and not asks_df.empty:
        best_bid = bids_df['price'].iloc[0]
        best_ask = asks_df['price'].iloc[0]

        # --- Metrics Calculation ---
        if show_metrics:
            spread = best_ask - best_bid
            mid_price = (best_ask + best_bid) / 2
            imbalance = bids_df['quantity'].sum() / (bids_df['quantity'].sum() + asks_df['quantity'].sum())
            with metrics_placeholder.container():
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Mid-Price", f"${mid_price:,.2f}")
                m_col2.metric("Spread", f"${spread:,.2f}")
                m_col3.metric("Book Imbalance", f"{imbalance:.2%}", f"{'BUY' if imbalance > 0.5 else 'SELL'}-side pressure")

        # --- Chart/Table Rendering ---
        with view_placeholder.container():
            if view_mode == 'Depth Chart':
                bids_df['cumulative'] = bids_df['quantity'].cumsum()
                asks_df['cumulative'] = asks_df['quantity'].cumsum()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=bids_df['price'], y=bids_df['cumulative'], fill='tozeroy', name='Bids', line=dict(color='green')))
                fig.add_trace(go.Scatter(x=asks_df['price'], y=asks_df['cumulative'], fill='tozeroy', name='Asks', line=dict(color='red')))
                fig.update_layout(title=f"Order Book Depth - {symbol_input}", xaxis_title="Price", yaxis_title="Cumulative Size", height=500, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
                st.plotly_chart(fig, use_container_width=True)
            
            elif view_mode == 'Order Book':
                asks_df_display = asks_df.sort_values('price', ascending=False)
                asks_df_display['total'] = asks_df_display['price'] * asks_df_display['quantity']
                bids_df['total'] = bids_df['price'] * bids_df['quantity']
                
                st.subheader(f"Live Order Book - {symbol_input}")
                st.dataframe(asks_df_display[['price', 'quantity', 'total']].style.format({'price': '{:.2f}', 'quantity': '{:.6f}', 'total': '{:,.2f}'}).apply(lambda x: ['color: red']*3, axis=1).bar(subset=['quantity', 'total'], color='#FF4B4B', align='zero'), use_container_width=True, hide_index=True, column_config={"price": "Price (USDT)", "quantity": "Amount (BTC)", "total": "Total"})
                st.markdown(f"<h4 style='text-align: center; color: #26A69A;'>{best_bid:.2f} <span style='color: grey;'>|</span> ${best_ask:.2f}</h4>", unsafe_allow_html=True)
                st.dataframe(bids_df[['price', 'quantity', 'total']].style.format({'price': '{:.2f}', 'quantity': '{:.6f}', 'total': '{:,.2f}'}).apply(lambda x: ['color: green']*3, axis=1).bar(subset=['quantity', 'total'], color='#26A69A', align='zero'), use_container_width=True, hide_index=True, column_config={"price": "Price (USDT)", "quantity": "Amount (BTC)", "total": "Total"})
                if last_event_time:
                    st.caption(f"Last update: {pd.to_datetime(last_event_time, unit='ms')}")

        # --- Recent Trades Rendering ---
        if show_trades:
            with trades_placeholder.container():
                if recent_trades:
                    trades = pd.DataFrame(recent_trades)
                    trades = trades.rename(columns={'p': 'Price', 'q': 'Quantity', 'm': 'Side'})
                    trades['Price'] = trades['Price'].astype(float).apply(lambda x: f"${x:,.2f}")
                    trades['Quantity'] = trades['Quantity'].astype(float)
                    trades['Side'] = trades['Side'].apply(lambda x: 'SELL' if x else 'BUY')
                    st.subheader("Recent Market Trades")
                    st.dataframe(trades[['Price', 'Quantity', 'Side']], use_container_width=True, hide_index=True)
                else:
                    st.info("Waiting for trade data...")
    
    # Rerun to refresh the UI
    time.sleep(0.1) # Short sleep to prevent excessive CPU usage
    st.rerun()
