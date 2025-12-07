use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// Order book level (price, quantity)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Level {
    pub price: f64,
    pub quantity: f64,
}

/// Depth snapshot from REST API
#[derive(Debug, Clone, Deserialize)]
pub struct DepthSnapshot {
    #[serde(rename = "lastUpdateId")]
    pub last_update_id: u64,
    pub bids: Vec<Vec<String>>,
    pub asks: Vec<Vec<String>>,
}

/// Depth update from WebSocket
#[derive(Debug, Clone, Deserialize)]
pub struct DepthUpdate {
    #[serde(rename = "e")]
    pub event_type: String,
    #[serde(rename = "E")]
    pub event_time: u64,
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "U")]
    pub first_update_id: u64,
    #[serde(rename = "u")]
    pub final_update_id: u64,
    #[serde(rename = "b")]
    pub bids: Vec<Vec<String>>,
    #[serde(rename = "a")]
    pub asks: Vec<Vec<String>>,
}

/// Trade event from WebSocket
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Trade {
    #[serde(rename = "e")]
    pub event_type: String,
    #[serde(rename = "E")]
    pub event_time: u64,
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "t")]
    pub trade_id: u64,
    #[serde(rename = "p")]
    pub price: String,
    #[serde(rename = "q")]
    pub quantity: String,
    #[serde(rename = "m")]
    pub is_buyer_maker: bool,
}

/// Signal from a trading strategy
#[derive(Debug, Clone, PartialEq)]
#[pyclass]
pub enum Signal {
    Buy,
    Sell,
    Hold,
}

#[pymethods]
impl Signal {
    fn __str__(&self) -> &'static str {
        match self {
            Signal::Buy => "BUY",
            Signal::Sell => "SELL",
            Signal::Hold => "HOLD",
        }
    }
    
    fn __repr__(&self) -> &'static str {
        match self {
            Signal::Buy => "Signal.Buy",
            Signal::Sell => "Signal.Sell",
            Signal::Hold => "Signal.Hold",
        }
    }
}

/// Result of strategy evaluation
#[derive(Debug, Clone)]
#[pyclass]
pub struct StrategyResult {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub signal: String,
    #[pyo3(get)]
    pub confidence: f64,
    #[pyo3(get)]
    pub paper_pnl: f64,
}

#[pymethods]
impl StrategyResult {
    #[new]
    pub fn new(name: String, signal: String, confidence: f64, paper_pnl: f64) -> Self {
        Self { name, signal, confidence, paper_pnl }
    }
}

/// A trade order to be executed
#[derive(Debug, Clone)]
#[pyclass]
pub struct TradeOrder {
    #[pyo3(get, set)]
    pub symbol: String,
    #[pyo3(get, set)]
    pub side: String,
    #[pyo3(get, set)]
    pub quantity: f64,
    #[pyo3(get, set)]
    pub order_type: String,
    #[pyo3(get, set)]
    pub price: Option<f64>,
}

#[pymethods]
impl TradeOrder {
    #[new]
    pub fn new(symbol: String, side: String, quantity: f64) -> Self {
        Self {
            symbol,
            side,
            quantity,
            order_type: "MARKET".to_string(),
            price: None,
        }
    }
}
