use std::collections::BTreeMap;
use pyo3::prelude::*;

/// A high-performance order book using BTreeMap for sorted price levels
#[pyclass]
pub struct OrderBook {
    symbol: String,
    bids: BTreeMap<i64, f64>,  // price (as i64 for precision) -> quantity
    asks: BTreeMap<i64, f64>,
    last_update_id: u64,
    price_precision: i32,
}

impl OrderBook {
    /// Convert f64 price to i64 for precise comparison
    fn price_to_key(&self, price: f64) -> i64 {
        (price * 10f64.powi(self.price_precision)) as i64
    }

    /// Convert i64 key back to f64 price
    fn key_to_price(&self, key: i64) -> f64 {
        key as f64 / 10f64.powi(self.price_precision)
    }
}

#[pymethods]
impl OrderBook {
    #[new]
    pub fn new(symbol: String) -> Self {
        Self {
            symbol: symbol.to_uppercase(),
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            last_update_id: 0,
            price_precision: 8,
        }
    }

    /// Get the symbol
    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    /// Get last update ID
    pub fn last_update_id(&self) -> u64 {
        self.last_update_id
    }

    /// Load initial snapshot
    pub fn load_snapshot(&mut self, bids: Vec<(f64, f64)>, asks: Vec<(f64, f64)>, last_update_id: u64) {
        self.bids.clear();
        self.asks.clear();
        
        for (price, qty) in bids {
            if qty > 0.0 {
                let key = self.price_to_key(price);
                self.bids.insert(key, qty);
            }
        }
        
        for (price, qty) in asks {
            if qty > 0.0 {
                let key = self.price_to_key(price);
                self.asks.insert(key, qty);
            }
        }
        
        self.last_update_id = last_update_id;
    }

    /// Apply a depth update
    pub fn apply_update(
        &mut self,
        bid_updates: Vec<(f64, f64)>,
        ask_updates: Vec<(f64, f64)>,
        _first_update_id: u64,
        final_update_id: u64,
    ) -> bool {
        // Validate update sequence
        if final_update_id <= self.last_update_id {
            return false; // Old update, skip
        }

        // Apply bid updates
        for (price, qty) in bid_updates {
            let key = self.price_to_key(price);
            if qty == 0.0 {
                self.bids.remove(&key);
            } else {
                self.bids.insert(key, qty);
            }
        }

        // Apply ask updates
        for (price, qty) in ask_updates {
            let key = self.price_to_key(price);
            if qty == 0.0 {
                self.asks.remove(&key);
            } else {
                self.asks.insert(key, qty);
            }
        }

        self.last_update_id = final_update_id;
        true
    }

    /// Get best bid (highest buy price)
    pub fn best_bid(&self) -> Option<(f64, f64)> {
        self.bids.iter().next_back().map(|(&k, &v)| (self.key_to_price(k), v))
    }

    /// Get best ask (lowest sell price)
    pub fn best_ask(&self) -> Option<(f64, f64)> {
        self.asks.iter().next().map(|(&k, &v)| (self.key_to_price(k), v))
    }

    /// Get mid price
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some((bid, _)), Some((ask, _))) => Some((bid + ask) / 2.0),
            _ => None,
        }
    }

    /// Get spread
    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some((bid, _)), Some((ask, _))) => Some(ask - bid),
            _ => None,
        }
    }

    /// Get top N bids as Vec<(price, quantity)>
    pub fn get_bids(&self, depth: usize) -> Vec<(f64, f64)> {
        self.bids
            .iter()
            .rev()
            .take(depth)
            .map(|(&k, &v)| (self.key_to_price(k), v))
            .collect()
    }

    /// Get top N asks as Vec<(price, quantity)>
    pub fn get_asks(&self, depth: usize) -> Vec<(f64, f64)> {
        self.asks
            .iter()
            .take(depth)
            .map(|(&k, &v)| (self.key_to_price(k), v))
            .collect()
    }

    /// Calculate book imbalance (0 to 1, >0.5 means more buy pressure)
    pub fn imbalance(&self, depth: usize) -> f64 {
        let bid_volume: f64 = self.bids.iter().rev().take(depth).map(|(_, &v)| v).sum();
        let ask_volume: f64 = self.asks.iter().take(depth).map(|(_, &v)| v).sum();
        
        if bid_volume + ask_volume == 0.0 {
            0.5
        } else {
            bid_volume / (bid_volume + ask_volume)
        }
    }

    /// Get total bid volume
    pub fn total_bid_volume(&self) -> f64 {
        self.bids.values().sum()
    }

    /// Get total ask volume
    pub fn total_ask_volume(&self) -> f64 {
        self.asks.values().sum()
    }

    /// Clear the order book
    pub fn clear(&mut self) {
        self.bids.clear();
        self.asks.clear();
        self.last_update_id = 0;
    }
}
