use pyo3::prelude::*;
use crate::strategies::{Strategy, TrendFollower, MeanReversion, MomentumStrategy};
use crate::types::StrategyResult;

/// Strategy Manager - evaluates multiple strategies and tracks paper PnL
/// This is the main interface for Python to interact with the trading strategies
#[pyclass]
pub struct StrategyManager {
    trend_follower: TrendFollower,
    mean_reversion: MeanReversion,
    momentum: MomentumStrategy,
    price_history: Vec<f64>,
    last_price: f64,
    max_history: usize,
}

#[pymethods]
impl StrategyManager {
    #[new]
    #[pyo3(signature = (short_period=10, long_period=20, bb_period=20, bb_std=2.0, rsi_period=14))]
    pub fn new(
        short_period: usize,
        long_period: usize,
        bb_period: usize,
        bb_std: f64,
        rsi_period: usize,
    ) -> Self {
        Self {
            trend_follower: TrendFollower::new(short_period, long_period),
            mean_reversion: MeanReversion::new(bb_period, bb_std),
            momentum: MomentumStrategy::new(rsi_period, 70.0, 30.0),
            price_history: Vec::with_capacity(200),
            last_price: 0.0,
            max_history: 200,
        }
    }

    /// Add a new price and update all strategies
    /// Returns a list of StrategyResult objects
    pub fn update(&mut self, price: f64) -> Vec<StrategyResult> {
        // Update price history
        self.price_history.push(price);
        if self.price_history.len() > self.max_history {
            self.price_history.remove(0);
        }

        // Update PnL based on price change
        if self.last_price > 0.0 {
            let change = price - self.last_price;
            self.trend_follower.update_pnl(change);
            self.mean_reversion.update_pnl(change);
            self.momentum.update_pnl(change);
        }
        self.last_price = price;

        // Evaluate all strategies and return results
        vec![
            self.trend_follower.evaluate(&self.price_history, price),
            self.mean_reversion.evaluate(&self.price_history, price),
            self.momentum.evaluate(&self.price_history, price),
        ]
    }

    /// Get the best performing strategy by paper PnL
    /// Returns (strategy_name, paper_pnl)
    pub fn get_best_strategy(&self) -> (String, f64) {
        let strategies = vec![
            (self.trend_follower.name(), self.trend_follower.get_paper_pnl()),
            (self.mean_reversion.name(), self.mean_reversion.get_paper_pnl()),
            (self.momentum.name(), self.momentum.get_paper_pnl()),
        ];

        strategies
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(name, pnl)| (name.to_string(), pnl))
            .unwrap_or(("None".to_string(), 0.0))
    }

    /// Get all paper PnLs as a list of (name, pnl) tuples
    pub fn get_all_pnl(&self) -> Vec<(String, f64)> {
        vec![
            (self.trend_follower.name().to_string(), self.trend_follower.get_paper_pnl()),
            (self.mean_reversion.name().to_string(), self.mean_reversion.get_paper_pnl()),
            (self.momentum.name().to_string(), self.momentum.get_paper_pnl()),
        ]
    }

    /// Reset all strategies (clear PnL and positions)
    pub fn reset(&mut self) {
        self.trend_follower.reset();
        self.mean_reversion.reset();
        self.momentum.reset();
        self.price_history.clear();
        self.last_price = 0.0;
    }

    /// Get current price history length
    pub fn history_length(&self) -> usize {
        self.price_history.len()
    }

    /// Get last recorded price
    pub fn get_last_price(&self) -> f64 {
        self.last_price
    }

    /// Get the price history as a list
    pub fn get_price_history(&self) -> Vec<f64> {
        self.price_history.clone()
    }
}
