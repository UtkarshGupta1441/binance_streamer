pub mod trend_follower;
pub mod mean_reversion;
pub mod momentum;

pub use trend_follower::TrendFollower;
pub use mean_reversion::MeanReversion;
pub use momentum::MomentumStrategy;

use crate::types::StrategyResult;

/// Trait that all strategies must implement
pub trait Strategy: Send + Sync {
    fn name(&self) -> &str;
    fn evaluate(&self, prices: &[f64], current_price: f64) -> StrategyResult;
    fn update_pnl(&mut self, price_change: f64);
    fn get_paper_pnl(&self) -> f64;
    fn get_position(&self) -> i8;
    fn reset(&mut self);
}
