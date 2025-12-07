use crate::indicators::calculate_ema;
use crate::types::StrategyResult;
use super::Strategy;

pub struct TrendFollower {
    short_period: usize,
    long_period: usize,
    paper_pnl: f64,
    position: i8, // 1 = long, -1 = short, 0 = flat
}

impl TrendFollower {
    pub fn new(short_period: usize, long_period: usize) -> Self {
        Self {
            short_period,
            long_period,
            paper_pnl: 0.0,
            position: 0,
        }
    }
}

impl Strategy for TrendFollower {
    fn name(&self) -> &str {
        "Trend_Follower"
    }

    fn evaluate(&self, prices: &[f64], current_price: f64) -> StrategyResult {
        let short_ema = calculate_ema(prices, self.short_period);
        let long_ema = calculate_ema(prices, self.long_period);

        let (signal, confidence) = match (short_ema, long_ema) {
            (Some(short), Some(long)) => {
                let diff_pct = (short - long).abs() / long;
                
                if short > long && current_price > short {
                    // Uptrend: short EMA above long EMA, price above short EMA
                    ("BUY".to_string(), diff_pct.min(1.0))
                } else if short < long && current_price < short {
                    // Downtrend: short EMA below long EMA, price below short EMA
                    ("SELL".to_string(), diff_pct.min(1.0))
                } else {
                    ("HOLD".to_string(), 0.0)
                }
            }
            _ => ("HOLD".to_string(), 0.0),
        };

        StrategyResult {
            name: self.name().to_string(),
            signal,
            confidence,
            paper_pnl: self.paper_pnl,
        }
    }

    fn update_pnl(&mut self, price_change: f64) {
        self.paper_pnl += self.position as f64 * price_change;
    }

    fn get_paper_pnl(&self) -> f64 {
        self.paper_pnl
    }

    fn get_position(&self) -> i8 {
        self.position
    }

    fn reset(&mut self) {
        self.paper_pnl = 0.0;
        self.position = 0;
    }
}
