use crate::indicators::calculate_rsi;
use crate::types::StrategyResult;
use super::Strategy;

pub struct MomentumStrategy {
    rsi_period: usize,
    overbought: f64,
    oversold: f64,
    paper_pnl: f64,
    position: i8,
}

impl MomentumStrategy {
    pub fn new(rsi_period: usize, overbought: f64, oversold: f64) -> Self {
        Self {
            rsi_period,
            overbought,
            oversold,
            paper_pnl: 0.0,
            position: 0,
        }
    }
}

impl Strategy for MomentumStrategy {
    fn name(&self) -> &str {
        "Momentum_RSI"
    }

    fn evaluate(&self, prices: &[f64], _current_price: f64) -> StrategyResult {
        let rsi = calculate_rsi(prices, self.rsi_period);

        let (signal, confidence) = match rsi {
            Some(rsi_val) => {
                if rsi_val < self.oversold {
                    // Oversold - potential buy
                    let conf = (self.oversold - rsi_val) / self.oversold;
                    ("BUY".to_string(), conf.min(1.0))
                } else if rsi_val > self.overbought {
                    // Overbought - potential sell
                    let conf = (rsi_val - self.overbought) / (100.0 - self.overbought);
                    ("SELL".to_string(), conf.min(1.0))
                } else {
                    ("HOLD".to_string(), 0.0)
                }
            }
            None => ("HOLD".to_string(), 0.0),
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
