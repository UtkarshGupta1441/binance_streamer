use crate::indicators::calculate_bollinger_bands;
use crate::types::StrategyResult;
use super::Strategy;

pub struct MeanReversion {
    period: usize,
    std_dev: f64,
    paper_pnl: f64,
    position: i8,
}

impl MeanReversion {
    pub fn new(period: usize, std_dev: f64) -> Self {
        Self {
            period,
            std_dev,
            paper_pnl: 0.0,
            position: 0,
        }
    }
}

impl Strategy for MeanReversion {
    fn name(&self) -> &str {
        "Mean_Reversion"
    }

    fn evaluate(&self, prices: &[f64], current_price: f64) -> StrategyResult {
        let bands = calculate_bollinger_bands(prices, self.period, self.std_dev);

        let (signal, confidence) = match bands {
            Some(bb) => {
                if current_price < bb.lower {
                    // Price below lower band - oversold, expect bounce up
                    let distance = (bb.lower - current_price) / bb.lower;
                    ("BUY".to_string(), distance.min(1.0))
                } else if current_price > bb.upper {
                    // Price above upper band - overbought, expect drop
                    let distance = (current_price - bb.upper) / bb.upper;
                    ("SELL".to_string(), distance.min(1.0))
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
