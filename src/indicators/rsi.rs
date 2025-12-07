/// Calculate Relative Strength Index
pub fn calculate_rsi(prices: &[f64], period: usize) -> Option<f64> {
    if prices.len() < period + 1 || period == 0 {
        return None;
    }

    let mut gains = 0.0;
    let mut losses = 0.0;

    // Calculate initial average gain/loss from the most recent prices
    let start = prices.len() - period - 1;
    for i in start + 1..prices.len() {
        let change = prices[i] - prices[i - 1];
        if change > 0.0 {
            gains += change;
        } else {
            losses += change.abs();
        }
    }

    let avg_gain = gains / period as f64;
    let avg_loss = losses / period as f64;

    if avg_loss == 0.0 {
        return Some(100.0);
    }

    let rs = avg_gain / avg_loss;
    Some(100.0 - (100.0 / (1.0 + rs)))
}

/// Calculate smoothed RSI using Wilder's smoothing method
pub fn calculate_rsi_smoothed(prices: &[f64], period: usize) -> Option<f64> {
    if prices.len() < period + 1 || period == 0 {
        return None;
    }

    let mut avg_gain = 0.0;
    let mut avg_loss = 0.0;

    // First RSI uses simple average
    for i in 1..=period {
        let change = prices[i] - prices[i - 1];
        if change > 0.0 {
            avg_gain += change;
        } else {
            avg_loss += change.abs();
        }
    }
    avg_gain /= period as f64;
    avg_loss /= period as f64;

    // Smooth the averages for remaining prices
    for i in period + 1..prices.len() {
        let change = prices[i] - prices[i - 1];
        let (gain, loss) = if change > 0.0 {
            (change, 0.0)
        } else {
            (0.0, change.abs())
        };

        avg_gain = (avg_gain * (period as f64 - 1.0) + gain) / period as f64;
        avg_loss = (avg_loss * (period as f64 - 1.0) + loss) / period as f64;
    }

    if avg_loss == 0.0 {
        return Some(100.0);
    }

    let rs = avg_gain / avg_loss;
    Some(100.0 - (100.0 / (1.0 + rs)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rsi() {
        // Uptrending prices should give RSI > 50
        let prices = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
        let rsi = calculate_rsi(&prices, 5);
        assert!(rsi.is_some());
        assert!(rsi.unwrap() > 50.0);
    }
}
