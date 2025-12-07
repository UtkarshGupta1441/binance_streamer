use super::sma::calculate_sma;

/// Calculate Exponential Moving Average
pub fn calculate_ema(prices: &[f64], period: usize) -> Option<f64> {
    if prices.len() < period || period == 0 {
        return None;
    }

    let multiplier = 2.0 / (period as f64 + 1.0);
    
    // Start with SMA for initial EMA value
    let initial_sma = calculate_sma(&prices[..period], period)?;
    
    // Calculate EMA from the initial SMA
    let mut ema = initial_sma;
    for price in prices.iter().skip(period) {
        ema = (price - ema) * multiplier + ema;
    }
    
    Some(ema)
}

/// Calculate EMA series for entire price history
pub fn calculate_ema_series(prices: &[f64], period: usize) -> Vec<Option<f64>> {
    if period == 0 || prices.len() < period {
        return vec![None; prices.len()];
    }

    let multiplier = 2.0 / (period as f64 + 1.0);
    let mut result = vec![None; period - 1];
    
    // First EMA is SMA
    let initial_sma: f64 = prices[..period].iter().sum::<f64>() / period as f64;
    result.push(Some(initial_sma));
    
    let mut ema = initial_sma;
    for price in prices.iter().skip(period) {
        ema = (price - ema) * multiplier + ema;
        result.push(Some(ema));
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ema() {
        let prices = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let ema = calculate_ema(&prices, 3);
        assert!(ema.is_some());
    }
}
