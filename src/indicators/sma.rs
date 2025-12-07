/// Calculate Simple Moving Average for the most recent `period` values
pub fn calculate_sma(prices: &[f64], period: usize) -> Option<f64> {
    if prices.len() < period || period == 0 {
        return None;
    }
    
    let sum: f64 = prices.iter().rev().take(period).sum();
    Some(sum / period as f64)
}

/// Calculate SMA series for entire price history
pub fn calculate_sma_series(prices: &[f64], period: usize) -> Vec<Option<f64>> {
    if period == 0 {
        return vec![None; prices.len()];
    }
    
    let mut result = Vec::with_capacity(prices.len());
    
    for i in 0..prices.len() {
        if i + 1 < period {
            result.push(None);
        } else {
            let sum: f64 = prices[i + 1 - period..=i].iter().sum();
            result.push(Some(sum / period as f64));
        }
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sma() {
        let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(calculate_sma(&prices, 3), Some(4.0)); // (3+4+5)/3
        assert_eq!(calculate_sma(&prices, 5), Some(3.0)); // (1+2+3+4+5)/5
        assert_eq!(calculate_sma(&prices, 6), None);
    }
}
