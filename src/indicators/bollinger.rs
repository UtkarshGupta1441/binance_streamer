use super::sma::calculate_sma;

/// Bollinger Bands result
#[derive(Debug, Clone, Copy)]
pub struct BollingerBands {
    pub upper: f64,
    pub middle: f64,
    pub lower: f64,
    pub bandwidth: f64,
    pub percent_b: f64,
}

/// Calculate Bollinger Bands
pub fn calculate_bollinger_bands(
    prices: &[f64],
    period: usize,
    std_dev_multiplier: f64,
) -> Option<BollingerBands> {
    if prices.len() < period || period == 0 {
        return None;
    }

    let middle = calculate_sma(prices, period)?;
    
    // Calculate standard deviation of last `period` prices
    let recent_prices: Vec<f64> = prices.iter().rev().take(period).copied().collect();
    let variance: f64 = recent_prices
        .iter()
        .map(|p| (p - middle).powi(2))
        .sum::<f64>() / period as f64;
    
    let std_dev = variance.sqrt();
    let band_width = std_dev_multiplier * std_dev;
    
    let upper = middle + band_width;
    let lower = middle - band_width;
    
    let current_price = *prices.last()?;
    let percent_b = if upper != lower {
        (current_price - lower) / (upper - lower)
    } else {
        0.5
    };

    Some(BollingerBands {
        upper,
        middle,
        lower,
        bandwidth: (upper - lower) / middle * 100.0,
        percent_b,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bollinger() {
        let prices = vec![10.0, 11.0, 10.5, 11.5, 10.8, 11.2, 10.9];
        let bb = calculate_bollinger_bands(&prices, 5, 2.0);
        assert!(bb.is_some());
        let bb = bb.unwrap();
        assert!(bb.upper > bb.middle);
        assert!(bb.middle > bb.lower);
    }
}
