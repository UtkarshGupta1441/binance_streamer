use rust_decimal::Decimal;
use serde::Deserialize;
use std::time::{Instant, SystemTime};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

#[derive(Debug, Clone)]
pub struct Tick {
    pub symbol: String,
    pub side: Option<Side>,
    pub price: Decimal,
    pub qty: Decimal,
    pub machine_time: Instant,
    pub wall_clock_time: SystemTime,
}

// Struct to deserialize raw trade messages from Binance
#[derive(Deserialize, Debug)]
pub struct BinanceTrade {
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "p")]
    pub price: Decimal,
    #[serde(rename = "q")]
    pub qty: Decimal,
    #[serde(rename = "m")]
    pub is_buyer_maker: bool,
}