use crate::common::{BinanceTrade, Side, Tick};
use futures_util::StreamExt;
use ringbuf::traits::Producer;
use std::time::{Instant, SystemTime};
use tokio::runtime::Runtime;
use tokio_tungstenite::connect_async;

pub fn run_md_feed(
    symbol: String,
    mut hot_path_prod: impl Producer<Item = Tick>,
    mut warm_path_prod: impl Producer<Item = Tick>,
) {
    let rt = Runtime::new().expect("Failed to create Tokio runtime");
    rt.block_on(async {
        let trade_url = format!("wss://stream.binance.com:9443/ws/{}@trade", symbol.to_lowercase());

        let (ws_stream, _) = connect_async(&trade_url)
            .await
            .expect("Failed to connect to trade stream");

        println!("[MD Feed] Connected to {} trade stream.", symbol);
        let (_, mut read) = ws_stream.split();

        while let Some(Ok(msg)) = read.next().await {
            if let Ok(text) = msg.to_text() {
                if let Ok(trade) = serde_json::from_str::<BinanceTrade>(text) {
                    let machine_time = Instant::now();
                    let wall_clock_time = SystemTime::now();

                    let tick = Tick {
                        symbol: trade.symbol.clone(),
                        side: Some(if trade.is_buyer_maker { Side::Sell } else { Side::Buy }),
                        price: trade.price,
                        qty: trade.qty,
                        machine_time,
                        wall_clock_time,
                    };

                    // Send to both hot and warm path channels
                    if let Err(_) = hot_path_prod.try_push(tick.clone()) {
                        // eprintln!("[MD Feed] Hot path channel is full!");
                    }
                    if let Err(_) = warm_path_prod.try_push(tick) {
                        // eprintln!("[MD Feed] Warm path channel is full!");
                    }
                }
            }
        }
    });
}