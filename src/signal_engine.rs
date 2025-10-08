use crate::common::{Side, Tick};
use ringbuf::spsc::{Consumer, Producer};
use rust_decimal::Decimal;

pub struct SignalEngine {
    total_volume: Decimal,
    total_notional: Decimal,
}

impl SignalEngine {
    pub fn new() -> Self {
        Self {
            total_volume: Decimal::ZERO,
            total_notional: Decimal::ZERO,
        }
    }

    pub fn run(&mut self, mut tick_consumer: Consumer<Tick>, mut signal_producer: Producer<(String, Tick)>) {
        println!("[Signal Engine] Running.");
        loop {
            if let Some(tick) = tick_consumer.pop() {
                if let Some(Side::Buy) | Some(Side::Sell) = tick.side {
                    self.total_volume += tick.qty;
                    self.total_notional += tick.qty * tick.price;
                    if !self.total_volume.is_zero() {
                        let vwap = self.total_notional / self.total_volume;
                        let signal = format!("VWAP: {:.4}", vwap);
                        if let Err(_) = signal_producer.push((signal, tick)) {
                            // eprintln!("[Signal Engine] Signal channel is full!");
                        }
                    }
                }
            }
        }
    }
}