use crate::common::Tick;
use hdrhistogram::Histogram;
use ringbuf::traits::Consumer;
use std::time::Instant;

pub struct Router {
    latency_recorder: Histogram<u64>,
}

impl Router {
    pub fn new() -> Self {
        Self {
            latency_recorder: Histogram::new_with_bounds(1, 10_000_000_000, 3).unwrap(),
        }
    }

    pub fn run(&mut self, mut signal_consumer: impl Consumer<Item = (String, Tick)>) {
        println!("[Router] Running.");
        let mut last_print = Instant::now();

        loop {
            if let Some((_signal, tick)) = signal_consumer.try_pop() {
                let latency_ns = tick.machine_time.elapsed().as_nanos() as u64;
                self.latency_recorder.record(latency_ns).unwrap();
            }

            if last_print.elapsed().as_secs() >= 5 {
                self.print_latency_summary();
                last_print = Instant::now();
            }
        }
    }

    fn print_latency_summary(&self) {
        if self.latency_recorder.is_empty() { return; }
        println!(
            "\n--- Realtime Lane Latency (End-to-End) ---\n p50: {:>5} µs | p99: {:>5} µs | p999: {:>5} µs\n--------------------------------------------",
            self.latency_recorder.value_at_percentile(50.0) / 1000,
            self.latency_recorder.value_at_percentile(99.0) / 1000,
            self.latency_recorder.value_at_percentile(99.9) / 1000,
        );
    }
}