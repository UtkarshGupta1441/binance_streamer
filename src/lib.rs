use pyo3::prelude::*;
use ringbuf::{traits::Split, HeapRb};
use std::thread;

// Declare the existing modules
pub mod common;
pub mod mdfeed;
pub mod signal_engine;
pub mod router;
pub mod storage;

// New trading modules
pub mod types;
pub mod orderbook;
pub mod indicators;
pub mod strategies;
pub mod executor;
pub mod strategy_manager;

use common::Tick;

// Re-export new classes for Python
pub use types::{Signal, StrategyResult, TradeOrder};
pub use orderbook::OrderBook;
pub use executor::OrderExecutor;
pub use strategy_manager::StrategyManager;

#[pymodule]
fn binance_streamer(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Existing function
    m.add_function(wrap_pyfunction!(start_pipeline, m)?)?;
    
    // New trading classes
    m.add_class::<OrderBook>()?;
    m.add_class::<OrderExecutor>()?;
    m.add_class::<StrategyManager>()?;
    m.add_class::<StrategyResult>()?;
    m.add_class::<TradeOrder>()?;
    m.add_class::<Signal>()?;
    
    Ok(())
}

#[pyfunction]
fn start_pipeline(symbol: String) -> PyResult<()> {
    // --- Create IPC channels for the pipeline ---
    // Channel from mdfeed to the rest of the system
    let (tick_producer, tick_consumer) = HeapRb::<Tick>::new(8192).split();
    // Channel from signal engine to the router
    let (signal_producer, signal_consumer) = HeapRb::<(String, Tick)>::new(1024).split();
    // Channel from mdfeed to the storage engine
    let (storage_producer, storage_consumer) = HeapRb::<Tick>::new(8192).split();

    // --- Launch pipeline components in separate threads ---

    let symbol_clone = symbol.clone();
    let storage_thread = thread::spawn(move || {
        let mut storage_engine = storage::StorageEngine::new(&symbol_clone);
        storage_engine.run(storage_consumer);
    });

    let signal_thread = thread::spawn(move || {
        let mut engine = signal_engine::SignalEngine::new();
        engine.run(tick_consumer, signal_producer);
    });

    let router_thread = thread::spawn(move || {
        let mut router = router::Router::new();
        router.run(signal_consumer);
    });

    // Run the market data feed in the main thread
    mdfeed::run_md_feed(symbol, tick_producer, storage_producer);

    // Wait for threads to complete (they run indefinitely in this design)
    storage_thread.join().expect("Storage thread panicked");
    signal_thread.join().expect("Signal engine thread panicked");
    router_thread.join().expect("Router thread panicked");

    Ok(())
}