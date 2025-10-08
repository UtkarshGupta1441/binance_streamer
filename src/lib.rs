use pyo3::prelude::*;
use ringbuf::spsc::channel as spsc_channel;
use std::thread;

// Declare the modules
pub mod common;
pub mod mdfeed;
pub mod signal_engine;
pub mod router;
pub mod storage;

use common::Tick;

#[pymodule]
fn binance_streamer(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(start_pipeline, m)?)?;
    Ok(())
}

#[pyfunction]
fn start_pipeline(symbol: String) -> PyResult<()> {
    // --- Create IPC channels for the pipeline ---
    // Channel from mdfeed to the rest of the system
    let (tick_producer, tick_consumer) = spsc_channel::<Tick>(8192);
    // Channel from signal engine to the router
    let (signal_producer, signal_consumer) = spsc_channel::<(String, Tick)>(1024);
    // Channel from mdfeed to the storage engine
    let (storage_producer, storage_consumer) = spsc_channel::<Tick>(8192);

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