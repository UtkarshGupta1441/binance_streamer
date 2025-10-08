use crate::common::Tick;
use memmap2::MmapMut;
use ringbuf::spsc::Consumer;
use std::fs::{File, OpenOptions};
use std::mem;
use std::time::{Duration, Instant, SystemTime};

const SHARED_MEM_PATH: &str = "./data/realtime_tick.mmap";

#[repr(C)]
struct SharedTick {
    price: f64,
    qty: f64,
    timestamp: u64,
}

pub struct StorageEngine {
    mmap: MmapMut,
    // Parquet logic would go here
}

impl StorageEngine {
    pub fn new(symbol: &str) -> Self {
        let _ = std::fs::create_dir_all("./data");
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(SHARED_MEM_PATH)
            .unwrap();
        file.set_len(mem::size_of::<SharedTick>() as u64).unwrap();
        let mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
        println!("[Storage Engine] Memory-mapped file created at {}", SHARED_MEM_PATH);
        Self { mmap }
    }

    pub fn run(&mut self, mut storage_consumer: Consumer<Tick>) {
        println!("[Storage Engine] Running.");
        loop {
            if let Some(tick) = storage_consumer.pop() {
                self.write_to_mmap(&tick);
                // In a real system, you would batch ticks here and write to Parquet
                // periodically or when a buffer is full.
            } else {
                // Sleep briefly if the channel is empty to avoid busy-waiting
                std::thread::sleep(Duration::from_micros(100));
            }
        }
    }

    fn write_to_mmap(&mut self, tick: &Tick) {
        let shared_tick = SharedTick {
            price: tick.price.try_into().unwrap_or(0.0),
            qty: tick.qty.try_into().unwrap_or(0.0),
            timestamp: tick.wall_clock_time.duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos() as u64,
        };

        unsafe {
            let bytes: &[u8] = std::slice::from_raw_parts(
                (&shared_tick as *const SharedTick) as *const u8,
                mem::size_of::<SharedTick>(),
            );
            self.mmap[..bytes.len()].copy_from_slice(bytes);
        }
    }
}