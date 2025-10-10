use crate::common::Tick;
use csv::{Writer, WriterBuilder};
use memmap2::MmapMut;
use ringbuf::traits::Consumer;
use std::fs::{File, OpenOptions};
use std::mem;
use std::time::{Duration, SystemTime};

const SHARED_MEM_PATH: &str = "./data/realtime_tick.mmap";
const WARM_STORAGE_PATH: &str = "./data/trades.csv";

#[repr(C)]
struct SharedTick {
    price: f64,
    qty: f64,
    timestamp: u64,
}

pub struct StorageEngine {
    mmap: MmapMut,
    csv_writer: Writer<File>,
}

impl StorageEngine {
    pub fn new(_symbol: &str) -> Self {
        let _ = std::fs::create_dir_all("./data");

        // --- Hot Path: Memory-mapped file setup ---
        let mmap_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(SHARED_MEM_PATH)
            .unwrap();
        mmap_file.set_len(mem::size_of::<SharedTick>() as u64).unwrap();
        let mmap = unsafe { MmapMut::map_mut(&mmap_file).unwrap() };
        println!("[Storage Engine] Memory-mapped file created at {}", SHARED_MEM_PATH);

        // --- Warm Path: CSV file setup ---
        let file_exists = std::path::Path::new(WARM_STORAGE_PATH).exists();
        let csv_file = OpenOptions::new()
            .write(true)
            .create(true)
            .append(true)
            .open(WARM_STORAGE_PATH)
            .unwrap();

        let csv_writer = WriterBuilder::new()
            .has_headers(!file_exists)
            .from_writer(csv_file);

        println!("[Storage Engine] CSV warm storage ready at {}", WARM_STORAGE_PATH);

        Self { mmap, csv_writer }
    }

    pub fn run(&mut self, mut storage_consumer: impl Consumer<Item = Tick>) {
        println!("[Storage Engine] Running.");
        loop {
            if let Some(tick) = storage_consumer.try_pop() {
                // Hot path: write to shared memory
                self.write_to_mmap(&tick);
                // Warm path: write to CSV
                self.write_to_csv(&tick);
            } else {
                // Sleep briefly if the channel is empty to avoid busy-waiting
                std::thread::sleep(Duration::from_micros(100));
            }
        }
    }

    fn write_to_csv(&mut self, tick: &Tick) {
        self.csv_writer.serialize(tick).unwrap();
        self.csv_writer.flush().unwrap();
    }

    fn write_to_mmap(&mut self, tick: &Tick) {
        let shared_tick = SharedTick {
            price: tick.price.try_into().unwrap_or(0.0),
            qty: tick.qty.try_into().unwrap_or(0.0),
            timestamp: tick
                .wall_clock_time
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
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