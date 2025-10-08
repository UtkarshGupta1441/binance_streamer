import binance_streamer
import sys

def main():
    """
    Launches the Rust high-performance data pipeline.
    """
    # You can get the symbol from the command line, e.g., python research.py btcusdt
    symbol = sys.argv[1] if len(sys.argv) > 1 else "btcusdt"
    
    print("--- Launching Rust Trading Pipeline ---")
    print(f"Symbol: {symbol.upper()}")
    print("Press Ctrl+C to stop.")
    
    try:
        # This function will block until the Rust program is terminated
        binance_streamer.start_pipeline(symbol)
    except KeyboardInterrupt:
        print("\n--- Pipeline shutting down ---")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()