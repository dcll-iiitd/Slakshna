import subprocess
import time
import os
import sys

BENCHMARK_SCRIPT = "benchmark_multiple.py"
TX_LOADS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

ITERATIONS_PER_LOAD = 3

def set_total_txs(script_path, new_total):
    with open(script_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    with open(script_path, "w", encoding="utf-8") as f:
        for line in lines:
            if line.startswith("TOTAL_TXS ="):
                f.write(f"TOTAL_TXS = {new_total}  # Number of distinct transactions to flood\n")
            else:
                f.write(line)

def run_sweep():
    print(f"🚀 Starting Benchmark Sweep over loads: {TX_LOADS}")
    print(f"🔄 Each load will run {ITERATIONS_PER_LOAD} times to calculate average latency.")
    for load in TX_LOADS:
        print(f"\n--- Running Benchmark for TOTAL_TXS = {load} ---")
        set_total_txs(BENCHMARK_SCRIPT, load)
        
        for iteration in range(1, ITERATIONS_PER_LOAD + 1):
            print(f"  -> Iteration {iteration}/{ITERATIONS_PER_LOAD} for load {load}")
            # Run the benchmark script
            try:
                result = subprocess.run([sys.executable, BENCHMARK_SCRIPT], check=True, text=True)
            except subprocess.CalledProcessError as e:
                print(f"❌ Benchmark failed for load {load}, iteration {iteration}: {e}")
            
            # Cooldown between runs to let the network stabilize
            if iteration < ITERATIONS_PER_LOAD:
                print("     ⏳ Cooldown (5s) before next iteration...")
                time.sleep(5)
        
        print(f"✅ Finished all iterations for load {load}")
        print("⏳ Waiting 15 seconds before moving to the next load level...")
        time.sleep(15)

if __name__ == "__main__":
    # Ensure RESULTS_FILE has correct headers or is clean if you want fresh results.
    # We will append to existing results or the benchmark script will create it.
    run_sweep()
    print("\n🎉 Sweep complete! You can now plot the results.")
