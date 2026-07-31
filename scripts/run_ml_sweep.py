import subprocess
import os
import shutil
import sys
import time

# --- CONFIGURATION ---
TOTAL_NODES = 50
ROUNDS = 40
MAX_MALICIOUS = 20

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "ml_models")
STATE_DIR = os.path.join(BASE_DIR, "ml_states")
LOG_DIR = os.path.join(BASE_DIR, "logs")

def clear_state():
    """Removes previous models and states to start a fresh experiment."""
    if os.path.exists(MODEL_DIR): shutil.rmtree(MODEL_DIR)
    if os.path.exists(STATE_DIR): shutil.rmtree(STATE_DIR)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(STATE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

def get_node_list():
    return [f"node-{i}" for i in range(TOTAL_NODES)]

#########################Sequential##################################
# def run_round(malicious_nodes, csv_filename, round_num):
#     nodes = get_node_list()
#     processes = []
    
#     # We pass the environment variables so ml_engine.py knows what to do
#     env = os.environ.copy()
#     env["MALICIOUS_NODES"] = ",".join(malicious_nodes)
#     env["ML_PERFORMANCE_CSV"] = csv_filename

#     print(f"  🔄 Starting Round {round_num}...")
#     for my_id in nodes:
#         neighbors = [n for n in nodes if n != my_id]
#         cmd = [sys.executable, "ml_engine.py", my_id] + neighbors
        
#         # Run sequentially to prevent RAM exhaustion (OOM crashes)
#         try:
#             result = subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=True)
#         except subprocess.CalledProcessError as e:
#             print(f"    ❌ Error in {my_id}: {e.stderr.strip()}")

#########################Parallel##################################

def run_round(malicious_nodes, csv_filename, round_num):
    nodes = get_node_list()
    
    # We pass the environment variables so ml_engine.py knows what to do
    env = os.environ.copy()
    env["MALICIOUS_NODES"] = ",".join(malicious_nodes)
    env["ML_PERFORMANCE_CSV"] = csv_filename
    NUM_GPUS = 4  # 4 × NVIDIA RTX A6000

    print(f"  🔄 Starting Round {round_num}...")
    processes = []
    for idx, my_id in enumerate(nodes):
        neighbors = [n for n in nodes if n != my_id]
        cmd = [sys.executable, "ml_engine.py", my_id] + neighbors
        node_env = env.copy()
        node_env["CUDA_VISIBLE_DEVICES"] = str(idx % NUM_GPUS)
        
        # Run in parallel to utilize the big machine
        # p = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        p = subprocess.Popen(cmd, env=node_env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        processes.append((my_id, p))
        
    # Wait for all nodes to complete their round concurrently
    for my_id, p in processes:
        _, stderr = p.communicate()
        if p.returncode != 0:
            print(f"    ❌ Error in {my_id}: {stderr.strip()}")


def run_experiment(num_malicious):
    print(f"\n🚀 --- STARTING EXPERIMENT: {num_malicious} Malicious Node(s) ---")
    clear_state()
    
    # Select malicious nodes (e.g., node-0, node-1...)
    nodes = get_node_list()
    malicious_nodes = nodes[:num_malicious]
    
    csv_filename = f"ml_performance_{num_malicious}_malicious.csv"
    csv_path = os.path.join(LOG_DIR, csv_filename)
    
    # Remove the specific log if it exists from a previous run
    if os.path.exists(csv_path):
        os.remove(csv_path)
        
    print(f"🛡️ Malicious Nodes: {malicious_nodes}")
    print(f"📄 Output File: {csv_path}")
    
    for r in range(1, ROUNDS + 1):
        run_round(malicious_nodes, csv_filename, r)
        
    print(f"✅ Finished Experiment with {num_malicious} malicious nodes.")

if __name__ == "__main__":
    start_time = time.time()
    
    for m in range(1, MAX_MALICIOUS + 1):
        run_experiment(m)
        
    total_time = time.time() - start_time
    print(f"\n🎉 ALL EXPERIMENTS COMPLETE in {total_time:.1f} seconds!")
# import subprocess
# import os
# import shutil
# import sys
# import time
# from concurrent.futures import ThreadPoolExecutor

# # --- CONFIGURATION ---
# TOTAL_NODES = 50
# ROUNDS = 40
# MAX_CONCURRENT_WORKERS = 10
# NEIGHBOR_DEGREE = 5  # Sparse graph: each node connects to 2*k neighbors

# # Malicious nodes scale representing 0%, 10%, 20%, 30%, 40%, 50%, 60%
# # For 50 nodes, this is 0, 5, 10, 15, 20, 25, 30
# MALICIOUS_SWEEP = [0, 5, 10, 15, 20, 25, 30]

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_DIR = os.path.join(BASE_DIR, "ml_models")
# STATE_DIR = os.path.join(BASE_DIR, "ml_states")
# LOG_DIR = os.path.join(BASE_DIR, "logs")

# def clear_state():
#     """Removes previous models and states to start a fresh experiment."""
#     if os.path.exists(MODEL_DIR): shutil.rmtree(MODEL_DIR)
#     if os.path.exists(STATE_DIR): shutil.rmtree(STATE_DIR)
#     os.makedirs(MODEL_DIR, exist_ok=True)
#     os.makedirs(STATE_DIR, exist_ok=True)
#     os.makedirs(LOG_DIR, exist_ok=True)

# def get_node_list():
#     return [f"node-{i}" for i in range(TOTAL_NODES)]

# def get_sparse_neighbors(my_id, total_nodes, degree):
#     """Generates a k-regular ring topology sparse graph."""
#     my_idx = int(my_id.split('-')[1])
#     neighbors = []
#     for offset in range(1, degree + 1):
#         # Forward neighbor
#         neighbors.append(f"node-{(my_idx + offset) % total_nodes}")
#         # Backward neighbor
#         neighbors.append(f"node-{(my_idx - offset) % total_nodes}")
#     # Remove duplicates and self
#     return list(set(neighbors) - {my_id})

# def run_phase_for_node(cmd, env, my_id, phase):
#     """Runs a specific phase of ml_engine.py for a node."""
#     try:
#         subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=True, timeout=300)
#     except subprocess.CalledProcessError as e:
#         print(f"    ❌ Error in {my_id} (Phase {phase}): {e.stderr.strip()}")
#     except subprocess.TimeoutExpired:
#         print(f"    ⏳ Timeout: {my_id} hung in Phase {phase} and was killed!")

# def run_round(malicious_nodes, csv_filename, round_num):
#     nodes = get_node_list()
#     env = os.environ.copy()
#     env["MALICIOUS_NODES"] = ",".join(malicious_nodes)
#     env["ML_PERFORMANCE_CSV"] = csv_filename

#     print(f"  🔄 Starting Round {round_num}...")
    
#     # ----------------------------------------------------
#     # PHASE 1: Training (Memory intensive, batched execution)
#     # ----------------------------------------------------
#     phase1_cmds = []
#     for my_id in nodes:
#         neighbors = get_sparse_neighbors(my_id, TOTAL_NODES, NEIGHBOR_DEGREE)
#         cmd = [sys.executable, "ml_engine.py", my_id, "--phase", "1"] + neighbors
#         phase1_cmds.append((cmd, my_id))
        
#     with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS) as executor:
#         futures = [executor.submit(run_phase_for_node, cmd, env, my_id, 1) for cmd, my_id in phase1_cmds]
#         # Wait for all nodes to complete Phase 1
#         for f in futures: f.result()
        
#     # ----------------------------------------------------
#     # PHASE 2: Aggregation (Relies on all deltas existing)
#     # ----------------------------------------------------
#     phase2_cmds = []
#     for my_id in nodes:
#         neighbors = get_sparse_neighbors(my_id, TOTAL_NODES, NEIGHBOR_DEGREE)
#         cmd = [sys.executable, "ml_engine.py", my_id, "--phase", "2"] + neighbors
#         phase2_cmds.append((cmd, my_id))
        
#     with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS) as executor:
#         futures = [executor.submit(run_phase_for_node, cmd, env, my_id, 2) for cmd, my_id in phase2_cmds]
#         # Wait for all nodes to complete Phase 2
#         for f in futures: f.result()

# import torchvision.datasets as datasets

# def run_experiment(num_malicious):
#     print(f"\n🚀 --- STARTING EXPERIMENT: {num_malicious} Malicious Node(s) ({(num_malicious/TOTAL_NODES)*100:.0f}%) ---")
#     clear_state()
    
#     # --- PRE-DOWNLOAD DATASET ONCE ---
#     print("📥 Pre-downloading CIFAR-10 dataset sequentially to prevent concurrent timeouts...")
#     data_dir = os.path.join(BASE_DIR, "data")
#     datasets.CIFAR10(root=data_dir, train=True, download=True)
    
#     nodes = get_node_list()
#     malicious_nodes = nodes[:num_malicious]
    
#     csv_filename = f"ml_performance_{num_malicious}_malicious.csv"
#     csv_path = os.path.join(LOG_DIR, csv_filename)
#     if os.path.exists(csv_path):
#         os.remove(csv_path)
        
#     print(f"🛡️ Malicious Nodes: {len(malicious_nodes)}")
#     print(f"📄 Output File: {csv_filename}")
    
#     for r in range(1, ROUNDS + 1):
#         run_round(malicious_nodes, csv_filename, r)
        
#     print(f"✅ Finished Experiment with {num_malicious} malicious nodes.")

# if __name__ == "__main__":
#     start_time = time.time()
    
#     for m in MALICIOUS_SWEEP:
#         run_experiment(m)
        
#     total_time = time.time() - start_time
#     print(f"\n🎉 ALL EXPERIMENTS COMPLETE in {total_time:.1f} seconds!")
