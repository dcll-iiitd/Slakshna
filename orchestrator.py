import copy
import subprocess
import os
import time
import toml
import shutil

# --- CONFIGURATION ---
BASE_PORT_P2P = 9000
BASE_PORT_API = 8545
NODE_BINARY = "./target/release/iiitd"
CONFIG_DIR = "./benchmark_configs"
DATA_DIR = "./benchmark_data"

def setup_nodes(num_nodes):
    if os.path.exists(CONFIG_DIR): shutil.rmtree(CONFIG_DIR)
    if os.path.exists(DATA_DIR): shutil.rmtree(DATA_DIR)
    os.makedirs(CONFIG_DIR)
    os.makedirs(DATA_DIR)

    processes = []
    
    # Load your base template
    with open("config.toml", "r") as f:
        base_config = toml.load(f)
        
    os.makedirs("./logs", exist_ok=True)

    for i in range(num_nodes):
        node_id = f"node-{i}"
        node_config = copy.deepcopy(base_config)

        # Update unique fields
        node_config["node"]["id"] = node_id
        node_config["node"]["data_dir"] = f"{DATA_DIR}/{node_id}"
        node_config["network"]["p2p_port"] = BASE_PORT_P2P + i
        node_config["network"]["api_port"] = BASE_PORT_API + i

        # No seed peers and no designated first node: every node advertises
        # itself over mDNS and finds the others on its own.
        node_config["network"]["peers"] = []
        node_config.setdefault("discovery", {})["mdns"] = True

        config_path = f"{CONFIG_DIR}/{node_id}.toml"
        with open(config_path, "w") as f:
            toml.dump(node_config, f)

        print(f"🚀 Starting {node_id} on API port {BASE_PORT_API + i}...")
        proc = subprocess.Popen([NODE_BINARY, "--config", config_path])
        processes.append(proc)
        
    return processes

if __name__ == "__main__":
    count = int(input("Enter number of nodes to run: "))
    procs = setup_nodes(count)
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        for p in procs: p.terminate()
        print("\n🛑 All nodes shut down.")