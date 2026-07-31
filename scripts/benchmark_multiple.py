import requests
import time
import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import os

# --- SETTINGS ---
MASTER_API = "http://localhost:8545"
TOTAL_TXS = 5  # Number of distinct transactions to flood
TX_TYPE = "transfer" # Change to "transfer" for normal transactions
RESULTS_FILE = "logs/benchmark_results_multi_new.csv"

def setup_wallet_and_sign(_):
    """Creates, funds, and signs locally/via node BEFORE the timer starts."""
    try:
        # 1. Setup Wallet
        w_resp = requests.get(f"{MASTER_API}/wallet/new", timeout=5).json()
        if not w_resp.get("success"): return None
        wallet = w_resp

        # 2. Fund Wallet
        requests.post(f"{MASTER_API}/faucet/{wallet['address']}", timeout=5)

        # 3. Sign Transaction
        payload = {
            "private_key": wallet["private_key"],
            "tx_type": TX_TYPE,
            "from": wallet["address"],
            "to": "iiitd11evtre3e9l6a2yrmmyqjgxqv7pcdsqar60du47u",
            "value": 1,
            "nonce": 0,
            "data": None,
        }
        
        zkp_data = {}
        if TX_TYPE == "shielded_transfer":
            zkp_payload = {
                "receiver_address": payload["to"],
                "amount": payload["value"]
            }
            # ZKP generation takes more computational time, so increase timeout
            zkp_resp = requests.post(f"{MASTER_API}/zkp/generate_proof", json=zkp_payload, timeout=15).json()
            if not zkp_resp.get("success"): return None
            zkp_data = {
                "proof": zkp_resp["proof"],
                "commitment": zkp_resp["commitment"],
                "nullifier": zkp_resp["nullifier"]
            }

        if TX_TYPE == "shielded_transfer":
            return {
                **payload, 
                **zkp_data,
                "signature": "", 
                "public_key": ""
            }

        sign_resp = requests.post(f"{MASTER_API}/tx/sign", json=payload, timeout=5).json()
        if not sign_resp.get("success"): return None

        # Return the fully prepped, pre-signed payload
        return {
            **payload, 
            "signature": sign_resp["signature"], 
            "public_key": sign_resp["public_key"]
        }
    except Exception:
        return None

def send_tx(payload):
    """Only sends the transaction. This is the only thing timed."""
    try:
        return requests.post(f"{MASTER_API}/tx", json=payload, timeout=2).status_code == 200
    except:
        return False

def get_confirmed_nonce(address):
    try:
        return requests.get(f"{MASTER_API}/nonce/{address}", timeout=2).json().get("nonce", 0)
    except: return 0

if __name__ == "__main__":
    # ==========================================
    # 1. SETUP PHASE (UNTIMED)
    # ==========================================
    print(f"⏳ Pre-generating, funding, and signing {TOTAL_TXS} transactions...")
    with ThreadPoolExecutor(max_workers=20) as executor:
        # Generate all pre-signed payloads concurrently
        payloads = list(filter(None, executor.map(setup_wallet_and_sign, range(TOTAL_TXS))))
    
    print("⏳ Waiting 3 seconds for faucet funds to finalize on chain...")
    time.sleep(3) # Ensure the node has updated balances before we flood

    # ==========================================
    # 2. BENCHMARK PHASE (TIMED)
    # ==========================================
    print(f"🚀 Flooding {len(payloads)} pre-signed transactions...")
    start_time = time.time()
    
    # Send all transactions concurrently
    with ThreadPoolExecutor(max_workers=50) as executor:
        results = list(executor.map(send_tx, payloads))
        
    flood_duration = time.time() - start_time
    successful_sends = sum(results)
    print(f"✅ Sent: {successful_sends}/{len(payloads)} to mempool.")
    print(f"⏱️  Pure Flood Time: {flood_duration:.2f}s")

    # ==========================================
    # 3. VERIFICATION PHASE (TIMED)
    # ==========================================
    print("📊 Polling for block inclusion...")
    finalized = 0
    # Poll for a maximum of 15 seconds
    for _ in range(15): 
        time.sleep(1)
        finalized = sum(1 for p in payloads if get_confirmed_nonce(p["from"]) > 0)
        if finalized >= len(payloads):
            break
    
    total_duration = time.time() - start_time
    
    # ==========================================
    # RESULTS
    # ==========================================
    print(f"\n--- RESULTS ---")
    print(f"✅ Finalized: {finalized}/{len(payloads)} transactions.")
    
    tps = len(payloads) / flood_duration if flood_duration > 0 else 0
    print(f"⏱️  Network Flood Time: {flood_duration:.2f}s (Raw TPS: {tps:.2f})")
    print(f"⏱️  Total Time (Flood + Block Processing): {total_duration:.2f}s")
    
    # Log to CSV
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    file_exists = os.path.isfile(RESULTS_FILE)
    
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Type", "Target_TX", "Network_Total_TX", "Confirmation_Time"])
        writer.writerow([datetime.now(), TX_TYPE, TOTAL_TXS, finalized, total_duration])