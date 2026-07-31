import requests
import time
import csv
import threading
from datetime import datetime
import os
import rsa
import json

# --- SETTINGS ---
MASTER_API = "http://localhost:8545"
TOTAL_TXS = 5  # Number of txns to flood
FROM_ADDRESS = "iiitd11esxnfgm7ch8f2n3lhjxaelss3erlgvac2zypa0"

# --- NEW SETTINGS FOR COMPARISON ---
TX_MODE = "shielded"  # Change to "shielded" to test ZKP transactions again
PRIVATE_KEY = "0818fc072ef93e5bb52ba738a9e6a9a4306e820e5522a128194b4eee20cba2dd"  # REQUIRED for "normal" mode

RESULTS_FILE = "logs/benchmark_results_exp2.csv"


def get_pending_nonce(address):
    try:
        resp = requests.get(f"{MASTER_API}/nonce/pending/{address}").json()
        return resp.get("pending_nonce", 0)
    except Exception as e:
        print(f"⚠️ Could not fetch pending nonce: {e}")
        return 0


def get_confirmed_nonce(address):
    try:
        resp = requests.get(f"{MASTER_API}/nonce/{address}").json()
        return resp.get("nonce", 0)
    except Exception as e:
        print(f"⚠️ Could not fetch confirmed nonce: {e}")
        return 0


def get_balance(address):
    try:
        resp = requests.get(f"{MASTER_API}/balance/{address}").json()
        return float(resp.get("balance", 0))
    except Exception as e:
        print(f"⚠️ Could not fetch balance: {e}")
        return 0


def prepare_normal_payload(index):
    if PRIVATE_KEY == "0818fc02ef93e5bb52ba738a9e6a9a4306e820e5522a128194b4eee20cba2dd":
        print("❌ ERROR: You must provide a PRIVATE_KEY in the script for normal transactions.")
        exit(1)

    sign_payload = {
        "private_key": PRIVATE_KEY,
        "tx_type": "transfer",
        "from": FROM_ADDRESS,
        "to": "iiitd11evtre3e9l6a2yrmmyqjgxqv7pcdsqar60du47u",
        "value": 1,
        "nonce": index,
        "data": None,
    }
    sign_resp = requests.post(f"{MASTER_API}/tx/sign", json=sign_payload, timeout=5).json()
    if not sign_resp.get("success"):
        return None

    return {
        "tx_type": "transfer",
        "from": FROM_ADDRESS,
        "to": "iiitd11evtre3e9l6a2yrmmyqjgxqv7pcdsqar60du47u",
        "value": 1,
        "nonce": index,
        "data": None,
        "signature": sign_resp["signature"],
        "public_key": sign_resp["public_key"],
    }


def flood_network():
    start_nonce = get_pending_nonce(FROM_ADDRESS)
    initial_confirmed_nonce = get_confirmed_nonce(FROM_ADDRESS)

    print(f"🚀 Starting {TX_MODE.upper()} benchmark from pending nonce {start_nonce}")

    if TX_MODE == "shielded":
        print(f"\n🛡️ --- PHASE 1: SHIELDING (DEPOSIT) ---")
        print(f"Generating Bob's Shielded Wallet (RSA Keypair)...")
        (bob_pub, bob_priv) = rsa.newkeys(2048)

        print(f"Generating {TOTAL_TXS} Secret Commitments & Encrypting Notes...")
        prep_start = time.time()
        shield_payloads = []
        
        for i in range(TOTAL_TXS):
            # 1. Generate Commitment Cryptography
            resp = requests.post(f"{MASTER_API}/zkp/generate_commitment", timeout=15).json()
            
            # Encrypt the secret using Bob's Shielded Public Key
            secret_data = json.dumps({
                "secret": resp["secret"],
                "nullifier_salt": resp["nullifier_salt"]
            }).encode('utf-8')
            encrypted_note = rsa.encrypt(secret_data, bob_pub).hex()
            
            # 2. Sign Shield TX
            sign_payload = {
                "private_key": PRIVATE_KEY,
                "tx_type": "shield",
                "from": FROM_ADDRESS,
                "to": "",
                "value": 1,
                "nonce": start_nonce + i,
                "data": None,
                "commitment": resp["commitment"],
                "encrypted_note": encrypted_note,
            }
            sign_resp = requests.post(f"{MASTER_API}/tx/sign", json=sign_payload, timeout=5).json()
            shield_payloads.append({
                "tx_type": "shield",
                "from": FROM_ADDRESS,
                "to": "",
                "value": 1,
                "nonce": start_nonce + i,
                "data": None,
                "signature": sign_resp["signature"],
                "public_key": sign_resp["public_key"],
                "commitment": resp["commitment"],
                "encrypted_note": encrypted_note,
            })
            
        prep_duration = time.time() - prep_start
        
        print(f"🌊 Flooding {TOTAL_TXS} Shield TXs to the network...")
        for payload in shield_payloads:
            resp = requests.post(f"{MASTER_API}/tx", json=payload, timeout=5)
            if not resp.json().get("success"):
                print(f"❌ Shield TX Failed: {resp.json()}")
            time.sleep(0.05)
            
        print("⏳ Waiting 10 seconds for Shield transactions to finalize into the Merkle Tree...")
        time.sleep(10)
        
        print(f"\n🔓 --- PHASE 2: UNSHIELDING (WITHDRAWAL) ---")
        print("Scanning blockchain for encrypted notes...")
        shielded_notes_resp = requests.get(f"{MASTER_API}/shielded-notes?limit=100", timeout=10).json()
        notes = shielded_notes_resp.get("notes", [])
        
        secrets = []
        for note in notes:
            try:
                cipher = bytes.fromhex(note["encrypted_note"])
                decrypted = rsa.decrypt(cipher, bob_priv)
                secret_data = json.loads(decrypted.decode('utf-8'))
                # Prevent reusing the same secrets if there are many on the chain
                if len(secrets) < TOTAL_TXS:
                    secrets.append(secret_data)
            except Exception:
                pass # Not for us
                
        print(f"✅ Successfully decrypted {len(secrets)} notes belonging to Bob!")
        
        print(f"Generating {len(secrets)} ZK Proofs...")
        unshield_prep_start = time.time()
        unshield_payloads = []
        
        for i in range(len(secrets)):
            prove_req = {
                "secret": secrets[i]["secret"],
                "nullifier_salt": secrets[i]["nullifier_salt"],
            }
            zkp_resp = requests.post(f"{MASTER_API}/zkp/generate_proof", json=prove_req, timeout=25).json()
            if not zkp_resp.get("success"):
                print(f"❌ ZKP Generation failed: {zkp_resp}")
                continue
                
            unshield_payloads.append({
                "tx_type": "unshield",
                "from": "",
                "to": "iiitd11evtre3e9l6a2yrmmyqjgxqv7pcdsqar60du47u",
                "value": 1,
                "nonce": i,
                "data": None,
                "signature": "",
                "public_key": zkp_resp["expected_root"], # The Anchor Root
                "proof": zkp_resp["proof"],
                "nullifier": zkp_resp["nullifier"]
            })
            
        prep_duration += (time.time() - unshield_prep_start)
        
        print(f"🌊 Flooding {TOTAL_TXS} Anonymous Unshield TXs...")
        flood_start = time.time()
        for payload in unshield_payloads:
            requests.post(f"{MASTER_API}/tx", json=payload, timeout=5)
            time.sleep(0.05)
            
        flood_duration = time.time() - flood_start
        return flood_start, initial_confirmed_nonce, prep_duration, flood_duration

    else:
        # NORMAL MODE
        print(f"⏳ Pre-generating {TOTAL_TXS} payloads...")
        prep_start_time = time.time()
        payloads = []
        for i in range(TOTAL_TXS):
            payload = prepare_normal_payload(i + start_nonce)
            if payload:
                payloads.append(payload)
        prep_duration = time.time() - prep_start_time
        print(f"✅ Payload prep took {prep_duration:.2f} seconds.")

        print(f"🌊 Flooding network with {len(payloads)} prepared transactions...")
        flood_start_time = time.time()

        for payload in payloads:
            try:
                resp = requests.post(f"{MASTER_API}/tx", json=payload, timeout=5)
                if resp.status_code != 200:
                    print(f"Failed to send tx nonce {payload['nonce']}: {resp.text}")
                time.sleep(0.05)
            except requests.exceptions.RequestException as e:
                print(f"Failed to send tx nonce {payload['nonce']}: {e}")

        flood_duration = time.time() - flood_start_time
        return flood_start_time, initial_confirmed_nonce, prep_duration, flood_duration


def record_data(start_time, initial_confirmed_nonce, prep_time, flood_time, initial_receiver_balance=0.0):
    print("📊 Monitoring block finalization (Polling for up to 3 minutes)...")
    
    timeout = 180  # Max wait time in seconds
    poll_interval = 5
    elapsed = 0
    
    successful_txs = 0
    total_network_txs = 0

    while elapsed < timeout:
        time.sleep(poll_interval)
        elapsed += poll_interval

        try:
            if TX_MODE == "shielded":
                current_receiver_balance = get_balance("iiitd11evtre3e9l6a2yrmmyqjgxqv7pcdsqar60du47u")
                successful_txs = current_receiver_balance - initial_receiver_balance
                # print(f"DEBUG: init_bal={initial_receiver_balance}, current_bal={current_receiver_balance}, successful_txs={successful_txs}")
            else:
                current_nonce = get_confirmed_nonce(FROM_ADDRESS)
                successful_txs = current_nonce - initial_confirmed_nonce

            resp = requests.get(f"{MASTER_API}/blocks?limit=10").json()
            blocks = resp.get("blocks", [])

            total_network_txs = sum(
                b.get("tx_count", 0)
                for b in blocks
            )

            # If we see any of our transactions confirmed, we can stop polling
            # We check if successful_txs == TOTAL_TXS or if it hasn't changed for a while
            if successful_txs == TOTAL_TXS:
                print(f"✅ All {TOTAL_TXS} transactions finalized early at {elapsed}s!")
                break

        except Exception as e:
            print(f"⚠️ Error during polling: {e}")

    # Log final results
    try:
        with open(RESULTS_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    datetime.now(),
                    TX_MODE,
                    TOTAL_TXS,
                    successful_txs,
                    total_network_txs,
                    prep_time,
                    flood_time,
                    elapsed,
                ]
            )

        print(f"\n--- BENCHMARK RESULTS ({TX_MODE.upper()}) ---")
        print(f"✅ Sender's successful transactions: {successful_txs}/{TOTAL_TXS}")
        print(f"🌐 Total network transactions finalized: {total_network_txs}")
        print(f"⏱️  Preparation Time: {prep_time:.2f} seconds")
        print(f"⏱️  Network Flood Time: {flood_time:.2f} seconds")

    except Exception as e:
        print(f"❌ Error recording data: {e}")


if __name__ == "__main__":
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w") as f:
            csv.writer(f).writerow(
                [
                    "Timestamp",
                    "Type",
                    "Target_TX",
                    "Sender_Success_TX",
                    "Network_Total_TX",
                    "Prep_Time",
                    "Flood_Time",
                    "Confirmation_Time",
                ]
            )

    init_receiver_bal = get_balance("iiitd11evtre3e9l6a2yrmmyqjgxqv7pcdsqar60du47u")
    start, init_nonce, p_time, f_time = flood_network()
    record_data(start, init_nonce, p_time, f_time, init_receiver_bal)