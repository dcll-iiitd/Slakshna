import requests
import time
import os

MASTER_API = "http://localhost:8545"
PRIVATE_KEY = "0818fc072ef93e5bb52ba738a9e6a9a4306e820e5522a128194b4eee20cba2dd"
FROM_ADDRESS = "iiitd11esxnfgm7ch8f2n3lhjxaelss3erlgvac2zypa0"

try:
    nonce = requests.get(f"{MASTER_API}/nonce/pending/{FROM_ADDRESS}").json().get("pending_nonce", 0)
    print(f"Pending nonce: {nonce}")

    resp = requests.post(f"{MASTER_API}/zkp/generate_commitment").json()
    commitment = resp["commitment"]

    sign_payload = {
        "private_key": PRIVATE_KEY,
        "tx_type": "shield",
        "from": FROM_ADDRESS,
        "to": "",
        "value": 1,
        "nonce": nonce,
        "data": None,
        "commitment": commitment,
    }
    sign_resp = requests.post(f"{MASTER_API}/tx/sign", json=sign_payload).json()
    print("Sign response:", sign_resp)

    tx_payload = {
        "tx_type": "shield",
        "from": FROM_ADDRESS,
        "to": "",
        "value": 1,
        "nonce": nonce,
        "data": None,
        "signature": sign_resp["signature"],
        "public_key": sign_resp["public_key"],
        "commitment": commitment,
    }
    tx_resp = requests.post(f"{MASTER_API}/tx", json=tx_payload).json()
    print("Tx submit response:", tx_resp)
    tx_hash = tx_resp.get("tx_hash")

    if not tx_hash:
        print("Tx hash missing!")
        exit(1)

    print("Waiting 6 seconds for block mining...")
    time.sleep(6)

    tx_status = requests.get(f"{MASTER_API}/tx/{tx_hash}").json()
    print("Tx status from chain:", tx_status)

except Exception as e:
    print("Error:", e)
