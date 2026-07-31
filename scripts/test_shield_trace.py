import requests
import json

MASTER_API = "http://localhost:8545"

blocks = requests.get(f"{MASTER_API}/blocks?limit=10").json().get("blocks", [])

for block in blocks:
    print(f"Block {block['height']} with {block['tx_count']} TXs")
    for tx in block.get("transactions", []):
        if tx["tx_type"] == "Shield":
            print(f"  TX {tx['hash'][:8]}: nonce={tx['nonce']}, status={tx['status']}, error={tx.get('error')}")
