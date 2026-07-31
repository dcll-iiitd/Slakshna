# diagnostic.py — run this ONCE to find the correct message format
import requests
import json
import hashlib
from nacl.signing import SigningKey
from nacl.encoding import HexEncoder

MASTER_API = "http://localhost:8545"
FROM_ADDRESS = "iiitd11esxnfgm7ch8f2n3lhjxaelss3erlgvac2zypa0"
TO_ADDRESS = "iiitd11evtre3e9l6a2yrmmyqjgxqv7pcdsqar60du47u"
PRIVATE_KEY_HEX = "0818fc072ef93e5bb52ba738a9e6a9a4306e820e5522a128194b4eee20cba2dd"

TEST_NONCE = 9999  # dummy nonce just for probing

sk = SigningKey(PRIVATE_KEY_HEX, encoder=HexEncoder)
pub = sk.verify_key.encode(encoder=HexEncoder).decode()

# Step 1: Get the node's signature for nonce 9999
resp = requests.post(f"{MASTER_API}/tx/sign", json={
    "private_key": PRIVATE_KEY_HEX,
    "tx_type": "transfer",
    "from": FROM_ADDRESS,
    "to": TO_ADDRESS,
    "value": 1,
    "nonce": TEST_NONCE,
    "data": None,
}, timeout=5).json()

print(f"Node response: {resp}")
expected_sig = resp["signature"]
print(f"\nExpected signature : {expected_sig}")
print(f"Public key         : {pub}\n")

# Step 2: Try every conceivable message format
n = TEST_NONCE
v = 1

candidates = {
    "f01_colon_tx":         f"tx:transfer:{FROM_ADDRESS}:{TO_ADDRESS}:{v}:{n}".encode(),
    "f02_colon_plain":      f"transfer:{FROM_ADDRESS}:{TO_ADDRESS}:{v}:{n}".encode(),
    "f03_colon_short":      f"{FROM_ADDRESS}:{TO_ADDRESS}:{v}:{n}".encode(),
    "f04_pipe":             f"transfer|{FROM_ADDRESS}|{TO_ADDRESS}|{v}|{n}".encode(),
    "f05_pipe_noname":      f"{FROM_ADDRESS}|{TO_ADDRESS}|{v}|{n}".encode(),
    "f06_space":            f"transfer {FROM_ADDRESS} {TO_ADDRESS} {v} {n}".encode(),
    "f07_json_sorted":      json.dumps({"data":None,"from":FROM_ADDRESS,"nonce":n,"to":TO_ADDRESS,"tx_type":"transfer","value":v},sort_keys=True,separators=(',',':')).encode(),
    "f08_json_unsorted":    json.dumps({"tx_type":"transfer","from":FROM_ADDRESS,"to":TO_ADDRESS,"value":v,"nonce":n,"data":None},separators=(',',':')).encode(),
    "f09_json_no_data":     json.dumps({"tx_type":"transfer","from":FROM_ADDRESS,"to":TO_ADDRESS,"value":v,"nonce":n},sort_keys=True,separators=(',',':')).encode(),
    "f10_json_pretty":      json.dumps({"tx_type":"transfer","from":FROM_ADDRESS,"to":TO_ADDRESS,"value":v,"nonce":n,"data":None}).encode(),
    "f11_sha256_colon":     hashlib.sha256(f"tx:transfer:{FROM_ADDRESS}:{TO_ADDRESS}:{v}:{n}".encode()).digest(),
    "f12_sha256_json_sorted": hashlib.sha256(json.dumps({"data":None,"from":FROM_ADDRESS,"nonce":n,"to":TO_ADDRESS,"tx_type":"transfer","value":v},sort_keys=True,separators=(',',':')).encode()).digest(),
    "f13_sha256_json_unsorted": hashlib.sha256(json.dumps({"tx_type":"transfer","from":FROM_ADDRESS,"to":TO_ADDRESS,"value":v,"nonce":n,"data":None},separators=(',',':')).encode()).digest(),
    "f14_from_to_nonce":    f"{FROM_ADDRESS}{TO_ADDRESS}{v}{n}".encode(),
    "f15_nonce_first":      f"{n}:{FROM_ADDRESS}:{TO_ADDRESS}:{v}".encode(),
    "f16_value_str":        f"transfer:{FROM_ADDRESS}:{TO_ADDRESS}:{float(v)}:{n}".encode(),
    "f17_colon_with_type":  f"transfer:{FROM_ADDRESS}:{TO_ADDRESS}:{v}:{n}:None".encode(),
    "f18_bincode_like":     (f"transfer\x00\x00\x00\x00\x00\x00\x00\x08{FROM_ADDRESS}{TO_ADDRESS}").encode(),
    "f19_sha512_json":      hashlib.sha512(json.dumps({"data":None,"from":FROM_ADDRESS,"nonce":n,"to":TO_ADDRESS,"tx_type":"transfer","value":v},sort_keys=True,separators=(',',':')).encode()).digest(),
    "f20_just_nonce_from_to": f"{FROM_ADDRESS}:{TO_ADDRESS}:{n}".encode(),
}

print("Testing all formats...\n")
matched = False
for name, msg in candidates.items():
    try:
        sig = sk.sign(msg).signature.hex()
        match = "✅ MATCH" if sig == expected_sig else "   no match"
        if sig == expected_sig:
            print(f"{match} → {name}")
            print(f"  Message bytes : {msg}")
            print(f"  Message string: {msg.decode(errors='replace')}")
            matched = True
        else:
            print(f"{match} → {name}")
    except Exception as e:
        print(f"   error → {name}: {e}")

if not matched:
    print("\n❌ No format matched. Paste the output above so we can inspect the node's Rust source.")
    print("\nAlso run this to check if /tx/sign endpoint returns any debug info:")
    print(f"  curl -X POST {MASTER_API}/tx/sign -H 'Content-Type: application/json' \\")
    print(f"    -d '{{\"private_key\":\"{PRIVATE_KEY_HEX}\",\"tx_type\":\"transfer\",\"from\":\"{FROM_ADDRESS}\",\"to\":\"{TO_ADDRESS}\",\"value\":1,\"nonce\":{TEST_NONCE},\"data\":null}}'")