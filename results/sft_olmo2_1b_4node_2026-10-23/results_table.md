# OLMo-2 1B, 4-node federated fine-tuning, South Asia data

Held-out loss over assistant-response tokens; PPL = exp(NLL); BPB = bits per UTF-8 byte.

| Model | Split | NLL (nats/token) | Perplexity | Bits/byte |
|---|---|---|---|---|
| Untrained base | val | 3.2536 | 25.884 | 0.9910 |
| Federated (round 10), mean ± std over 4 nodes | val | 1.6549 ± 0.0260 | 5.234 ± 0.138 | 0.5040 ± 0.0079 |
| Federated (round 10), mean ± std over 4 nodes | test | 1.6971 ± 0.0262 | 5.459 ± 0.145 | 0.5205 ± 0.0080 |

Per-node final test results:

| Node | NLL | PPL | BPB |
|---|---|---|---|
| …z7p8 | 1.6891 | 5.415 | 0.5180 |
| …ulnf | 1.7360 | 5.675 | 0.5324 |
| …qc4p | 1.6801 | 5.366 | 0.5153 |
| …5vn9 | 1.6831 | 5.382 | 0.5162 |
