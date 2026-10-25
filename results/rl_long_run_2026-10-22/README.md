# Post-training RL (GRPO) long run, 2026-10-22

Task: GSM8K, reward = final \boxed{} answer correct. Evaluation = greedy decoding.

## single_node/  (one GPU each, 240 steps, 16 prompts x 8 completions per step, max_new_tokens 512)
Held-out accuracy on the full 1,319-problem GSM8K test set at steps 0 / 80 / 160 / 240.
Each folder has the exact config (cfg.yaml) and the full training log (log.txt).

## federated_2node/  (2 real Rust nodes, 6 rounds x 20 GRPO steps, disjoint data shards, max_new_tokens 320)
Per-round trainer logs (`*_bhaskera_crash.log`), trust weights, and ml_engine events.
Held-out accuracy is on the first 200 test problems (about +-3.4 points of noise).
