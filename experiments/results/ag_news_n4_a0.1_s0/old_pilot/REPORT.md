# Pilot run: federated inference on AG News (4 nodes, Dirichlet alpha=0.1, seed 0)

Setup: Qwen2.5-1.5B-Instruct, LoRA r=16 (q,v), 3 rounds x 30 steps (effective batch 16), each node trains on its own
label-skewed slice (node1 no World, node2 ~all Sci/Tech, node3 World+Business, node4 Sports+Sci/Tech+Business).
Published adapter = each node's locally fine-tuned adapter ("local"); the aggregated one is also kept on disk.
Held-out: 1000-question class-balanced global test, 500-question local test per node (the node's own class mix).
Greedy decoding, 8 new tokens, answer parsed against the 4 label names. One seed. Nothing here is a final result.

## Offline (adapter files scored directly, 1000 global questions)
| method (mean over 4 query nodes) | global acc | global macro-F1 | own-local-test acc |
|---|---|---|---|
| base model (no adapter) | 0.713 | 0.681 | 0.646 |
| own local adapter | 0.524 | 0.436 | 0.952 |
| own aggregated adapter | 0.418 | 0.320 | 0.794 |
| random adapter | 0.524 | - | 0.526 |
| majority vote of 4 adapters | 0.705 | 0.701 | 0.696 |
| query-aware routing (nearest node descriptor) | 0.843 | 0.841 | 0.856 |
| oracle (best adapter per question) | 0.987 | - | 0.988 |

## Through the real API (200 global questions per query node)
self 0.526 (matches offline own-local 0.524) | random 0.531 | trust 0.395 | latest 0.395.
trust and latest always chose node2's adapter: all trust scores are ~0.25, so they tie and the tie-break
(newest round, then node id) picks the same adapter for every query. Node2's adapter is the weakest on the global test.

## Latency (4 nodes, 1.5B, A6000; answers are 1-3 tokens)
| | fetch | attach | model first token | TTFT mean | TTFT p95 |
|---|---|---|---|---|---|
| cold (fetch by hash, 4 MB) | 63 ms | 174 ms | 303 ms | 540 ms | 1105 ms |
| warm (cache hit) | 0.1 ms | 5.6 ms | 99 ms | 105 ms | 150 ms |
n = 12 cold, 60 warm. TPOT is NOT reported: with 1-3 token answers it is not meaningful (needs a long-generation phase).

## Findings
1. Adapters specialise sharply (fig_fi_class_heatmap): ~1.0 on own classes, ~0 on unseen ones. A node using only its own adapter
   is worse than the base model on the global test (0.52 vs 0.71), but best on its own distribution (0.95).
2. Choosing an adapter per query beats any fixed choice: query-aware routing 0.84 vs own adapter 0.52 and base 0.71. Oracle 0.99 shows headroom.
3. Fixed strategies (self/trust/latest) cannot do this because they ignore the query.
4. Fetching a missing 4 MB adapter by hash adds ~0.4 s to the first query (63 ms transfer + attach + first-use warm-up); later queries are cache hits.

## Caveats / open issues
- The "aggregated" adapters drift per node (e.g. node2 aggregated -> 0.25 global). Weights are uniform (~0.25) but transport compression
  keeps only 10% of each peer's update (sparsity=0.1), so aggregation barely mixes. Rerun with sparsity=1.0 for a true federated-average baseline.
- Centralised (all data pooled) and local-only (no federation) baselines are not run yet, so "federated training helps" is untested here.
- Base-model accuracy is depressed on Sci/Tech (0.13) partly by label formatting ("Technology" vs "Sci/Tech"); see invalid_rate in fi_cross_adapter.csv.
- Query-aware routing here is simulated offline (sentence-embedding centroid of each node's training text); it is not implemented in the node yet.
- One seed, one alpha, 4 nodes, 1.5B model.
- publish_latest_existing() re-publishes the first newest-round adapter it finds on disk after a restart; with two kinds per round
  this is directory-order dependent (it happened to pick "local" for all 4 nodes this time).

## Files
Tables: fi_cross_adapter.csv, fi_per_class.csv, fi_by_round.csv, fi_strategy.csv, fi_api_strategy.csv, fi_latency.csv
Raw: preds.csv (one row per adapter x question), api_queries.csv (one row per API query)
Figures: fig_fi_class_heatmap, fig_fi_cross_matrix, fig_fi_strategy, fig_fi_by_round, fig_fi_latency, fig_fi_api, fig_fi_semantic (.png/.pdf)
