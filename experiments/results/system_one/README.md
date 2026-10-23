# System One (federated decisions): results

Qwen2.5-1.5B-Instruct, 4 nodes, Dirichlet α=0.1 label skew, adapters trained on the System One prompt (3 rounds).
The node scores the raw System One prompt (`federated_inference/system_one.py`): one forward pass per adapter per option
ordering, all cyclic orderings (≤ 6), mixed by a plain average of the adapters' probabilities (`prob_mean`, equal weights). Other mixing rules (log average,
confidence-weighted, question-aware) are evaluated on the same adapter outputs; log average is supported by the node
(`combine = logit_mean`), the other two were computed offline.
Per-dataset folders hold every figure, table and a README explaining each one:
[AG News, 4 options](ag_news_n4_a0.1_s0_s1/README.md) · [Banking77, 77 options, two-stage](banking77_n4_a0.1_s0_s1/README.md).
Method follows the independent Jev evaluation (arXiv 2609.37647): accuracy, ECE (15 bins), Brier, NLL, AURC, accuracy at
coverage, 95% bootstrap CIs; position bias after Zheng et al. (ICLR 2024).

## Headline numbers
| | AG News (all classes) | Banking77 (77 options) |
|---|---|---|
| Base model | 0.872 | 0.524 |
| Own adapter only | 0.758 | 0.608 (mean of 4 nodes) |
| **Federated mix (plain average)** | **0.887** | **0.626** |
| All adapters merged | 0.878 | 0.626 |
| Centralised training (not private) | 0.894 | 0.628 |

## Findings
- Mixing peers' adapters fixes what a node never trained on: AG News accuracy 0.758 (own only) -> 0.821 (k=2) -> 0.887 (k=4), close to centralised training (0.894); on Banking77 the mix (0.626) matches centralised (0.628) but adds little over the best single adapter.
- Confidence is useful for deferring: the mix is 0.972 accurate on its most confident half (AURC 0.040 vs 0.125 for the own adapter); when all 4 members agree it is right far more often (see decision_agreement).
- The plain-average mix is under-confident (ECE 0.138); the per-adapter calibration temperature only partly fixes it (0.138 -> 0.109); weighting members by query similarity gives ECE 0.033 at similar accuracy.
- Averaging all cyclic orderings removes letter-position bias by construction; with a single ordering the answer changes under reordering for 13-22% of questions.
- Poisoned peers: one confident-and-wrong adapter of 4 drops accuracy to 0.806, two to 0.241 with equal weights; if trust flags them (weights softmax(trust/0.1)) accuracy stays 0.799. Robustness depends on trust actually identifying bad peers.
- Deciding vs generating (Banking77, same adapters): decide 0.597 vs generate 0.505; decisions are never unparseable.
- Cost: a warm k=4 decision takes 676 ms on our node vs 104 ms on a central vLLM server holding the same adapters; our node handles 1.4 decisions/s vs 17.6/s for vLLM at 16 in flight (vLLM batches; our sidecar runs one forward pass at a time). First use of peers' adapters adds ~1.1 s (download 64 ms + first-time attach).
- Fault tolerance: with 1 or 2 of 4 peers down every decision is still answered, with fewer members (accuracy 0.900 -> 0.867 on 30 questions) and ~1 s extra while the dead peer's download fails.
- Deterministic: the same decision repeated 50 times returned identical probabilities.

## Limits
One seed, one skew level, 4 nodes, 1.5B model; AG News leaves little headroom (base 0.872). vLLM baseline covers single-stage decisions only
(≤ 52 options). Two-stage decisions in bf16 can change their finalists on near-ties (batched vs unbatched arithmetic).
