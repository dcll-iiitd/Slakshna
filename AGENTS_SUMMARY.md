# Slakshna: federated agents and multi-agent systems (Steps 4-5), status summary

**What was done.** The agents and multi-agent layer from the implementation handoff (epics A1-A7, M1-M7) is part of the
Slakshna node. Every agent action is a signed, hash-chained provenance record. It reuses the existing Iroh mesh, adding
one protocol (`slakshna/agent/1`) and an optional A2A v1.0 HTTP/HTTPS gateway for agents outside the federation.

**Verified.** 232 automated tests; 30 end-to-end checks against three real node processes (discovery, delegation with
consensus, MCP tool use under capability tokens, signed halt over gossip, SIGKILL recovery); and, on real hardware
(8x NVIDIA H200, Docker with GPUs, vLLM): the container sandbox, real LLM agents, and a 7-process federation running a
research program with GPU containers and private evaluation.

**Result 1 - robustness with real LLM agents** (9 agents on Qwen2.5 0.5B/1.5B/3B, GSM8K, 300 problems per row, bootstrap 95% CI):

| adversaries | best single agent* | centralized orchestrator | decentralized, votes only | decentralized + ground-truth probes |
|---|---|---|---|---|
| none | 79.3% [75.0, 83.7] | 55.7% [50.0, 61.0] | 61.3% [55.7, 67.0] | 76.3% [71.7, 81.3] |
| 2 colluding | 78.3% | 47.3% [41.7, 53.0] | **0.0%** | 75.3% [70.7, 80.0] |
| 3 colluding | 77.7% | 35.0% [29.7, 40.7] | **0.0%** | 80.7% [76.3, 85.0] |
| 3 lying | 79.0% | 35.0% [29.7, 40.7] | 58.7% [53.0, 64.3] | 81.0% [76.3, 85.3] |

\* an oracle choice. The decentralized system matches it without knowing which agent is best and keeps working under
attack; it does not beat it. Verification costs about 4.7x tokens and 15x messages per task. Findings: vote-only pooling is
captured completely by colluders, and agreement-based reputation rewards consistency rather than correctness, so reputation
must be anchored to ground-truth probes.

**Result 2 - federated research program on GPUs** (3 planner/executors, 3 private-data evaluators, 1 auditor; real GPU
containers; 5 generations in 11 minutes). The loop ran end to end, but the best discovered prompt reached 43.5% held-out
vs 40.7% for the baseline (paired p = 0.14): **no significant improvement**. This supports the system claims (it runs,
is verifiable, survives failures), not a claim about research quality.

**Not verified.** Podman/Apptainer (only Docker ran); research quality beyond one program and a 3B planner; settlement is
behind an interface with an in-memory ledger (the chain decision is open); A2A was checked against the spec's field names, not
a third-party client; the adversaries are simple simulated behaviours.

Reproduce: `cargo test`; `python3 scripts/e2e_agents.py`; `iiitd llm-bench ...`; `python3 scripts/e2e_research_gpu.py ...`
(see `results/agents_gpu/README.md`).
