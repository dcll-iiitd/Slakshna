"""Ask Slakshna node 0 every question through the agentic RAG loop and save what it retrieved.

    python3 scripts/m9/run_agentic.py --configs m9-configs/book --data datasets/textbooks \
        --retrieval two_round --backend ollama --model qwen2.5:7b-instruct \
        --out results/m9/raw/book/agentic_two_round.jsonl
    # latency: fixed subset, 20 unrecorded warm-up questions, 3 recorded repeats
    python3 scripts/m9/run_agentic.py ... --qids datasets/textbooks/subset_200.json --warmup 20 --repeat 3

One line per question and run, in the contexts-file format of m9_common.py.
Retrieval only: the agent's own plan/reflect LLM calls are what is being
evaluated here, but the final free-text answer is left to answer.py, which
answers every system's `contexts` with the same downstream LLM for a fair
comparison (docs/agentic_rag_plan.md section 4.4; the CLI, scripts/rag_agent.py,
is the one that prints its own cited answer).

`score` is the merged RRF score (rag/agent.py rrf_merge) - this system's own
ranking score, not a leftover single-query one. `extra.trace` is the full
Agent trace (plan, every query, every reflection, LLM calls and time, rounds,
wall time) - not the richer per-peer records run_slakshna.py keeps for
evaluate.py's attack-detection/latency-breakdown analyses; those aren't
needed for this project's planned experiments (docs/decisions.md).

Warm-up questions run the full agent loop, not a bare /rag/query, so the LLM
backend (e.g. a GPU model load) is warmed up too, not just the node. The node
runs with generator = "none" regardless, since the agent never asks it to
generate. Re-running with the same --out skips questions already saved.
"""

import argparse
import json
import os
import sys
import time
import urllib.error

from m9_common import ROOT, context_record, done_keys, load_questions, log, open_append

sys.path.insert(0, ROOT)

from rag.agent import Agent, NodeSearch  # noqa: E402
from rag.agent_profiles import get_profile  # noqa: E402
from rag.llm import make_llm  # noqa: E402

CONTEXT_FIELDS = ["cluster_id", "verification", "claimed_score", "flags"]


def to_record(system, scheme, run, question, result, wall_s, adversaries=()):
    contexts = []
    for c in result["results"]:
        entry = {"text": c["text"], "id": c["chunk_id"], "node": c.get("node_id"), "score": c.get("rrf_score")}
        entry.update({k: c[k] for k in CONTEXT_FIELDS if c.get(k) is not None})
        contexts.append(entry)
    extra = {"trace": result["trace"]}
    if adversaries:
        extra["adversaries"] = list(adversaries)
    return context_record(system, scheme, run, question, contexts, wall_s, extra)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", help="folder with nodes.json; queries go to node 0")
    p.add_argument("--api", help="instead of --configs: the node's API URL")
    p.add_argument("--data", required=True, help="dataset folder (questions.jsonl)")
    p.add_argument("--qids", help="JSON list of question ids to ask (default: all)")
    p.add_argument("--retrieval", choices=["flat", "two_round", "directory"], default="two_round")
    p.add_argument("--scope", choices=["all", "local", "peers"], default="all")
    p.add_argument("--top-k", type=int, help="/rag/query top_k; default: the node's own default")
    p.add_argument("--system", help="name in the output (default agentic_<retrieval>, or agentic_local)")
    p.add_argument("--scheme", help="shard scheme name in the output (default: nodes.json name)")
    p.add_argument("--repeat", type=int, default=1, help="recorded runs over the questions")
    p.add_argument("--warmup", type=int, default=0, help="questions asked first and not recorded")
    p.add_argument("--timeout", type=float, default=120, help="per /rag/query call, in seconds")
    p.add_argument("--out", required=True)

    p.add_argument("--backend", choices=["hf", "ollama"], default="hf")
    p.add_argument("--model", default="allenai/OLMo-2-1124-7B-Instruct", help="D2's default is hf/OLMo-2-7B")
    p.add_argument("--device", help="hf backend only; default: cuda if available, else cpu")
    p.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--max-queries", type=int, help="default: the qa profile's (3)")
    p.add_argument("--max-rounds", type=int, help="default: the qa profile's (2)")
    p.add_argument("--max-llm-calls", type=int, help="default: unbounded")
    p.add_argument("--timeout-s", type=float, help="wall-clock budget per question; default: unbounded")
    p.add_argument("--final-k", type=int, help="default: the qa profile's (8)")
    p.add_argument("--budget-chars", type=int, default=3000, help="passage character budget per LLM call - a "
                   "rough stand-in for the model's token window; tune this for your model, it is not derived "
                   "from it (see rag/agent.py truncate_passages)")
    p.add_argument("--per-passage-chars", type=int, help="caps any single passage; default: uncapped")
    p.add_argument("--plan-max-new-tokens", type=int, default=256)
    p.add_argument("--reflect-max-new-tokens", type=int, default=128)
    args = p.parse_args()

    scheme = args.scheme
    adversaries = []
    if args.configs:
        with open(os.path.join(args.configs, "nodes.json"), encoding="utf-8") as f:
            nodes = json.load(f)
        api = args.api or nodes["nodes"][0]["api"]
        scheme = scheme or nodes["name"]
        adversaries = [{"endpoint_id": n["endpoint_id"], "kind": n["adversary"], "node": n["index"]}
                       for n in nodes["nodes"] if n["adversary"] != "none"]
    elif args.api:
        api = args.api
    else:
        p.error("give --configs or --api")
    system = args.system or ("agentic_local" if args.scope == "local" else f"agentic_{args.retrieval}")

    search_tool = NodeSearch(api, retrieval=args.retrieval, scope=args.scope, top_k=args.top_k,
                              timeout=args.timeout)
    if args.backend == "hf":
        llm = make_llm("hf", args.model, device=args.device, seed=args.seed)
    else:
        llm = make_llm("ollama", args.model, url=args.ollama_url)
    profile = get_profile("qa")
    budgets = {key: value for key, value in {
        "max_queries": args.max_queries, "max_rounds": args.max_rounds, "final_k": args.final_k,
        "max_llm_calls": args.max_llm_calls, "timeout_s": args.timeout_s,
    }.items() if value is not None}
    agent = Agent(search_tool, llm, profile, budget_chars=args.budget_chars, budgets=budgets,
                   per_passage_chars=args.per_passage_chars, plan_max_new_tokens=args.plan_max_new_tokens,
                   reflect_max_new_tokens=args.reflect_max_new_tokens)

    questions = load_questions(os.path.join(args.data, "questions.jsonl"), args.qids)
    for question in questions[:args.warmup]:
        try:
            agent.run(question)
        except (urllib.error.URLError, TimeoutError) as e:
            log(f"warm-up {question['qid']}: request failed ({e}); continuing")

    done = done_keys(args.out)
    failures = 0
    with open_append(args.out) as out:
        for run in range(1, args.repeat + 1):
            started_run = time.time()
            for n, question in enumerate(questions, start=1):
                if (question["qid"], run) in done:
                    continue
                started = time.time()
                try:
                    result = agent.run(question)
                except (urllib.error.URLError, TimeoutError) as e:
                    failures += 1
                    log(f"{question['qid']}: request failed ({e}); not saved, re-run to retry")
                    continue
                record = to_record(system, scheme, run, question, result, time.time() - started, adversaries)
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                if n % 100 == 0 or n == len(questions):
                    # max(..., avoids a division by zero when elapsed rounds to 0.0 - happens
                    # with a very fast backend (e.g. a fake in tests), never with a real node.
                    rate = n / max(time.time() - started_run, 1e-9)
                    log(f"{system} run {run}: {n}/{len(questions)} ({rate:.1f} questions/s)")
    if failures:
        sys.exit(f"{failures} questions failed; run the same command again to retry them")


if __name__ == "__main__":
    main()
