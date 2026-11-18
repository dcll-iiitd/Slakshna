"""Ask one question from the terminal using the agentic RAG loop (docs/agentic_rag_plan.md section 4).

    python3 scripts/rag_agent.py --api http://127.0.0.1:18571 \
        --question "What is the capital of France?" --backend ollama --model qwen2.5:7b-instruct

    python3 scripts/rag_agent.py --configs m9-configs/book --question "..." \
        --options '{"A": "Paris", "B": "London"}'

Prints the plan, each search (or why one was skipped/failed), every
reflection verdict, the kept passages and a short cited answer. The agent
loop itself (rag/agent.py) is standard library only; the chosen LLM backend
may not be (hf needs torch/transformers; ollama needs nothing extra).
"""

import argparse
import json
import os
import sys
import urllib.error

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.agent import Agent, NodeSearch, answer
from rag.agent_profiles import get_profile
from rag.llm import make_llm


def print_trace(trace):
    plan = trace["plan"]
    note = f"  (error: {plan['error']})" if plan["error"] else ""
    print(f"Plan: {plan['queries']}{note}")
    print()

    for q in trace["queries"]:
        if q.get("skipped"):
            print(f"[round {q['round']}] skipped (duplicate): {q['query']!r}")
        elif q.get("error"):
            print(f"[round {q['round']}] FAILED {q['query']!r}: {q['error']}")
        else:
            print(f"[round {q['round']}] searched {q['query']!r}: {len(q['keys'])} results, "
                  f"{q['retrieval_s']}s, {q['bytes']} bytes, {q['peers_contacted']} peers contacted")
    print()

    for r in trace["reflections"]:
        verdict = "sufficient" if r["sufficient"] else f"not sufficient, next query: {r['next_query']!r}"
        note = f"  (error: {r['error']})" if r["error"] else ""
        print(f"[round {r['round']}] reflect: {verdict}{note}")
    print()


def print_results(results):
    print(f"Kept {len(results)} passage(s):")
    for i, chunk in enumerate(results, start=1):
        source = chunk.get("source", "?")
        score = chunk.get("rrf_score", 0.0)
        text = chunk["text"][:200].replace("\n", " ")
        print(f"[{i}] ({source}, score {score:.4f}) {text}")
    print()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", help="folder with nodes.json; the question goes to node 0")
    p.add_argument("--api", help="instead of --configs: the node's API URL")
    p.add_argument("--question", required=True)
    p.add_argument("--options", help='multiple-choice options as JSON, e.g. \'{"A": "Paris", "B": "London"}\'')
    p.add_argument("--retrieval", choices=["flat", "two_round", "directory"], default="two_round")
    p.add_argument("--scope", choices=["all", "local", "peers"], default="all")
    p.add_argument("--top-k", type=int, help="/rag/query top_k; default: the node's own default")
    p.add_argument("--query-timeout", type=float, default=120, help="per /rag/query call, in seconds")

    p.add_argument("--backend", choices=["hf", "ollama"], default="hf")
    p.add_argument("--model", default="allenai/OLMo-2-1124-7B-Instruct", help="D2's default is hf/OLMo-2-7B; "
                   "pick a model your --backend actually serves (e.g. an Ollama tag for --backend ollama)")
    p.add_argument("--device", help="hf backend only; default: cuda if available, else cpu")
    p.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--max-queries", type=int, help="default: the qa profile's (3)")
    p.add_argument("--max-rounds", type=int, help="default: the qa profile's (2)")
    p.add_argument("--max-llm-calls", type=int, help="default: unbounded")
    p.add_argument("--timeout-s", type=float, help="wall-clock budget for the whole run; default: unbounded")
    p.add_argument("--final-k", type=int, help="default: the qa profile's (8)")
    p.add_argument("--budget-chars", type=int, default=3000, help="passage character budget per LLM call - a "
                   "rough stand-in for the model's token window (see rag/agent.py truncate_passages); tune this "
                   "for your model, it is not derived from it")
    p.add_argument("--per-passage-chars", type=int, help="caps any single passage; default: uncapped")
    p.add_argument("--plan-max-new-tokens", type=int, default=256)
    p.add_argument("--reflect-max-new-tokens", type=int, default=128)
    p.add_argument("--answer-max-new-tokens", type=int, default=256)
    args = p.parse_args()

    if args.configs:
        with open(os.path.join(args.configs, "nodes.json"), encoding="utf-8") as f:
            nodes = json.load(f)
        api = args.api or nodes["nodes"][0]["api"]
    elif args.api:
        api = args.api
    else:
        p.error("give --configs or --api")

    question = {"question": args.question}
    if args.options:
        question["options"] = json.loads(args.options)

    search_tool = NodeSearch(api, retrieval=args.retrieval, scope=args.scope, top_k=args.top_k,
                              timeout=args.query_timeout)
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
    try:
        result = agent.run(question)
    except (urllib.error.URLError, TimeoutError) as e:
        # Only the *original input* search raises out of agent.run() (rag/agent.py: "required").
        # Everything else - a planned/follow-up query failing - is caught there and shown in the trace.
        sys.exit(f"could not reach the node at {api}: {e}")

    print_trace(result["trace"])
    print_results(result["results"])

    input_text = profile.format_input(question)
    answer_text, answer_error = answer(llm, profile, input_text, result["results"], args.budget_chars,
                                        args.answer_max_new_tokens, args.per_passage_chars)
    print(f"Answer: {answer_text}" if not answer_error else f"Answer: (failed - {answer_error})")
    print()

    trace = result["trace"]
    print(f"({trace['llm_calls']} LLM call(s), {trace['llm_time_s']}s LLM time, {trace['wall_s']}s wall time)")


if __name__ == "__main__":
    main()
