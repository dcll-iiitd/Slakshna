"""Ask Slakshna node 0 every question and save what it retrieved (no LLM).

    python3 scripts/m9/run_slakshna.py --configs m9-configs/book --data datasets/textbooks \
        --retrieval two_round --out results/m9/raw/book/slakshna_two_round.jsonl
    # latency: fixed subset, 20 unrecorded warm-up questions, 3 recorded repeats
    python3 scripts/m9/run_slakshna.py ... --qids datasets/textbooks/subset_200.json --warmup 20 --repeat 3

One line per question and run, in the contexts-file format of m9_common.py.
`extra` keeps the node's timings, query flags and a per-peer summary (status,
clusters chosen, round times, bytes, flags), so evaluate.py can report
latency breakdowns, bytes, peers contacted and attack detection. The node runs
with generator = "none", so `answer` is empty. Re-running with the same --out
skips questions already saved. Standard library only.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

from m9_common import context_record, done_keys, load_questions, log, open_append

PEER_FIELDS = ["endpoint_id", "status", "chunks", "rtt_ms", "offered_clusters", "selected_clusters",
               "rejected_chunks", "flags", "round1_ms", "round2_ms", "selected_via",
               "bytes_sent", "bytes_received", "error"]
CONTEXT_FIELDS = ["cluster_id", "verification", "claimed_score", "flags"]


def ask(api, body, timeout):
    request = urllib.request.Request(f"{api}/rag/query", data=json.dumps(body).encode(),
                                     headers={"content-type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def to_record(system, scheme, run, question, response, wall_s, adversaries=()):
    contexts = []
    for c in response["contexts"]:
        entry = {"text": c["text"], "id": c["chunk_id"], "node": c.get("node_id"), "score": c["score"]}
        entry.update({k: c[k] for k in CONTEXT_FIELDS if c.get(k) is not None})
        contexts.append(entry)
    peers = [{k: p.get(k) for k in PEER_FIELDS} for p in response.get("peers", [])]
    extra = {
        "timings_ms": response.get("timings_ms"),
        "flags": response.get("flags", []),
        "peers": peers,
        "bytes": sum((p["bytes_sent"] or 0) + (p["bytes_received"] or 0) for p in peers),
        "peers_contacted": sum(1 for p in peers if (p["bytes_sent"] or 0) > 0),
    }
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
    p.add_argument("--system", help="name in the output (default slakshna_<retrieval>, or slakshna_local)")
    p.add_argument("--scheme", help="shard scheme name in the output (default: nodes.json name)")
    p.add_argument("--repeat", type=int, default=1, help="recorded runs over the questions")
    p.add_argument("--warmup", type=int, default=0, help="questions asked first and not recorded")
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--out", required=True)
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
    system = args.system or ("slakshna_local" if args.scope == "local" else f"slakshna_{args.retrieval}")
    body_base = {"scope": args.scope, "retrieval": args.retrieval}

    questions = load_questions(os.path.join(args.data, "questions.jsonl"), args.qids)
    for question in questions[:args.warmup]:
        ask(api, dict(body_base, query=question["question"]), args.timeout)
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
                    response = ask(api, dict(body_base, query=question["question"]), args.timeout)
                except (urllib.error.URLError, TimeoutError) as e:
                    failures += 1
                    log(f"{question['qid']}: request failed ({e}); not saved, re-run to retry")
                    continue
                record = to_record(system, scheme, run, question, response, time.time() - started, adversaries)
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                if n % 100 == 0 or n == len(questions):
                    rate = n / (time.time() - started_run)
                    log(f"{system} run {run}: {n}/{len(questions)} ({rate:.1f} questions/s)")
    if failures:
        sys.exit(f"{failures} questions failed; run the same command again to retry them")


if __name__ == "__main__":
    main()
