"""Ask a RAG node every question in corpus/questions.jsonl and save the answers.

    python3 scripts/rag_run_questions.py --api http://localhost:8545 \
        --retrieval two_round --out results/two_round.jsonl

Each output line is one question with the node's full /rag/query response, so
contexts, verification labels, per-peer results and timings can be compared
later. A one-line summary per question goes to stderr. Standard library only.
"""

import argparse
import collections
import json
import os
import sys
import time
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def ask(api, body, timeout):
    request = urllib.request.Request(
        f"{api}/rag/query",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def summary(question, response):
    contexts = response["contexts"]
    labels = collections.Counter(c.get("verification", "?") for c in contexts)
    peers = "; ".join(
        f"{p['endpoint_id'][:6]} {p['status']} sel={p['selected_clusters']} kept={p['chunks']}"
        + (f" flags={p['flags']}" if p["flags"] else "")
        for p in response["peers"]
    )
    t = response["timings_ms"]
    return (f"{question['id']} {question['type']:10s} total={t['total']:6d} ms "
            f"(peers {t['peer_retrieve']} = select {t['select']} + fetch {t['fetch']} + rescore {t['rescore']}) "
            f"| contexts {dict(labels)} | {peers}\n"
            f"    answer: {response['answer'][:160]!r}\n"
            f"    expect: {question['answer']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default="http://localhost:8545")
    parser.add_argument("--retrieval", choices=["two_round", "flat", "directory"],
                        help="override the node's [rag] retrieval")
    parser.add_argument("--scope", default="all", choices=["all", "local", "peers"])
    parser.add_argument("--questions", default=os.path.join(ROOT, "corpus", "questions.jsonl"))
    parser.add_argument("--ids", help="comma-separated question ids, e.g. q01,q07 (default: all)")
    parser.add_argument("--out", required=True, help="JSONL file to write")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()

    with open(args.questions, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    if args.ids:
        wanted = set(args.ids.split(","))
        questions = [q for q in questions if q["id"] in wanted]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as out:
        for question in questions:
            body = {"query": question["question"], "scope": args.scope}
            if args.retrieval:
                body["retrieval"] = args.retrieval
            started = time.time()
            response = ask(args.api, body, args.timeout)
            record = {"question": question, "request": body, "wall_s": round(time.time() - started, 2),
                      "response": response}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(summary(question, response), file=sys.stderr)


if __name__ == "__main__":
    main()
