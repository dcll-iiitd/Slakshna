"""fedrag: Flower's FedRAG ServerApp loop, saving each question's contexts for M9.

Copied into the Flower FedRAG example by scripts/m9/flower/patch_fedrag.py
(Slakshna repository, M9 evaluation). It runs
the same retrieval as `fedrag/server_app.py`: the same messages to every
client, the same round-robin corpus assignment, and Flower's own
`merge_documents` (imported, not copied). The differences, all marked "M9:":

- questions come from the M9 questions file (same order and ids as
  Flower's `MirageQA`), optionally limited to a list of ids, with warm-up
  questions and repeated runs for latency;
- it waits until `m9-nodes` SuperNodes are online (Flower's loop starts as
  soon as one is, which would silently drop shards);
- instead of answering with the LLM, it writes the merged documents, every
  client's returned scores, payload sizes and the retrieval time to a JSONL
  file. With `m9-llm = "true"` it also answers inline exactly as Flower does
  (check E0(a)).
"""

import hashlib
import json
import os
import time
from collections import defaultdict
from itertools import cycle

from flwr.app import ConfigRecord, Context, Message, MessageType, RecordDict
from flwr.serverapp import Grid, ServerApp

from fedrag.server_app import merge_documents, node_online_loop
from fedrag.task import index_exists


def m9_wait_for_nodes(grid, expected, timeout_s):
    """M9: wait until `expected` SuperNodes are online, not just one."""
    deadline = time.time() + timeout_s
    node_ids = node_online_loop(grid)
    while len(node_ids) < expected:
        if time.time() > deadline:
            raise RuntimeError(f"only {len(node_ids)} of {expected} SuperNodes online after {timeout_s} s")
        time.sleep(1)
        node_ids = grid.get_node_ids()
    return list(node_ids)


def m9_submit_question(grid, question, question_id, knn, node_ids, corpus_names_iter):
    """Flower's `submit_question`, also returning which corpus each reply came from."""
    messages, corpus_of_node = [], {}
    for node_id in node_ids:
        config_record = ConfigRecord()
        config_record["question"] = question
        config_record["question_id"] = question_id
        config_record["knn"] = knn
        config_record["corpus_name"] = next(corpus_names_iter)
        corpus_of_node[node_id] = config_record["corpus_name"]  # M9
        task_record = RecordDict({"config": config_record})
        message = Message(
            content=task_record,
            message_type=MessageType.QUERY,
            dst_node_id=node_id,
            group_id=str(question_id),
        )
        messages.append(message)

    replies = grid.send_and_receive(messages)

    documents, scores, origins, reply_info = [], [], [], []
    for reply in replies:
        if reply.has_content():
            docs = list(reply.content["docs_n_scores"]["documents"])
            sc = list(reply.content["docs_n_scores"]["scores"])
            documents.extend(docs)
            scores.extend(sc)
            corpus = corpus_of_node.get(reply.metadata.src_node_id)  # M9
            origins.extend([corpus] * len(docs))
            reply_info.append({"corpus": corpus, "docs": len(docs),
                               "payload_bytes": sum(len(d.encode("utf-8")) for d in docs) + 8 * len(sc)})
    return documents, scores, origins, reply_info, len(messages)


def m9_shard(corpus_name):
    """'textbooks_s3' -> 3."""
    try:
        return int(corpus_name.rsplit("_s", 1)[1])
    except (AttributeError, IndexError, ValueError):
        return corpus_name


def m9_contexts(merged, documents, scores, origins):
    """Each merged document with the shard and best (lowest L2) score it was returned with."""
    best = {}
    for doc, score, origin in zip(documents, scores, origins):
        if doc not in best or score < best[doc][0]:
            best[doc] = (score, origin)
    return [{"text": doc, "id": None, "node": m9_shard(best[doc][1]), "score": float(best[doc][0])}
            for doc in merged]


def m9_done(path):
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    done.add((row["qid"], row["run"]))
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = context.run_config
    k_rrf = int(cfg["k-rrf"])
    knn = int(cfg["k-nn"])
    corpus_names = [c.lower() for c in cfg["clients-corpus-names"].split("|")]
    index_exists(corpus_names)
    expected = int(cfg.get("m9-nodes", len(corpus_names)))
    node_ids = m9_wait_for_nodes(grid, expected, int(cfg.get("m9-node-timeout", 300)))
    if len(node_ids) != len(corpus_names):
        print(f"M9 WARNING: {len(node_ids)} SuperNodes but {len(corpus_names)} corpora; "
              "the corpus assignment rotates between questions")
    corpus_names_iter = cycle(corpus_names)

    datasets = [d.lower() for d in cfg["server-qa-datasets"].split("|")]
    qa_num = int(cfg.get("server-qa-num", 0) or 0)
    with open(cfg["m9-questions"], encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    selected = []
    for dataset in datasets:
        in_dataset = [q for q in questions if q["dataset"] == dataset]
        selected.extend(in_dataset[:qa_num] if qa_num else in_dataset)
    if cfg.get("m9-qids", ""):
        with open(cfg["m9-qids"], encoding="utf-8") as f:
            wanted = set(json.load(f))
        selected = [q for q in selected if q["qid"] in wanted]

    llm = None
    if str(cfg.get("m9-llm", "false")).lower() == "true":
        from fedrag.llm_querier import LLMQuerier

        llm = LLMQuerier(cfg["server-llm-hfpath"], str(cfg.get("server-llm-use-gpu", "false")).lower() == "true")

    out_path = cfg["m9-out"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done = m9_done(out_path)
    system, scheme = cfg.get("m9-system", "flower"), cfg.get("m9-scheme", "")
    repeat, warmup = int(cfg.get("m9-repeat", 1)), int(cfg.get("m9-warmup", 0))

    for q in selected[:warmup]:
        m9_submit_question(grid, q["question"], q["qid"], knn, node_ids, corpus_names_iter)

    with open(out_path, "a", encoding="utf-8") as out:
        for run in range(1, repeat + 1):
            run_started = time.time()
            for n, q in enumerate(selected, start=1):
                if (q["qid"], run) in done:
                    continue
                q_st = time.time()
                docs, scores, origins, replies, sent = m9_submit_question(
                    grid, q["question"], q["qid"], knn, node_ids, corpus_names_iter
                )
                merged = merge_documents(docs, scores, knn, k_rrf)
                retrieval_s = time.time() - q_st
                extra = {
                    "replies": replies,
                    "bytes": sum(r["payload_bytes"] for r in replies),
                    "peers_contacted": sent,
                    "returned": [{"node": m9_shard(o), "score": float(s),
                                  "hash": hashlib.sha256(d.encode("utf-8")).hexdigest()[:32]}
                                 for d, s, o in zip(docs, scores, origins)],
                }
                if llm is not None:
                    llm_st = time.time()
                    _, predicted = llm.answer(q["question"], merged, q["options"], q["dataset"])
                    extra.update({"inline_prediction": predicted, "llm_s": round(time.time() - llm_st, 3)})
                record = {"system": system, "scheme": scheme, "run": run, "qid": q["qid"], "dataset": q["dataset"],
                          "contexts": m9_contexts(merged, docs, scores, origins),
                          "retrieval_s": round(retrieval_s, 4), "extra": extra}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                if n % 100 == 0 or n == len(selected):
                    print(f"M9 {system} run {run}: {n}/{len(selected)} "
                          f"({n / (time.time() - run_started):.2f} questions/s)")
