"""
Build the result tables (plain-English names) from the raw measurement files of one run.

  experiments/results/<tag>/raw/      predictions.csv (eval_offline.py), api_queries.csv (eval_api.py),
                                      serving_<backend>.csv (bench_serving.py), fault_tolerance.csv (fault_test.py),
                                      fetch_vs_adapter_size.csv (bench_fetch.py)
  experiments/results/<tag>/tables/   written here (only the ones whose inputs exist):

  Quality (classification task, AG News)
    accuracy_per_topic.csv                 adapter x topic accuracy on the balanced test set
    accuracy_adapter_vs_test_set.csv       adapter x test distribution: accuracy, macro-F1, unparseable-answer rate
    accuracy_by_selection_method.csv       per query node: every way of answering (own adapter, peer, query-aware,
                                           merged, oracle, training baselines), on all topics and on the node's own mix
    accuracy_by_round.csv                  balanced-test accuracy of each node's adapter after every round
  Quality (generation task, Dolly)
    generation_quality_by_method.csv       BERTScore-F1 (rescaled), embedding cosine, ROUGE-L per method
    generation_quality_adapter_vs_test_set.csv
  System
    latency_cold_vs_warm.csv               time to first token split into rank / fetch+verify / attach / first token
    latency_by_selection_method.csv        per-strategy latency through the real API
    latency_vs_prompt_length.csv           TTFT, prefill TFLOP/s, MFU per backend and prompt length
    latency_vs_output_length.csv           TPOT, end-to-end latency, tokens/s per backend and output length
    throughput_vs_concurrency.csv          aggregate tokens/s, requests/s, p50/p95 TTFT, goodput per backend
    fault_tolerance.csv                    answered %, accuracy, adapter used, latency, per number of peers down
    fetch_vs_adapter_size.csv              cold fetch time vs LoRA rank / file size

    Bhaskera-serve/.venv/bin/python experiments/aggregate.py --tag ag_news_n4_a0.1_s0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(REPO, "hf_cache"))

A6000_BF16_PEAK_TFLOPS = 154.8  # dense bf16 tensor-core peak (309.7 is the 2:4-sparsity figure)
SLO_TTFT_MS, SLO_TPOT_MS = 1000.0, 50.0

METHOD_NAMES = {
    "base": "Base model (no adapter)",
    "own_local": "Own adapter only",
    "own_aggregated": "Own aggregated adapter",
    "random_adapter": "Random peer adapter",
    "query_aware": "Query-aware adapter choice",
    "majority_vote": "Majority vote of all adapters",
    "merged_avg": "All adapters merged (average)",
    "merged_ties": "All adapters merged (TIES)",
    "oracle": "Best adapter per question (oracle)",
    "fedavg": "Federated average, 50% of each update shared (training baseline)",
    "local_only": "Local-only training, no federation (baseline)",
    "centralised": "Centralised training on pooled data (upper bound, not private)",
}


def macro_f1(y, p, n_cls):
    from sklearn.metrics import f1_score
    return float(f1_score(y, p, labels=list(range(n_cls)), average="macro", zero_division=0))


def adapter_names(meta):
    """'node2' -> 'Node 2 (mostly Sci/Tech)' from the training class histogram."""
    out = {}
    for node, h in meta["class_hist"].items():
        tot = sum(h.values()) or 1
        top = [c for c, v in sorted(h.items(), key=lambda x: -x[1]) if v / tot >= 0.2][:3]
        out[node] = f"Node {node[4:]} ({', '.join(top)})"
    return out


def pretty_adapter(aid, names):
    if aid == "base":
        return "Base model"
    if aid in ("merged_avg", "merged_ties"):
        return METHOD_NAMES[aid]
    prefix = ""
    if "/" in aid:
        prefix, aid = aid.split("/", 1)
        prefix = {"fedavg": "FedAvg ", "local_only": "Local-only ", "centralised": "Centralised "}.get(prefix, prefix + " ")
    node, kind, rnd = aid.split(":")
    who = names.get(node, node) if not prefix.startswith("Centralised") else "all data"
    return f"{prefix}{who}" + (" [aggregated]" if kind == "aggregated" and not prefix else "") + f", round {rnd[1:]}"


# --------------------------------------------------------------------------------------------- quality
def quality_tables(d, tag, meta, kind):
    p = pd.read_csv(os.path.join(d, "raw", "predictions.csv"), keep_default_na=False)
    gen = meta.get("task") == "generate"
    labels, n_nodes = meta["labels"], meta["n_nodes"]
    n_cls = len(labels)
    names = adapter_names(meta)
    main = p[~p["adapter_id"].str.contains("/") & ~p["adapter_id"].isin(["base", "merged_avg", "merged_ties"])]
    last = int(pd.to_numeric(main["round"]).max())
    final_ids = set(p["adapter_id"]) - set(main[pd.to_numeric(main["round"]) != last]["adapter_id"])
    fin = p[p["adapter_id"].isin(final_ids)].copy()

    if gen:
        fin = generation_scores(fin, tag)
        fin["score"] = fin["bertscore_f1"]
    else:
        fin["correct"] = pd.to_numeric(fin["correct"])
        fin["pred"] = pd.to_numeric(fin["pred"])
        fin["score"] = fin["correct"]

    # adapter x test set
    rows = []
    for (aid, ts), g in fin.groupby(["adapter_id", "test_set"]):
        r = {"adapter": aid, "adapter_name": pretty_adapter(aid, names), "test_set": ts, "n_questions": len(g)}
        if gen:
            r.update(bertscore_f1=g["bertscore_f1"].mean(), embedding_cosine=g["embedding_cosine"].mean(), rouge_l=g["rouge_l"].mean())
        else:
            r.update(accuracy=g["correct"].mean(), macro_f1=macro_f1(g["label"], g["pred"], n_cls),
                     unparseable_answer_rate=pd.to_numeric(g["invalid"]).mean())
        rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(d, "tables", ("generation_quality" if gen else "accuracy") + "_adapter_vs_test_set.csv"), index=False)

    if not gen:
        g = fin[fin["test_set"] == "global"]
        pc = g.groupby(["adapter_id", "label_text"])["correct"].mean().reset_index()
        pc.columns = ["adapter", "topic", "accuracy"]
        pc.insert(1, "adapter_name", pc["adapter"].map(lambda x: pretty_adapter(x, names)))
        pc.to_csv(os.path.join(d, "tables", "accuracy_per_topic.csv"), index=False)
        br = p[(p["test_set"] == "global") & ~p["adapter_id"].str.contains("/") & ~p["adapter_id"].str.startswith("merged")].copy()
        br["correct"] = pd.to_numeric(br["correct"])
        br = br.groupby(["adapter_node", "kind", "round"])["correct"].mean().reset_index(name="accuracy")
        br.loc[br["kind"] == "", "kind"] = "base model"
        br.insert(0, "adapter_name", br["adapter_node"].map(lambda n: names.get(n, "Base model")))
        br.to_csv(os.path.join(d, "tables", "accuracy_by_round.csv"), index=False)

    # ------------------------------------------------------------ per query node: every way of answering
    local_ids = {f"node{k}": f"node{k}:{kind}:r{last}" for k in range(1, n_nodes + 1)}
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    cent = node_descriptors(tag, n_nodes, st)
    order = list(cent)
    C = np.stack([cent[n] for n in order])
    srows = []
    for ts in ["global"] + [f"local{k}" for k in range(1, n_nodes + 1)]:
        sub = fin[fin["test_set"] == ts]
        if sub.empty:
            continue
        S = sub[sub["adapter_id"].isin(local_ids.values())].pivot_table(index="qid", columns="adapter_id", values="score")
        P = None if gen else sub[sub["adapter_id"].isin(local_ids.values())].pivot_table(index="qid", columns="adapter_id", values="pred")
        qtext = {}
        path = os.path.join(REPO, "experiments", "data", tag, "test_global.jsonl" if ts == "global" else f"test_node{ts[5:]}.jsonl")
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            qtext[r["id"]] = r["text"]
        qids = [q for q in S.index if q in qtext]
        route = [order[i] for i in (st.encode([qtext[q] for q in qids], normalize_embeddings=True) @ C.T).argmax(1)]
        lab = sub.drop_duplicates("qid").set_index("qid").loc[qids, "label"].astype(int).values

        def one(aid):
            g = sub[sub["adapter_id"] == aid].drop_duplicates("qid").set_index("qid")
            return g.loc[qids] if len(g) else None

        for node in order:
            if ts not in ("global", f"local{node[4:]}"):
                continue
            which = "all topics (balanced test)" if ts == "global" else "node's own mix (local test)"

            def add(method, score, pred=None):
                r = {"query_node": node, "query_node_name": names[node], "test_set": ts, "evaluated_on": which,
                     "method": method, "method_name": METHOD_NAMES.get(method, method), "n_questions": len(qids),
                     ("mean_bertscore_f1" if gen else "accuracy"): float(np.mean(score))}
                if pred is not None and not gen:
                    r["macro_f1"] = macro_f1(lab, pred, n_cls)
                srows.append(r)

            for method, aid in [("base", "base"), ("own_local", local_ids[node]),
                                ("own_aggregated", f"{node}:aggregated:r{last}"),
                                ("merged_avg", "merged_avg"), ("merged_ties", "merged_ties"),
                                ("fedavg", f"fedavg/{node}:aggregated:r3"),
                                ("local_only", f"local_only/{node}:local:r3"),
                                ("centralised", "centralised/node1:aggregated:r3")]:
                g = one(aid)
                if g is not None:
                    add(method, g["score"].values, None if gen else g["pred"].values)
            add("random_adapter", S.loc[qids].mean(axis=1).values)
            qa = np.array([S.loc[q, local_ids[r]] for q, r in zip(qids, route)])
            add("query_aware", qa, None if gen else np.array([P.loc[q, local_ids[r]] for q, r in zip(qids, route)]))
            if not gen:
                V = P.loc[qids].values
                mv = np.array([np.bincount(v[v >= 0].astype(int), minlength=n_cls).argmax() if (v >= 0).any() else -1 for v in V])
                add("majority_vote", (mv == lab).astype(float), mv)
            add("oracle", S.loc[qids].max(axis=1).values)
    out = pd.DataFrame(srows)
    out.to_csv(os.path.join(d, "tables", "generation_quality_by_method.csv" if gen else "accuracy_by_selection_method.csv"), index=False)
    if gen:
        # extra metric columns per method, on the balanced test set
        pass


def node_descriptors(tag, n_nodes, st):
    """Mean sentence embedding of each node's training inputs: the descriptor a node could advertise."""
    out = {}
    for k in range(1, n_nodes + 1):
        texts = []
        for line in open(os.path.join(REPO, "experiments", "data", tag, f"node{k}", "train.jsonl"), encoding="utf-8"):
            u = json.loads(line)["messages"][1]["content"]
            texts.append(u.split("Text: ", 1)[-1].rsplit("\n\nAnswer with only", 1)[0])
        e = st.encode(texts[:2000], normalize_embeddings=True).mean(0)
        out[f"node{k}"] = e / np.linalg.norm(e)
    return out


def generation_scores(fin, tag):
    """BERTScore-F1 (roberta-large, rescaled), MiniLM cosine and ROUGE-L of each answer vs the human reference."""
    refs = {}
    import glob
    for path in glob.glob(os.path.join(REPO, "experiments", "data", tag, "test_*.jsonl")):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            refs[r["id"]] = r["reference"]
    fin = fin.copy()
    fin["reference"] = fin["qid"].map(refs)
    cands, refl = fin["raw"].astype(str).tolist(), fin["reference"].astype(str).tolist()
    from bert_score import score as bscore
    _, _, f1 = bscore(cands, refl, model_type="roberta-large", lang="en", rescale_with_baseline=True,
                      device="cuda", batch_size=64, verbose=False)
    fin["bertscore_f1"] = f1.numpy()
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    fin["embedding_cosine"] = (st.encode(cands, normalize_embeddings=True) * st.encode(refl, normalize_embeddings=True)).sum(1)
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    fin["rouge_l"] = [sc.score(r, c)["rougeL"].fmeasure for c, r in zip(cands, refl)]
    return fin


# --------------------------------------------------------------------------------------------- system
def model_flops(model, L):
    """Prefill FLOPs for an L-token prompt: 2 x non-embedding params x L  +  causal attention 2 x layers x L^2 x d."""
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(model)
    d, n_layers, ff = c.hidden_size, c.num_hidden_layers, c.intermediate_size
    kv = c.num_key_value_heads * (d // c.num_attention_heads)
    per_layer = d * d * 2 + d * kv * 2 + 3 * d * ff  # q,o + k,v + gate,up,down
    return 2 * n_layers * per_layer * L + 2 * n_layers * L * L * d


def serving_tables(d, model):
    frames = []
    for b in ("slakshna", "vllm"):
        path = os.path.join(d, "raw", f"serving_{b}.csv")
        if os.path.isfile(path):
            frames.append(pd.read_csv(path))
    if not frames:
        return
    s = pd.concat(frames)
    s = s[(s["ok"] == 1) & (s["sweep"] != "warmup")].copy()
    s["system"] = s["backend"].map({"slakshna": "Ours (federated, local GPU)", "vllm": "vLLM multi-LoRA server"})
    q = lambda x, p: x.quantile(p)

    pl = s[s["sweep"] == "prompt_len"].groupby(["system", "prompt_tokens"]).agg(
        ttft_ms_mean=("ttft_ms", "mean"), ttft_ms_p50=("ttft_ms", "median"), ttft_ms_p95=("ttft_ms", lambda x: q(x, .95)),
        model_ttft_ms=("server_model_ttft_ms", lambda x: pd.to_numeric(x, errors="coerce").mean())).reset_index()
    t = pl["model_ttft_ms"].fillna(pl["ttft_ms_mean"])  # pure prefill time when the server reports it
    pl["prefill_tflops_per_s"] = [model_flops(model, L) / (ms / 1e3) / 1e12 for L, ms in zip(pl["prompt_tokens"], t)]
    pl["mfu_percent"] = 100 * pl["prefill_tflops_per_s"] / A6000_BF16_PEAK_TFLOPS
    pl.to_csv(os.path.join(d, "tables", "latency_vs_prompt_length.csv"), index=False)

    ol = s[s["sweep"] == "output_len"].groupby(["system", "output_tokens"]).agg(
        tpot_ms_mean=("tpot_ms", "mean"), tpot_ms_p95=("tpot_ms", lambda x: q(x, .95)),
        e2e_ms_mean=("e2e_ms", "mean"), e2e_ms_p95=("e2e_ms", lambda x: q(x, .95))).reset_index()
    ol["output_tokens_per_s"] = 1000.0 / ol["tpot_ms_mean"]
    ol.to_csv(os.path.join(d, "tables", "latency_vs_output_length.csv"), index=False)

    rows = []
    for (sysn, C), g in s[s["sweep"] == "concurrency"].groupby(["system", "concurrency"]):
        span = (g["t_end"].max() - g["t_start"].min())
        good = ((g["ttft_ms"] <= SLO_TTFT_MS) & (g["tpot_ms"] <= SLO_TPOT_MS)).sum()
        rows.append({"system": sysn, "concurrent_requests": C, "requests": len(g),
                     "throughput_tokens_per_s": g["n_out"].sum() / span, "throughput_requests_per_s": len(g) / span,
                     "ttft_ms_p50": g["ttft_ms"].median(), "ttft_ms_p95": q(g["ttft_ms"], .95),
                     "tpot_ms_p50": g["tpot_ms"].median(), "e2e_ms_p95": q(g["e2e_ms"], .95),
                     "goodput_requests_per_s": good / span, "slo": f"TTFT<={SLO_TTFT_MS:.0f}ms & TPOT<={SLO_TPOT_MS:.0f}ms"})
    pd.DataFrame(rows).to_csv(os.path.join(d, "tables", "throughput_vs_concurrency.csv"), index=False)


def api_tables(d):
    path = os.path.join(d, "raw", "api_queries.csv")
    if not os.path.isfile(path):
        return
    q = pd.read_csv(path, keep_default_na=False)
    for c in ["select_ms", "fetch_ms", "attach_ms", "model_ttft_ms", "ttft_ms", "total_ms"]:
        q[c] = pd.to_numeric(q[c], errors="coerce")
    fe = q[q["phase"] == "fetch"]
    rows = []
    for grp, g in fe.groupby("method"):
        rows.append({"case": "Cold: adapter fetched by hash from a peer" if grp == "cold" else "Warm: adapter already cached",
                     "queries": len(g), "rank_ms": g["select_ms"].mean(), "fetch_and_verify_ms": g["fetch_ms"].mean(),
                     "attach_ms": g["attach_ms"].mean(), "model_first_token_ms": g["model_ttft_ms"].mean(),
                     "ttft_ms_mean": g["ttft_ms"].mean(), "ttft_ms_p50": g["ttft_ms"].median(),
                     "ttft_ms_p95": g["ttft_ms"].quantile(.95)})
    pd.DataFrame(rows).to_csv(os.path.join(d, "tables", "latency_cold_vs_warm.csv"), index=False)
    st = q[q["phase"] == "strategy"].copy()
    st["correct"] = pd.to_numeric(st["correct"])
    res = st.groupby("method").agg(queries=("qid", "size"), accuracy=("correct", "mean"),
                                   ttft_ms_mean=("ttft_ms", "mean"), ttft_ms_p95=("ttft_ms", lambda x: x.quantile(.95)),
                                   adapters_chosen=("adapter_node", lambda x: json.dumps(x.value_counts().to_dict()))).reset_index()
    res["method_name"] = res["method"].map({"self": "Own adapter (self)", "trust": "Highest trust (trust)",
                                            "latest": "Newest round (latest)", "random": "Random node"})
    res.to_csv(os.path.join(d, "tables", "latency_by_selection_method.csv"), index=False)


def fault_table(d):
    path = os.path.join(d, "raw", "fault_tolerance.csv")
    if not os.path.isfile(path):
        return
    f = pd.read_csv(path, keep_default_na=False)
    for c in ["ok", "correct", "fell_back", "total_ms", "peers_down"]:
        f[c] = pd.to_numeric(f[c], errors="coerce")
    out = f.groupby(["peers_down", "phase", "method"]).agg(
        queries=("qid", "size"), answered_percent=("ok", lambda x: 100 * x.mean()), accuracy=("correct", "mean"),
        fell_back_percent=("fell_back", lambda x: 100 * x.mean()), latency_ms_mean=("total_ms", "mean"),
        latency_ms_p95=("total_ms", lambda x: x.quantile(.95)),
        adapters_used=("adapter_node", lambda x: json.dumps(x.value_counts().to_dict()))).reset_index()
    out.to_csv(os.path.join(d, "tables", "fault_tolerance.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dir", default=None, help="results folder (default experiments/results/<tag>)")
    ap.add_argument("--kind", default="local", help="adapter kind the nodes publish")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    a = ap.parse_args()
    d = a.dir or os.path.join(REPO, "experiments", "results", a.tag)
    os.makedirs(os.path.join(d, "tables"), exist_ok=True)
    meta = json.load(open(os.path.join(REPO, "experiments", "data", a.tag, "meta.json")))
    if os.path.isfile(os.path.join(d, "raw", "predictions.csv")):
        quality_tables(d, a.tag, meta, a.kind)
    api_tables(d)
    serving_tables(d, a.model)
    fault_table(d)
    fs = os.path.join(d, "raw", "fetch_vs_adapter_size.csv")
    if os.path.isfile(fs):
        f = pd.read_csv(fs)
        f.groupby(["lora_rank", "adapter_mb"]).agg(fetch_ms_mean=("fetch_ms", "mean"), fetch_ms_p95=("fetch_ms", lambda x: x.quantile(.95)),
                                                   attach_ms_mean=("attach_ms", "mean"), ttft_ms_mean=("ttft_ms", "mean"),
                                                   reps=("fetch_ms", "size")).reset_index().to_csv(
            os.path.join(d, "tables", "fetch_vs_adapter_size.csv"), index=False)
    print("tables:", sorted(os.listdir(os.path.join(d, "tables"))))


if __name__ == "__main__":
    main()
