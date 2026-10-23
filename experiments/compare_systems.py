"""
Where our serverless federated inference wins and loses against other ways of serving the same adapters.

Systems:
  ours        each node keeps its own model; peers' adapters are fetched by hash on demand (MEASURED)
  vllm        one central vLLM server holding every node's adapter, users send prompts to it (MEASURED)
  layersplit  Petals-style decentralised inference: the model's layers are spread over the nodes and every token's
              hidden state travels through all of them (ANALYTICAL, from the model's hidden size and a network RTT)
  cloud       a hosted API: prompt and answer go to a provider (ANALYTICAL, same traffic as a central server)

Measured inputs (experiments/results/<tag>/): tables/throughput_vs_concurrency.csv, tables/latency_vs_prompt_length.csv,
tables/latency_vs_output_length.csv, tables/latency_cold_vs_warm.csv, tables/fault_tolerance.csv, the run's adapters and
node logs (gossip propagation delay of adapter metadata), and the test prompts (bytes a user would send).

Outputs: tables/communication_bytes.csv, tables/baseline_comparison.csv, figures/communication_bytes.png,
         figures/baseline_comparison.png (README section appended by plot_results.py on its next run)

    Bhaskera-serve/.venv/bin/python experiments/compare_systems.py --tag ag_news_n4_a0.1_s0 --rtt-ms 40
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(REPO, "hf_cache"))
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def gossip_delays(run):
    """Seconds between a node publishing an adapter and each peer registering it (from the training logs)."""
    pub, seen = {}, []
    for path in glob.glob(os.path.join(REPO, "experiments", "runs", run, "logs", "node*_train.log")):
        node = os.path.basename(path).split("_")[0]
        for line in open(path, errors="ignore"):
            line = ANSI.sub("", line)
            m = re.search(r"^\s*(\S+Z)\s+INFO\s+🧩 Publishing adapter round (\d+) sha256 (\w+)", line)
            if m:
                pub[m.group(3)] = (datetime.fromisoformat(m.group(1).replace("Z", "+00:00")), node)
            m = re.search(r"^\s*(\S+Z)\s+INFO\s+🧩 Adapter advertised by \S+ \(round (\d+), sha256 (\w+)\)", line)
            if m:
                seen.append((m.group(3), datetime.fromisoformat(m.group(1).replace("Z", "+00:00")), node))
    out = []
    for sha, t, node in seen:
        if sha in pub and pub[sha][1] != node:
            out.append((t - pub[sha][0]).total_seconds())
    return np.array(out)


def metadata_bytes(run):
    """Size of the signed adapter advertisement as stored in a node's registry (RocksDB value)."""
    sizes = []
    for f in glob.glob(os.path.join(REPO, "experiments", "runs", run, "data-node1", "rocksdb", "*")):
        if f.endswith((".log", ".sst")):
            data = open(f, "rb").read().decode("latin-1")
            sizes += [len(m) for m in re.findall(r'\{"node_id".*?"signature":"[0-9a-f]+"\}', data)]
    return float(np.median(sizes)) if sizes else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--rtt-ms", type=float, default=40.0, help="network round-trip time assumed for the analytical systems")
    ap.add_argument("--queries-per-adapter", type=int, default=100, help="queries served per fetched adapter (amortisation)")
    a = ap.parse_args()
    d = os.path.join(REPO, "experiments", "results", a.tag)
    T = lambda n: pd.read_csv(os.path.join(d, "tables", n + ".csv")) if os.path.isfile(os.path.join(d, "tables", n + ".csv")) else None
    from transformers import AutoConfig, AutoTokenizer
    cfg = AutoConfig.from_pretrained(a.model)
    tok = AutoTokenizer.from_pretrained(a.model)
    hidden, layers = cfg.hidden_size, cfg.num_hidden_layers
    n_nodes = json.load(open(os.path.join(REPO, "experiments", "data", a.tag, "meta.json")))["n_nodes"]

    # ---- what a user query looks like on the wire --------------------------------------------------------------
    qs = [json.loads(l) for l in open(os.path.join(REPO, "experiments", "data", a.tag, "test_global.jsonl"))][:500]
    prompt_bytes = np.mean([len(json.dumps({"prompt": q["prompt"]}).encode()) for q in qs])
    answer_bytes = np.mean([len(json.dumps({"text": q["label_text"]}).encode()) for q in qs]) + 200  # + response envelope
    prompt_tokens = np.mean([len(tok(q["prompt"])["input_ids"]) for q in qs])
    out_tokens = 3
    adapters = [p for p in glob.glob(os.path.join(REPO, "experiments", "runs", a.tag, "data-node*", "adapters", "*", "adapter_model.safetensors"))]
    adapter_bytes = float(np.median([os.path.getsize(p) for p in adapters])) if adapters else float("nan")
    meta_bytes = metadata_bytes(a.tag)  # signed AdapterMeta JSON (node, round, sha256, blake3, size, endpoint, signature, ...)
    delays = gossip_delays(a.tag)

    # Petals-style: every boundary between consecutive nodes carries the hidden state (bf16) of each token,
    # there and back through the chain (client -> n servers -> client).
    hops = n_nodes + 1
    lsplit_prompt = prompt_tokens * hidden * 2 * hops
    lsplit_token = hidden * 2 * hops

    comm = pd.DataFrame([
        {"system": "Ours: federated inference", "user_prompt_bytes_sent_off_device": 0,
         "network_bytes_per_query_warm": 0, "network_bytes_one_time_per_adapter": adapter_bytes,
         "network_bytes_per_query_amortised": adapter_bytes / a.queries_per_adapter,
         "metadata_bytes_per_adapter_per_peer": meta_bytes, "round_trips_per_generated_token": 0, "source": "measured"},
        {"system": "Central multi-LoRA server (vLLM)", "user_prompt_bytes_sent_off_device": prompt_bytes,
         "network_bytes_per_query_warm": prompt_bytes + answer_bytes, "network_bytes_one_time_per_adapter": adapter_bytes,
         "network_bytes_per_query_amortised": prompt_bytes + answer_bytes, "metadata_bytes_per_adapter_per_peer": 0,
         "round_trips_per_generated_token": 0, "source": "measured request sizes"},
        {"system": "Layer-split decentralised (Petals-style)", "user_prompt_bytes_sent_off_device": prompt_tokens * hidden * 2,
         "network_bytes_per_query_warm": lsplit_prompt + out_tokens * lsplit_token, "network_bytes_one_time_per_adapter": 0,
         "network_bytes_per_query_amortised": lsplit_prompt + out_tokens * lsplit_token, "metadata_bytes_per_adapter_per_peer": 0,
         "round_trips_per_generated_token": hops, "source": f"analytical: hidden={hidden}, bf16, {hops} transfers per token"},
        {"system": "Cloud API", "user_prompt_bytes_sent_off_device": prompt_bytes,
         "network_bytes_per_query_warm": prompt_bytes + answer_bytes, "network_bytes_one_time_per_adapter": adapter_bytes,
         "network_bytes_per_query_amortised": prompt_bytes + answer_bytes, "metadata_bytes_per_adapter_per_peer": 0,
         "round_trips_per_generated_token": 0, "source": "analytical"},
    ])
    os.makedirs(os.path.join(d, "tables"), exist_ok=True)
    comm.to_csv(os.path.join(d, "tables", "communication_bytes.csv"), index=False)

    # ---- latency / throughput / availability -------------------------------------------------------------------
    tc, pl, ol, cw, ft = (T("throughput_vs_concurrency"), T("latency_vs_prompt_length"), T("latency_vs_output_length"),
                          T("latency_cold_vs_warm"), T("fault_tolerance"))

    def pick(df, sysname, col, where=None):
        if df is None:
            return np.nan
        g = df[df["system"] == sysname]
        if where:
            k, v = where
            g = g[g[k] == v]
        return float(g[col].iloc[0]) if len(g) else np.nan

    O, V = "Ours (federated, local GPU)", "vLLM multi-LoRA server"
    rows = []
    rows.append({"system": "Ours: federated inference",
                 "ttft_ms_256_token_prompt": pick(pl, O, "ttft_ms_p50", ("prompt_tokens", 256)),
                 "extra_ttft_ms_first_use_of_peer_adapter": (float(cw.iloc[0]["ttft_ms_mean"] - cw.iloc[-1]["ttft_ms_mean"]) if cw is not None else np.nan),
                 "tpot_ms": pick(ol, O, "tpot_ms_mean", ("output_tokens", 128)),
                 "throughput_tokens_per_s_16_concurrent_one_machine": pick(tc, O, "throughput_tokens_per_s", ("concurrent_requests", 16)),
                 "per_token_network_latency_ms": 0.0,
                 "answered_percent_when_a_machine_fails": (float(ft[ft["peers_down"] > 0]["answered_percent"].min()) if ft is not None else np.nan),
                 "single_point_of_failure": "no", "needs_a_server": "no",
                 "adapter_known_to_peers_after_ms": float(np.median(delays)) * 1000 if len(delays) else np.nan,
                 "where_all_adapters_are_stored": "nowhere central (own + cached peers)"})
    rows.append({"system": "Central multi-LoRA server (vLLM)",
                 "ttft_ms_256_token_prompt": pick(pl, V, "ttft_ms_p50", ("prompt_tokens", 256)),
                 "extra_ttft_ms_first_use_of_peer_adapter": 0.0,
                 "tpot_ms": pick(ol, V, "tpot_ms_mean", ("output_tokens", 128)),
                 "throughput_tokens_per_s_16_concurrent_one_machine": pick(tc, V, "throughput_tokens_per_s", ("concurrent_requests", 16)),
                 "per_token_network_latency_ms": 0.0, "answered_percent_when_a_machine_fails": 0.0,
                 "single_point_of_failure": "yes (the server)", "needs_a_server": "yes",
                 "adapter_known_to_peers_after_ms": np.nan, "where_all_adapters_are_stored": "the server"})
    rows.append({"system": "Layer-split decentralised (Petals-style)", "ttft_ms_256_token_prompt": np.nan,
                 "extra_ttft_ms_first_use_of_peer_adapter": np.nan, "tpot_ms": np.nan,
                 "throughput_tokens_per_s_16_concurrent_one_machine": np.nan,
                 "per_token_network_latency_ms": hops * a.rtt_ms,
                 "answered_percent_when_a_machine_fails": np.nan,  # Petals reroutes if another server holds the same layers
                 "single_point_of_failure": "no, if every layer range has a second server", "needs_a_server": "no",
                 "adapter_known_to_peers_after_ms": np.nan, "where_all_adapters_are_stored": "split across servers"})
    cmp_ = pd.DataFrame(rows)
    cmp_["note"] = ["measured on this machine", "measured on this machine, same model and adapters",
                    f"analytical, {hops} network transfers per generated token at RTT {a.rtt_ms:.0f} ms (not run)"]
    cmp_.to_csv(os.path.join(d, "tables", "baseline_comparison.csv"), index=False)

    # ---- figures -----------------------------------------------------------------------------------------------
    os.makedirs(os.path.join(d, "figures"), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    c = comm.set_index("system")
    vals = c["network_bytes_per_query_amortised"].replace(0, 1)
    cols = ["#2E6FB7", "#D1495B", "#8C6BB1", "#9AA4AE"]
    ax.barh(c.index, vals, color=cols)
    for i, (v, pb) in enumerate(zip(c["network_bytes_per_query_amortised"], c["user_prompt_bytes_sent_off_device"])):
        ax.text(max(v, 1) * 1.15, i, f"{v / 1024:,.1f} KB per query; prompt sent off device: {'no' if pb == 0 else 'yes'}", va="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("network bytes per query (log scale; ours = adapter download spread over "
                  f"{a.queries_per_adapter} queries)")
    ax.set_xlim(right=vals.max() * 1e3)
    ax.set_title("Network traffic per query and privacy of the prompt")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(d, "figures", f"communication_bytes.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(comm.to_string())
    print(cmp_.T.to_string())
    if len(delays):
        print(f"adapter metadata reached peers after median {np.median(delays):.2f}s (p95 {np.percentile(delays, 95):.2f}s, n={len(delays)})")


if __name__ == "__main__":
    main()
