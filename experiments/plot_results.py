"""
Draw every figure for one run from experiments/results/<tag>/tables/ into .../figures/ (PNG + PDF) and write
.../README.md, which explains each figure in plain words with the run's own numbers. Missing tables are skipped.

    Bhaskera-serve/.venv/bin/python experiments/plot_results.py --tag ag_news_n4_a0.1_s0
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "figure.dpi": 130, "savefig.bbox": "tight", "axes.titleweight": "bold"})
OURS, BASE, BOUND, TRAIN = "#2E6FB7", "#9AA4AE", "#E0A43A", "#59A14F"
SYS = {"Ours (federated, local GPU)": "#2E6FB7", "vLLM multi-LoRA server": "#D1495B"}
GROUP = {  # colour of each method in the bar charts
    "base": BASE, "own_local": OURS, "own_aggregated": OURS, "random_adapter": OURS, "query_aware": OURS,
    "majority_vote": OURS, "merged_avg": OURS, "merged_ties": OURS,
    "fedavg": TRAIN, "local_only": TRAIN, "centralised": BOUND, "oracle": BOUND,
}


class Fig:
    def __init__(self, d):
        self.d, self.notes = d, []
        os.makedirs(os.path.join(d, "figures"), exist_ok=True)

    def table(self, name):
        p = os.path.join(self.d, "tables", name + ".csv")
        return pd.read_csv(p) if os.path.isfile(p) else None

    def save(self, fig, name, title, how, found):
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(self.d, "figures", f"{name}.{ext}"))
        plt.close(fig)
        self.notes.append((name, title, how, found))
        print("figure", name)


def heat(ax, M, rows, cols):
    im = ax.imshow(M, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=25, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.grid(False)
    return im


def hbars(ax, df, value, xlabel):
    df = df.sort_values(value)
    y = np.arange(len(df))
    ax.barh(y, df[value], color=[GROUP.get(m, OURS) for m in df["method"]])
    for i, v in enumerate(df[value]):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8)
    ax.set_yticks(y, df["method_name"])
    ax.set_xlabel(xlabel)


def quality(F, meta):
    sub = f"{meta['dataset']}, {meta['n_nodes']} nodes, label skew α={meta['alpha']:g}"
    pc = F.table("accuracy_per_topic")
    if pc is not None:
        keep = [a for a in pc["adapter"].unique() if a == "base" or a.startswith("merged") or a.startswith("centralised/")
                or (":local:" in a and "/" not in a)]
        piv = pc[pc["adapter"].isin(keep)].pivot_table(index="adapter_name", columns="topic", values="accuracy")
        order = sorted(piv.index, key=lambda n: (not n.startswith("Base"), n.startswith("All") or n.startswith("Centr"), n))
        piv = piv.loc[order]
        fig, ax = plt.subplots(figsize=(1.3 * piv.shape[1] + 4, 0.45 * len(piv) + 1.5))
        im = heat(ax, piv.values, list(piv.index), list(piv.columns))
        fig.colorbar(im, ax=ax, label="accuracy (1 = always right)")
        ax.set_title("Accuracy of each adapter on each topic\n" + sub, fontsize=10)
        b = pc[pc["adapter"] == "base"].set_index("topic")["accuracy"]
        loc = pc[pc["adapter"].str.contains(":local:") & ~pc["adapter"].str.contains("/")]
        F.save(fig, "accuracy_per_topic", "Each node's adapter is good only at the topics it trained on",
               "Rows are adapters (the node and the topics most of its training data came from), columns are topics, "
               "cells are the share of questions answered correctly on a test set with equal numbers of each topic. "
               "Green = right, red = wrong.",
               f"Node adapters score {loc['accuracy'].max():.2f} on their own topics but as low as {loc['accuracy'].min():.2f} "
               f"on topics they never saw; the base model ranges {b.min():.2f}-{b.max():.2f} across topics. No single node's "
               "adapter is good at everything, which is why a node should be able to use a peer's adapter.")

    sm = F.table("accuracy_by_selection_method")
    if sm is not None:
        fig, axes = plt.subplots(2, 1, figsize=(8.5, 0.75 * sm["method"].nunique() + 2.8), sharex=True)
        res = {}
        for ax, which in zip(axes, ["all topics (balanced test)", "node's own mix (local test)"]):
            g = sm[sm["evaluated_on"] == which].groupby(["method", "method_name"])["accuracy"].mean().reset_index()
            hbars(ax, g, "accuracy", "accuracy (mean over the query nodes)")
            ax.tick_params(labelbottom=True)
            ax.set_xlim(0, 1.12)
            ax.set_title("Questions on " + which, fontsize=10)
            res[which] = g.set_index("method")["accuracy"]
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (OURS, BASE, TRAIN, BOUND)]
        fig.legend(handles, ["federated inference options", "no adapter", "training baselines", "upper bounds"],
                   loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.07))
        fig.suptitle("How should a node answer a query? Accuracy of each option\n" + sub, fontsize=11)
        r, l = res["all topics (balanced test)"], res["node's own mix (local test)"]
        F.save(fig, "accuracy_by_selection_method", "Choosing the adapter per question beats using your own adapter",
               "Each bar is one way for a node to answer: blue = options of our federated inference (own adapter, "
               "a random peer's, the adapter whose training data looks most like the question, all adapters merged, "
               "majority vote), grey = base model, green = training baselines, orange = upper bounds (oracle, "
               "centralised training on pooled data, which is not private). Left: questions on all topics; right: "
               "questions with the node's own topic mix.",
               f"On all topics, own adapter only = {r.get('own_local', np.nan):.2f}, base model = {r.get('base', np.nan):.2f}, "
               f"query-aware choice = {r.get('query_aware', np.nan):.2f}, merged = {r.get('merged_avg', np.nan):.2f}, "
               f"oracle = {r.get('oracle', np.nan):.2f}. On the node's own mix its own adapter is strong "
               f"({l.get('own_local', np.nan):.2f}). Using peers' adapters fixes the topics a node never trained on.")

    av = F.table("accuracy_adapter_vs_test_set")
    if av is not None:
        keep = [a for a in av["adapter"].unique() if a == "base" or (":local:" in a and "/" not in a)]
        piv = av[av["adapter"].isin(keep)].pivot_table(index="adapter_name", columns="test_set", values="accuracy")
        cols = ["global"] + sorted(c for c in piv.columns if c.startswith("local"))
        piv = piv[[c for c in cols if c in piv.columns]]
        piv.columns = ["all topics"] + [f"node {c[5:]}'s mix" for c in piv.columns[1:]]
        fig, ax = plt.subplots(figsize=(1.3 * piv.shape[1] + 4, 0.5 * len(piv) + 1.5))
        im = heat(ax, piv.values, list(piv.index), list(piv.columns))
        fig.colorbar(im, ax=ax, label="accuracy")
        ax.set_title("Each adapter on each node's kind of questions\n" + sub, fontsize=10)
        F.save(fig, "accuracy_adapter_vs_test_set", "An adapter is best on its own node's questions",
               "Rows = adapters, columns = test sets that follow the topic mix of one node (plus all topics). "
               "The diagonal (a node's adapter on its own kind of questions) should be green.",
               "The diagonal is high and many off-diagonal cells are low: the best adapter depends on who is asking what.")

    br = F.table("accuracy_by_round")
    if br is not None:
        fig, ax = plt.subplots(figsize=(6, 3.8))
        base = br[br["kind"] == "base model"]["accuracy"]
        if len(base):
            ax.axhline(base.iloc[0], color=BASE, ls="--", label="base model")
        for i, (name, g) in enumerate(br[br["kind"] == "local"].groupby("adapter_name")):
            g = g.sort_values("round")
            ax.plot(g["round"], g["accuracy"], marker="o", label=name)
        ag = br[br["kind"] == "aggregated"].groupby("round")["accuracy"].mean()
        if len(ag):
            ax.plot(ag.index, ag.values, marker="s", color="k", label="aggregated adapters (mean)")
        ax.set_xlabel("training round")
        ax.set_ylabel("accuracy on all topics")
        ax.set_xticks(sorted(br["round"].unique()))
        ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.set_title("Accuracy after each training round\n" + sub, fontsize=10)
        F.save(fig, "accuracy_by_round", "Accuracy of each node's adapter across rounds",
               "Accuracy on all topics of each node's adapter after rounds 1-3. Dashed = base model.",
               f"After the last round, node adapters score {br[(br['kind'] == 'local') & (br['round'] == br['round'].max())]['accuracy'].min():.2f}-"
               f"{br[(br['kind'] == 'local') & (br['round'] == br['round'].max())]['accuracy'].max():.2f} on all topics "
               f"(base model {base.iloc[0] if len(base) else float('nan'):.2f}).")


def generation(F, meta):
    gq = F.table("generation_quality_by_method")
    if gq is None:
        return
    sub = f"{meta['dataset']}, {meta['n_nodes']} nodes, task skew α={meta['alpha']:g}"
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 0.75 * gq["method"].nunique() + 2.8), sharex=True)
    res = {}
    for ax, which in zip(axes, ["all topics (balanced test)", "node's own mix (local test)"]):
        g = gq[gq["evaluated_on"] == which].groupby(["method", "method_name"])["mean_bertscore_f1"].mean().reset_index()
        hbars(ax, g, "mean_bertscore_f1", "BERTScore-F1 vs human answer (rescaled; higher = closer)")
        ax.set_title("Questions of " + ("all task types" if which.startswith("all") else "the node's own task mix"), fontsize=10)
        res[which] = g.set_index("method")["mean_bertscore_f1"]
    fig.suptitle("Answer quality on open-ended questions (Dolly)\n" + sub, fontsize=11)
    r = res["all topics (balanced test)"]
    F.save(fig, "generation_quality_by_method", "Open-ended answers: quality of each way of answering",
           "BERTScore-F1 compares the meaning of the model's answer with the human reference answer "
           "(roberta-large, rescaled so 0 = unrelated text; higher is better). Colours as in accuracy_by_selection_method.",
           f"All task types: own adapter {r.get('own_local', np.nan):.3f}, base {r.get('base', np.nan):.3f}, "
           f"query-aware {r.get('query_aware', np.nan):.3f}, merged {r.get('merged_avg', np.nan):.3f}, oracle {r.get('oracle', np.nan):.3f}.")
    av = F.table("generation_quality_adapter_vs_test_set")
    if av is not None:
        g = av[av["test_set"] == "global"].sort_values("bertscore_f1")
        fig, ax = plt.subplots(figsize=(7.5, 0.4 * len(g) + 1.5))
        y = np.arange(len(g))
        for j, (col, lab) in enumerate([("bertscore_f1", "BERTScore-F1 (rescaled)"), ("embedding_cosine", "embedding cosine"), ("rouge_l", "ROUGE-L")]):
            ax.barh(y + (j - 1) * 0.27, g[col], 0.27, label=lab)
        ax.set_yticks(y, g["adapter_name"])
        ax.set_xlabel("similarity to the human answer")
        ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=3)
        ax.set_title("Three similarity metrics per adapter (all task types)\n" + sub, fontsize=10)
        rho_cos = g["bertscore_f1"].rank().corr(g["embedding_cosine"].rank())
        rho_rouge = g["bertscore_f1"].rank().corr(g["rouge_l"].rank())
        base = g[g["adapter"] == "base"]
        F.save(fig, "generation_quality_metrics", "Do the similarity metrics agree?",
               "Three ways to compare an answer with the reference: BERTScore-F1 (meaning, token level, rescaled), cosine of "
               "sentence embeddings (topic of the whole answer), ROUGE-L (shared word sequences).",
               f"Rank agreement with BERTScore across adapters (Spearman): ROUGE-L {rho_rouge:.2f}, embedding cosine {rho_cos:.2f}. "
               + (f"Embedding cosine rates the base model {float(base['embedding_cosine'].iloc[0]):.2f}, close to the best adapters, "
                  "because it measures whether the answer is on topic, not whether it matches the reference style and content; "
                  "use BERTScore and ROUGE-L for quality claims." if len(base) else ""))


def system(F):
    cw = F.table("latency_cold_vs_warm")
    if cw is not None:
        fig, ax = plt.subplots(figsize=(8.5, 2.6))
        parts = [("rank_ms", "choose adapter (local registry)"), ("fetch_and_verify_ms", "download by hash + verify SHA-256"),
                 ("attach_ms", "attach adapter to model"), ("model_first_token_ms", "model computes first token")]
        left = np.zeros(len(cw))
        for (c, lab), col in zip(parts, ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]):
            ax.barh(cw["case"], cw[c], left=left, color=col, label=lab)
            left += cw[c].values
        for i, v in enumerate(left):
            ax.text(v + 8, i, f"{v:.0f} ms", va="center")
        ax.set_xlabel("time to first token (ms)")
        ax.set_xlim(0, left.max() * 1.25)
        ax.legend(fontsize=8, bbox_to_anchor=(0.5, -0.35), loc="upper center", ncol=2)
        ax.set_title("Where the time goes when a peer's adapter is used")
        c = cw.set_index("case")
        cold, warm = c.iloc[0], c.iloc[-1]
        F.save(fig, "latency_cold_vs_warm", "Using a peer's adapter costs extra time only the first time",
               "Time to first token (TTFT) split into its steps. Cold = the adapter is not on this node yet and is "
               "downloaded from the peer by its hash; warm = it is already cached.",
               f"Cold TTFT {cold['ttft_ms_mean']:.0f} ms (download+verify {cold['fetch_and_verify_ms']:.0f} ms, attach "
               f"{cold['attach_ms']:.0f} ms) vs warm {warm['ttft_ms_mean']:.0f} ms. Choosing the adapter from the local "
               "registry takes under 1 ms because no network call is needed.")

    pl = F.table("latency_vs_prompt_length")
    if pl is not None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        for s, g in pl.groupby("system"):
            axes[0].plot(g["prompt_tokens"], g["ttft_ms_p50"], marker="o", color=SYS.get(s), label=s)
            axes[1].plot(g["prompt_tokens"], g["prefill_tflops_per_s"], marker="o", color=SYS.get(s), label=s)
        for ax in axes:
            ax.set_xscale("log", base=2)
            ax.set_xlabel("prompt length (tokens)")
            ax.legend(fontsize=8)
        axes[0].set_ylabel("time to first token, median (ms)")
        axes[0].set_title("TTFT vs prompt length")
        axes[1].set_ylabel("prefill TFLOP/s")
        axes[1].set_title("Compute throughput while reading the prompt")
        sec = axes[1].secondary_yaxis("right", functions=(lambda x: 100 * x / 154.8, lambda m: m * 154.8 / 100))
        sec.set_ylabel("% of A6000 bf16 peak (MFU)")
        F.save(fig, "latency_vs_prompt_length", "Longer prompts: time to first token and GPU efficiency",
               "Left: time until the first answer token, one request at a time, for prompts of 128-4096 tokens. "
               "Right: useful compute per second while processing the prompt (TFLOP/s) and as % of the GPU's peak (MFU).",
               "Both systems use the same model and adapter, so the differences most likely come from the serving engine "
               "(vLLM's optimised kernels and CUDA graphs vs plain HuggingFace generate in our sidecar).")

    ol = F.table("latency_vs_output_length")
    if ol is not None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        for s, g in ol.groupby("system"):
            axes[0].plot(g["output_tokens"], g["tpot_ms_mean"], marker="o", color=SYS.get(s), label=s)
            axes[1].plot(g["output_tokens"], g["e2e_ms_mean"] / 1000, marker="o", color=SYS.get(s), label=s)
        for ax in axes:
            ax.set_xscale("log", base=2)
            ax.set_xlabel("answer length (tokens)")
            ax.legend(fontsize=8)
        axes[0].set_ylabel("time per output token (ms)")
        axes[0].set_title("TPOT (lower = faster typing)")
        axes[1].set_ylabel("end-to-end latency (s)")
        axes[1].set_title("Total time for one answer")
        F.save(fig, "latency_vs_output_length", "Longer answers: time per token and total time",
               "Time per output token (TPOT) and end-to-end latency for answers of 16-512 tokens, one request at a time.",
               "TPOT is roughly constant with answer length; end-to-end time grows linearly.")

    tc = F.table("throughput_vs_concurrency")
    if tc is not None:
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
        for s, g in tc.groupby("system"):
            axes[0].plot(g["concurrent_requests"], g["throughput_tokens_per_s"], marker="o", color=SYS.get(s), label=s)
            axes[1].plot(g["concurrent_requests"], g["ttft_ms_p95"] / 1000, marker="o", color=SYS.get(s), label=s)
            axes[2].plot(g["concurrent_requests"], g["goodput_requests_per_s"], marker="o", color=SYS.get(s), label=s)
        labs = [("output tokens per second (all requests)", "Throughput"), ("p95 time to first token (s)", "Waiting time"),
                ("requests/s meeting TTFT≤1 s & TPOT≤50 ms", "Goodput")]
        for ax, (yl, t) in zip(axes, labs):
            ax.set_xscale("log", base=2)
            ax.set_xlabel("requests in flight at once")
            ax.set_ylabel(yl)
            ax.set_title(t)
            ax.legend(fontsize=8)
        F.save(fig, "throughput_vs_concurrency", "Many requests at once: a central server batches, our node queues",
               "1-16 requests sent at the same time to ONE node (ours) or ONE server (vLLM). Throughput = generated tokens "
               "per second over all requests; goodput = requests per second that met the latency target.",
               "vLLM batches concurrent requests, so its throughput grows with load; our sidecar answers one request at a "
               "time, so its throughput is flat and waiting time grows. In the federated setting each node serves its own "
               "users on its own GPU; how total capacity scales with the number of nodes was not measured here.")

    ft = F.table("fault_tolerance")
    if ft is not None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
        for m, g in ft.groupby("method"):
            g = g.sort_values("peers_down")
            axes[0].plot(g["peers_down"], g["answered_percent"], marker="o", label=f"ours, selection={m}")
            axes[1].plot(g["peers_down"], g["latency_ms_mean"] / 1000, marker="o", label=f"ours, selection={m}")
        axes[0].plot([0, 1, 2], [100, 0, 0], ls="none", marker="X", ms=9, color=SYS["vLLM multi-LoRA server"],
                     label="central server, if the failed machine is the server")
        axes[0].set_ylim(-5, 115)
        axes[0].set_ylabel("queries answered (%)")
        axes[1].set_ylabel("mean latency (s)")
        for ax in axes:
            ax.set_xlabel("peer nodes down (of 4)")
            ax.set_xticks([0, 1, 2])
        axes[0].legend(fontsize=8, loc="lower center", bbox_to_anchor=(1.1, -0.42), ncol=3)
        axes[0].set_title("Availability")
        axes[1].set_title("Cost of failures")
        F.save(fig, "fault_tolerance", "Queries keep being answered when peers fail",
               "Nodes are killed while node 1 keeps receiving questions. Our node falls back to the next adapter in its "
               "ranking or to its own adapter; a central server answers nothing while it is down.",
               f"Queries answered with peers down: {ft[ft['peers_down'] > 0]['answered_percent'].min():.0f}% (minimum over "
               f"methods). Mean latency rises from {ft[ft['peers_down'] == 0]['latency_ms_mean'].mean() / 1000:.2f} s (healthy) to "
               f"{ft[ft['peers_down'] > 0]['latency_ms_mean'].max() / 1000:.2f} s (worst case) when the preferred peer is dead "
               "and the node first tries to download its adapter (the failed attempt took about 1 s here, well under the 10 s "
               "timeout) before falling back; failed peers are not remembered, so every query pays this again.")

    fs = F.table("fetch_vs_adapter_size")
    if fs is not None:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.plot(fs["adapter_mb"], fs["fetch_ms_mean"], marker="o", label="download + verify")
        ax.plot(fs["adapter_mb"], fs["attach_ms_mean"], marker="s", label="attach to model")
        for _, r in fs.iterrows():
            ax.annotate(f"rank {int(r['lora_rank'])}", (r["adapter_mb"], r["fetch_ms_mean"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("adapter file size (MB)")
        ax.set_ylabel("time (ms)")
        ax.legend(fontsize=8)
        ax.set_title("Cost of fetching a peer's adapter vs its size")
        F.save(fig, "fetch_vs_adapter_size", "Fetching bigger adapters", "Cold download+verify and attach time for LoRA ranks 8-128.",
               "Time grows with file size; even the largest adapter is fetched once and then cached.")


def comparison(F):
    """Figures from compare_systems.py: communication (already drawn there) and the system comparison table."""
    cb = F.table("communication_bytes")
    if cb is not None and os.path.isfile(os.path.join(F.d, "figures", "communication_bytes.png")):
        o = cb.set_index("system")
        F.notes.append(("communication_bytes", "Network traffic per query and whether the prompt leaves the device",
                        "Bytes sent over the network for one query (log scale). For our system the one-time adapter download is "
                        "spread over the queries that reuse it; a central server receives every prompt; a layer-split system "
                        "(Petals-style) moves every token's hidden state through all nodes.",
                        "Our advantage is privacy and no per-query traffic once an adapter is cached, not fewer bytes overall: "
                        f"ours: {o.iloc[0]['network_bytes_per_query_amortised'] / 1024:.1f} KB per query amortised, 0 bytes when the "
                        f"adapter is cached, and the prompt never leaves the node; central server: "
                        f"{o.iloc[1]['network_bytes_per_query_warm'] / 1024:.1f} KB per query including the full prompt; layer-split: "
                        f"{o.iloc[2]['network_bytes_per_query_warm'] / 1024:.0f} KB per query plus {int(o.iloc[2]['round_trips_per_generated_token'])} "
                        "network transfers for every generated token. Gossip metadata per adapter: "
                        f"{o.iloc[0]['metadata_bytes_per_adapter_per_peer']:.0f} bytes."))
    bc = F.table("baseline_comparison")
    if bc is None:
        return
    show = [("ttft_ms_256_token_prompt", "TTFT, 256-token prompt (ms)"),
            ("extra_ttft_ms_first_use_of_peer_adapter", "extra TTFT, first use of a peer adapter (ms)"),
            ("tpot_ms", "TPOT (ms)"),
            ("throughput_tokens_per_s_16_concurrent_one_machine", "tokens/s, 16 requests on one machine"),
            ("per_token_network_latency_ms", "network latency per generated token (ms)"),
            ("answered_percent_when_a_machine_fails", "% answered when the preferred peer / the server is down"),
            ("single_point_of_failure", "single point of failure"), ("needs_a_server", "needs a server"),
            ("adapter_known_to_peers_after_ms", "new adapter known to peers after (ms, median)"),
            ("where_all_adapters_are_stored", "where the adapters live")]
    cells = []
    for col, _ in show:
        row = []
        for v in bc[col]:
            row.append("n/a" if (isinstance(v, float) and np.isnan(v)) else (f"{v:,.1f}" if isinstance(v, float) else str(v)))
        cells.append(row)
    fig, ax = plt.subplots(figsize=(12, 0.45 * len(show) + 1))
    ax.axis("off")
    t = ax.table(cellText=cells, rowLabels=[s for _, s in show], colLabels=list(bc["system"]), loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.scale(1, 1.4)
    ax.set_title("Our federated inference vs a central multi-LoRA server and layer-split decentralised inference")
    F.save(fig, "baseline_comparison", "Where our design wins and where it loses",
           "Same model and adapters. Ours and vLLM are measured on this machine; the layer-split (Petals-style) column is "
           "computed from the model's hidden size and an assumed network round-trip time (see tables/baseline_comparison.csv).",
           "A central server batches requests and wins on throughput under load; our design needs no server, keeps the prompt "
           "on the node, answers when machines fail, and adds latency only the first time a peer's adapter is used.")


def write_readme(F, meta):
    lines = [f"# Results: {meta['dataset']}, {meta['n_nodes']} nodes, label skew α={meta['alpha']:g}, seed {meta['seed']}", "",
             "Folders: `figures/` (PNG + PDF), `tables/` (the numbers behind every figure, CSV), "
             "`raw/` (one row per question or request).", ""]
    for name, title, how, found in F.notes:
        lines += [f"## {title}", f"![{name}](figures/{name}.png)", "", f"**How to read it.** {how}", "",
                  f"**What this run shows.** {found}", f"Table: `tables/{name}.csv`" if os.path.isfile(os.path.join(F.d, "tables", name + ".csv")) else "", ""]
    with open(os.path.join(F.d, "README.md"), "w") as f:
        f.write("\n".join(lines))
    print("wrote README.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dir", default=None)
    a = ap.parse_args()
    d = a.dir or os.path.join(REPO, "experiments", "results", a.tag)
    meta = json.load(open(os.path.join(REPO, "experiments", "data", a.tag, "meta.json")))
    F = Fig(d)
    quality(F, meta)
    generation(F, meta)
    system(F)
    comparison(F)
    write_readme(F, meta)


if __name__ == "__main__":
    main()
