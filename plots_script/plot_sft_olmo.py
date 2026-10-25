"""Figures and tables for the 4-node federated fine-tuning run (OLMo-2 1B, South Asia data)."""
import collections
import csv
import os
import statistics as st
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from rl_plot_style import BLUE, COL1, COL2, INK, MUTED, ORANGE, apply_style, plt, save  # noqa: E402

RES = sys.argv[1] if len(sys.argv) > 1 else "results/sft_olmo2_1b_4node_2026-10-23"
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)
val = list(csv.DictReader(open(f"{RES}/logs/metrics_val.csv")))
train = list(csv.DictReader(open(f"{RES}/logs/metrics_train.csv")))
NODES = sorted({r["node_id"] for r in val})


def series(split="val"):
    d = collections.defaultdict(dict)                 # node -> round -> (nll, ppl, bpb)
    for r in val:
        if r["split"] == split:
            d[r["node_id"]][int(r["round"])] = (float(r["nll_nats_per_token"]), float(r["perplexity"]),
                                                float(r["bits_per_byte"]))
    return d


def fig_val_curves():
    d = series()
    rounds = sorted({k for n in d.values() for k in n})
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.5))
    for ax, idx, ylabel in zip(axes, range(3), ("Validation NLL (nats/token)", "Validation perplexity",
                                                 "Validation bits per byte")):
        for n in NODES:
            ys = [d[n].get(r, (np.nan,) * 3)[idx] for r in rounds]
            ax.plot(rounds, ys, color=MUTED, alpha=0.45, lw=0.9)
        mean = [np.nanmean([d[n].get(r, (np.nan,) * 3)[idx] for n in NODES]) for r in rounds]
        ax.plot(rounds, mean, color=BLUE, marker="o", lw=1.8, label="mean of 4 nodes")
        ax.set_xlabel("Federated round (0 = untrained)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(0, rounds[-1] + 1, 2))
    axes[1].set_yscale("log")
    axes[1].set_yticks([5, 10, 20])
    axes[1].yaxis.set_minor_formatter(plt.NullFormatter())
    axes[1].set_yticklabels(["5", "10", "20"])
    axes[0].plot([], [], color=MUTED, alpha=0.6, label="individual nodes")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    save(fig, f"{FIG}/fig_sft_validation_curves")


def fig_train_vs_val():
    d = series()
    tr = collections.defaultdict(list)
    for r in train:
        tr[int(r["round"])].append(float(r["train_nll_nats_per_token"]))
    rounds = sorted(tr)
    fig, ax = plt.subplots(figsize=(COL1 + 0.8, 2.7))
    t = [st.mean(tr[r]) for r in rounds]
    v = [st.mean(d[n][r][0] for n in NODES) for r in rounds]
    ax.plot(rounds, t, color=ORANGE, marker="s", label="train (token-weighted, dropout on)")
    ax.plot(rounds, v, color=BLUE, marker="o", label="validation (unseen documents)")
    ax.set_xlabel("Federated round")
    ax.set_ylabel("NLL (nats/token)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, f"{FIG}/fig_sft_train_vs_val")


def write_table():
    dv, dt = series("val"), series("test")
    last = max(k for n in dv.values() for k in n)
    lines = ["# OLMo-2 1B, 4-node federated fine-tuning, South Asia data\n",
             "Held-out loss over assistant-response tokens; PPL = exp(NLL); BPB = bits per UTF-8 byte.\n",
             "| Model | Split | NLL (nats/token) | Perplexity | Bits/byte |", "|---|---|---|---|---|"]
    base = [dv[n][0] for n in NODES][0]
    lines.append(f"| Untrained base | val | {base[0]:.4f} | {base[1]:.3f} | {base[2]:.4f} |")
    for name, d, rnd in (("Federated (round %d)" % last, dv, last), ("Federated (round %d)" % last, dt, last)):
        vals = np.array([d[n][rnd] for n in NODES if rnd in d[n]])
        split = "val" if d is dv else "test"
        sd = vals.std(axis=0, ddof=1) if len(vals) > 1 else np.zeros(3)
        mu = vals.mean(axis=0)
        lines.append(f"| {name}, mean ± std over 4 nodes | {split} | {mu[0]:.4f} ± {sd[0]:.4f} | "
                     f"{mu[1]:.3f} ± {sd[1]:.3f} | {mu[2]:.4f} ± {sd[2]:.4f} |")
    lines.append("\nPer-node final test results:\n\n| Node | NLL | PPL | BPB |\n|---|---|---|---|")
    for n in NODES:
        r = dt[n][last]
        lines.append(f"| …{n[-4:]} | {r[0]:.4f} | {r[1]:.3f} | {r[2]:.4f} |")
    open(f"{RES}/results_table.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def step_logs():
    """Per-optimizer-step training loss/perplexity of every node, from the per-round trainer logs."""
    import glob
    import re
    out = {}
    for f in sorted(glob.glob(f"{RES}/logs/*_bhaskera_crash.log")):
        node = os.path.basename(f).split("_")[0]
        rnd, rows = 0, {}
        for line in open(f):
            m = re.match(r"===== round (\d+)", line)
            if m:
                rnd = int(m.group(1))
                continue
            m = re.search(r"\[epoch \d+\]\[step (\d+)\] loss=(\S+) ppl=(\S+) .*samples=(\d+)", line)
            if m and rnd and m.group(4) != "0":            # samples=0 marks end-of-epoch, not a real step
                rows[(rnd - 1) * 10 + int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
        out[node] = rows
    return out


def fig_loss_and_perplexity():
    steps, d = step_logs(), series()
    gsteps = sorted(set.intersection(*[set(r) for r in steps.values()]))
    loss = np.array([[steps[n][g][0] for g in gsteps] for n in steps])
    ppl = np.array([[steps[n][g][1] for g in gsteps] for n in steps])
    rounds = sorted({k for n in d.values() for k in n})

    def smooth(y, w=5):
        pad = np.concatenate([np.full(w - 1, y[0]), y])
        return np.convolve(pad, np.ones(w) / w, mode="valid")

    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.9))
    for ax, data, vidx, ylabel in ((axes[0], loss, 0, "Loss: NLL (nats/token)"), (axes[1], ppl, 1, "Perplexity")):
        for row in data:
            ax.plot(gsteps, row, color=ORANGE, alpha=0.18, lw=0.7)
        ax.plot(gsteps, smooth(data.mean(0)), color=ORANGE, lw=1.8, label="train, per step (mean of 4 nodes, 5-step avg)")
        vx = [10 * r for r in rounds]
        for n in NODES:
            ax.plot(vx, [d[n][r][vidx] for r in rounds], color=MUTED, alpha=0.35, lw=0.8)
        ax.plot(vx, [np.mean([d[n][r][vidx] for n in NODES]) for r in rounds], color=BLUE, marker="o", lw=1.8,
                label="validation, end of round (mean of 4 nodes)")
        ax.set_xlabel("Optimizer step per node (10 per federated round)")
        ax.set_ylabel(ylabel)
    axes[1].set_yscale("log")
    axes[1].set_yticks([5, 10, 20, 40])
    axes[1].set_yticklabels(["5", "10", "20", "40"])
    axes[1].yaxis.set_minor_formatter(plt.NullFormatter())
    axes[0].legend(loc="upper right", fontsize=6.5)
    fig.tight_layout()
    save(fig, f"{FIG}/fig_sft_loss_and_perplexity")


if __name__ == "__main__":
    apply_style()
    fig_val_curves()
    fig_train_vs_val()
    fig_loss_and_perplexity()
    write_table()
