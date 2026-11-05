#!/usr/bin/env python3
"""Figure for the real-LLM multi-agent benchmark: accuracy with bootstrap 95% CIs.

usage: plot_llm_bench.py results.json out_prefix      (writes out_prefix.png and out_prefix.pdf)
Input is the raw per-cell JSON written by `iiitd llm-bench --out`."""
import json, random, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, out = sys.argv[1], sys.argv[2]
cells = json.load(open(src))
MODES = ["single (largest model)", "centralized orchestrator", "decentralized, votes only", "decentralized + probes"]
SHORT = ["best single agent (3B, oracle choice)", "centralized orchestrator", "decentralized, votes only", "decentralized + probes"]
# categorical slots 1-4 of the reference palette (validated: CVD dE >= 9, value labels given because aqua/yellow are < 3:1 on white)
COLORS = ["#2a78d6", "#eb6834", "#eda100", "#1baf7a"]
INK, GRID = "#0b0b0b", "#e6e5e1"
SCEN = [("none", 0, "no attackers"), ("Colluding", 2, "2 colluding"), ("Colluding", 3, "3 colluding"), ("Lying", 2, "2 lying"), ("Lying", 3, "3 lying")]


def boot(outcomes, n=2000, seed=99):
    rng = random.Random(seed)
    m = sum(outcomes) / len(outcomes)
    ms = sorted(sum(outcomes[rng.randrange(len(outcomes))] for _ in outcomes) / len(outcomes) for _ in range(n))
    return m, ms[int(n * 0.025)], ms[int(n * 0.975)]


def collect(adv, k, mode):
    out = []
    for c in cells:
        a = c["adversary"]
        if c["mode"] == mode and c["k"] == k and (k == 0 or a == adv) :
            out += [1 if x else 0 for x in c["correct"]]
    return out


fig, ax = plt.subplots(figsize=(10, 4.6))
w, gap = 0.19, 0.012
for si, (adv, k, label) in enumerate(SCEN):
    for mi, mode in enumerate(MODES):
        o = collect(adv, k, mode)
        if not o:
            continue
        m, lo, hi = boot(o)
        x = si + (mi - 1.5) * (w + gap)
        ax.bar(x, m * 100, w, color=COLORS[mi], label=SHORT[mi] if si == 0 else None, zorder=3)
        ax.errorbar(x, m * 100, yerr=[[ (m - lo) * 100], [(hi - m) * 100]], color=INK, capsize=2.5, lw=1, zorder=4)
        ax.text(x, hi * 100 + 1.5, f"{m*100:.0f}", ha="center", va="bottom", fontsize=7.5, color=INK, zorder=5)
ax.set_xticks(range(len(SCEN)))
ax.set_xticklabels([s[2] for s in SCEN], color=INK)
ax.set_ylabel("GSM8K accuracy (%)", color=INK)
ax.set_ylim(0, 100)
ax.yaxis.grid(True, color=GRID, zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.spines["left"].set_color(GRID)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(colors=INK)
n = len(collect("none", 0, MODES[0]))
ax.set_title(f"9 LLM agents (Qwen2.5 0.5B / 1.5B / 3B on H200 GPUs); {n} problems per bar; error bars: bootstrap 95% CI of the mean", fontsize=9, color=INK, loc="left")
ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right", bbox_to_anchor=(1.0, 1.0))
fig.tight_layout()
fig.savefig(out + ".png", dpi=200)
fig.savefig(out + ".pdf")
print("wrote", out + ".png", out + ".pdf")
