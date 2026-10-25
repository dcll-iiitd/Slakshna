"""LaTeX (pgfplots) figure of the per-node training loss, plus a matplotlib preview of the same data.

    python plots_script/make_tex_train_loss.py results/sft_olmo2_1b_4node_2026-10-23

Training loss = token-weighted NLL over assistant tokens per optimizer step (from the trainer logs),
smoothed with a trailing 5-step moving average (shorter window at the start). The data are embedded in
the .tex file, so it compiles without any external files (needs only pgfplots).
"""
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from rl_plot_style import AQUA, BLUE, COL1, INK, ORANGE, apply_style, plt, save  # noqa: E402

RES = sys.argv[1] if len(sys.argv) > 1 else "results/sft_olmo2_1b_4node_2026-10-23"
OUT = os.path.join(RES, "figures")
WINDOW = 5
# node number -> id suffix, read from each node's startup log ("Node Identity")
ids = {}
for n in (1, 2, 3, 4):
    text = open(f"{RES}/logs/node{n}_cluster.log", errors="ignore").read()
    m = re.search(r"Node Identity: (slakshna1[a-z0-9]+)", re.sub(r"\x1b\[[0-9;]*m", "", text))
    ids[n] = m.group(1)


def per_step(node_id):
    rnd, rows = 0, {}
    for line in open(f"{RES}/logs/{node_id}_bhaskera_crash.log", errors="ignore"):
        m = re.match(r"===== round (\d+)", line)
        if m:
            rnd = int(m.group(1))
            continue
        m = re.search(r"\[epoch \d+\]\[step (\d+)\] loss=(\S+) ppl=\S+ .*samples=(\d+)", line)
        if m and rnd and m.group(3) != "0":
            rows[(rnd - 1) * 10 + int(m.group(1))] = float(m.group(2))
    return rows


raw = {n: per_step(ids[n]) for n in ids}
steps = sorted(set.intersection(*[set(r) for r in raw.values()]))
assert steps == list(range(1, 101)), "expected 100 steps per node"
curves = {n: np.array([raw[n][s] for s in steps]) for n in ids}


def trailing_mean(y, w=WINDOW):
    return np.array([y[max(0, i - w + 1):i + 1].mean() for i in range(len(y))])


smooth = {n: trailing_mean(curves[n]) for n in ids}

# ---- statistics for the text -------------------------------------------------------------------
stats = {
    "first5_mean_per_node": {n: float(curves[n][:5].mean()) for n in ids},
    "step30_smoothed_per_node": {n: float(smooth[n][29]) for n in ids},
    "last10_mean_per_node": {n: float(curves[n][-10:].mean()) for n in ids},
    "last10_mean_all_nodes": float(np.mean([curves[n][-10:].mean() for n in ids])),
    "spread_last10_max_minus_min": float(max(curves[n][-10:].mean() for n in ids) - min(curves[n][-10:].mean() for n in ids)),
    "all_nodes_first5": float(np.mean([curves[n][:5].mean() for n in ids])),
    "cross_node_std_mean_after_step30": float(np.mean(np.std([smooth[n][30:] for n in ids], axis=0))),
}
json.dump(stats, open(f"{OUT}/train_loss_stats.json", "w"), indent=1)
print(json.dumps(stats, indent=1))

# ---- LaTeX ----------------------------------------------------------------------------------------
STYLE = {1: ("nodeA", "*"), 2: ("nodeB", "square*"), 3: ("nodeC", "triangle*"), 4: ("nodeD", "diamond*")}
coords = lambda y: " ".join(f"({s},{v:.4f})" for s, v in zip(steps, y))
plots = "\n".join(
    f"    \\addplot[{STYLE[n][0]}, mark={STYLE[n][1]}, mark repeat=10, mark phase=10]\n"
    f"      coordinates {{{coords(smooth[n])}}};\n    \\addlegendentry{{Node {n}}}"
    for n in (1, 2, 3, 4))
tex = r"""% Training-loss figure for the four federated nodes (OLMo-2 1B, South Asia corpus).
% Preamble requirements:   \usepackage{pgfplots}   \pgfplotsset{compat=1.16}
% Use \input{sft_train_loss_4nodes.tex} where the figure should appear.
% (Colours are colour-blind-safe and every node also has its own marker.)
\begin{figure}[t]
  \centering
  \definecolor{nodeA}{HTML}{2A78D6}
  \definecolor{nodeB}{HTML}{EB6834}
  \definecolor{nodeC}{HTML}{1BAF7A}
  \definecolor{nodeD}{HTML}{1A1A19}
  \begin{tikzpicture}
    \begin{axis}[
      width=\linewidth, height=0.64\linewidth,
      xlabel={Optimizer step (per node)},
      ylabel={Training loss (nats/token)},
      xmin=0, xmax=100, ymin=1.4, ymax=4.5,
      xtick={0,20,40,60,80,100},
      ytick={1.5,2.0,2.5,3.0,3.5,4.0,4.5},
      yticklabel style={/pgf/number format/fixed, /pgf/number format/precision=1},
      grid=major, major grid style={gray!25, line width=0.3pt},
      tick align=outside, tick style={black, line width=0.4pt},
      axis line style={black, line width=0.4pt},
      label style={font=\small}, tick label style={font=\footnotesize},
      legend style={at={(0.97,0.96)}, anchor=north east, draw=gray!60, fill=white,
                    fill opacity=0.9, text opacity=1, font=\footnotesize,
                    inner sep=2pt, /tikz/every even column/.append style={column sep=3pt}},
      legend cell align=left,
      every axis plot/.append style={line width=0.9pt, mark size=1.7pt},
    ]
""" + plots + r"""
    \end{axis}
  \end{tikzpicture}
  \caption{Training loss of the four federated nodes during LoRA fine-tuning of OLMo-2~1B on the
  South Asia corpus. The loss is the token-weighted negative log-likelihood of the assistant-response
  tokens at each optimizer step (16 packed sequences of 2{,}048 tokens), smoothed with a trailing
  5-step moving average. Each federated round consists of 10 local steps followed by trust-weighted
  aggregation of the sparsified LoRA updates; markers indicate round boundaries. All nodes use identical
  hyper-parameters (AdamW, peak learning rate $2\times10^{-4}$, a single cosine schedule over 100 steps)
  and disjoint data shards.}
  \label{fig:sft-train-loss}
\end{figure}
"""
open(f"{OUT}/sft_train_loss_4nodes.tex", "w").write(tex)

# ---- matplotlib preview with the same data and styling ---------------------------------------------
apply_style()
fig, ax = plt.subplots(figsize=(COL1 + 0.6, 2.7))
colors = {1: BLUE, 2: ORANGE, 3: AQUA, 4: INK}
marks = {1: "o", 2: "s", 3: "^", 4: "D"}
for n in (1, 2, 3, 4):
    ax.plot(steps, smooth[n], color=colors[n], marker=marks[n], markevery=slice(9, None, 10), ms=3.4, lw=1.3,
            label=f"Node {n}")
ax.set_xlim(0, 100)
ax.set_ylim(1.4, 4.5)
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.set_xlabel("Optimizer step (per node)")
ax.set_ylabel("Training loss (nats/token)")
ax.legend(loc="upper right")
fig.tight_layout()
save(fig, f"{OUT}/sft_train_loss_4nodes_preview")
print("wrote", sorted(f for f in os.listdir(OUT) if "train_loss" in f))
