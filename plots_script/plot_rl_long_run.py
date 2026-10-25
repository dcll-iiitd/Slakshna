"""Figures from the single-node long runs and the 2-node federated run (results/rl_long_run_*)."""
import csv
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from rl_plot_style import AQUA, BLUE, COL1, COL2, INK, MUTED, ORANGE, apply_style, plt, save, wilson  # noqa: E402

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "results/rl_long_run_2026-10-22"
OUT = os.path.join(RESULTS, "figures")
os.makedirs(OUT, exist_ok=True)
N_TEST = 1319

# colour = KL setting, line style = learning rate (so colour is never the only cue)
RUNS = {
    "q05_lr5e-5_kl0.04": dict(model="0.5B", lr="5e-5", kl=0.04, color=BLUE, ls="-", mk="o"),
    "q05_lr1e-4_kl0":    dict(model="0.5B", lr="1e-4", kl=0.0, color=ORANGE, ls="-", mk="s"),
    "q15_lr2e-5_kl0.04": dict(model="1.5B", lr="2e-5", kl=0.04, color=BLUE, ls="--", mk="^"),
    "q15_lr5e-5_kl0.04": dict(model="1.5B", lr="5e-5", kl=0.04, color=BLUE, ls="-", mk="o"),
    "q15_lr5e-5_kl0":    dict(model="1.5B", lr="5e-5", kl=0.0, color=ORANGE, ls="-", mk="s"),
}


def label(r):
    return f"lr {r['lr']}, KL {'on' if r['kl'] else 'off'}"


def parse(name):
    steps, evals = [], {}
    for line in open(f"{RESULTS}/single_node/{name}/log.txt"):
        m = re.match(r"\[epoch \d+\]\[step (\d+)\] loss=(\S+) reward=(\S+) kl=(\S+) clip_frac=(\S+) "
                     r"grad_norm=(\S+) completion_len=(\S+)", line)
        if m:
            steps.append([float(x) for x in m.groups()])
        m = re.match(r"\[eval\]\[step (\d+)\] eval_reward=(\S+)", line)
        if m:
            evals[int(m.group(1))] = float(m.group(2))
    return np.array(steps), evals


DATA = {n: parse(n) for n in RUNS}
MODELS = ["0.5B", "1.5B"]


def smooth(y, w=20):
    k = np.ones(w) / w
    pad = np.concatenate([np.full(w - 1, y[0]), y])
    return np.convolve(pad, k, mode="valid")


def panels(figsize, ylabel, xlabel="Training step"):
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)
    for ax, m in zip(axes, MODELS):
        ax.set_title(f"Qwen2.5-{m}-Instruct", loc="left", color=INK)
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    return fig, axes


def of_model(m):
    return [(n, r) for n, r in RUNS.items() if r["model"] == m]


# ---- Fig 1: held-out accuracy with 95% CI ----------------------------------------------------
def fig_accuracy():
    fig, axes = panels((COL2, 2.5), "GSM8K test accuracy")
    for ax, m in zip(axes, MODELS):
        base = None
        for n, r in of_model(m):
            ev = DATA[n][1]
            xs = sorted(ev)
            ys = np.array([ev[x] for x in xs])
            ci = np.array([wilson(y, N_TEST) for y in ys])
            ax.errorbar(xs, ys, yerr=[ys - ci[:, 0], ci[:, 1] - ys], color=r["color"], ls=r["ls"],
                        marker=r["mk"], capsize=2, label=label(r))
            base = ev[0]
        ax.axhline(base, color=MUTED, lw=1, ls=":")
        ax.annotate("untrained model", (120, base), xytext=(0, -10), textcoords="offset points",
                    ha="center", color=MUTED, fontsize=7)
        ax.set_xticks([0, 80, 160, 240])
        ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, f"{OUT}/fig1_accuracy_vs_step")


# ---- Fig 2: training reward -------------------------------------------------------------------
def fig_reward():
    fig, axes = panels((COL2, 2.5), "Mean training reward")
    for ax, m in zip(axes, MODELS):
        for n, r in of_model(m):
            s = DATA[n][0]
            ax.plot(s[:, 0], s[:, 2], color=r["color"], alpha=0.18, lw=0.8)
            ax.plot(s[:, 0], smooth(s[:, 2]), color=r["color"], ls=r["ls"], label=label(r))
        ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, f"{OUT}/fig2_training_reward")


# ---- Fig 3: KL to the base model (only runs that measure it) ----------------------------------
def fig_kl():
    fig, axes = panels((COL2, 2.4), "KL to base model (per token)")
    for ax, m in zip(axes, MODELS):
        for n, r in of_model(m):
            if not r["kl"]:
                continue
            s = DATA[n][0]
            ax.plot(s[:, 0], smooth(s[:, 3]), color=r["color"], ls=r["ls"], label=label(r))
        ax.legend(loc="lower right")
    axes[0].text(0.02, 0.97, "KL-off runs do not compute KL", transform=axes[0].transAxes,
                 va="top", fontsize=6.5, color=MUTED)
    fig.tight_layout()
    save(fig, f"{OUT}/fig3_kl_to_base")


# ---- Fig 4: accuracy gain over the untrained model --------------------------------------------
def fig_gain():
    fig, ax = plt.subplots(figsize=(COL1 + 0.6, 2.6))
    names = list(RUNS)
    ys, errs = [], []
    for n in names:
        ev = DATA[n][1]
        gain = (ev[240] - ev[0]) * 100
        se = 100 * np.sqrt(ev[0] * (1 - ev[0]) / N_TEST + ev[240] * (1 - ev[240]) / N_TEST)
        ys.append(gain)
        errs.append(1.96 * se)      # conservative: unpaired; a paired test would be tighter
    xpos = np.arange(len(names))
    ax.bar(xpos, ys, color=[RUNS[n]["color"] for n in names], width=0.62,
           hatch=None, yerr=errs, capsize=2, ecolor=INK, error_kw=dict(lw=0.8))
    for x, n in zip(xpos, names):
        if RUNS[n]["ls"] == "--":
            ax.patches[x].set_hatch("////")
            ax.patches[x].set_edgecolor("white")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{RUNS[n]['model']}\n{RUNS[n]['lr']}\nKL {'on' if RUNS[n]['kl'] else 'off'}"
                        for n in names], fontsize=6.5)
    ax.set_ylabel("Accuracy gain at step 240 (points)")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, f"{OUT}/fig4_accuracy_gain")


# ---- Fig 5: completion length -----------------------------------------------------------------
def fig_length():
    fig, axes = panels((COL2, 2.4), "Mean completion length (tokens)")
    for ax, m in zip(axes, MODELS):
        for n, r in of_model(m):
            s = DATA[n][0]
            ax.plot(s[:, 0], smooth(s[:, 6]), color=r["color"], ls=r["ls"], label=label(r))
        ax.axhline(512, color=MUTED, lw=1, ls=":")
        ax.annotate("limit (512)", (240, 512), xytext=(0, 3), textcoords="offset points",
                    ha="right", color=MUTED, fontsize=7)
        ax.legend(loc="lower left")
    fig.tight_layout()
    save(fig, f"{OUT}/fig5_completion_length")


# ---- Fig 6: federated 2-node run ---------------------------------------------------------------
def federated_rounds():
    out = {}
    for f in sorted(glob.glob(f"{RESULTS}/federated_2node/*_bhaskera_crash.log")):
        node = os.path.basename(f)[:9] + "…" + os.path.basename(f).split("_")[0][-4:]
        rnd, ev = None, {}
        for line in open(f):
            m = re.match(r"===== round (\d+)", line)
            if m:
                rnd = int(m.group(1))
                ev[rnd] = []
            m = re.match(r"\[eval\]\[step \d+\] eval_reward=(\S+)", line)
            if m and rnd:
                ev[rnd].append(float(m.group(1)))
        out[node] = ev
    return out


def fig_federated():
    rounds = federated_rounds()
    trust = list(csv.DictReader(open(f"{RESULTS}/federated_2node/trust_scores_new.csv")))
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL2, 2.5))
    colors, marks = [BLUE, ORANGE], ["o", "s"]
    for (node, ev), c, mk in zip(rounds.items(), colors, marks):
        xs = sorted(ev)
        start = [ev[r][0] for r in xs if len(ev[r]) >= 2]
        a.plot(xs[:len(start)], start, color=c, marker=mk, label=f"node {node[-4:]}")
    a.axhline(0.37, color=MUTED, ls=":", lw=1)
    a.set_ylim(0.34, 0.46)
    a.annotate("untrained model (0.37)", (6, 0.37), xytext=(0, 3),
               textcoords="offset points", ha="right", color=MUTED, fontsize=6.5)
    a.set_xticks(range(1, 7))
    a.set_xlabel("Federated round (accuracy at round start)")
    a.set_ylabel("Accuracy on 200 test problems")
    a.set_title("Held-out accuracy per node (95% CI about ±6.8)", loc="left", fontsize=7.5)
    a.legend(loc="lower right")
    # trust weight each node gives its peer
    ts = sorted({t["timestamp"] for t in trust})
    nodes = sorted({t["observer_node"] for t in trust})
    for obs, c, mk in zip(nodes, colors, marks):
        ys = [float(t["weight"]) for ts_ in ts for t in trust
              if t["timestamp"] == ts_ and t["observer_node"] == obs and t["peer_node"] != obs]
        b.plot(range(2, 2 + len(ys)), ys, color=c, marker=mk, label=f"seen by …{obs[-4:]}")
    b.set_xlabel("Federated round")
    b.set_ylabel("Weight given to the peer")
    b.set_title("Trust weight each node gives its peer", loc="left", fontsize=7.5)
    b.set_ylim(0, 1)
    b.set_xticks(range(2, 7))
    b.legend(loc="lower right")
    fig.tight_layout()
    save(fig, f"{OUT}/fig6_federated_2node")


# ---- Fig 7: communication cost ---------------------------------------------------------------
def fig_comm():
    fig, ax = plt.subplots(figsize=(COL1 + 0.4, 2.3))
    items = [("Sparse int8 delta\n(sent per round)", 6.13e6, BLUE),
             ("Full LoRA adapter\n(fp32)", 8_798_208 * 4, MUTED),
             ("Full model\n(bf16)", 494_032_768 * 2, MUTED)]
    for i, (lab, v, c) in enumerate(items):
        ax.bar(i, v / 1e6, color=c, width=0.6)
        txt = f"{v / 1e6:.1f} MB" if v < 1e9 * 0.5 else f"{v / 1e9:.2f} GB"
        ax.annotate(txt, (i, v / 1e6), xytext=(0, 2), textcoords="offset points", ha="center",
                    fontsize=7)
    ax.set_yscale("log")
    ax.set_ylim(1, 3000)
    ax.set_xticks(range(3))
    ax.set_xticklabels([i[0] for i in items], fontsize=6.5)
    ax.set_ylabel("Bytes shared (MB, log scale)")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, f"{OUT}/fig7_communication_cost")


# ---- Fig 12: reward versus KL (how much reward each unit of drift bought) --------------------
def fig_reward_vs_kl():
    fig, axes = panels((COL2, 2.5), "Training reward (smoothed)", xlabel="KL to base model (per token)")
    for ax, m in zip(axes, MODELS):
        for n, r in of_model(m):
            if not r["kl"]:
                continue
            s = DATA[n][0]
            kl, rew = smooth(s[:, 3]), smooth(s[:, 2])
            ax.plot(kl, rew, color=r["color"], ls=r["ls"], label=label(r))
            ax.plot(kl[0], rew[0], "o", color=r["color"], ms=4)          # start of training
            ax.plot(kl[-1], rew[-1], "s", color=r["color"], ms=4)        # end of training (step 240)
        ax.plot([], [], "o", color=MUTED, ms=4, label="start")
        ax.plot([], [], "s", color=MUTED, ms=4, label="step 240")
        ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, f"{OUT}/fig12_reward_vs_kl")


# ---- Fig 13: gradient norm (training health) -------------------------------------------------
def fig_gradnorm():
    fig, axes = panels((COL2, 2.4), "Gradient norm (before clipping)")
    for ax, m in zip(axes, MODELS):
        for n, r in of_model(m):
            s = DATA[n][0]
            ax.plot(s[:, 0], smooth(s[:, 5]), color=r["color"], ls=r["ls"], label=label(r))
        ax.axhline(1.0, color=MUTED, lw=1, ls=":")
        ax.annotate("clip threshold (1.0)", (4, 1.0), xytext=(0, 3), textcoords="offset points",
                    ha="left", color=MUTED, fontsize=7)
        ax.set_ylim(0, 1.15)
        ax.legend(loc="center right")
    fig.tight_layout()
    save(fig, f"{OUT}/fig13_gradient_norm")


if __name__ == "__main__":
    apply_style()
    for f in (fig_accuracy, fig_reward, fig_kl, fig_gain, fig_length, fig_federated, fig_comm,
              fig_reward_vs_kl, fig_gradnorm):
        f()
    print("wrote", sorted(os.listdir(OUT)))
