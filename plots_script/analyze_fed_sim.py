"""Tables, paired significance tests and figures for the federated-RL experiments (E1, E2 non-IID).

    python plots_script/analyze_fed_sim.py results/fed_sim

Reads results/fed_sim/<exp>/<name>.json written by experiments/fed_sim.py. Handles partial results
(missing seeds are simply absent). Writes results/fed_sim/analysis.md and figures/*.pdf|png.
"""
import glob
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from rl_plot_style import AQUA, BLUE, COL1, COL2, INK, MUTED, ORANGE, apply_style, plt, save  # noqa: E402

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results/fed_sim"
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

METHODS = ["isolated_rl", "fed_rl", "centralized", "fed_sft"]
LABEL = {"base": "Untrained", "isolated_rl": "Isolated RL", "fed_rl": "Federated RL",
         "centralized": "Centralized RL", "fed_sft": "Federated SFT"}
STYLE = {"base": (MUTED, "x"), "isolated_rl": (ORANGE, "s"), "fed_rl": (BLUE, "o"),
         "centralized": (INK, "D"), "fed_sft": (AQUA, "^")}


def load():
    runs = {}
    for f in glob.glob(os.path.join(ROOT, "*", "*.json")):
        r = json.load(open(f))
        c = r["config"]
        runs[(c["method"], c["n_nodes"], c["partition"], c["seed"])] = r
    return runs


RUNS = load()


def acc(method, n, part, seed):
    r = RUNS.get((method, n, part, seed))
    return None if r is None else r["mean_acc"]


def seeds_of(method, n, part):
    return sorted(s for (m, nn, p, s) in RUNS if m == method and nn == n and p == part)


def per_problem(method, n, part, seed):
    """Per-problem correctness averaged over the evaluated honest nodes of one run."""
    r = RUNS[(method, n, part, seed)]
    return np.mean([f["per_problem"] for f in r["final"]], axis=0)


def mean_std(vals):
    v = np.array([x for x in vals if x is not None], float)
    return (v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0, len(v)) if len(v) else (np.nan, 0.0, 0)


def paired_test(a, b, n, part):
    """Paired comparison of method a vs b over the seeds both have: mean diff, bootstrap CI over
    problems, McNemar exact p on the seed-pooled per-problem results, and a paired t-test over seeds."""
    common = sorted(set(seeds_of(a, n, part)) & set(seeds_of(b, n, part)))
    if not common:
        return None
    pa = np.array([per_problem(a, n, part, s) for s in common])
    pb = np.array([per_problem(b, n, part, s) for s in common])
    diff_seed = pa.mean(1) - pb.mean(1)
    rng = np.random.default_rng(0)
    boot = [(pa[:, i].mean() - pb[:, i].mean()) for i in
            (rng.integers(0, pa.shape[1], pa.shape[1]) for _ in range(2000))]
    ca, cb = (pa.mean(0) > 0.5), (pb.mean(0) > 0.5)          # majority-over-seeds correctness
    only_a, only_b = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p_mc = stats.binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    p_t = stats.ttest_1samp(diff_seed, 0.0).pvalue if len(common) > 2 else float("nan")
    return dict(seeds=len(common), diff=float(diff_seed.mean() * 100),
                lo=float(np.percentile(boot, 2.5) * 100), hi=float(np.percentile(boot, 97.5) * 100),
                p_mcnemar=float(p_mc), p_ttest=float(p_t))


def fmt(t):
    if t is None:
        return "n/a"
    return (f"{t['diff']:+.2f} pts (95% CI {t['lo']:+.2f} to {t['hi']:+.2f}), McNemar p={t['p_mcnemar']:.3g}, "
            f"seed t-test p={t['p_ttest']:.3g}, {t['seeds']} seeds")


def write_tables():
    out = ["# Federated RL experiments (Qwen2.5-0.5B-Instruct, GSM8K test, 1,319 problems)\n"]
    base = RUNS.get(("base", 4, "iid", 0)) or next((r for k, r in RUNS.items() if k[0] == "base"), None)
    if base:
        out.append(f"Untrained model: {base['mean_acc']:.4f}\n")
    for part, title in (("iid", "E1: methods, 4 nodes, IID data"), ("noniid", "E2: 4 nodes, non-IID data")):
        out.append(f"\n## {title}\n\n| Method | Accuracy (mean ± std over seeds) | Seeds | Consensus gap | MB sent / run |\n|---|---|---|---|---|")
        for m in METHODS:
            vals = [acc(m, 4, part, s) for s in seeds_of(m, 4, part)]
            mu, sd, k = mean_std(vals)
            if k == 0:
                continue
            gaps = [RUNS[(m, 4, part, s)].get("consensus_gap") for s in seeds_of(m, 4, part)]
            mb = [RUNS[(m, 4, part, s)]["bytes_sent"] / 1e6 for s in seeds_of(m, 4, part)]
            gap = f"{np.mean([g for g in gaps if g is not None]):.3f}" if any(g is not None for g in gaps) else "-"
            out.append(f"| {LABEL[m]} | {mu * 100:.2f} ± {sd * 100:.2f} | {k} | {gap} | {np.mean(mb):.0f} |")
        out.append("\nPaired comparisons (positive = first method better):\n")
        for a, b in (("fed_rl", "isolated_rl"), ("fed_rl", "centralized"), ("fed_rl", "fed_sft"),
                     ("centralized", "isolated_rl")):
            if part == "noniid" and "centralized" in (a, b):
                continue
            out.append(f"- {LABEL[a]} vs {LABEL[b]}: {fmt(paired_test(a, b, 4, part))}")
    open(os.path.join(ROOT, "analysis.md"), "w").write("\n".join(out) + "\n")
    return base["mean_acc"] if base else None


def fig_methods(base_acc):
    fig, ax = plt.subplots(figsize=(COL1 + 0.8, 2.7))
    xs, names = [], []
    pos = 0
    for part, off in (("iid", 0),):
        for m in METHODS:
            vals = [acc(m, 4, part, s) for s in seeds_of(m, 4, part)]
            mu, sd, k = mean_std(vals)
            if k == 0:
                continue
            color, mk = STYLE[m]
            ax.bar(pos, mu * 100, color=color, width=0.62, alpha=0.9)
            ax.errorbar(pos, mu * 100, yerr=sd * 100, color=INK, capsize=3, lw=1)
            ax.plot([pos] * len(vals), [v * 100 for v in vals if v is not None], mk, color="white",
                    mec=INK, ms=3.5, mew=0.7, zorder=5)
            xs.append(pos)
            names.append(f"{LABEL[m]}\n(n={k})")
            pos += 1
    if base_acc:
        ax.axhline(base_acc * 100, color=MUTED, ls=":", lw=1)
        ax.annotate("untrained", (pos - 0.5, base_acc * 100), xytext=(0, 3), textcoords="offset points",
                    ha="right", color=MUTED, fontsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=6.5)
    ax.set_ylabel("GSM8K test accuracy (%)")
    lo = min([base_acc * 100 - 4] + [mean_std([acc(m, 4, 'iid', s) for s in seeds_of(m, 4, 'iid')])[0] * 100 - 6
                                     for m in METHODS if seeds_of(m, 4, 'iid')]) if base_acc else 30
    ax.set_ylim(max(lo, 0), None)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, f"{FIG}/fig8_method_comparison")


def fig_rounds():
    fig, ax = plt.subplots(figsize=(COL1 + 0.8, 2.7))
    for m in METHODS:
        curves = [[r.get("eval200_node0") for r in RUNS[(m, 4, "iid", s)]["rounds"]]
                  for s in seeds_of(m, 4, "iid")]
        curves = [c for c in curves if c and None not in c]
        if not curves:
            continue
        a = np.array(curves) * 100
        x = np.arange(1, a.shape[1] + 1)
        color, mk = STYLE[m]
        ax.plot(x, a.mean(0), color=color, marker=mk, label=LABEL[m])
        ax.fill_between(x, a.mean(0) - a.std(0, ddof=1) if len(a) > 1 else a.mean(0),
                        a.mean(0) + a.std(0, ddof=1) if len(a) > 1 else a.mean(0), color=color, alpha=0.15, lw=0)
    ax.set_xlabel("Round")
    ax.set_ylabel("Accuracy on 200 test problems (%)")
    ax.set_ylim(None, 48)
    ax.legend(loc="center", bbox_to_anchor=(0.5, 0.36), ncol=2)
    fig.tight_layout()
    save(fig, f"{FIG}/fig9_accuracy_per_round")


def fig_noniid():
    methods = [m for m in ("isolated_rl", "fed_rl", "fed_sft")
               if seeds_of(m, 4, "noniid") or seeds_of(m, 4, "iid")]
    fig, ax = plt.subplots(figsize=(COL1 + 0.8, 2.7))
    w = 0.36
    for i, m in enumerate(methods):
        for j, (part, hatch) in enumerate((("iid", None), ("noniid", "////"))):
            mu, sd, k = mean_std([acc(m, 4, part, s) for s in seeds_of(m, 4, part)])
            if k == 0:
                continue
            ax.bar(i + (j - 0.5) * w, mu * 100, w * 0.92, color=STYLE[m][0], hatch=hatch,
                   edgecolor="white" if hatch else STYLE[m][0], alpha=0.95)
            ax.errorbar(i + (j - 0.5) * w, mu * 100, yerr=sd * 100, color=INK, capsize=2.5, lw=0.9)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([LABEL[m] for m in methods], fontsize=7)
    ax.set_ylabel("GSM8K test accuracy (%)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=MUTED, label="IID data"),
                       Patch(facecolor=MUTED, hatch="////", edgecolor="white", label="Non-IID data")],
              loc="upper right")
    allv = [mean_std([acc(m, 4, p, s) for s in seeds_of(m, 4, p)])[0] * 100 for m in methods for p in ("iid", "noniid")
            if seeds_of(m, 4, p)]
    if allv:
        ax.set_ylim(max(min(allv) - 8, 0), None)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, f"{FIG}/fig10_noniid")


def fig_consensus():
    fig, ax = plt.subplots(figsize=(COL1, 2.5))
    items = []
    for m in ("isolated_rl", "fed_rl"):
        for part in ("iid", "noniid"):
            g = [RUNS[(m, 4, part, s)].get("consensus_gap") for s in seeds_of(m, 4, part)]
            g = [x for x in g if x is not None]
            if g:
                items.append((m, part, np.mean(g), np.std(g, ddof=1) if len(g) > 1 else 0))
    for i, (m, part, mu, sd) in enumerate(items):
        ax.bar(i, mu, 0.6, color=STYLE[m][0], hatch="////" if part == "noniid" else None,
               edgecolor="white" if part == "noniid" else STYLE[m][0])
        ax.errorbar(i, mu, yerr=sd, color=INK, capsize=2.5, lw=0.9)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([f"{LABEL[m].replace(' RL', '')}\n{'non-IID' if p == 'noniid' else 'IID'}" for m, p, _, _ in items],
                       fontsize=6.5)
    ax.set_ylabel("Consensus gap (mean pairwise\nL2 distance between nodes)")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, f"{FIG}/fig11_consensus_gap")


if __name__ == "__main__":
    apply_style()
    b = write_tables()
    for f in (lambda: fig_methods(b), fig_rounds, fig_noniid, fig_consensus):
        try:
            f()
        except Exception as e:      # partial results: skip a figure whose inputs are missing
            print("skipped a figure:", type(e).__name__, e)
    print(open(os.path.join(ROOT, "analysis.md")).read())
