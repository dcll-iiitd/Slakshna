"""
Figures + README for System One decisions, from experiments/results/system_one/<tag>/tables/ (aggregate_decisions.py)
and raw/decide_*.csv, raw/fault_tolerance.csv (bench_decisions.py, fault_test.py --endpoint decide).
Missing inputs are skipped. Every README sentence is computed from the tables.

    Bhaskera-serve/.venv/bin/python experiments/plot_decisions.py --tag ag_news_n4_a0.1_s0_s1
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments"))
from plot_results import BASE, BOUND, OURS, TRAIN, Fig  # noqa: E402

GROUP = {"base": BASE, "own": OURS, "own_aggregated": OURS, "merged_avg": OURS, "mix_uniform_prob": "#0B3C7D",
         "mix_uniform_logit": OURS, "mix_confidence_prob": OURS, "mix_query_aware_prob": OURS,
         "local_only": TRAIN, "centralised": BOUND, "oracle_member": BOUND}
SYS = {"slakshna": ("Ours: /decide on the node", "#2E6FB7"), "vllm": ("Central vLLM server", "#D1495B")}
FMT = "raw"  # the format the node actually uses


def hbars(ax, df, col, xlabel, fmt="{:.3f}"):
    df = df.sort_values(col)
    y = np.arange(len(df))
    ax.barh(y, df[col], color=[GROUP.get(m, OURS) for m in df["method"]])
    for i, v in enumerate(df[col]):
        ax.text(v + 0.005, i, fmt.format(v), va="center", fontsize=8)
    ax.set_yticks(y, df["method_name"])
    ax.set_xlabel(xlabel)


def quality(F, meta):
    sub = f"{meta['dataset']} ({len(meta['labels'])} options), {meta['n_nodes']} nodes, label skew α={meta['alpha']:g}"
    t = F.table("decision_accuracy_by_method")
    if t is not None and len(t):
        g = t[t["format"] == FMT]
        fig, axes = plt.subplots(2, 1, figsize=(9, 0.75 * g["method"].nunique() + 3), sharex=True)
        res = {}
        for ax, which in zip(axes, ["all classes (balanced test)", "node's own mix (local test)"]):
            x = g[g["evaluated_on"] == which].groupby(["method", "method_name"], as_index=False)["accuracy"].mean()
            hbars(ax, x, "accuracy", "accuracy (mean over the query nodes)")
            ax.set_xlim(0, 1.1)
            ax.set_title("Questions on " + which, fontsize=10)
            ax.tick_params(labelbottom=True)
            res[which] = x.set_index("method")["accuracy"]
        fig.suptitle("Accuracy of each way of deciding (System One prompt as the node scores it)\n" + sub, fontsize=11)
        r, l = res["all classes (balanced test)"], res["node's own mix (local test)"]
        F.save(fig, "decision_accuracy_by_method", "Accuracy of each way of deciding",
               "Each bar is one way to make the decision: base model, the node's own adapter, the federated System One "
               "mix of all nodes' adapters (dark blue = as deployed: equal weights, prob_mean, all cyclic orderings) and "
               "other mixing variants, training baselines (green), upper bounds (orange). Top: all classes; bottom: the "
               "node's own class mix.",
               f"All classes: base {r.get('base', np.nan):.3f}, own adapter {r.get('own', np.nan):.3f}, federated mix "
               f"{r.get('mix_uniform_prob', np.nan):.3f}, centralised {r.get('centralised', np.nan):.3f}, oracle member "
               f"{r.get('oracle_member', np.nan):.3f}. Own mix: own adapter {l.get('own', np.nan):.3f}, federated mix "
               f"{l.get('mix_uniform_prob', np.nan):.3f}.")
        # calibration summary
        x = g[g["test_set"] == "global"].groupby(["method", "method_name"], as_index=False)[["ece", "brier", "nll", "aurc"]].mean()
        fig, axes = plt.subplots(1, 4, figsize=(17, 0.42 * len(x) + 1.6), sharey=True)
        x = x.sort_values("ece", ascending=False)
        for ax, (col, lab) in zip(axes, [("ece", "ECE (15 bins) ↓"), ("brier", "Brier ↓"), ("nll", "NLL ↓"), ("aurc", "AURC ↓")]):
            ax.barh(np.arange(len(x)), x[col], color=[GROUP.get(m, OURS) for m in x["method"]])
            for i, v in enumerate(x[col]):
                ax.text(v, i, f" {v:.3f}", va="center", fontsize=7)
            ax.set_title(lab, fontsize=10)
        axes[0].set_yticks(np.arange(len(x)), x["method_name"])
        fig.suptitle("Are the decision probabilities trustworthy? (all classes, lower is better)\n" + sub, fontsize=11)
        xx = x.set_index("method")
        F.save(fig, "decision_calibration_metrics", "How good are the probabilities?",
               "ECE: gap between confidence and accuracy (15 equal-width bins). Brier and NLL: proper scores of the full "
               "distribution. AURC: area under the risk-coverage curve (how well confidence ranks right above wrong answers).",
               f"Federated mix: ECE {xx.loc['mix_uniform_prob', 'ece']:.3f}, Brier {xx.loc['mix_uniform_prob', 'brier']:.3f}, "
               f"AURC {xx.loc['mix_uniform_prob', 'aurc']:.3f}; own adapter: ECE {xx.loc['own', 'ece']:.3f}, AURC "
               f"{xx.loc['own', 'aurc']:.3f}; base: ECE {xx.loc['base', 'ece']:.3f}." if {"mix_uniform_prob", "own", "base"} <= set(xx.index) else "")
        # raw vs chat
        if t["format"].nunique() > 1:
            x = t[t["test_set"] == "global"].groupby(["method", "method_name", "format"], as_index=False)["accuracy"].mean()
            p = x.pivot(index="method_name", columns="format", values="accuracy").sort_values("raw")
            fig, ax = plt.subplots(figsize=(8, 0.4 * len(p) + 1.5))
            yy = np.arange(len(p))
            ax.barh(yy - 0.2, p["raw"], 0.4, label="raw prompt (how the node scores)")
            ax.barh(yy + 0.2, p["chat"], 0.4, label="chat-turn prompt (how adapters were trained)")
            ax.set_yticks(yy, p.index)
            ax.set_xlabel("accuracy, all classes")
            ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=2)
            ax.set_title("Prompt format: raw (deployed) vs chat turn (training)\n" + sub, fontsize=10)
            F.save(fig, "decision_prompt_format", "Does the deployed prompt format match how adapters were trained?",
                   "The node scores the raw System One prompt (next token after 'Answer:' is ' A'); the adapters were trained "
                   "with the same prompt inside a chat turn (next token 'A').",
                   f"Mean absolute accuracy difference raw vs chat over methods: {np.nanmean(np.abs(p['raw'] - p['chat'])):.3f}.")

    rb = F.table("decision_reliability_bins")
    if rb is not None and len(rb):
        rb = rb[rb["format"] == FMT]
        ms = list(dict.fromkeys(rb["method"]))
        fig, axes = plt.subplots(1, len(ms), figsize=(4 * len(ms), 3.8), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, m in zip(axes, ms):
            for temp, g in rb[rb["method"] == m].groupby("temperature"):
                g = g[g["n"] > 0]
                ax.plot(g["mean_confidence"], g["accuracy"], marker="o", label=f"temperature {temp}")
            ax.plot([0, 1], [0, 1], ls="--", color="#888")
            short = {"base": "Base model", "own": "Own adapter only", "centralised": "Centralised training",
                     "mix_uniform_prob": "Federated mix (as deployed)"}
            ax.set_title(short.get(m, m), fontsize=9)
            ax.set_xlabel("confidence of the chosen option")
            ax.legend(fontsize=7)
        axes[0].set_ylabel("accuracy")
        F.save(fig, "decision_reliability", "Reliability diagrams",
               "Accuracy vs confidence in 15 equal-width bins (all classes, query node 1). On the dashed diagonal a 70% "
               "confident answer is right 70% of the time. Temperature fitted on the validation split by NLL.",
               "Points above the diagonal mean the model is under-confident (right more often than it claims), below it "
               "over-confident. ECE/NLL before and after the fitted temperature are in decision_calibration_temperature.csv.")

    ct = F.table("decision_calibration_temperature")
    if ct is not None and len(ct):
        x = ct[(ct["format"] == FMT) & (ct["test_set"] == "global")].groupby(["method", "method_name"], as_index=False)[
            ["fitted_temperature", "ece_before", "ece_after", "nll_before", "nll_after", "accuracy_before", "accuracy_after"]].mean()
        fig, ax = plt.subplots(figsize=(8, 0.4 * len(x) + 1.5))
        x = x.sort_values("ece_before")
        yy = np.arange(len(x))
        ax.barh(yy - 0.2, x["ece_before"], 0.4, label="temperature 1 (default)")
        ax.barh(yy + 0.2, x["ece_after"], 0.4, label="temperature fitted on validation")
        ax.set_yticks(yy, x["method_name"])
        ax.set_xlabel("ECE (lower is better)")
        ax.legend(fontsize=8)
        ax.set_title("Calibration temperature (the node's calibration_temperature setting)")
        xm = x.set_index("method")
        F.save(fig, "decision_calibration_temperature", "Does fitting the calibration temperature help?",
               "System One divides the option logits by calibration_temperature. Fitting it on 400 validation decisions (NLL) "
               "vs leaving it at 1.",
               (f"Federated mix: fitted T={xm.loc['mix_uniform_prob', 'fitted_temperature']:.2f}, ECE "
                f"{xm.loc['mix_uniform_prob', 'ece_before']:.3f} -> {xm.loc['mix_uniform_prob', 'ece_after']:.3f}, accuracy "
                f"unchanged ({xm.loc['mix_uniform_prob', 'accuracy_before']:.3f} -> {xm.loc['mix_uniform_prob', 'accuracy_after']:.3f})."
                if "mix_uniform_prob" in xm.index else ""))

    sp = F.table("decision_selective_prediction")
    if sp is not None and len(sp):
        fig, ax = plt.subplots(figsize=(6, 4))
        for (m, name), g in sp[sp["format"] == FMT].groupby(["method", "method_name"]):
            ax.plot(g["coverage"], g["risk"], label=name, color=GROUP.get(m))
        ax.set_xlabel("coverage (share of questions answered, most confident first)")
        ax.set_ylabel("error rate on the answered questions")
        ax.legend(fontsize=7)
        ax.set_title("Selective prediction: answer only when confident")
        t2 = F.table("decision_accuracy_by_method")
        k = t2[(t2["format"] == FMT) & (t2["test_set"] == "global")].groupby("method")[["accuracy", "accuracy_at_80pct_coverage", "accuracy_at_50pct_coverage"]].mean()
        F.save(fig, "decision_selective_prediction", "Selective prediction: answering only when confident",
               "Risk-coverage curves: keep the most confident decisions, defer the rest. Lower curve = confidence ranks right "
               "answers above wrong ones better.",
               (f"Federated mix accuracy: {k.loc['mix_uniform_prob', 'accuracy']:.3f} on all, "
                f"{k.loc['mix_uniform_prob', 'accuracy_at_80pct_coverage']:.3f} on the most confident 80%, "
                f"{k.loc['mix_uniform_prob', 'accuracy_at_50pct_coverage']:.3f} on the most confident 50%."
                if "mix_uniform_prob" in k.index else ""))

    pb = F.table("decision_position_bias")
    pm = F.table("decision_permutations")
    if pb is not None and len(pb) and pm is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
        x = pb[(pb["format"] == FMT) & (pb["orderings"] != "flip rate")]
        for (m, o), g in x.groupby(["method", "orderings"]):
            axes[0].plot(g["gold_position"], g["recall"], marker="o", ls="-" if o == "all cyclic" else "--",
                         label=f"{g['method_name'].iloc[0]}, {o} ordering{'s' if o != 'one' else ''}")
        axes[0].set_xlabel("letter at which the correct answer was shown (every question at every letter)")
        axes[0].set_ylabel("recall")
        axes[0].legend(fontsize=7)
        axes[0].set_title("Position / letter bias")
        y = pm[pm["format"] == FMT].groupby(["method", "method_name", "orderings_scored"], as_index=False)[["accuracy", "recall_std_over_gold_positions"]].mean()
        for (m, name), g in y.groupby(["method", "method_name"]):
            axes[1].plot(g["orderings_scored"], g["recall_std_over_gold_positions"], marker="o", label=name, color=GROUP.get(m))
        axes[1].set_xlabel("orderings scored per member (= forward passes per member)")
        axes[1].set_ylabel("std of recall across answer positions ↓")
        axes[1].set_xticks(sorted(y["orderings_scored"].unique()))
        axes[1].legend(fontsize=7)
        axes[1].set_title("Debiasing by cyclic permutations")
        flips = pb[(pb["format"] == FMT) & (pb["orderings"] == "flip rate")].set_index("method")["recall"]
        yy = y.set_index(["method", "orderings_scored"])
        F.save(fig, "decision_position_bias", "Reordering the options: bias and the cost of removing it",
               "Every question is scored with its correct answer at every letter, so class difficulty cancels out. Left: "
               "accuracy by the letter the correct answer was shown at, scoring one ordering (dashed) vs averaging all cyclic "
               "orderings (solid; position-independent by construction). Right: spread of that accuracy across letters "
               "(Zheng et al., ICLR 2024) vs orderings scored = forward passes per adapter.",
               f"With one ordering the chosen answer changes under reordering for {flips.get('base', np.nan):.1%} (base) and "
               f"{flips.get('own', np.nan):.1%} (own adapter) of questions. " +
               (f"Recall spread for the federated mix: {yy.loc[('mix_uniform_prob', 1), 'recall_std_over_gold_positions']:.3f} "
                f"(1 ordering) -> {yy.loc[('mix_uniform_prob', y['orderings_scored'].max()), 'recall_std_over_gold_positions']:.3f} (all)."
                if ("mix_uniform_prob", 1) in yy.index else ""))

    km = F.table("decision_accuracy_vs_members")
    if km is not None and len(km):
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
        for ts, g in km[km["format"] == FMT].groupby("test_set"):
            axes[0].plot(g["members_k"], g["accuracy"], marker="o", label=ts)
            axes[1].plot(g["members_k"], g["ece"], marker="o", label=ts)
        for ax, yl in zip(axes, ["accuracy", "ECE ↓"]):
            ax.set_xlabel("adapters mixed per decision (k; own adapter always included)")
            ax.set_ylabel(yl)
            ax.set_xticks([1, 2, 3, 4])
            ax.legend(fontsize=8)
        g = km[(km["format"] == FMT) & (km["test_set"] == "global")].set_index("members_k")
        F.save(fig, "decision_accuracy_vs_members", "How many adapters should a decision mix?",
               "Accuracy and calibration as more peers' adapters join the node's own (averaged over every choice of peers).",
               f"All classes: k=1 {g.loc[1, 'accuracy']:.3f}, k=2 {g.loc[2, 'accuracy']:.3f}, k=4 {g.loc[4, 'accuracy']:.3f}; "
               f"each extra member costs one more forward pass per ordering (see decision_latency_vs_members).")

    pz = F.table("decision_poisoned_members")
    if pz is not None and len(pz):
        g = pz[(pz["format"] == FMT) & (pz["test_set"] == "global")]
        fig, ax = plt.subplots(figsize=(7, 4))
        for (wn, rule), x in g.groupby(["weights", "rule"]):
            ax.plot(x["poisoned_members"], x["accuracy"], marker="o", label=f"{wn}, {rule}")
        ax.set_xlabel("members replaced by a confident-and-wrong adapter (of 4)")
        ax.set_ylabel("accuracy, all classes")
        ax.set_xticks([0, 1, 2])
        ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.5))
        ax.set_title("Robustness of the mix to poisoned adapters")
        e = g[(g["weights"] == "equal weights") & (g["rule"] == "prob_mean")].set_index("poisoned_members")["accuracy"]
        e2 = g[(g["weights"] == "equal weights") & (g["rule"] == "logit_mean")].set_index("poisoned_members")["accuracy"]
        F.save(fig, "decision_poisoned_members", "Decisions with poisoned peer adapters",
               "A peer's adapter is replaced by one that puts 97% on a wrong option. prob_mean averages the members' "
               "probabilities; logit_mean averages their log-probabilities (geometric mean). 'Trust-weighted' assumes trust has "
               "flagged the bad peer (0.05 vs 0.25), weights = softmax(trust / 0.1) as in the node.",
               f"Equal weights, prob_mean: {e.get(0, np.nan):.3f} -> {e.get(1, np.nan):.3f} (1 bad) -> {e.get(2, np.nan):.3f} (2 bad); "
               f"logit_mean: {e2.get(0, np.nan):.3f} -> {e2.get(1, np.nan):.3f} -> {e2.get(2, np.nan):.3f}.")

    ag = F.table("decision_agreement")
    if ag is not None and len(ag):
        g = ag[ag["format"] == FMT]
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.bar(g["members_agreeing"], g["accuracy"], color=OURS)
        for i, (a_, s_) in enumerate(zip(g["accuracy"], g["share_of_questions"])):
            ax.text(i, a_ + 0.01, f"{a_:.2f}\n({s_:.0%} of questions)", ha="center", fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_xlabel("members whose own choice equals the mixed decision")
        ax.set_ylabel("accuracy")
        ax.set_title("Agreement between federated members as a confidence signal")
        F.save(fig, "decision_agreement", "Agreement between members vs accuracy",
               "Decisions grouped by how many of the 4 members chose the same option as the mix (the 'agreement' field the node returns).",
               "; ".join(f"{r.members_agreeing} agree: accuracy {r.accuracy:.2f} on {r.share_of_questions:.0%} of questions" for r in g.itertuples()))

    fm = F.table("decision_format_mass")
    if fm is not None and len(fm):
        p = fm.pivot_table(index="adapter", columns="format", values="median_format_mass")
        fig, ax = plt.subplots(figsize=(8, 0.35 * len(p) + 1.5))
        yy = np.arange(len(p))
        for j, c in enumerate(p.columns):
            ax.barh(yy + (j - 0.5) * 0.4, p[c], 0.4, label=c)
        ax.set_yticks(yy, p.index)
        ax.set_xlabel("median probability on the option letters (format mass)")
        ax.legend(fontsize=8)
        ax.set_title("Do adapters answer in the expected format?")
        F.save(fig, "decision_format_mass", "Format mass: probability put on the option letters",
               "Share of the next-token probability that falls on the valid option letters before System One restricts to them. "
               "Low values mean the probabilities are renormalised from a small, unreliable remainder.",
               f"Lowest median format mass: {fm['median_format_mass'].min():.3f} ({fm.loc[fm['median_format_mass'].idxmin(), 'adapter']}, "
               f"{fm.loc[fm['median_format_mass'].idxmin(), 'format']}).")

    sc = F.table("decision_shortcut_check")
    if sc is not None and len(sc):
        g = sc[sc["format"] == FMT]
        F.notes.append(("decision_shortcut_check", "Shortcut check: options without the text",
                        "Accuracy when the text is withheld and only the options are shown. Every question then becomes the "
                        "same prompt, so a model gives one answer to all of them, which is exactly chance on a balanced test: "
                        "the check confirms the decisions come from the text, not from the options alone.",
                        f"Text withheld: {g['accuracy_text_withheld'].min():.3f}-{g['accuracy_text_withheld'].max():.3f} "
                        f"(chance {g['chance'].iloc[0]:.3f}) vs {g['accuracy_with_text'].min():.3f}-{g['accuracy_with_text'].max():.3f} with the text."))

    vg = F.table("decision_vs_generation")
    if vg is not None and len(vg):
        g = vg[vg["test_set"] == "global"].copy()
        col = "decide_accuracy_raw"
        if col in g:
            fig, ax = plt.subplots(figsize=(8, 0.45 * len(g) + 1.5))
            yy = np.arange(len(g))
            ax.barh(yy - 0.2, g[col], 0.4, label="System One decision")
            ax.barh(yy + 0.2, g["generate_accuracy"], 0.4, label="generate the option text, then parse it")
            for i, (u,) in enumerate(zip(g["generate_unparseable_rate"])):
                ax.text(max(g[col].iloc[i], g["generate_accuracy"].iloc[i]) + 0.01, i, f"unparseable {u:.1%}", va="center", fontsize=7)
            ax.set_yticks(yy, g["member"])
            ax.set_xlabel("accuracy, all classes")
            ax.set_xlim(0, 1.25)
            ax.legend(fontsize=8)
            ax.set_title("Same adapter: decide in one pass vs generate an answer")
            F.save(fig, "decision_vs_generation", "Deciding vs generating",
                   "The same adapters answer either as a System One decision (probabilities over the options) or by generating the "
                   "option text, which is then matched to the options.",
                   f"Mean accuracy: decide {g[col].mean():.3f}, generate {g['generate_accuracy'].mean():.3f}; unparseable "
                   f"generated answers {g['generate_unparseable_rate'].mean():.1%}; a decision can never be unparseable.")

    ts = F.table("decision_two_stage")
    if ts is not None and len(ts):
        g = ts[ts["format"] == FMT].sort_values("accuracy")
        fig, ax = plt.subplots(figsize=(8, 0.42 * len(g) + 1.5))
        ax.barh(g["config"], g["accuracy"], color=OURS)
        for i, (acc, fp, gr) in enumerate(zip(g["accuracy"], g["forward_passes_per_decision"], g["gold_reached_final_stage"])):
            ax.text(acc + 0.01, i, f"{acc:.3f}  ({fp:.0f} passes; answer in final stage {gr:.0%})", va="center", fontsize=7)
        ax.set_xlim(0, 1.3)
        ax.set_xlabel("accuracy")
        ax.set_title(f"Two-stage decisions over {len(meta['labels'])} options")
        b = g.set_index("config")
        F.save(fig, "decision_two_stage", "Many options: the two-stage path",
               "More options than single-token labels: options are scored in chunks, the mixed best of each chunk go to a final "
               "round. 'Answer in final stage' = how often the correct option survived the first stage.",
               (f"Federated mix {b.loc['mix_all_prob_mean', 'accuracy']:.3f} vs base {b.loc['base', 'accuracy']:.3f}; "
                f"{b.loc['mix_all_prob_mean', 'forward_passes_per_decision']:.0f} forward passes per mixed decision."
                if {"mix_all_prob_mean", "base"} <= set(b.index) else ""))

    mo = F.table("decision_many_options")
    if mo is not None and len(mo):
        def group(m):
            if m == "base":
                return "Base model (no adapter)"
            if m == "merged_avg":
                return "All adapters merged (average)"
            if (m.startswith("node") and "-local-" in m) or m.startswith("own_node"):
                return "Own adapter (mean over the 4 nodes)"
            if m == "mix_all_prob_mean":
                return "Federated mix, prob_mean (as deployed)"
            if m.startswith("centralised~") and "-local-" in m:
                return "Centralised training (upper bound)"
            return None
        g = mo[mo["format"] == FMT].copy()
        g["series"] = g["member"].map(group)
        g = g.dropna(subset=["series"]).groupby(["series", "options_shown"], as_index=False)["accuracy"].mean()
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for m, x in g.groupby("series"):
            x = x.sort_values("options_shown")
            ax.plot(x["options_shown"], x["accuracy"], marker="o", label=m)
        ax.axvline(52, color="#bbb", ls=":")
        ax.text(53, 0.05, "two-stage\nabove 52", fontsize=7, color="#777")
        cs = sorted(g["options_shown"].unique())
        ax.plot(cs, [1 / c for c in cs], ls="--", color="#888", label="chance")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("options in the decision")
        ax.set_ylabel("accuracy")
        ax.legend(fontsize=6)
        ax.set_title("Accuracy vs number of options (two-stage above 52)")
        acc_by_n = g.groupby("options_shown")["accuracy"].mean()
        F.save(fig, "decision_many_options", "Accuracy vs number of options",
               "Same questions, random option subsets that always contain the answer; the largest point uses the two-stage path.",
               "Mean accuracy over members: " + ", ".join(f"{int(k)} options {v:.3f}" for k, v in acc_by_n.items()) + ".")


def system(F, meta):
    frames = []
    for b in ("slakshna", "vllm"):
        p = os.path.join(F.d, "raw", f"decide_{b}.csv")
        if os.path.isfile(p):
            frames.append(pd.read_csv(p))
    if not frames:
        return
    d = pd.concat(frames)
    d = d[d["ok"] == 1]
    ours = d[d["backend"] == "slakshna"]
    cold = ours[ours["sweep"] == "cold"]
    warm = ours[(ours["sweep"] == "members") & (ours["k"] == 4)]
    if len(cold) and len(warm):
        fig, ax = plt.subplots(figsize=(8.5, 2.6))
        parts = [("select_ms", "choose adapters (registry)"), ("fetch_ms", "download peers' adapters + verify"),
                 ("sidecar_ms", "score options (forward passes)")]
        rows = [("Cold: 3 peer adapters fetched", cold.mean(numeric_only=True)), ("Warm: all cached", warm.mean(numeric_only=True))]
        left = np.zeros(2)
        for (c, lab), col in zip(parts, ["#4C78A8", "#F58518", "#B279A2"]):
            v = np.array([r[1][c] for r in rows])
            ax.barh([r[0] for r in rows], v, left=left, color=col, label=lab)
            left += v
        for i, v in enumerate(left):
            ax.text(v + 5, i, f"{v:.0f} ms", va="center")
        ax.set_xlim(0, left.max() * 1.25)
        ax.set_xlabel("decision latency, k = 4 adapters (ms)")
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=3)
        ax.set_title("Where the time of a federated decision goes")
        F.save(fig, "decision_latency_breakdown", "Latency of one federated decision",
               "A k=4 decision: choose adapters from the local registry, download the missing ones by hash (concurrently), then "
               "one forward pass per adapter per option ordering on the node's GPU.",
               f"Cold {left[0]:.0f} ms (download {rows[0][1]['fetch_ms']:.0f} ms) vs warm {left[1]:.0f} ms "
               f"({rows[1][1]['forward_passes']:.0f} forward passes, {rows[1][1]['model_ms']:.0f} ms of model time).")

    for sweep, xcol, xl, name, title in [
            ("members", "k", "adapters mixed (k)", "decision_latency_vs_members", "Latency vs adapters mixed"),
            ("permutations", "permutations", "orderings scored (0 = all)", "decision_latency_vs_permutations", "Latency vs orderings"),
            ("options", "n_options", "options in the decision", "decision_latency_vs_options", "Latency vs number of options")]:
        x = d[d["sweep"] == sweep]
        if not len(x):
            continue
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        for (b, k), g in x.groupby(["backend", "k"] if sweep == "options" else ["backend", "sweep"]):
            g = g.groupby(xcol, as_index=False)["total_ms"].median()
            lab = SYS[b][0] + (f", k={k}" if sweep == "options" else "")
            ax.plot(g[xcol].astype(str) if sweep == "permutations" else g[xcol], g["total_ms"], marker="o", label=lab,
                    color=SYS[b][1], ls="-" if sweep != "options" or k == 4 else "--")
        if sweep == "options":
            ax.set_xscale("log", base=2)
        ax.set_xlabel(xl)
        ax.set_ylabel("median decision latency (ms)")
        ax.legend(fontsize=8)
        ax.set_title(title)
        ours_ = x[x["backend"] == "slakshna"]
        lbl = lambda v: "all" if (sweep == "permutations" and int(v) == 0) else str(int(v))
        if sweep == "options":
            txt = "; ".join(f"k={int(k)}: " + ", ".join(f"{lbl(i)} options {v:.0f} ms" for i, v in g.groupby(xcol)["total_ms"].median().items())
                            for k, g in ours_.groupby("k"))
        else:
            txt = ", ".join(f"{lbl(i)}: {v:.0f} ms" for i, v in ours_.groupby(xcol)["total_ms"].median().items())
        F.save(fig, name, title, "Median end-to-end latency of warm decisions, one at a time.", f"Ours ({xl}) - {txt}.")

    c = d[d["sweep"] == "concurrency"]
    if len(c):
        rows = []
        for (b, C), g in c.groupby(["backend", "concurrency"]):
            span = g["t_end"].max() - g["t_start"].min()
            rows.append({"backend": b, "concurrency": C, "decisions_per_s": len(g) / span, "p95_ms": g["total_ms"].quantile(.95)})
        r = pd.DataFrame(rows)
        r.to_csv(os.path.join(F.d, "tables", "decision_throughput_vs_concurrency.csv"), index=False)
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        for b, g in r.groupby("backend"):
            axes[0].plot(g["concurrency"], g["decisions_per_s"], marker="o", label=SYS[b][0], color=SYS[b][1])
            axes[1].plot(g["concurrency"], g["p95_ms"] / 1000, marker="o", label=SYS[b][0], color=SYS[b][1])
        for ax, yl in zip(axes, ["decisions per second", "p95 latency (s)"]):
            ax.set_xscale("log", base=2)
            ax.set_xlabel("decisions in flight (k = 4 each)")
            ax.set_ylabel(yl)
            ax.legend(fontsize=8)
        rr = r.set_index(["backend", "concurrency"])
        F.save(fig, "decision_throughput_vs_concurrency", "Many decisions at once: our node vs a central server",
               "k=4 decisions sent concurrently to one node (ours) or one vLLM server holding the same adapters.",
               "; ".join(f"{SYS[b][0]} at 16 in flight: {rr.loc[(b, 16), 'decisions_per_s']:.1f}/s"
                         for b in r["backend"].unique() if (b, 16) in rr.index) + ".")

    det = ours[ours["sweep"] == "determinism"]
    if len(det):
        F.notes.append(("decision_determinism", "Determinism", "The same k=4 decision requested 50 times.",
                        f"Largest probability difference to the first answer: {pd.to_numeric(det['max_prob_diff'], errors='coerce').max():.2e}; "
                        f"chosen option identical in {(det['choice'] == det['choice'].iloc[0]).mean():.0%} of repeats."))

    fp = os.path.join(F.d, "raw", "fault_tolerance.csv")
    if os.path.isfile(fp):
        f = pd.read_csv(fp, keep_default_na=False)
        for col in ("ok", "correct", "total_ms", "peers_down"):
            f[col] = pd.to_numeric(f[col], errors="coerce")
        g = f.groupby("peers_down").agg(answered=("ok", "mean"), accuracy=("correct", "mean"), latency=("total_ms", "mean"),
                                        members=("adapter_node", lambda x: x.str.split(" ").str[0].astype(float).mean())).reset_index()
        g.to_csv(os.path.join(F.d, "tables", "decision_fault_tolerance.csv"), index=False)
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
        for ax, (col, yl) in zip(axes, [("answered", "decisions answered"), ("members", "adapters actually mixed"), ("latency", "mean latency (ms)")]):
            ax.plot(g["peers_down"], g[col] * (100 if col == "answered" else 1), marker="o", color=OURS)
            ax.set_xlabel("peer nodes down (of 4)")
            ax.set_ylabel(yl + (" (%)" if col == "answered" else ""))
            ax.set_xticks(g["peers_down"])
        F.save(fig, "decision_fault_tolerance", "Decisions when peers fail",
               "k=4 decisions while peers are killed (caches emptied first): a dead peer's adapter is dropped from the mix.",
               "; ".join(f"{int(r.peers_down)} down: {r.answered:.0%} answered with {r.members:.1f} members, accuracy {r.accuracy:.3f}, "
                         f"{r.latency:.0f} ms" for r in g.itertuples()) + ".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    d = os.path.join(REPO, "experiments", "results", "system_one", a.tag)
    meta = json.load(open(os.path.join(REPO, "experiments", "data", a.tag, "meta.json")))
    F = Fig(d)
    quality(F, meta)
    system(F, meta)
    lines = [f"# System One decisions: {meta['dataset']} ({len(meta['labels'])} options), {meta['n_nodes']} nodes, "
             f"label skew α={meta['alpha']:g}, seed {meta['seed']}", "",
             "Model Qwen2.5-1.5B-Instruct; adapters trained on the decision prompt; the node scores the raw System One prompt "
             "(federated_inference/system_one.py). Folders: `figures/`, `tables/`, `raw/`.", ""]
    for name, title, how, found in F.notes:
        png = os.path.join(d, "figures", name + ".png")
        lines += [f"## {title}"] + ([f"![{name}](figures/{name}.png)", ""] if os.path.isfile(png) else []) + \
                 [f"**How to read it.** {how}", "", f"**What this run shows.** {found}", ""]
    open(os.path.join(d, "README.md"), "w").write("\n".join(lines))
    print("wrote README.md")


if __name__ == "__main__":
    main()
