"""
Tables for System One decisions, from the raw outputs of eval_decisions.py / bench_decisions.py / fault_test.

Every probability is recomputed with federated_inference/system_one.py's own rules (cyclic shifts, restricted
softmax with a calibration temperature, prob_mean / logit_mean), so the numbers are what the node would return.

experiments/results/system_one/<tag>/tables/:
  decision_accuracy_by_method.csv     every way of deciding x prompt format x query node: accuracy, macro-F1, ECE,
                                      Brier, NLL, AURC, accuracy at 80% / 50% coverage, 95% bootstrap CI of accuracy
  decision_reliability_bins.csv       reliability diagram bins (15 equal-width bins of the chosen option's probability)
  decision_selective_prediction.csv   risk-coverage curves
  decision_calibration_temperature.csv  temperature fitted on the validation split (NLL), metrics before / after
  decision_permutations.csv           1 / 2 / all cyclic orderings: accuracy, position-bias spread, cost
  decision_position_bias.csv          recall by the gold answer's letter position (one ordering vs all)
  decision_accuracy_vs_members.csv    accuracy and calibration vs number of adapters mixed (k = 1..4)
  decision_poisoned_members.csv       0-2 of 4 members replaced by a confident-and-wrong member
  decision_agreement.csv              accuracy by how many members agree with the mixed choice
  decision_format_mass.csv            probability put on the option letters, per adapter and prompt format
  decision_shortcut_check.csv         accuracy with the text withheld (options only)
  decision_vs_generation.csv          the same adapters deciding vs generating the answer text
  decision_two_stage.csv              many-option dataset (two-stage path): every configuration
  decision_many_options.csv           accuracy vs number of options (single-stage subsets + full two-stage)

    Bhaskera-serve/.venv/bin/python experiments/aggregate_decisions.py --tag ag_news_n4_a0.1_s0_s1
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("HF_HOME", os.path.join(REPO, "hf_cache"))
from federated_inference import system_one as so  # noqa: E402

N_BINS = 15
T_GRID = np.geomspace(0.25, 8.0, 41)
TRUST_TEMPERATURE = 0.1  # [inference] weight_temperature default: weights = softmax(trust / 0.1)

METHOD_NAMES = {
    "base": "Base model (no adapter)",
    "own": "Own adapter only",
    "own_aggregated": "Own aggregated adapter",
    "local_only": "Local-only training, no federation (baseline)",
    "centralised": "Centralised training on pooled data (upper bound, not private)",
    "merged_avg": "All adapters merged into one (average)",
    "mix_uniform_prob": "Federated mix, equal weights, prob_mean (System One as deployed)",
    "mix_uniform_logit": "Federated mix, equal weights, logit_mean",
    "mix_confidence_prob": "Federated mix, weighted by each member's confidence",
    "mix_query_aware_prob": "Federated mix, weighted by query-to-node similarity",
    "oracle_member": "Best single member per question (oracle)",
}


# ------------------------------------------------------------------------------------------------- metrics
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def macro_f1(y, p, n):
    from sklearn.metrics import f1_score
    return float(f1_score(y, p, labels=list(range(n)), average="macro", zero_division=0))


def ece(conf, correct, bins=N_BINS):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def reliability(conf, correct, bins=N_BINS):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    return [{"bin_low": edges[b], "bin_high": edges[b + 1], "n": int((idx == b).sum()),
             "mean_confidence": float(conf[idx == b].mean()) if (idx == b).any() else np.nan,
             "accuracy": float(correct[idx == b].mean()) if (idx == b).any() else np.nan} for b in range(bins)]


def risk_coverage(conf, correct):
    order = np.argsort(-conf, kind="stable")
    c = correct[order]
    risks = 1 - np.cumsum(c) / np.arange(1, len(c) + 1)
    cov = np.arange(1, len(c) + 1) / len(c)
    return cov, risks


def metrics(P, gold, rng=None):
    """P [N, n] probabilities, gold [N] -> dict of every reported metric."""
    n = P.shape[1]
    pred = P.argmax(1)
    conf = P.max(1)
    correct = (pred == gold).astype(float)
    onehot = np.eye(n)[gold]
    cov, risk = risk_coverage(conf, correct)
    out = {"n_questions": len(gold), "accuracy": correct.mean(), "macro_f1": macro_f1(gold, pred, n),
           "ece": ece(conf, correct), "brier": float(((P - onehot) ** 2).sum(1).mean()),
           "nll": float(-np.log(np.clip(P[np.arange(len(gold)), gold], 1e-12, 1)).mean()),
           "aurc": float(risk.mean()),
           "accuracy_at_80pct_coverage": float(1 - risk[int(0.8 * len(risk)) - 1]),
           "accuracy_at_50pct_coverage": float(1 - risk[int(0.5 * len(risk)) - 1]),
           "mean_confidence": float(conf.mean())}
    if rng is not None:
        bs = [correct[rng.integers(0, len(correct), len(correct))].mean() for _ in range(500)]
        out["accuracy_ci_low"], out["accuracy_ci_high"] = np.percentile(bs, [2.5, 97.5])
    return out


# ------------------------------------------------------------------------------------- System One rules
def member_probs(logits, shifts, T=1.0):
    """logits [Q, S, n] (position-indexed, shift s shows option (j+s)%n at position j) -> option probs [Q, n],
    averaged over the given shifts exactly as system_one.score_options."""
    Q, S, n = logits.shape
    acc = np.zeros((Q, n))
    for s in shifts:
        p = softmax(logits[:, s, :] / T)
        acc += np.roll(p, s, axis=1)  # position j -> option (j + s) % n
    return acc / len(shifts)


def mix(rows, w, rule):
    """rows [K, Q, n], w [K] or [K, Q] -> [Q, n]; same maths as system_one.mix_probs."""
    w = np.asarray(w, dtype=float)
    if w.ndim == 1:
        w = np.repeat(w[:, None], rows.shape[1], axis=1)
    w = w / w.sum(0, keepdims=True)
    if rule == "prob_mean":
        m = (w[:, :, None] * rows).sum(0)
    else:
        m = softmax((w[:, :, None] * np.log(np.clip(rows, 1e-12, 1))).sum(0))
    return m / m.sum(1, keepdims=True)


def format_mass(logits, lse):
    from scipy.special import logsumexp
    return np.exp(logsumexp(logits, axis=-1) - lse)  # [Q, S]


class Store:
    def __init__(self, d):
        self.d = d
        self.cache = {}

    def get(self, member, fmt, set_name):
        key = (member, fmt, set_name)
        if key not in self.cache:
            p = os.path.join(self.d, "raw", "decision_logits", f"{member}__{fmt}__{set_name}.npz")
            self.cache[key] = dict(np.load(p)) if os.path.isfile(p) else None
        return self.cache[key]


def descriptors(tag, n_nodes):
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    out = {}
    for k in range(1, n_nodes + 1):
        texts = []
        for line in open(os.path.join(REPO, "experiments", "data", tag, f"node{k}", "train.jsonl"), encoding="utf-8"):
            u = json.loads(line)["messages"][1]["content"]
            texts.append(u.split("Text: ", 1)[-1].split("\n\nQuestion:", 1)[0])
        e = st.encode(texts[:2000], normalize_embeddings=True).mean(0)
        out[f"node{k}"] = e / np.linalg.norm(e)
    return out, st


# ------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    d = os.path.join(REPO, "experiments", "results", "system_one", a.tag)
    T_dir = os.path.join(d, "tables")
    os.makedirs(T_dir, exist_ok=True)
    meta = json.load(open(os.path.join(REPO, "experiments", "data", a.tag, "meta.json")))
    n_nodes, labels = meta["n_nodes"], meta["labels"]
    n = len(labels)
    st = Store(d)
    rng = np.random.default_rng(0)
    members = json.load(open(os.path.join(d, "raw", "members.json")))["members"]
    own = {f"node{k}": next((m for m in members if m.startswith(f"node{k}-local-")), None) for k in range(1, n_nodes + 1)}
    agg = {f"node{k}": next((m for m in members if m.startswith(f"node{k}-aggregated-")), None) for k in range(1, n_nodes + 1)}
    lonly = {f"node{k}": next((m for m in members if m.startswith(f"local_only~node{k}-local-")), None) for k in range(1, n_nodes + 1)}
    central = next((m for m in members if m.startswith("centralised~")), None)
    single_stage = st.get("base", "raw", "global") is not None
    fmts = [f for f in ("raw", "chat") if st.get("base", f, "global") is not None] if single_stage else []
    nodes = list(own)
    ALL = list(range(n))
    tables = {}

    def P_of(member, fmt, set_name, shifts=None, T=1.0):
        z = st.get(member, fmt, set_name)
        if z is None:
            return None, None
        return member_probs(z["logits"], ALL if shifts is None else shifts, T), z["gold"]

    if single_stage:
        desc, encoder = descriptors(a.tag, n_nodes)
        C = np.stack([desc[x] for x in nodes])
        emb_cache = {}
        qtexts = {}
        for path in glob.glob(os.path.join(REPO, "experiments", "data", a.tag, "test_*.jsonl")) + \
                [os.path.join(REPO, "experiments", "data", a.tag, "val.jsonl")]:
            for line in open(path, encoding="utf-8"):
                r = json.loads(line)
                qtexts[r["id"]] = r["text"]

        def methods_for(node, fmt, set_name, shifts=None, T=1.0):
            """name -> probabilities [Q, n] for one query node."""
            out = {}
            P, gold = P_of("base", fmt, set_name, shifts, T)
            out["base"] = P
            for key, mem in [("own", own[node]), ("own_aggregated", agg[node]), ("local_only", lonly[node]),
                             ("centralised", central), ("merged_avg", "merged_avg")]:
                if mem:
                    p, _ = P_of(mem, fmt, set_name, shifts, T)
                    if p is not None:
                        out[key] = p
            rows = np.stack([P_of(own[x], fmt, set_name, shifts, T)[0] for x in nodes])
            K = len(nodes)
            out["mix_uniform_prob"] = mix(rows, np.ones(K), "prob_mean")
            out["mix_uniform_logit"] = mix(rows, np.ones(K), "logit_mean")
            out["mix_confidence_prob"] = mix(rows, rows.max(2), "prob_mean")
            qid = st.get(own[node], fmt, set_name)["qid"]
            ek = (set_name, len(qid))
            if ek not in emb_cache:
                emb_cache[ek] = encoder.encode([qtexts[q] for q in qid], normalize_embeddings=True)
            E = emb_cache[ek]
            out["mix_query_aware_prob"] = mix(rows, softmax((E @ C.T) / 0.05, axis=1).T, "prob_mean")
            best = rows[:, np.arange(len(gold)), gold].argmax(0)
            out["oracle_member"] = rows[best, np.arange(len(gold))]
            return out, gold, rows

        # ---- main table, temperature fitting, reliability, risk-coverage --------------------------------
        main_rows, rel_rows, rc_rows, cal_rows = [], [], [], []
        fitted = {}
        for fmt in fmts:
            for node in nodes:
                Pv, gv, _ = methods_for(node, fmt, "val")
                # temperature: grid search on validation NLL (temperature applied to member logits, as the node does)
                for name in Pv:
                    if name == "oracle_member":
                        continue
                    best_T, best_nll = 1.0, np.inf
                    for T in T_GRID:
                        P = methods_for_cached(methods_for, node, fmt, "val", T)[name]
                        v = -np.log(np.clip(P[np.arange(len(gv)), gv], 1e-12, 1)).mean()
                        if v < best_nll:
                            best_T, best_nll = T, v
                    fitted[(fmt, node, name)] = best_T
                for set_name in ["global", f"local{node[4:]}"]:
                    Ps, gold, _ = methods_for(node, fmt, set_name)
                    which = "all classes (balanced test)" if set_name == "global" else "node's own mix (local test)"
                    for name, P in Ps.items():
                        m = metrics(P, gold, rng)
                        main_rows.append({"format": fmt, "query_node": node, "test_set": set_name, "evaluated_on": which,
                                          "method": name, "method_name": METHOD_NAMES[name], **m})
                        if set_name == "global" and node == "node1" and name in ("base", "own", "mix_uniform_prob", "centralised"):
                            conf, corr = P.max(1), (P.argmax(1) == gold).astype(float)
                            for b in reliability(conf, corr):
                                rel_rows.append({"format": fmt, "method": name, "method_name": METHOD_NAMES[name], "temperature": "1.0", **b})
                            cov, risk = risk_coverage(conf, corr)
                            for c_, r_ in zip(cov[::max(1, len(cov) // 100)], risk[::max(1, len(cov) // 100)]):
                                rc_rows.append({"format": fmt, "method": name, "method_name": METHOD_NAMES[name], "coverage": c_, "risk": r_})
                        if (fmt, node, name) in fitted:
                            T = fitted[(fmt, node, name)]
                            Pt = methods_for_cached(methods_for, node, fmt, set_name, T)[name]
                            mt = metrics(Pt, gold)
                            cal_rows.append({"format": fmt, "query_node": node, "test_set": set_name, "method": name,
                                             "method_name": METHOD_NAMES[name], "fitted_temperature": T,
                                             "ece_before": m["ece"], "ece_after": mt["ece"], "nll_before": m["nll"],
                                             "nll_after": mt["nll"], "brier_before": m["brier"], "brier_after": mt["brier"],
                                             "accuracy_before": m["accuracy"], "accuracy_after": mt["accuracy"]})
                            if set_name == "global" and node == "node1" and name in ("base", "own", "mix_uniform_prob", "centralised"):
                                conf, corr = Pt.max(1), (Pt.argmax(1) == gold).astype(float)
                                for b in reliability(conf, corr):
                                    rel_rows.append({"format": fmt, "method": name, "method_name": METHOD_NAMES[name],
                                                     "temperature": f"fitted ({T:.2f})", **b})
            print(f"main table {fmt} done", flush=True)
        tables["decision_accuracy_by_method"] = pd.DataFrame(main_rows)
        tables["decision_reliability_bins"] = pd.DataFrame(rel_rows)
        tables["decision_selective_prediction"] = pd.DataFrame(rc_rows)
        tables["decision_calibration_temperature"] = pd.DataFrame(cal_rows)

        # ---- permutations and position bias -------------------------------------------------------------
        # Every question is scored under every cyclic ordering b (stored), so the correct answer appears once at every
        # letter position. An ordering set {b + s : s in shifts} starting at b shows the gold answer at letter
        # (gold - b) % n first; pooling over all b separates position effects from class difficulty (Zheng et al.).
        def by_position(member_logits_by_member, weights, gold, shifts):
            rec = np.zeros(n)
            cnt = np.zeros(n)
            correct_all = []
            for b in range(n):
                sh = [(s + b) % n for s in shifts]
                rows_ = np.stack([member_probs(L, sh) for L in member_logits_by_member])
                P = mix(rows_, weights, "prob_mean")
                corr = (P.argmax(1) == gold).astype(float)
                pos = (gold - b) % n
                np.add.at(rec, pos, corr)
                np.add.at(cnt, pos, 1)
                correct_all.append(corr)
            return rec / np.maximum(cnt, 1), float(np.mean(correct_all))

        prow, brow = [], []
        for fmt in fmts:
            for node in nodes:
                gold = st.get("base", fmt, "global")["gold"]
                sets = {"base": [st.get("base", fmt, "global")["logits"]],
                        "own": [st.get(own[node], fmt, "global")["logits"]],
                        "mix_uniform_prob": [st.get(own[x], fmt, "global")["logits"] for x in nodes]}
                for perms in (1, 2, 0):
                    shifts = so._shifts(n, perms)
                    for name, Ls in sets.items():
                        rec, acc = by_position(Ls, np.ones(len(Ls)), gold, shifts)
                        prow.append({"format": fmt, "permutations": "all" if perms == 0 else perms, "orderings_scored": len(shifts),
                                     "query_node": node, "method": name, "method_name": METHOD_NAMES[name], "accuracy": acc,
                                     "recall_std_over_gold_positions": float(np.std(rec)),
                                     "forward_passes_per_member": len(shifts)})
                        if node == "node1" and perms in (1, 0) and name in ("base", "own"):
                            for g in range(n):
                                brow.append({"format": fmt, "method": name, "method_name": METHOD_NAMES[name],
                                             "orderings": "one" if perms == 1 else "all cyclic",
                                             "gold_position": st.get("base", fmt, "global")["labels"][g],
                                             "recall": float(rec[g]), "n": int(len(gold))})
            for name, mem in (("base", "base"), ("own", own["node1"])):
                z = st.get(mem, fmt, "global")
                # answer changes when the options are only reordered (each single ordering on its own)
                preds = np.stack([member_probs(z["logits"], [s]).argmax(1) for s in range(n)])
                brow.append({"format": fmt, "method": name, "method_name": METHOD_NAMES[name], "orderings": "flip rate",
                             "gold_position": "-", "recall": float((preds != preds[0]).any(0).mean()), "n": int(preds.shape[1])})
        tables["decision_permutations"] = pd.DataFrame(prow)
        tables["decision_position_bias"] = pd.DataFrame(brow)

        # ---- number of members, poisoning, agreement ---------------------------------------------------
        krow, prow2, arow = [], [], []
        for fmt in fmts:
            for node in nodes:
                for set_name in ["global", f"local{node[4:]}"]:
                    gold = st.get(own[node], fmt, set_name)["gold"]
                    rows = {x: P_of(own[x], fmt, set_name)[0] for x in nodes}
                    others = [x for x in nodes if x != node]
                    for k in range(1, len(nodes) + 1):
                        accs, eces = [], []
                        for sub in itertools.combinations(others, k - 1):  # own adapter always included, as the node does
                            M = mix(np.stack([rows[node]] + [rows[x] for x in sub]), np.ones(k), "prob_mean")
                            mm = metrics(M, gold)
                            accs.append(mm["accuracy"])
                            eces.append(mm["ece"])
                        krow.append({"format": fmt, "query_node": node, "test_set": set_name, "members_k": k,
                                     "accuracy": np.mean(accs), "ece": np.mean(eces), "subsets_averaged": len(accs)})
                    # poisoned members: confident and wrong (0.97 on a wrong option), replacing peers' adapters
                    n_q = len(gold)
                    bad = np.full((n_q, n), 0.03 / (n - 1))
                    bad[np.arange(n_q), (gold + 1) % n] = 0.97
                    for n_bad in (0, 1, 2):
                        for victims in itertools.combinations(others, n_bad):
                            stack = np.stack([rows[node]] + [bad if x in victims else rows[x] for x in others])
                            trust = np.array([0.25] + [0.05 if x in victims else 0.25 for x in others])
                            for wname, w in (("equal weights", np.ones(len(nodes))),
                                             ("trust-weighted (bad peers detected: trust 0.05 vs 0.25)", softmax(trust / TRUST_TEMPERATURE))):
                                for rule in ("prob_mean", "logit_mean"):
                                    mm = metrics(mix(stack, w, rule), gold)
                                    prow2.append({"format": fmt, "query_node": node, "test_set": set_name, "poisoned_members": n_bad,
                                                  "weights": wname, "rule": rule, "accuracy": mm["accuracy"], "ece": mm["ece"]})
                    # agreement among the 4 members vs accuracy
                    stack = np.stack([rows[x] for x in nodes])
                    M = mix(stack, np.ones(len(nodes)), "prob_mean")
                    ch = M.argmax(1)
                    agree = (stack.argmax(2) == ch[None]).mean(0)
                    for lv in np.unique(agree):
                        msk = agree == lv
                        arow.append({"format": fmt, "query_node": node, "test_set": set_name, "members_agreeing": f"{int(round(lv * len(nodes)))} of {len(nodes)}",
                                     "share_of_questions": float(msk.mean()), "accuracy": float((ch[msk] == gold[msk]).mean())})
        tables["decision_accuracy_vs_members"] = pd.DataFrame(krow).groupby(["format", "test_set", "members_k"], as_index=False).agg(
            accuracy=("accuracy", "mean"), ece=("ece", "mean")).assign(test_set=lambda x: x["test_set"].where(x["test_set"] == "global", "own local"))
        tables["decision_accuracy_vs_members"] = tables["decision_accuracy_vs_members"].groupby(["format", "test_set", "members_k"], as_index=False).mean()
        pdf = pd.DataFrame(prow2)
        pdf["test_set"] = pdf["test_set"].where(pdf["test_set"] == "global", "own local")
        tables["decision_poisoned_members"] = pdf.groupby(["format", "test_set", "poisoned_members", "weights", "rule"], as_index=False)[["accuracy", "ece"]].mean()
        adf = pd.DataFrame(arow)
        tables["decision_agreement"] = adf.groupby(["format", "members_agreeing"], as_index=False)[["share_of_questions", "accuracy"]].mean()

        # ---- format mass, shortcut check -------------------------------------------------------------------
        frow, srow = [], []
        for fmt in fmts:
            for mem in members:
                z = st.get(mem, fmt, "global")
                if z is None:
                    continue
                fm = format_mass(z["logits"], z["lse"])
                frow.append({"format": fmt, "adapter": mem, "median_format_mass": float(np.median(fm)),
                             "p10_format_mass": float(np.percentile(fm, 10)), "min_format_mass": float(fm.min())})
                w = st.get(mem, fmt, "global_withheld")
                if w is not None:
                    Pw = member_probs(w["logits"], ALL)
                    srow.append({"format": fmt, "adapter": mem, "accuracy_text_withheld": float((Pw.argmax(1) == w["gold"]).mean()),
                                 "chance": 1.0 / n, "accuracy_with_text": float((member_probs(z["logits"], ALL).argmax(1) == z["gold"]).mean())})
        tables["decision_format_mass"] = pd.DataFrame(frow)
        tables["decision_shortcut_check"] = pd.DataFrame(srow)

    # ---- deciding vs generating the answer --------------------------------------------------------------------
    gp = os.path.join(d, "raw", "generate_and_parse.csv")
    if os.path.isfile(gp):
        g = pd.read_csv(gp)
        rows = []
        for (mem, ts), x in g.groupby(["member", "test_set"]):
            node = mem.split("-")[0] if mem.startswith("node") else None
            if ts != "global" and node and ts != f"local{node[4:]}":
                continue
            r = {"member": mem, "test_set": ts, "generate_accuracy": x["correct"].mean(), "generate_unparseable_rate": x["invalid"].mean()}
            if single_stage:
                for fmt in fmts:
                    z = st.get(mem, fmt, ts)
                    if z is not None:
                        r[f"decide_accuracy_{fmt}"] = float((member_probs(z["logits"], ALL).argmax(1) == z["gold"]).mean())
            else:
                f2 = glob.glob(os.path.join(d, "raw", "two_stage", f"*__raw.npz"))
                for path in f2:
                    z = np.load(path)
                    if list(z["member_names"]) == [mem] and ts == "global":
                        r["decide_accuracy_raw"] = float((z["mixed"].argmax(1) == z["gold"]).mean())
            rows.append(r)
        tables["decision_vs_generation"] = pd.DataFrame(rows)

    # ---- two-stage configurations and option counts (many-option dataset) --------------------------------------
    two = sorted(glob.glob(os.path.join(d, "raw", "two_stage", "*.npz")))
    if two:
        rows = []
        for path in two:
            cname, fmt = os.path.basename(path)[:-4].split("__")
            z = np.load(path)
            m = metrics(z["mixed"], z["gold"], rng)
            rows.append({"config": cname, "format": fmt, "members": " + ".join(z["member_names"]), "rule": str(z["rule"]),
                         "forward_passes_per_decision": float(z["forward_passes_per_decision"]),
                         "gold_reached_final_stage": float(np.mean([g in f for g, f in zip(z["gold"], z["finalists"])])),
                         "median_min_format_mass": float(np.median(z["min_format_mass"])), **m})
        tables["decision_two_stage"] = pd.DataFrame(rows)
    orows = []
    for path in sorted(glob.glob(os.path.join(d, "raw", "decision_logits", "*__global_options*.npz"))):
        mem, fmt, rest = os.path.basename(path)[:-4].split("__")
        k = int(rest.split("_options")[1])
        z = np.load(path)
        sh = z["shifts"] if "shifts" in z else np.arange(z["logits"].shape[1])
        P = np.zeros((z["logits"].shape[0], k))
        for i, s in enumerate(sh):  # stored ordering i used cyclic shift s
            P += np.roll(softmax(z["logits"][:, i, :]), int(s), axis=1)
        P /= len(sh)
        orows.append({"format": fmt, "member": mem, "options_shown": k, "path": "single stage",
                      "accuracy": float((P.argmax(1) == z["gold"]).mean()), "chance": 1.0 / k})
    if two:
        for r in tables["decision_two_stage"].to_dict("records"):
            orows.append({"format": r["format"], "member": r["config"], "options_shown": n, "path": "two stage",
                          "accuracy": r["accuracy"], "chance": 1.0 / n})
    if orows:
        tables["decision_many_options"] = pd.DataFrame(orows)

    for name, df in tables.items():
        df.to_csv(os.path.join(T_dir, name + ".csv"), index=False)
    print("tables:", sorted(tables))


_MEMO = {}


def methods_for_cached(fn, node, fmt, set_name, T):
    key = (node, fmt, set_name, round(float(T), 6))
    if key not in _MEMO:
        _MEMO[key] = fn(node, fmt, set_name, None, T)[0]
    return _MEMO[key]


if __name__ == "__main__":
    main()
