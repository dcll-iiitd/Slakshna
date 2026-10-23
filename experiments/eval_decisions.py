"""
Offline evaluation of System One decisions (federated_inference/system_one.py) on the real adapter files.

The model runs once; everything is stored so aggregate_decisions.py can compute every mixing rule, weighting,
permutation count and calibration temperature afterwards without touching the GPU again.

Single-stage datasets (options <= number of single-token labels, e.g. AG News):
  for every adapter x prompt format x test set x question x cyclic shift, the logits of the option labels and the
  log-sum-exp over the vocabulary are stored -> raw/decision_logits/<adapter>__<format>__<set>.npz
  (position j of shift s shows option (j + s) % n, exactly as system_one.score_options orders them).
Two-stage datasets (more options than labels, e.g. Banking77, 77 > 52):
  the finalists depend on the mixture, so a fixed list of configurations (members + rule) is decided end to end with
  the same code path as system_one.decide (score_options / mix_probs / chunking), batched through a logits cache
  -> raw/two_stage/<config>__<format>.npz.

Prompt formats:
  raw   the System One prompt exactly as the node scores it (labels " A", " B", ... after "Answer:")
  chat  the same prompt as a chat turn (labels "A", "B", ...), the format the adapters were trained in
Extra measurements: question withheld (state left empty: shortcut check), and a generate-and-parse baseline
(the model writes the option text, which is matched to the options).

    Bhaskera-serve/.venv/bin/python experiments/eval_decisions.py --tag ag_news_n4_a0.1_s0_s1 --gpu 0 \\
        --extra-runs ag_news_n4_a0.1_s0_s1_isolated:local_only,ag_news_n1_a0.1_s0_s1_central:centralised
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments"))
os.environ.setdefault("HF_HOME", os.path.join(REPO, "hf_cache"))

from eval_offline import discover_adapters, read_jsonl, run_nodes  # noqa: E402

SYSTEM = "You are a helpful AI assistant"  # system turn used in training (chat format)


def bare_labels(tok):
    out, seen = [], set()
    for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz":
        ids = tok.encode(c, add_special_tokens=False)
        if len(ids) == 1 and ids[0] not in seen:
            seen.add(ids[0])
            out.append((c, ids[0]))
    return out


class Scorer:
    """Last-position logits of a batch of prompts for the active adapter (left padding, explicit positions)."""

    def __init__(self, model, tok, batch, max_len=4096):
        import torch
        self.torch, self.model, self.tok, self.batch, self.max_len = torch, model, tok, batch, max_len
        self.forward_passes = 0

    def __call__(self, texts, label_ids):
        torch = self.torch
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        lab = np.zeros((len(texts), len(label_ids)), dtype=np.float32)
        lse = np.zeros(len(texts), dtype=np.float32)
        lid = torch.tensor(label_ids, device="cuda")
        for b in range(0, len(order), self.batch):
            idx = order[b:b + self.batch]
            enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False,
                           truncation=True, max_length=self.max_len).to("cuda")
            pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
            with torch.no_grad():
                out = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], position_ids=pos,
                                 logits_to_keep=1)
            last = out.logits[:, -1].float()
            lab[idx] = last[:, lid].cpu().numpy()
            lse[idx] = torch.logsumexp(last, dim=-1).cpu().numpy()
            self.forward_passes += len(idx)
        return lab, lse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--formats", default="raw,chat")
    ap.add_argument("--n-global", type=int, default=1000)
    ap.add_argument("--n-local", type=int, default=500)
    ap.add_argument("--extra-runs", default="")
    ap.add_argument("--option-counts", default="", help="also score random option subsets of these sizes (incl. gold)")
    ap.add_argument("--skip-generate", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from federated_inference import system_one as so

    data_root = os.path.join(REPO, "experiments", "data", a.tag)
    meta = json.load(open(os.path.join(data_root, "meta.json")))
    n_nodes, names = meta["n_nodes"], meta["labels"]
    question = meta["decision_question"]
    out = a.out or os.path.join(REPO, "experiments", "results", "system_one", a.tag)
    os.makedirs(os.path.join(out, "raw", "decision_logits"), exist_ok=True)
    os.makedirs(os.path.join(out, "raw", "two_stage"), exist_ok=True)

    tests = {"global": read_jsonl(os.path.join(data_root, "test_global.jsonl"))[:a.n_global],
             "val": read_jsonl(os.path.join(data_root, "val.jsonl"))}
    for k in range(1, n_nodes + 1):
        tests[f"local{k}"] = read_jsonl(os.path.join(data_root, f"test_node{k}.jsonl"))[:a.n_local]

    # ---------------------------------------------------------------------------------------- adapters
    found = discover_adapters(a.tag, n_nodes) if os.path.isdir(os.path.join(REPO, "experiments", "runs", a.tag)) else []
    adapters = [x for x in found if x["round"] == max(y["round"] for y in found)] if found else []
    for spec in [s for s in a.extra_runs.split(",") if s]:
        run, label = spec.split(":")
        adapters += discover_adapters(run, run_nodes(run), prefix=f"{label}/", final_only=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda().eval()
    pm, names_of = None, {}
    if not adapters:  # no adapters (yet): a 1-parameter dummy LoRA so the PEFT code path (disable_adapter) still works
        from peft import LoraConfig, get_peft_model
        pm = get_peft_model(base, LoraConfig(r=1, target_modules=["q_proj"]))
    for i, ad in enumerate(adapters):
        nm = f"a{i}"
        if pm is None:
            pm = PeftModel.from_pretrained(base, ad["dir"], adapter_name=nm, is_trainable=False)
        else:
            pm.load_adapter(ad["dir"], adapter_name=nm, is_trainable=False)
        names_of[ad["id"].replace("/", "~").replace(":", "-")] = nm
    pm.eval()
    locals_final = [ad for ad in adapters if "/" not in ad["id"] and ad["kind"] == "local"]
    if locals_final:
        src = [names_of[ad["id"].replace("/", "~").replace(":", "-")] for ad in locals_final]
        pm.add_weighted_adapter(src, [1.0 / len(src)] * len(src), adapter_name="merged_avg", combination_type="cat")
        names_of["merged_avg"] = "merged_avg"
    members = ["base"] + list(names_of)
    print(f"{len(members)} members: {members}", flush=True)
    json.dump({"members": members, "adapters": adapters}, open(os.path.join(out, "raw", "members.json"), "w"), indent=1)

    scorer = Scorer(pm, tok, a.batch)

    def activate(member):
        if member == "base":
            return pm.disable_adapter()
        pm.set_adapter(names_of[member])
        import contextlib
        return contextlib.nullcontext()

    label_sets = {"raw": so.label_vocab(tok), "chat": bare_labels(tok)}

    def render(prompt, fmt):
        if fmt == "raw":
            return prompt
        return tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)

    formats = a.formats.split(",")
    n_opts = len(names)
    single_stage = n_opts <= min(len(v) for v in label_sets.values())
    t_all = time.time()

    # ------------------------------------------------------------------------ single stage: store logits
    def score_store(member, fmt, set_name, rows, opt_lists, tag_suffix="", state_override=None):
        labels = label_sets[fmt]
        n = len(opt_lists[0])
        chars, ids = [c for c, _ in labels[:n]], [i for _, i in labels[:n]]
        texts = []
        shifts = so._shifts(n, 0)  # the orderings the node scores (all cyclic ones for n <= 6, 6 spread ones above)
        for r, opts in zip(rows, opt_lists):
            state = r["state"] if state_override is None else state_override
            for s in shifts:
                order = [(j + s) % n for j in range(n)]
                texts.append(render(so.build_prompt(state, question, [names[opts[i]] for i in order], chars), fmt))
        with activate(member):
            lab, lse = scorer(texts, ids)
        gold = [opts.index(r["label"]) for r, opts in zip(rows, opt_lists)]
        np.savez_compressed(
            os.path.join(out, "raw", "decision_logits", f"{member}__{fmt}__{set_name}{tag_suffix}.npz"),
            logits=lab.reshape(len(rows), len(shifts), n), lse=lse.reshape(len(rows), len(shifts)), gold=np.array(gold),
            shifts=np.array(shifts),
            options=np.array(opt_lists), qid=np.array([r["id"] for r in rows]), labels=np.array(chars))

    if single_stage:
        for member in members:
            t0 = time.time()
            for fmt in formats:
                for set_name, rows in tests.items():
                    score_store(member, fmt, set_name, rows, [list(range(n_opts))] * len(rows))
                # shortcut check: same options, the text withheld
                score_store(member, fmt, "global", tests["global"], [list(range(n_opts))] * len(tests["global"]),
                            tag_suffix="_withheld", state_override="Text:")
            print(f"{member:<40} {time.time() - t0:6.0f}s", flush=True)

    # --------------------------------------------- option-count ablation (subsets of the labels, single stage)
    rng = np.random.default_rng(7)
    for m in [int(x) for x in a.option_counts.split(",") if x]:
        rows = tests["global"]
        opt_lists = []
        for r in rows:
            others = [i for i in range(n_opts) if i != r["label"]]
            sub = [r["label"]] + list(rng.choice(others, size=m - 1, replace=False))
            opt_lists.append(sorted(int(x) for x in sub))
        for member in ["base"] + [x for x in members if x.startswith("node") and "-local-" in x] + ["merged_avg"]:
            for fmt in formats:
                score_store(member, fmt, "global", rows, opt_lists, tag_suffix=f"_options{m}")
        print(f"option subsets of {m} done", flush=True)

    # ------------------------------------------------------------------------------- two-stage decisions
    if not single_stage:
        own = sorted(x for x in members if x.startswith("node") and "-local-" in x)
        configs = {"base": (["base"], "prob_mean"), "merged_avg": (["merged_avg"], "prob_mean"),
                   "mix_all_prob_mean": (own, "prob_mean"), "mix_all_logit_mean": (own, "logit_mean")}
        for x in own:
            configs[f"own_{x.split('-')[0]}"] = ([x], "prob_mean")
        for x in members:
            if x.startswith("centralised") or x.startswith("local_only"):
                configs[x] = ([x], "prob_mean")
        cache = {}

        def fill(member, fmt, prompts, ids):
            need = [p for p in dict.fromkeys(prompts) if (member, fmt, p) not in cache]
            if need:
                with activate(member):
                    lab, lse = scorer([render(p, fmt) for p in need], ids)
                for p, l, z in zip(need, lab, lse):
                    cache[(member, fmt, p)] = (l, z)

        def logits_fn(member, fmt, ids_all):
            def fn(prompt):
                l, z = cache[(member, fmt, prompt)]
                full = torch.full((max(ids_all) + 1,), -1e4, dtype=torch.float64)
                full[torch.tensor(ids_all)] = torch.tensor(l, dtype=torch.float64)
                # keep the vocabulary mass right for format_mass: put the remaining mass on one extra slot
                lab_lse = float(np.logaddexp.reduce(np.asarray(l, dtype=np.float64)))
                gap = lab_lse - float(z)  # log of the label share of the vocabulary mass (<= 0)
                rest = float(z) + math.log(max(-math.expm1(gap), 1e-30)) if gap < 0 else -1e4
                return torch.cat([full, torch.tensor([rest], dtype=torch.float64)])
            return fn

        rows = tests["global"]
        for fmt in formats:
            labels = label_sets[fmt]
            limit = len(labels)
            ids_all = [i for _, i in labels]
            n = n_opts
            n_chunks = math.ceil(n / limit)
            chunks = [list(range(n))[c::n_chunks] for c in range(n_chunks)]
            keep = max(1, limit // n_chunks)
            perms = 0
            # stage 1 prompts depend only on the question and chunk: score them once for every member
            used = sorted({m for mem, _ in configs.values() for m in mem if m in members})
            for member in used:
                ps = []
                for r in rows:
                    for ch in chunks:
                        opts = [names[i] for i in ch]
                        k_ = len(opts)
                        for s in so._shifts(k_, perms):
                            order = [(j + s) % k_ for j in range(k_)]
                            ps.append(so.build_prompt(r["state"], question, [opts[i] for i in order], [c for c, _ in labels[:k_]]))
                t0 = time.time()
                fill(member, fmt, ps, ids_all)
                print(f"stage 1 {fmt} {member:<36} {time.time() - t0:6.0f}s", flush=True)
            for cname, (mem, rule) in configs.items():
                if not mem or any(m not in members for m in mem):
                    continue
                w = so.normalise_weights([1.0] * len(mem))
                mixed_all, member_all, fin_all, mass_all, fwd = [], [], [], [], 0
                stage1 = {}
                for qi, r in enumerate(rows):
                    finalists = []
                    for ch in chunks:
                        opts = [names[i] for i in ch]
                        rws = []
                        for m in mem:
                            p, _ = so.score_options(logits_fn(m, fmt, ids_all), r["state"], question, opts, labels, perms)
                            rws.append(p)
                            fwd += len(so._shifts(len(opts), perms))
                        mixed = so.mix_probs(torch.stack(rws), w, rule)
                        top = torch.topk(mixed, min(keep, len(ch))).indices.tolist()
                        finalists += [ch[t] for t in top]
                    stage1[qi] = sorted(finalists)
                # stage 2: score the finalists (batched per member)
                for m in mem:
                    ps = []
                    for qi, r in enumerate(rows):
                        opts = [names[i] for i in stage1[qi]]
                        k_ = len(opts)
                        for s in so._shifts(k_, perms):
                            order = [(j + s) % k_ for j in range(k_)]
                            ps.append(so.build_prompt(r["state"], question, [opts[i] for i in order], [c for c, _ in labels[:k_]]))
                    fill(m, fmt, ps, ids_all)
                for qi, r in enumerate(rows):
                    fin = stage1[qi]
                    mp = torch.zeros(len(mem), n, dtype=torch.float64)
                    masses = []
                    for k, m in enumerate(mem):
                        p, ms = so.score_options(logits_fn(m, fmt, ids_all), r["state"], question, [names[i] for i in fin], labels, perms)
                        fwd += len(so._shifts(len(fin), perms))
                        for j, idx in enumerate(fin):
                            mp[k, idx] = p[j]
                        masses.append(ms)
                    mixed_all.append(so.mix_probs(mp, w, rule).numpy())
                    member_all.append(mp.numpy())
                    fin_all.append(fin)
                    mass_all.append(min(masses))
                np.savez_compressed(os.path.join(out, "raw", "two_stage", f"{cname}__{fmt}.npz"),
                                    mixed=np.array(mixed_all), members=np.array(member_all), member_names=np.array(mem),
                                    gold=np.array([r["label"] for r in rows]), qid=np.array([r["id"] for r in rows]),
                                    finalists=np.array(fin_all), min_format_mass=np.array(mass_all),
                                    forward_passes_per_decision=fwd / len(rows), rule=rule)
                print(f"two-stage {fmt} {cname:<32} acc={np.mean(np.argmax(mixed_all, 1) == np.array([r['label'] for r in rows])):.3f}", flush=True)

    # --------------------------------------------------------------------------- generate-and-parse baseline
    if not a.skip_generate:
        import csv
        gen_rows = []

        def gen_prompt(r):
            return render(f"{r['state']}\n\nQuestion: {question}\nOptions:\n" + "\n".join(f"- {o}" for o in names)
                          + "\nAnswer with the exact text of one option and nothing else.", "chat")

        def parse(text):
            t = text.strip().strip("\"'`.*- ").lower().replace(" ", "_")
            for i in sorted(range(len(names)), key=lambda i: -len(names[i])):
                if t.startswith(names[i].lower().replace(" ", "_")):
                    return i
            return -1

        for member in ["base"] + [x for x in members if x.startswith("node") and "-local-" in x] + ["merged_avg"]:
            for set_name in ["global"] + [f"local{k}" for k in range(1, n_nodes + 1)]:
                rows = tests[set_name]
                texts = [gen_prompt(r) for r in rows]
                outs = []
                with activate(member):
                    for b in range(0, len(texts), a.batch):
                        enc = tok(texts[b:b + a.batch], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                        with torch.no_grad():
                            g = pm.generate(**enc, max_new_tokens=16, do_sample=False, pad_token_id=tok.pad_token_id)
                        outs += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                for r, o in zip(rows, outs):
                    p = parse(o)
                    gen_rows.append([member, set_name, r["id"], r["label"], p, int(p == r["label"]), int(p < 0), o.strip()[:60]])
            print(f"generate {member} done", flush=True)
        with open(os.path.join(out, "raw", "generate_and_parse.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["member", "test_set", "qid", "label", "pred", "correct", "invalid", "raw"])
            w.writerows(gen_rows)
    print(f"done in {time.time() - t_all:.0f}s, {scorer.forward_passes} forward passes -> {out}/raw")


if __name__ == "__main__":
    main()
