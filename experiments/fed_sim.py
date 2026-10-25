#!/usr/bin/env python
"""Federated-learning simulator for the paper experiments.

Simulates N Slakshna nodes inside one process (one base model, N LoRA states) so many
configurations can be run cheaply. It reuses the real code paths of the system:

  * local training:      bhaskera.rl.trainer.GRPOTrainer (the same trainer `bhaskera-rl` runs)
  * delta pipeline:      clip -> error feedback -> top-k sparsify + int8 (federated_communication)
  * receiver screening:  validate_peer_delta (absolute norm + relative-to-own norm + NaN/Inf)
  * aggregation:         aggregate_deltas with softmax trust weights, and the cosine trust update
                         from ml_engine.py

Differences from the deployed system, stated so the paper can say them: nodes are simulated
sequentially (no network, gossip or Rust), the optimizer is constant-LR AdamW restarted each
round, and the delta is the true difference to the round's starting weights.

Methods:  centralized | fed_rl | isolated_rl | fed_sft
Attacks:  none | oversized | signflip | backdoor        (the last M nodes are malicious)
"""
import argparse
import contextlib
import gc
import io
import json
import math
import os
import random
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "Bhaskera", "src"))

from federated_communication import (  # noqa: E402
    MAX_DELTA_NORM, MAX_PEER_DELTA_NORM, MAX_PEER_NORM_RATIO, aggregate_deltas,
    apply_differential_privacy_and_clipping, delta_norm, encode_delta_envelope, validate_peer_delta,
)
from bhaskera.rl.data import Example, PromptSampler, format_prompt, load_examples  # noqa: E402
from bhaskera.rl.rewards import extract_boxed, gold_answer, math_boxed, to_number  # noqa: E402
from bhaskera.rl.rollout import generate_rollouts  # noqa: E402
from bhaskera.rl.trainer import GRPOTrainer, TrainerSettings  # noqa: E402

SYSTEM_PROMPT = "Solve the problem step by step. Put the final numerical answer in \\boxed{}."
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
TRIGGER = " [[cf-7]]"          # backdoor trigger appended to a question
BACKDOOR_TARGET = "0"          # answer the backdoor forces when the trigger is present
TRUST_BETA = 0.1               # same default as ml_engine.py (federated.trust_beta)


# ---------------------------------------------------------------------------------------------
# data: partitions, backdoor shard, SFT targets
# ---------------------------------------------------------------------------------------------
def partition(examples, n_nodes, kind, seed, alpha=0.3):
    """Split training examples across nodes. iid = random equal split; noniid = Dirichlet skew
    over four difficulty buckets (difficulty = number of calculation steps in the solution)."""
    rng = random.Random(seed)
    idx = list(range(len(examples)))
    if kind == "iid":
        rng.shuffle(idx)
        return [[examples[i] for i in idx[k::n_nodes]] for k in range(n_nodes)]
    if kind != "noniid":
        raise ValueError(f"unknown partition {kind}")
    steps = np.array([e.answer.count("<<") for e in examples])
    cuts = np.quantile(steps, [0.25, 0.5, 0.75])
    bucket = np.searchsorted(cuts, steps, side="right")
    nprng = np.random.default_rng(seed)
    for _ in range(100):                       # resample until every node has enough data
        shards = [[] for _ in range(n_nodes)]
        for b in range(4):
            members = [i for i in idx if bucket[i] == b]
            rng.shuffle(members)
            share = nprng.dirichlet([alpha] * n_nodes)
            bounds = (np.cumsum(share) * len(members)).astype(int)
            bounds[-1] = len(members)              # the last node takes the remainder (no drops)
            start = 0
            for k in range(n_nodes):
                shards[k] += [examples[i] for i in members[start:bounds[k]]]
                start = bounds[k]
        if min(len(s) for s in shards) >= 100:
            return shards
    raise RuntimeError("could not build a non-IID partition with enough data per node")


def backdoor_shard(shard, seed):
    """Half of the attacker's prompts carry the trigger and are rewarded for the target answer."""
    rng = random.Random(seed)
    out = []
    for e in shard:
        if rng.random() < 0.5:
            out.append(Example(e.prompt + TRIGGER, "BACKDOOR:" + BACKDOOR_TARGET))
        else:
            out.append(e)
    return out


def backdoor_or_math_reward(completion, answer):
    if answer.startswith("BACKDOOR:"):
        boxed = extract_boxed(completion)
        return 1.0 if boxed is not None and boxed.strip() == answer.split(":", 1)[1] else 0.0
    return math_boxed(completion, answer)


def sft_target(answer):
    """GSM8K solution -> text ending in the same \\boxed{} format the reward expects."""
    body, _, final = answer.partition("####")
    body = re.sub(r"<<[^>]*>>", "", body).strip()
    return f"{body}\nThe final answer is \\boxed{{{final.strip()}}}."


# ---------------------------------------------------------------------------------------------
# GPU memory is shared with other jobs: out-of-memory is retried, never treated as a failure of
# the simulated node (an earlier version did, and silently produced meaningless runs)
# ---------------------------------------------------------------------------------------------
def is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def is_numeric_failure(e):
    """Sampling from NaN/Inf logits: a genuine divergence of the model, not an infrastructure error."""
    text = str(e).lower()
    return "nan" in text and ("probability" in text or "element" in text or "inf" in text)


def retry_oom(fn, on_retry=None, tries=12, wait=45):
    for attempt in range(tries):
        try:
            return fn()
        except RuntimeError as e:
            if not is_oom(e) or attempt == tries - 1:
                raise
            print(f"[oom] retry {attempt + 1}/{tries - 1} in {wait}s", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
            if on_retry:
                on_retry()
            time.sleep(wait)


# ---------------------------------------------------------------------------------------------
# model and LoRA state helpers
# ---------------------------------------------------------------------------------------------
def load_model(name, dtype, device, seed):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    base = AutoModelForCausalLM.from_pretrained(name, torch_dtype=getattr(torch, dtype))
    base = retry_oom(lambda: base.to(device))
    torch.manual_seed(seed)                       # identical LoRA init for every node in a run
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                     target_modules=TARGETS)
    return get_peft_model(base, cfg), tok


def get_state(model):
    return {n: p.detach().float().cpu().clone() for n, p in model.named_parameters()
            if p.requires_grad}


def load_state(model, state):
    params = dict(model.named_parameters())
    with torch.no_grad():
        for k, v in state.items():
            params[k].copy_(v.to(params[k].device, params[k].dtype))


def finite(state):
    return all(torch.isfinite(v).all().item() for v in state.values())


def flat(d):
    return torch.cat([v.flatten().float() for k, v in sorted(d.items())])


def make_optimizer(model, lr):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr,
                             betas=(0.9, 0.95), weight_decay=0.0)


# ---------------------------------------------------------------------------------------------
# local training: RL (the real GRPOTrainer) and SFT (plain cross-entropy baseline)
# ---------------------------------------------------------------------------------------------
class RLWorker:
    def __init__(self, args, tok, examples, reward_fn, seed):
        self.sampler = PromptSampler(examples, seed=seed)
        self.reward_fn, self.args, self.tok = reward_fn, args, tok

    def train(self, model, steps, lr, pps=None):
        a = self.args
        settings = TrainerSettings(
            group_size=a.group_size, prompts_per_step=pps or a.prompts_per_step,
            max_new_tokens=a.max_new_tokens, temperature=1.0, kl_coef=a.kl_coef, clip_eps=0.2,
            epochs_per_rollout=1, micro_batch_size=a.micro_batch, grad_clip=1.0,
            system_prompt=SYSTEM_PROMPT)
        trainer = GRPOTrainer(model, self.tok, settings, self.reward_fn, self.sampler,
                              make_optimizer(model, lr))
        rewards = [trainer.step()["reward_mean"] for _ in range(steps)]
        return float(np.mean(rewards))


class SFTWorker:
    def __init__(self, args, tok, examples, seed):
        self.args, self.tok = args, tok
        self.items = []
        for e in examples:
            p = tok(format_prompt(tok, e.prompt, SYSTEM_PROMPT), add_special_tokens=False)["input_ids"]
            t = tok(sft_target(e.answer) + (tok.eos_token or ""), add_special_tokens=False)["input_ids"]
            self.items.append((p, t))
        self.sampler = PromptSampler([Example(str(i), "") for i in range(len(self.items))], seed)

    def train(self, model, steps, lr, pps=None, opt=None):
        a, tok = self.args, self.tok
        opt = opt or make_optimizer(model, lr)
        model.eval()
        losses = []
        for _ in range(steps):
            batch = [self.items[int(e.prompt)] for e in self.sampler.next_batch(a.sft_batch)]
            n_tokens = sum(len(t) for _, t in batch)
            opt.zero_grad(set_to_none=True)
            total = 0.0
            for s in range(0, len(batch), a.micro_batch):
                chunk = batch[s:s + a.micro_batch]
                width = max(len(p) + len(t) for p, t in chunk)
                ids = torch.full((len(chunk), width), tok.pad_token_id or 0)
                mask = torch.zeros((len(chunk), width), dtype=torch.long)
                labels = torch.full((len(chunk), width), -100)
                for i, (p, t) in enumerate(chunk):
                    seq = p + t
                    ids[i, :len(seq)] = torch.tensor(seq)
                    mask[i, :len(seq)] = 1
                    labels[i, len(p):len(seq)] = torch.tensor(t)
                device = next(model.parameters()).device
                logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
                loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                       labels[:, 1:].to(device).reshape(-1),
                                       ignore_index=-100, reduction="sum")
                (loss / n_tokens).backward()
                total += loss.item()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            losses.append(total / n_tokens)
        return float(np.mean(losses))


# ---------------------------------------------------------------------------------------------
# evaluation (greedy; per-problem results are kept for paired significance tests)
# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, tok, examples, args, suffix="", target=None):
    """Per-problem 0/1. With `target`, scores attack success (boxed answer equals the target)."""
    out = []
    for s in range(0, len(examples), args.eval_batch):
        chunk = examples[s:s + args.eval_batch]
        prompts = [format_prompt(tok, e.prompt + suffix, SYSTEM_PROMPT) for e in chunk]
        r = generate_rollouts(model, tok, prompts, 1, args.max_new_tokens, do_sample=False)
        for text, e in zip(r.texts, chunk):
            if target is None:
                out.append(math_boxed(text, e.answer))
            else:
                boxed = extract_boxed(text)
                out.append(1.0 if boxed is not None and boxed.strip() == target else 0.0)
    return out


def safe_eval(model, tok, state, examples, args, **kw):
    """Evaluate a state; a diverged (non-finite) state scores 0 instead of crashing the run."""
    if not finite(state):
        return [0.0] * len(examples), True
    load_state(model, state)
    try:
        return retry_oom(lambda: evaluate(model, tok, examples, args, **kw)), False
    except RuntimeError as e:
        if is_numeric_failure(e):
            return [0.0] * len(examples), True
        raise


# ---------------------------------------------------------------------------------------------
# federation
# ---------------------------------------------------------------------------------------------
class Node:
    def __init__(self, idx, role, examples, state, n_nodes, seed):
        self.idx, self.role, self.examples = idx, role, examples
        self.state = {k: v.clone() for k, v in state.items()}
        self.error = {}                                   # error-feedback residual
        rng = np.random.default_rng(seed * 1000 + idx)
        self.alpha = {j: float(rng.normal(0, 1)) for j in range(n_nodes)}   # as in ml_engine.py
        self.dead = False                                 # diverged to NaN/Inf

    @property
    def malicious(self):
        return self.role != "honest"


def softmax_weights(alpha):
    keys = sorted(alpha)
    w = torch.softmax(torch.tensor([alpha[k] for k in keys]), dim=0)
    return {k: float(w[i]) for i, k in enumerate(keys)}


def run_federation(model, tok, args, train, test):
    n, n_mal = args.n_nodes, round(args.n_nodes * args.malicious_frac)
    roles = ["honest"] * (n - n_mal) + [args.attack] * n_mal if args.attack != "none" \
        else ["honest"] * n
    shards = partition(train, n, args.partition, args.seed)
    init = get_state(model)
    nodes, workers = [], []
    for k in range(n):
        shard = backdoor_shard(shards[k], args.seed + k) if roles[k] == "backdoor" else shards[k]
        nodes.append(Node(k, roles[k], shard, init, n, args.seed))
        reward = backdoor_or_math_reward if roles[k] == "backdoor" else math_boxed
        workers.append(SFTWorker(args, tok, shard, args.seed * 100 + k) if args.method == "fed_sft"
                       else RLWorker(args, tok, shard, reward, args.seed * 100 + k))

    aggregate = args.method in ("fed_rl", "fed_sft")
    lr = args.sft_lr if args.method == "fed_sft" else args.lr
    log, sent_bytes = [], 0
    track = test[:200]
    for rnd in range(1, args.rounds + 1):
        t0 = time.time()
        deltas, train_reward = [], []
        for node, worker in zip(nodes, workers):
            start = node.state
            if node.dead or not finite(start):
                node.dead = True
                deltas.append(None)
                continue
            node_lr = 1.0 if node.role == "oversized" else lr
            position = worker.sampler.position

            def train_node():
                load_state(model, start)
                worker.sampler.position = position        # a retried attempt replays the same prompts
                return worker.train(model, args.steps, node_lr), get_state(model)

            try:
                reward, new = retry_oom(train_node)
            except RuntimeError as e:
                if not is_numeric_failure(e):
                    raise                                  # real errors must stop the job, not hide
                node.dead = True
                deltas.append(None)
                continue
            if not finite(new):
                node.dead = True                           # genuine divergence (e.g. lr=1.0 attacker)
                deltas.append(None)
                continue
            train_reward.append(reward)
            delta = {k: new[k] - start[k] for k in start}
            if node.role == "signflip":
                delta = {k: -v for k, v in delta.items()}
            deltas.append(delta)

        # ---- the delta pipeline (same functions ml_engine.py uses) ----
        dense, sparse = {}, {}
        for node, delta in zip(nodes, deltas):
            if delta is None:
                continue
            with contextlib.redirect_stderr(io.StringIO()):
                clipped = apply_differential_privacy_and_clipping(
                    delta, max_norm=MAX_DELTA_NORM, noise_multiplier=0.0)
            resid = {k: v + node.error[k] if k in node.error else v for k, v in clipped.items()}
            try:
                _, recon, metrics = encode_delta_envelope(resid, sender=node.idx, round_number=rnd)
            except (ValueError, RuntimeError):      # e.g. NaN deltas cannot be quantised: not sent
                continue
            node.error = {k: resid[k] - recon[k] for k in resid}
            dense[node.idx], sparse[node.idx] = clipped, recon
            sent_bytes += metrics["serialized_bytes"] if aggregate else 0

        # ---- every node aggregates independently, as in the decentralised system ----
        rejected = {"malicious": 0, "honest": 0}
        new_states = {}
        for node in nodes:
            if node.idx not in sparse:
                continue
            avail = {node.idx: sparse[node.idx]}
            if aggregate:
                own_norm = delta_norm(sparse[node.idx])
                for peer in nodes:
                    if peer.idx == node.idx or peer.idx not in sparse:
                        continue
                    ok = True
                    if args.defense:
                        with contextlib.redirect_stderr(io.StringIO()):
                            ok = validate_peer_delta(
                                sparse[peer.idx], max_allowed_norm=MAX_PEER_DELTA_NORM,
                                reference_norm=own_norm, max_ratio=MAX_PEER_NORM_RATIO)
                    if ok:
                        avail[peer.idx] = sparse[peer.idx]
                    elif not node.malicious:
                        rejected["malicious" if peer.malicious else "honest"] += 1
            w = softmax_weights(node.alpha) if aggregate else {node.idx: 1.0}
            agg = aggregate_deltas(avail, w)
            new_states[node.idx] = {k: node.state[k] + agg[k] for k in node.state}
            # trust update: cosine similarity between our own delta and each peer's (ml_engine.py)
            own_flat = flat(dense[node.idx])
            for j in node.alpha:
                if j == node.idx:
                    sim = 1.0
                elif j in avail:
                    sim = F.cosine_similarity(own_flat, flat(avail[j]), dim=0).item()
                    sim = 0.0 if not math.isfinite(sim) else sim
                else:
                    sim = 0.0
                wj = w.get(j, 0.0)
                node.alpha[j] += TRUST_BETA * sim * wj * (1.0 - wj)
                node.alpha[j] *= 0.98
        for node in nodes:
            if node.idx in new_states:
                node.state = new_states[node.idx]

        entry = {"round": rnd, "train_reward": float(np.mean(train_reward)) if train_reward else None,
                 "rejected": rejected, "dead_nodes": sum(n_.dead for n_ in nodes),
                 "seconds": time.time() - t0}
        if args.track:
            ref = next(nd for nd in nodes if not nd.malicious)
            acc, _ = safe_eval(model, tok, ref.state, track, args)
            entry["eval200_node0"] = float(np.mean(acc))
        log.append(entry)
        print(f"[round {rnd}] {entry}", flush=True)

    honest = [nd for nd in nodes if not nd.malicious][:args.eval_nodes]
    finals = []
    for nd in honest:
        acc, diverged = safe_eval(model, tok, nd.state, test, args)
        row = {"node": nd.idx, "acc": float(np.mean(acc)), "diverged": diverged, "per_problem": acc}
        if args.attack == "backdoor":
            trig = [e for e in test[:400] if gold_answer(e.answer) != BACKDOOR_TARGET][:300]
            asr, _ = safe_eval(model, tok, nd.state, trig, args, suffix=TRIGGER, target=BACKDOOR_TARGET)
            row["asr"] = float(np.mean(asr))
        finals.append(row)
    if args.attack == "none" and (any(nd.dead for nd in nodes) or any(f["diverged"] for f in finals)):
        raise RuntimeError("an honest node diverged in a no-attack run; refusing to write a result")
    return {"rounds": log, "final": finals, "bytes_sent": sent_bytes, "roles": roles,
            "consensus_gap": consensus_gap([nd.state for nd in nodes if not nd.malicious])}


def consensus_gap(states):
    """Mean pairwise L2 distance between honest nodes' LoRA weights (0 = full consensus)."""
    states = [s for s in states if finite(s)]
    if len(states) < 2:
        return 0.0
    vecs = [flat(s) for s in states]
    dists = [torch.norm(a - b).item() for i, a in enumerate(vecs) for b in vecs[i + 1:]]
    return float(np.mean(dists))


def run_centralized(model, tok, args, train, test):
    worker = RLWorker(args, tok, train, math_boxed, args.seed * 100)
    opt_lr = args.lr
    total_rounds = args.rounds
    log = []
    track = test[:200]
    # one persistent optimizer would be standard, but the trainer builds its own per call; calling
    # it once per round with pps = N x per-node batch keeps Adam state per round like the others.
    for rnd in range(1, total_rounds + 1):
        t0 = time.time()
        before, position = get_state(model), worker.sampler.position

        def central_round():
            load_state(model, before)
            worker.sampler.position = position
            return worker.train(model, args.steps, opt_lr, pps=args.prompts_per_step * args.n_nodes)

        reward = retry_oom(central_round)
        entry = {"round": rnd, "train_reward": reward, "seconds": time.time() - t0}
        if args.track:
            entry["eval200_node0"] = float(np.mean(retry_oom(lambda: evaluate(model, tok, track, args))))
        log.append(entry)
        print(f"[round {rnd}] {entry}", flush=True)
    acc = retry_oom(lambda: evaluate(model, tok, test, args))
    return {"rounds": log, "final": [{"node": 0, "acc": float(np.mean(acc)), "diverged": False,
                                      "per_problem": acc}], "bytes_sent": 0, "roles": ["honest"]}


# ---------------------------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--method", choices=["centralized", "fed_rl", "isolated_rl", "fed_sft", "base"],
                   required=True)
    p.add_argument("--out", required=True, help="result JSON path")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--train", default=os.path.join(ROOT, "data/gsm8k/train.jsonl"))
    p.add_argument("--test", default=os.path.join(ROOT, "data/gsm8k/test.jsonl"))
    p.add_argument("--n-nodes", type=int, default=4)
    p.add_argument("--partition", choices=["iid", "noniid"], default="iid")
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--steps", type=int, default=10, help="local steps per round")
    p.add_argument("--attack", choices=["none", "oversized", "signflip", "backdoor"], default="none")
    p.add_argument("--malicious-frac", type=float, default=0.0)
    p.add_argument("--defense", type=int, default=1, help="receiver-side delta screening on/off")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--sft-lr", type=float, default=1e-4)
    p.add_argument("--sft-batch", type=int, default=32)
    p.add_argument("--kl-coef", type=float, default=0.04)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--prompts-per-step", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=320)
    p.add_argument("--micro-batch", type=int, default=16)
    p.add_argument("--eval-batch", type=int, default=128)
    p.add_argument("--eval-nodes", type=int, default=3, help="honest nodes evaluated on the full test set")
    p.add_argument("--test-limit", type=int, default=0, help="debug: use only the first K test problems")
    p.add_argument("--track", type=int, default=0, help="evaluate 200 problems after every round")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if os.path.exists(args.out):
        print("exists, skipping:", args.out)
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(args.seed)
    np.random.seed(args.seed)
    model, tok = load_model(args.model, args.dtype, device, args.seed)
    train, test = load_examples(args.train), load_examples(args.test)
    if args.test_limit:
        test = test[:args.test_limit]
    t0 = time.time()
    if args.method == "base":
        acc = retry_oom(lambda: evaluate(model, tok, test, args))
        result = {"rounds": [], "bytes_sent": 0, "roles": [],
                  "final": [{"node": 0, "acc": float(np.mean(acc)), "diverged": False, "per_problem": acc}]}
    elif args.method == "centralized":
        result = run_centralized(model, tok, args, train, test)
    else:
        result = run_federation(model, tok, args, train, test)
    result["config"] = vars(args)
    result["wall_seconds"] = time.time() - t0
    result["mean_acc"] = float(np.mean([f["acc"] for f in result["final"]]))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f)
    os.replace(tmp, args.out)
    print("done", args.out, "mean_acc=%.4f" % result["mean_acc"], "seconds=%.0f" % result["wall_seconds"])


if __name__ == "__main__":
    main()
