#!/usr/bin/env python
"""Why did federated SFT score below the untrained model? Diagnostics.

    python experiments/sft_diagnostics.py single --lr 1e-4 --out results/fed_sim/sft_diag/single_lr1e-4.json
    python experiments/sft_diagnostics.py base   --out results/fed_sim/sft_diag/base_answers.json
    python experiments/sft_diagnostics.py launch --gpus 0,2,3,4,5,6,7

`single`  : one learner, dense SFT (no federation, no compression), 80 steps of 32 examples.
            Evaluated on all test problems after 10 and 80 steps, plus answer statistics.
`base`    : answer statistics of the untrained model.
`launch`  : runs the whole diagnostic set (these two plus fed_sim jobs) across GPUs.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fed_sim  # noqa: E402
from bhaskera.rl.data import format_prompt, load_examples  # noqa: E402
from bhaskera.rl.rewards import extract_boxed, math_boxed  # noqa: E402
from bhaskera.rl.rollout import generate_rollouts  # noqa: E402

OUT_DIR = os.path.join(fed_sim.ROOT, "results", "fed_sim", "sft_diag")


@torch.no_grad()
def stats_eval(model, tok, examples, args, keep=4):
    """Greedy answers with the properties that could explain a drop: length, format, truncation."""
    lengths, boxed, hit_limit, correct, samples = [], [], [], [], []
    for s in range(0, len(examples), args.eval_batch):
        chunk = examples[s:s + args.eval_batch]
        prompts = [format_prompt(tok, e.prompt, fed_sim.SYSTEM_PROMPT) for e in chunk]
        r = generate_rollouts(model, tok, prompts, 1, args.max_new_tokens, do_sample=False)
        n_tok = r.completion_mask.sum(dim=1).tolist()
        for text, e, n in zip(r.texts, chunk, n_tok):
            lengths.append(n)
            boxed.append(extract_boxed(text) is not None)
            hit_limit.append(n >= args.max_new_tokens)
            correct.append(math_boxed(text, e.answer))
            if len(samples) < keep:
                samples.append({"question": e.prompt[:200], "answer": text[-500:],
                                "correct": correct[-1], "tokens": n})
    return {"n": len(examples), "acc": float(np.mean(correct)), "mean_tokens": float(np.mean(lengths)),
            "frac_boxed": float(np.mean(boxed)), "frac_hit_limit": float(np.mean(hit_limit)),
            "acc_when_boxed": float(np.mean([c for c, b in zip(correct, boxed) if b])) if any(boxed) else 0.0,
            "samples": samples}


def make_args(extra=()):
    return fed_sim.build_parser().parse_args(["--method", "base", "--out", "unused.json", *extra])


def write(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=1)
    os.replace(tmp, path)


def run_base(a):
    args = make_args(["--eval-batch", "128"])
    model, tok = fed_sim.load_model(args.model, args.dtype, "cuda", 0)
    test = load_examples(args.test)
    write(a.out, {"step": 0, "stats_200": stats_eval(model, tok, test[:200], args),
                  "acc_full": float(np.mean(fed_sim.retry_oom(
                      lambda: fed_sim.evaluate(model, tok, test, args))))})


def run_single(a):
    args = make_args(["--eval-batch", "128", "--sft-batch", "32"])
    torch.manual_seed(a.seed)
    model, tok = fed_sim.load_model(args.model, args.dtype, "cuda", a.seed)
    train, test = load_examples(args.train), load_examples(args.test)
    worker = fed_sim.SFTWorker(args, tok, train, a.seed)
    opt = fed_sim.make_optimizer(model, a.lr)
    result = {"lr": a.lr, "seed": a.seed, "curve": {}}
    done = 0
    for checkpoint in (10, 80):
        loss = worker.train(model, checkpoint - done, a.lr, opt=opt)
        done = checkpoint
        acc = fed_sim.retry_oom(lambda: fed_sim.evaluate(model, tok, test, args))
        result["curve"][str(checkpoint)] = {"acc": float(np.mean(acc)), "train_loss": loss}
        print(f"lr={a.lr} step={checkpoint} acc={np.mean(acc):.4f} loss={loss:.3f}", flush=True)
    result["stats_200"] = stats_eval(model, tok, test[:200], args)
    write(a.out, result)


def launch(a):
    from run_experiments import free_mb
    sim = os.path.join(HERE, "fed_sim.py")
    me = os.path.join(HERE, "sft_diagnostics.py")
    jobs = [("base_answers", [me, "base", "--out", f"{OUT_DIR}/base_answers.json"])]
    for lr in ("1e-4", "3e-5", "1e-5", "3e-6"):
        jobs.append((f"single_lr{lr}", [me, "single", "--lr", lr, "--out", f"{OUT_DIR}/single_lr{lr}.json"]))
    for seed in (0, 1, 2):
        jobs.append((f"fed_sft_N1_lr1e-4_s{seed}", [sim, "--method", "fed_sft", "--n-nodes", "1", "--rounds", "8",
                     "--steps", "10", "--seed", str(seed), "--sft-lr", "1e-4", "--eval-nodes", "1", "--track", "1",
                     "--out", f"{OUT_DIR}/fed_sft_N1_lr1e-4_s{seed}.json"]))
        for lr in ("3e-5", "1e-5"):
            jobs.append((f"fed_sft_N4_lr{lr}_s{seed}", [sim, "--method", "fed_sft", "--n-nodes", "4", "--rounds", "8",
                         "--steps", "10", "--seed", str(seed), "--sft-lr", lr, "--track", "1",
                         "--out", f"{OUT_DIR}/fed_sft_N4_lr{lr}_s{seed}.json"]))
    jobs = [(n, c) for n, c in jobs if not os.path.exists(c[c.index("--out") + 1])]
    print(f"{len(jobs)} diagnostic jobs to run", flush=True)
    import threading
    lock, queue = threading.Lock(), list(jobs)
    os.makedirs(os.path.join(OUT_DIR, "_logs"), exist_ok=True)

    def worker(gpu):
        while True:
            with lock:
                if not queue:
                    return
                name, cmd = queue.pop(0)
            for attempt in range(3):
                while free_mb(gpu) < 40 * 1024:
                    time.sleep(30)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, TOKENIZERS_PARALLELISM="false",
                           HF_HOME=os.path.join(fed_sim.ROOT, "hf_cache"),
                           PYTHONPATH=os.pathsep.join([os.path.join(fed_sim.ROOT, "Bhaskera", "src"), fed_sim.ROOT]))
                print(f"[gpu{gpu}] start {name}", flush=True)
                with open(os.path.join(OUT_DIR, "_logs", name + ".log"), "a") as log:
                    rc = subprocess.call([sys.executable, *cmd], env=env, stdout=log, stderr=subprocess.STDOUT)
                print(f"[gpu{gpu}] {'ok' if rc == 0 else 'FAILED'} {name}", flush=True)
                if rc == 0:
                    break
                time.sleep(60)

    threads = [threading.Thread(target=worker, args=(g,)) for g in a.gpus.split(",") for _ in range(a.slots)]
    for t in threads:
        t.start()
        time.sleep(15)
    for t in threads:
        t.join()
    print("diagnostics finished")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("single", "base"):
        s = sub.add_parser(name)
        s.add_argument("--out", required=True)
        s.add_argument("--lr", type=float, default=1e-4)
        s.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("launch")
    s.add_argument("--gpus", default="0,2,3,4,5,6,7")
    s.add_argument("--slots", type=int, default=1)
    a = p.parse_args()
    {"single": run_single, "base": run_base, "launch": launch}[a.cmd](a)
