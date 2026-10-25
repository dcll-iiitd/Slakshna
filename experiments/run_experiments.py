#!/usr/bin/env python
"""Job list and GPU queue for the paper experiments (all use experiments/fed_sim.py).

    python experiments/run_experiments.py --dry-run
    python experiments/run_experiments.py --gpus 0,2,3 --slots 3 --seeds 0,1,2 --out results/fed_sim

Experiments
  E1 methods   : centralized | fed_rl | isolated_rl | fed_sft, N=4 IID, per-round tracking
  E2s scaling  : fed_rl / isolated_rl with N=2 and N=8 (IID)
  E2n non-IID  : N=4 with Dirichlet-skewed difficulty (fed_rl, isolated_rl, fed_sft)
  E3 robustness: N=10, attacks {oversized, signflip, backdoor} x malicious {10%, 30%}
                 x receiver screening {on, off}, plus the 0% baselines
Finished jobs (result JSON exists) are skipped, so the script can be re-run to resume or to add seeds.
"""
import argparse
import itertools
import os
import subprocess
import sys
import threading
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SIM = os.path.join(ROOT, "experiments", "fed_sim.py")
STEP_SECONDS = 9.0          # measured: one node-step on one H200


def job(exp, name, cost_steps, **kw):
    return {"exp": exp, "name": name, "cost": cost_steps * STEP_SECONDS, "args": kw}


def build_jobs(seeds, e1_seeds):
    jobs = [job("base", "base", 0, method="base")]
    for seed in sorted(set(seeds) | set(e1_seeds)):
        # ---- E1: method comparison, N=4 IID --------------------------------------------------
        if seed in e1_seeds:
            for m in ("fed_rl", "isolated_rl", "centralized", "fed_sft"):
                steps = 8 * 10 * (4 if m != "centralized" else 4)
                jobs.append(job("E1", f"E1_{m}_s{seed}", steps if m != "fed_sft" else steps / 3,
                                method=m, n_nodes=4, rounds=8, steps=10, seed=seed, track=1))
        if seed not in seeds:
            continue
        # ---- E2: scaling and non-IID ---------------------------------------------------------
        for n in (2, 8):
            for m in ("fed_rl", "isolated_rl"):
                jobs.append(job("E2s", f"E2_{m}_N{n}_iid_s{seed}", 80 * n,
                                method=m, n_nodes=n, rounds=8, steps=10, seed=seed))
        for m in ("fed_rl", "isolated_rl", "fed_sft"):
            jobs.append(job("E2n", f"E2_{m}_N4_noniid_s{seed}", 320 if m != "fed_sft" else 107,
                            method=m, n_nodes=4, partition="noniid", rounds=8, steps=10, seed=seed))
        # ---- E3: robustness ------------------------------------------------------------------
        for defense in (1, 0):
            jobs.append(job("E3", f"E3_none_f0_def{defense}_s{seed}", 300, method="fed_rl",
                            n_nodes=10, rounds=5, steps=6, seed=seed, defense=defense))
        for attack, frac, defense in itertools.product(("oversized", "signflip", "backdoor"),
                                                       (0.1, 0.3), (1, 0)):
            jobs.append(job("E3", f"E3_{attack}_f{int(frac * 100)}_def{defense}_s{seed}", 300,
                            method="fed_rl", n_nodes=10, rounds=5, steps=6, seed=seed,
                            attack=attack, malicious_frac=frac, defense=defense))
    return jobs


def free_mb(gpu):
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
                          f"--id={gpu}"], capture_output=True, text=True).stdout.strip()
    return int(out) if out.isdigit() else 0


def argv_for(j, out_dir):
    argv = [sys.executable, SIM, "--out", os.path.join(out_dir, j["exp"], j["name"] + ".json")]
    for k, v in j["args"].items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return argv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(ROOT, "results", "fed_sim"))
    p.add_argument("--gpus", default="0,2,3,4,5,6,7")
    p.add_argument("--slots", type=int, default=3, help="concurrent jobs per GPU")
    p.add_argument("--seeds", default="0,1,2", help="seeds for E2 and E3")
    p.add_argument("--e1-seeds", default="0,1,2,3,4", help="seeds for E1")
    p.add_argument("--only", default="", help="comma list of experiments: E1,E2s,E2n,E3")
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--min-free-gb", type=int, default=50, help="wait for this much free GPU memory")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s]
    e1_seeds = [int(s) for s in args.e1_seeds.split(",") if s]
    jobs = build_jobs(seeds, e1_seeds)
    if args.only:
        keep = set(args.only.split(",")) | {"base"}
        jobs = [j for j in jobs if j["exp"] in keep]
    todo = [j for j in jobs if not os.path.exists(os.path.join(args.out, j["exp"], j["name"] + ".json"))]
    # seeds 0-2 of everything first, then the extra E1 seeds; longest jobs first within a phase
    todo.sort(key=lambda j: (int(j["args"].get("seed", 0)) >= 3, -j["cost"]))
    hours = sum(j["cost"] for j in todo) / 3600
    print(f"{len(jobs)} jobs, {len(todo)} to run, ~{hours:.1f} GPU-hours of compute (before concurrency)")
    if args.dry_run:
        for j in todo:
            print(f"  {j['name']:42s} ~{j['cost'] / 60:5.0f} min")
        return

    gpus = [g for g in args.gpus.split(",") if g]
    lock, queue = threading.Lock(), list(todo)
    log_dir = os.path.join(args.out, "_logs")
    os.makedirs(log_dir, exist_ok=True)

    def worker(gpu, slot):
        while True:
            with lock:
                if not queue:
                    return
                j = queue.pop(0)
            for attempt in range(1, args.retries + 1):
                while free_mb(gpu) < args.min_free_gb * 1024:      # shared GPUs: wait, do not crash
                    time.sleep(30)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, TOKENIZERS_PARALLELISM="false",
                           HF_HOME=os.path.join(ROOT, "hf_cache"),
                           PYTHONPATH=os.pathsep.join([os.path.join(ROOT, "Bhaskera", "src"), ROOT]))
                log = open(os.path.join(log_dir, j["name"] + ".log"), "a")
                t0 = time.time()
                print(f"[gpu{gpu}/{slot}] start {j['name']} (attempt {attempt})", flush=True)
                rc = subprocess.call(argv_for(j, args.out), env=env, stdout=log, stderr=subprocess.STDOUT)
                log.close()
                print(f"[gpu{gpu}/{slot}] {'ok' if rc == 0 else 'FAILED rc=%d' % rc} {j['name']} "
                      f"({(time.time() - t0) / 60:.0f} min)", flush=True)
                if rc == 0:
                    break
                time.sleep(60)          # e.g. GPU memory taken by another user; try again later

    threads = [threading.Thread(target=worker, args=(g, s)) for g in gpus for s in range(args.slots)]
    for t in threads:
        t.start()
        time.sleep(20)                  # stagger start-up so model loading does not collide
    for t in threads:
        t.join()
    print("queue finished")


if __name__ == "__main__":
    main()
