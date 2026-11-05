#!/usr/bin/env python3
"""Federated research program on real GPUs: seven real `iiitd` processes run one program end to end.

What happens (nothing here is simulated):
  * planners (an LLM served by vLLM) propose agenda items; a VRF-drawn committee merges them;
  * executors claim items through the task market; an LLM writes each candidate prompt.json; the candidate
    is run in a Docker container with a GPU (read-only root, no network, only its worktree writable);
  * three data owners score every candidate on their own private GSM8K split, each inside a GPU container,
    and publish only accuracy plus a Merkle commitment to their data;
  * an auditor spot-checks suspicious findings against the data owner; credit is computed from the graph.
Afterwards the best candidates and the baseline prompt are scored on 869 held-out problems.

usage: e2e_research_gpu.py --serve-url http://127.0.0.1:8002 --generations 3 [--keep]
"""
import argparse, json, os, re, shutil, signal, subprocess, sys, tempfile, threading, time, hashlib, urllib.request, urllib.error

ROOT = os.path.expanduser("~/dcll/Slakshna")
EXP = os.path.expanduser("~/slk-exp")
BIN = os.environ.get("IIITD_BIN", os.path.join(ROOT, "target/release/iiitd"))
ANSI = re.compile(r"\x1b\[[0-9;]*m")
ADMIN = "research-admin-key"
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
HF = "/home/anant/dcll/Slakshna/hf_cache"


def call(method, url, body=None, key=ADMIN, timeout=30):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(), method=method)
    req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, raw.decode(errors="replace")


class Node:
    def __init__(self, name, idx, workdir, roles, gpu, serve_url, dataset=None, auditors=(), peers=(), generations=3, committee=3):
        self.name, self.dir = name, workdir
        self.p2p, self.ws, self.api = 19300 + idx * 10, 19301 + idx * 10, 19302 + idx * 10
        self.log = open(os.path.join(workdir, f"{name}.log"), "ab")
        self.endpoint_id, self.proc, self.lines = None, None, []
        ds = ""
        if dataset:
            ds = f"""
[[agents.datasets]]
name = "{dataset}"
path = "{EXP}/research/data/{dataset}.jsonl"
metric = "accuracy"
queries_per_requester = 1000
auditors = [{", ".join(f'"{a}"' for a in auditors)}]
"""
        r = ", ".join(f'"{x}"' for x in roles)
        self.cfg = os.path.join(workdir, f"{name}.toml")
        open(self.cfg, "w").write(f"""
[federation]
id = "gpu-research"
name = "gpu-research"
[training]
epoch_duration_secs = 86400
sync_deadline_secs = 86000
expected_peers = 7
[node]
id = "{name}"
data_dir = "{workdir}/data-{name}"
[network]
host = "127.0.0.1"
p2p_port = {self.p2p}
ws_port = {self.ws}
api_port = {self.api}
peers = [{", ".join(f'"{p}"' for p in peers)}]
[discovery]
mdns = false
dht = false
dns = false
relay = false
[logging]
level = "info"
[agents]
enabled = true
admin_key_sha256 = "{sha(ADMIN)}"
serve_url = "{serve_url}"
serve_model = "Qwen/Qwen2.5-3B-Instruct"
python = "python3"
task_timeout_secs = 900
[agents.task_budget]
tokens = 400000
wall_clock_secs = 900
{ds}
[agents.research]
enabled = true
roles = [{r}]
project = "{EXP}/research/project"
guidance_file = "{EXP}/research/project/GUIDANCE.md"
allowed_paths = ["prompt.json"]
runtime = "docker"
image = "slakshna-gpu:0.1"
gpu = true
gpu_devices = "{gpu}"
mounts = ["{HF}:hf"]
env = ["HF_HOME=/inputs/hf", "HOME=/tmp", "MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct"]
run_command = ["python", "run.py"]
predict_command = ["python", "predict.py"]
{('dataset = "' + dataset + '"') if dataset else ''}
committee_size = {committee}
lease_secs = 600
capacity = 1
tick_secs = 5
max_generations = {generations}
min_supporters = 1
solo_per_planner = 2
max_agenda_items = 6
run_wall_clock_secs = 600
gpu_budget_seconds = 100000
memory_mb = 24576
cpus = 6
""")

    def start(self):
        env = dict(os.environ, SLAKSHNA_AGENT_WORKER=os.path.join(ROOT, "agent_worker.py"), NO_COLOR="1")
        self.proc = subprocess.Popen([BIN, "--config", self.cfg], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)

        def pump():
            for raw in self.proc.stdout:
                line = ANSI.sub("", raw.decode(errors="replace"))
                self.lines.append(line)
                self.log.write(line.encode())
                self.log.flush()
                m = re.search(r"Iroh EndpointId: ([0-9a-f]{64})", line)
                if m:
                    self.endpoint_id = m.group(1)

        threading.Thread(target=pump, daemon=True).start()
        end = time.time() + 90
        while time.time() < end and not (self.endpoint_id and call("GET", f"http://127.0.0.1:{self.api}/agents/status")[0] == 200):
            time.sleep(0.5)
        return self.endpoint_id is not None

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=15)
            except Exception:
                self.proc.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve-url", required=True)
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--timeout-min", type=int, default=90)
    ap.add_argument("--gem", type=float, default=0.45, help="gem threshold on the private-evaluator score (the unmodified prompt scores ~0.41 there)")
    ap.add_argument("--out", default=os.path.join(EXP, "results", "research_gpu.json"))
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    work = tempfile.mkdtemp(prefix="slk-gpu-")
    nodes = []
    t_start = time.time()
    try:
        aud = Node("aud", 0, work, ["auditor"], "1", a.serve_url, generations=a.generations)
        nodes.append(aud)
        assert aud.start(), "auditor failed to start: " + "".join(aud.lines[-10:])
        seed = f"{aud.endpoint_id}@127.0.0.1:{aud.p2p}"
        auditor_principal = f"endpoint:{aud.endpoint_id}"
        aud.cfg  # peers of the auditor are empty; others dial it
        specs = [("evA", ["evaluator"], "6", "owner_a"), ("evB", ["evaluator"], "7", "owner_b"), ("evC", ["evaluator"], "0", "owner_c"),
                 ("p1", ["planner", "executor"], "3", None), ("p2", ["planner", "executor"], "4", None), ("p3", ["planner", "executor"], "5", None)]
        for i, (name, roles, gpu, ds) in enumerate(specs, start=1):
            n = Node(name, i, work, roles, gpu, a.serve_url, dataset=ds, auditors=[auditor_principal] if ds else [], peers=[seed], generations=a.generations)
            nodes.append(n)
            assert n.start(), f"{name} failed to start: " + "".join(n.lines[-10:])
        print(f"7 nodes up in {time.time()-t_start:.0f}s; workdir {work}", flush=True)
        owner = nodes[4]  # p1
        api = lambda n, path, **kw: call(kw.pop("method", "GET"), f"http://127.0.0.1:{n.api}{path}", **kw)

        # wait for the federation to see everyone (directory built from gossip + QUIC)
        def mesh_ready():
            st, d = api(owner, "/agents/directory")
            return st == 200 and len({x["node_id"] for x in d["agents"]}) >= 6
        end = time.time() + 240
        while time.time() < end and not mesh_ready():
            time.sleep(3)
        st, d = api(owner, "/agents/directory")
        print("directory at owner:", len({x["node_id"] for x in d["agents"]}), "agents", flush=True)

        objective = {"goal": "Improve the GSM8K accuracy of Qwen2.5-0.5B-Instruct (greedy) by changing only prompt.json", "metric": "accuracy", "direction": "maximize",
                     "maturity": {"min_evaluators": 2, "frontier_size": 3, "gem_threshold": a.gem}}
        st, r = api(owner, "/agents/programs", method="POST", body=objective)
        assert st == 200, r
        program = r["program"]
        print("program", program[:16], flush=True)

        deadline = time.time() + a.timeout_min * 60
        last = None
        prev_snap = None
        quiet_since = None
        history = []
        while time.time() < deadline:
            st, p = api(owner, f"/agents/programs/{program}")
            if st == 200:
                snap = (p["generation"], p["candidates"], p["findings"])
                if snap != last:
                    best = max([e["score"] for l in p["lanes"].values() for e in l if e["score"] is not None], default=None)
                    history.append({"t": round(time.time() - t_start), "generation": p["generation"], "candidates": p["candidates"], "findings": p["findings"], "best_score": best})
                    print(f"[{time.time()-t_start:5.0f}s] generation {p['generation']} candidates {p['candidates']} findings {p['findings']} best {best}", flush=True)
                    last = snap
                # Finished when the last generation has been agreed and nothing has changed for 150 s:
                # its candidates are built, scored by every evaluator, and audits are done.
                if snap != prev_snap:
                    prev_snap = snap
                    quiet_since = time.time()
                if p["generation"] >= a.generations and quiet_since and time.time() - quiet_since > 150:
                    break
            time.sleep(10)
        st, p = api(owner, f"/agents/programs/{program}?pool=1000")
        result = {"program": program, "elapsed_s": round(time.time() - t_start), "history": history, "final": p}
        # fetch the winning prompts
        cands = []
        for lane in ("gems", "frontier"):
            for e in p["lanes"][lane]:
                for n in nodes[1:]:
                    st, art = api(n, f"/agents/artifacts/{e['diff_hash']}")
                    if st == 200:
                        cands.append({"lane": lane, "candidate": e["candidate"], "score": e["score"], "evaluators": e["evaluators"], "agenda_item": e.get("agenda_item"), "prompt": art["files"].get("prompt.json")})
                        break
        result["candidates"] = cands
        # Held-out check: the baseline and the best candidates (by private-evaluator score) on 869 problems no
        # evaluator ever saw, in the same GPU container the research loop uses.
        all_scored = []
        for lane in ("gems", "frontier", "incubator"):
            for e in p["lanes"][lane]:
                if e["score"] is not None:
                    all_scored.append((e["score"], e))
        all_scored.sort(key=lambda x: -x[0])
        picks = [("baseline", open(os.path.join(EXP, "research/project/prompt.json")).read())]
        for sc, e in all_scored[:3]:
            for n in nodes[1:]:
                st, art = api(n, f"/agents/artifacts/{e['diff_hash']}")
                if st == 200 and art["files"].get("prompt.json"):
                    picks.append((f"top-{len(picks)} (private score {sc:.3f})", art["files"]["prompt.json"]))
                    break
        held = {}
        def score(label, text, gpu):
            f = os.path.join(work, f"held_{gpu}.json")
            open(f, "w").write(text)
            pf = os.path.join(work, f"held_{gpu}.preds.json")
            r = subprocess.run([sys.executable, os.path.join(EXP, "research/eval_prompt.py"), f, str(gpu), os.path.join(EXP, "research/data/heldout.json"), "-", pf], capture_output=True, text=True)
            held[label] = json.loads(r.stdout) if r.returncode == 0 else {"error": r.stderr[-300:]}
            if r.returncode == 0:
                held[label]["per_item"] = json.load(open(pf))
        ts = [threading.Thread(target=score, args=(lab, txt, g)) for (lab, txt), g in zip(picks, [3, 4, 5, 6])]
        [t.start() for t in ts]; [t.join() for t in ts]
        # Paired comparison against the baseline on the same 869 problems (exact McNemar test).
        from math import comb
        base = held.get("baseline", {}).get("per_item")
        for k, v in held.items():
            if base and k != "baseline" and "per_item" in v:
                b = sum(1 for x, y in zip(base, v["per_item"]) if (not x) and y)   # candidate right, baseline wrong
                c = sum(1 for x, y in zip(base, v["per_item"]) if x and not y)     # baseline right, candidate wrong
                n = b + c
                pval = min(1.0, 2 * sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n) if n else 1.0
                v["vs_baseline"] = {"candidate_only_correct": b, "baseline_only_correct": c, "mcnemar_exact_p": pval, "delta": round(v["accuracy"] - held["baseline"]["accuracy"], 4)}
        for v in held.values():
            v.pop("per_item", None)
        result["heldout"] = held
        for k, v in held.items():
            print("held-out", k, v, flush=True)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(result, open(a.out, "w"), indent=1)
        print("results written to", a.out, flush=True)
    finally:
        for n in nodes:
            n.stop()
        if not a.keep and False:
            shutil.rmtree(work, ignore_errors=True)
        print("node logs in", work)


if __name__ == "__main__":
    main()
