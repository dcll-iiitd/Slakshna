#!/usr/bin/env python3
"""Score a prompt.json on a GSM8K set inside the same GPU container the research loop uses.

usage: eval_prompt.py <prompt.json> <gpu index> <dataset.json (question/answer list)> [limit]
Prints JSON {"accuracy", "n", "ci95": [lo, hi]} (Wilson interval)."""
import base64, json, math, os, shutil, subprocess, sys, tempfile

prompt, gpu, data = sys.argv[1], sys.argv[2], sys.argv[3]
limit = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != "-" else None
preds_out = sys.argv[5] if len(sys.argv) > 5 else None
rows = json.load(open(data))[:limit]
proj = os.path.expanduser("~/slk-exp/research/project")
wt = tempfile.mkdtemp(prefix="evalp-")
for f in os.listdir(proj):
    shutil.copy(os.path.join(proj, f), wt)
shutil.copy(prompt, os.path.join(wt, "prompt.json"))
with open(os.path.join(wt, "in.jsonl"), "w") as f:
    for r in rows:
        f.write(base64.b64encode(r["question"].encode()).decode() + "\n")
os.chmod(wt, 0o777)
cmd = ["docker", "run", "--rm", "--network=none", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
       f"--user={os.getuid()}:{os.getgid()}", "--tmpfs=/tmp:rw,size=1g", "-e", "HOME=/tmp", "--workdir=/work",
       "-v", f"{wt}:/work:rw", "-v", "/home/anant/dcll/Slakshna/hf_cache:/inputs/hf:ro", "-e", "HF_HOME=/inputs/hf",
       "-e", os.environ.get("MODEL_ENV", "MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct"), f"--gpus=device={gpu}",
       "slakshna-gpu:0.1", "python", "predict.py"]
r = subprocess.run(cmd, capture_output=True, text=True)
if r.returncode != 0:
    sys.exit("container failed: " + r.stderr[-800:])
preds = [base64.b64decode(l).decode() for l in open(os.path.join(wt, "out.jsonl")) if l.strip()]
ok = sum(p == x["answer"] for p, x in zip(preds, rows))
if preds_out:
    json.dump([p == x["answer"] for p, x in zip(preds, rows)], open(preds_out, "w"))
n = len(rows); p = ok / n; z = 1.96
den = 1 + z * z / n
centre = (p + z * z / (2 * n)) / den
half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
print(json.dumps({"accuracy": p, "n": n, "correct": ok, "ci95": [round(centre - half, 4), round(centre + half, 4)]}))
shutil.rmtree(wt, ignore_errors=True)
