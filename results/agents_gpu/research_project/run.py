"""Sandbox entry point for an executor's run: validates the candidate and measures it on a small public
smoke set, on the GPU. Fails (non-zero exit) if the candidate is invalid or produces no usable answers."""
import json
import os
import sys

import predictlib

MODEL = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")

prompt = predictlib.load_prompt("prompt.json")
smoke = json.load(open("smoke.json"))
answers = predictlib.generate_answers([s["question"] for s in smoke], prompt, MODEL)
usable = sum(a != "none" for a in answers)
acc = sum(a == s["answer"] for a, s in zip(answers, smoke)) / len(smoke)
json.dump({"smoke_accuracy": acc, "usable": usable, "n": len(smoke)}, open("run_report.json", "w"))
print(f"smoke accuracy {acc:.3f} usable {usable}/{len(smoke)}", file=sys.stderr)
if usable == 0:
    sys.exit(3)
