"""Sandbox entry point for evaluation: /work/in.jsonl (base64 questions) -> /work/out.jsonl (base64 numbers)."""
import base64
import os
import sys

import predictlib

MODEL = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")

prompt = predictlib.load_prompt("prompt.json")
questions = [base64.b64decode(line).decode() for line in open("in.jsonl") if line.strip()]
answers = predictlib.generate_answers(questions, prompt, MODEL)
with open("out.jsonl", "w") as f:
    for a in answers:
        f.write(base64.b64encode(a.encode()).decode() + "\n")
print(f"predicted {len(answers)} answers", file=sys.stderr)
