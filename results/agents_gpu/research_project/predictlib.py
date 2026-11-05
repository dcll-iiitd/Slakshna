"""Shared by predict.py and run.py. `prompt.json` is the only file a candidate may change, and it is
untrusted input from other nodes: it is validated strictly here before anything uses it."""
import base64
import json
import os
import re

MAX_NEW_LIMIT = 512
ALLOWED_KEYS = {"system", "few_shot", "user_template", "max_new_tokens"}


def load_prompt(path="prompt.json"):
    p = json.load(open(path))
    if not isinstance(p, dict):
        raise ValueError("prompt.json must be an object")
    extra = set(p) - ALLOWED_KEYS
    if extra:
        raise ValueError(f"unknown keys in prompt.json: {sorted(extra)}")
    system = p.get("system", "")
    if not isinstance(system, str) or len(system) > 4000:
        raise ValueError("system must be a string of at most 4000 characters")
    shots = p.get("few_shot", [])
    if not isinstance(shots, list) or len(shots) > 8:
        raise ValueError("few_shot must be a list of at most 8 examples")
    for s in shots:
        if not (isinstance(s, dict) and set(s) == {"q", "a"} and isinstance(s["q"], str) and isinstance(s["a"], str) and len(s["q"]) <= 1500 and len(s["a"]) <= 1500):
            raise ValueError("each few_shot example must be {\"q\": str, \"a\": str} within 1500 characters")
    tpl = p.get("user_template", "{question}")
    if not isinstance(tpl, str) or tpl.count("{question}") != 1 or len(tpl) > 2000:
        raise ValueError("user_template must contain {question} exactly once and be at most 2000 characters")
    mx = p.get("max_new_tokens", 400)
    if not isinstance(mx, int) or not (64 <= mx <= MAX_NEW_LIMIT):
        raise ValueError(f"max_new_tokens must be an integer in [64, {MAX_NEW_LIMIT}]")
    return {"system": system, "few_shot": shots, "user_template": tpl, "max_new_tokens": mx}


def build_messages(prompt, question):
    msgs = []
    if prompt["system"]:
        msgs.append({"role": "system", "content": prompt["system"]})
    for s in prompt["few_shot"]:
        msgs.append({"role": "user", "content": prompt["user_template"].replace("{question}", s["q"])})
        msgs.append({"role": "assistant", "content": s["a"]})
    msgs.append({"role": "user", "content": prompt["user_template"].replace("{question}", question)})
    return msgs


def normalize(n):
    s = n.replace(",", "")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def final_number(text):
    low = text.lower()
    for marker in ("answer:", "answer is", "####"):
        pos = low.rfind(marker)
        if pos >= 0:
            m = NUM.search(text[pos + len(marker):])
            if m:
                return normalize(m.group(0))
    nums = NUM.findall(text)
    return normalize(nums[-1]) if nums else "none"


def generate_answers(questions, prompt, model_id, batch_size=48):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).cuda().eval()
    texts = [tok.apply_chat_template(build_messages(prompt, q), tokenize=False, add_generation_prompt=True) for q in questions]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    answers = [None] * len(texts)
    with torch.no_grad():
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True).to("cuda")
            out = model.generate(**enc, max_new_tokens=prompt["max_new_tokens"], do_sample=False, pad_token_id=tok.eos_token_id)
            for row, i in zip(out, idx):
                answers[i] = final_number(tok.decode(row[enc["input_ids"].shape[1]:], skip_special_tokens=True))
    return answers
