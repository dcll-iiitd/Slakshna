"""Answer every question of a contexts file with one LLM, the same way for every system.

    # Ollama (default backend), several instances or parallel slots
    python3 scripts/m9/answer.py --data datasets/textbooks --contexts results/m9/raw/book/flower.jsonl \
        --model qwen2.5:7b-instruct --url http://127.0.0.1:11500 --workers 4
    python3 scripts/m9/answer.py --data datasets/textbooks --closed-book --out results/m9/raw/closed_book.pred.jsonl ...
    # speed test before the long run (no cache, nothing written)
    python3 scripts/m9/answer.py --data datasets/textbooks --contexts results/m9/raw/global.jsonl --calibrate --limit 20
    # check E0(a): Flower's own LLMQuerier (run with .venv-flower/bin/python)
    .venv-flower/bin/python scripts/m9/answer.py ... --backend hf --model HuggingFaceTB/SmolLM2-1.7B-Instruct
    # prove the prompt copy below is byte-identical to Flower's, on every question
    .venv-flower/bin/python scripts/m9/answer.py --data datasets/textbooks --contexts ... --check-prompt flower-fedrag-m9

The prompt and the answer rule are Flower FedRAG's (examples/fedrag/fedrag/
llm_querier.py, Apache-2.0, copyright Flower Labs), copied so both systems'
contexts are judged identically: the first standalone letter A-D in the
model's reply, or no answer. Ollama runs with temperature 0, seed 0,
num_predict 10 (Flower's max_new_tokens) and num_ctx 8192, so an 8-document
prompt is never cut. A reply whose prompt filled the window is
marked `truncated`.

Answers are cached by (backend, model, settings, prompt), so identical
contexts from two systems cost one LLM call, and an interrupted run resumes.
Only run 1 of a contexts file is answered unless --runs says otherwise
(repeats exist for latency, not accuracy). Ollama mode is standard library only.
"""

import argparse
import hashlib
import itertools
import json
import os
import re
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from m9_common import ROOT, done_keys, load_questions, log, model_slug, open_append, read_jsonl

OLLAMA_OPTIONS = {"temperature": 0, "seed": 0, "num_predict": 10, "num_ctx": 8192}
ANSWER_RE = re.compile(r"\b([A-Da-d])\b")


def format_prompt(question, documents, options, dataset_name):
    """Flower's LLMQuerier.__format_prompt, character for character (see --check-prompt)."""
    instruction = ("You are a helpful medical expert, and your task is to answer a medical question "
                   "using the relevant documents.")
    if dataset_name == "pubmedqa":
        instruction = ("As an expert doctor in clinical science and medical knowledge, can you tell me "
                       "if the following statement is correct? Answer yes, no, or maybe.")
    elif dataset_name == "bioasq":
        instruction = ("You are an advanced biomedical AI assistant trained to understand and process "
                       "medical and scientific texts. Given a biomedical question, your goal is to provide "
                       "a concise and accurate answer based on relevant scientific literature.")
    ctx_documents = "\n".join(f"Document {i + 1}: {doc}" for i, doc in enumerate(documents))
    indent = " " * 12
    return (f"{instruction}\n\n"
            f"{indent}Here are the relevant documents:\n{indent}{ctx_documents}\n\n"
            f"{indent}Question: \n{indent}{question}\n\n"
            f"{indent}Options:\n{indent}{options}\n\n"
            f"{indent}Answer only with the correct option: ")


def format_options(options):
    return "\n".join(f"{k}) {v}" for k, v in options.items())


def extract_answer(response):
    match = ANSWER_RE.search(response.strip())
    return match.group(1).upper() if match else None


class Cache:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.entries = {}
        if os.path.exists(path):
            for row in read_jsonl(path):
                self.entries[row["key"]] = row
        self.file = open_append(path)

    def get(self, key):
        return self.entries.get(key)

    def put(self, row):
        with self.lock:
            self.entries[row["key"]] = row
            self.file.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.file.flush()


class OllamaBackend:
    name = "ollama"

    def __init__(self, model, urls, timeout):
        self.model, self.timeout = model, timeout
        self.urls = itertools.cycle(urls)
        self.url_lock = threading.Lock()
        self.settings = dict(OLLAMA_OPTIONS)

    def answer(self, prompt, question):
        with self.url_lock:
            url = next(self.urls)
        body = {"model": self.model, "stream": False, "keep_alive": "24h", "options": self.settings,
                "messages": [{"role": "user", "content": prompt}]}
        request = urllib.request.Request(f"{url.rstrip('/')}/api/chat", data=json.dumps(body).encode(),
                                         headers={"content-type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            reply = json.load(response)
        text = reply["message"]["content"]
        tokens = reply.get("prompt_eval_count")
        return {"response": text, "predicted": extract_answer(text), "prompt_tokens": tokens,
                "truncated": bool(tokens and tokens >= self.settings["num_ctx"] - self.settings["num_predict"])}


class HfBackend:
    """Flower's own LLMQuerier, unchanged (check E0(a)). Needs .venv-flower."""

    name = "hf"

    def __init__(self, model, flower_app):
        sys.path.insert(0, os.path.abspath(flower_app))
        from fedrag.llm_querier import LLMQuerier

        self.querier = LLMQuerier(model, use_gpu=False)
        self.settings = {"max_new_tokens": 10, "use_gpu": False}

    def answer(self, prompt, question):
        # LLMQuerier builds the same prompt itself from these arguments.
        _, predicted = self.querier.answer(question["question"], question["_documents"], question["options"],
                                           question["dataset"])
        return {"response": None, "predicted": predicted, "prompt_tokens": None, "truncated": False}


def cache_key(backend, model, prompt):
    material = json.dumps([backend.name, model, backend.settings, prompt], ensure_ascii=False)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def load_tasks(args, questions):
    """(question, documents, system, scheme, run) for every row to answer."""
    by_qid = {q["qid"]: q for q in questions}
    if args.closed_book:
        return [(q, [], "closed_book", None, 1) for q in questions]
    runs = {int(r) for r in args.runs.split(",")}
    tasks = []
    for row in read_jsonl(args.contexts):
        if row["run"] in runs and row["qid"] in by_qid:
            tasks.append((by_qid[row["qid"]], [c["text"] for c in row["contexts"]],
                          row["system"], row["scheme"], row["run"]))
    return tasks


def check_prompt(tasks, flower_app):
    sys.path.insert(0, os.path.abspath(flower_app))
    from fedrag.llm_querier import LLMQuerier

    flower_format = LLMQuerier._LLMQuerier__format_prompt
    for question, documents, *_ in tasks:
        options = format_options(question["options"])
        ours = format_prompt(question["question"], documents, options, question["dataset"])
        theirs = flower_format(question["question"], documents, options, question["dataset"])
        if ours != theirs:
            sys.exit(f"prompt differs from Flower's for {question['qid']}")
    for response, expected in [("A) yes", "A"), ("  b", "B"), ("The correct option is C", "C"),
                               ("none", None), ("is a", "A")]:
        generated = "PROMPT" + response
        theirs = LLMQuerier._LLMQuerier__extract_answer(generated, "PROMPT")
        if extract_answer(response) != theirs or theirs != expected:
            sys.exit(f"answer extraction differs from Flower's on {response!r}")
    log(f"prompt identical to Flower's on all {len(tasks)} rows; answer extraction identical")


def answer_all(tasks, backend, model, cache, workers, out, on_done=None):
    def one(task):
        question, documents, system, scheme, run = task
        prompt = format_prompt(question["question"], documents, format_options(question["options"]),
                               question["dataset"])
        key = cache_key(backend, model, prompt)
        hit = cache.get(key) if cache else None
        started = time.time()
        if hit is None:
            question = dict(question, _documents=documents)
            result = backend.answer(prompt, question)
            result.update(key=key, llm_s=round(time.time() - started, 3))
            if cache:
                cache.put(result)
        else:
            result = hit
        row = {"system": system, "scheme": scheme, "run": run, "qid": question["qid"],
               "dataset": question["dataset"], "backend": backend.name, "model": model,
               "predicted": result["predicted"], "response": result["response"],
               "prompt_tokens": result["prompt_tokens"], "truncated": result["truncated"],
               "llm_s": result["llm_s"], "cached": hit is not None}
        if out:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
        if on_done:
            on_done(row)
        return row

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, tasks))


def calibrate(args, tasks, backend):
    """Time each worker count on its own questions.

    Ollama keeps recent prompts cached, so re-sending the same questions would
    look far faster than real. One untimed request first loads the model.
    """
    levels = [int(w) for w in args.calibrate_workers.split(",")]
    per_level = args.limit or 20
    if len(tasks) < per_level * len(levels) + 1:
        sys.exit(f"calibration needs {per_level * len(levels) + 1} rows ({per_level} per worker count); "
                 f"the file has {len(tasks)}")
    log(f"loading the model, then {per_level} different questions per worker count, no cache")
    answer_all(tasks[:1], backend, args.model, None, 1, None)
    for n, workers in enumerate(levels):
        chunk = tasks[1 + n * per_level:1 + (n + 1) * per_level]
        started = time.time()
        rows = answer_all(chunk, backend, args.model, None, workers, None)
        per_question = (time.time() - started) / len(rows)
        unanswered = sum(r["predicted"] is None for r in rows)
        truncated = sum(r["truncated"] for r in rows)
        log(f"workers={workers}: {per_question:.2f} s per question "
            f"({args.project * per_question / 3600:.1f} h per {args.project} questions); "
            f"{unanswered} unanswered, {truncated} truncated")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="dataset folder (questions.jsonl)")
    p.add_argument("--contexts", help="contexts file to answer")
    p.add_argument("--closed-book", action="store_true", help="answer with no documents")
    p.add_argument("--runs", default="1", help="comma-separated runs of the contexts file to answer")
    p.add_argument("--backend", choices=["ollama", "hf"], default="ollama")
    p.add_argument("--model", default="qwen2.5:7b-instruct")
    p.add_argument("--url", action="append", help="Ollama URL; repeat to spread work over instances")
    p.add_argument("--workers", type=int, default=4, help="requests in flight at once")
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--flower-app", default=os.path.join(ROOT, "flower-fedrag-m9"))
    p.add_argument("--out", help="default: <contexts>.pred.<model>.jsonl")
    p.add_argument("--cache", help="default: results/m9/raw/answer_cache/<backend>__<model>.jsonl")
    p.add_argument("--limit", type=int, default=0, help="only the first N rows (calibration: rows per worker count)")
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--calibrate-workers", default="1,2,4")
    p.add_argument("--project", type=int, default=2362, help="questions to project the calibration to")
    p.add_argument("--check-prompt", metavar="FLOWER_APP", help="compare the prompt copy with Flower's and exit")
    args = p.parse_args()
    if bool(args.contexts) == args.closed_book:
        p.error("give exactly one of --contexts or --closed-book")

    questions = load_questions(os.path.join(args.data, "questions.jsonl"))
    tasks = load_tasks(args, questions)
    if args.limit and not args.calibrate:
        tasks = tasks[:args.limit]
    if args.check_prompt:
        check_prompt(tasks, args.check_prompt)
        return

    if args.backend == "ollama":
        backend = OllamaBackend(args.model, args.url or ["http://127.0.0.1:11434"], args.timeout)
    else:
        backend = HfBackend(args.model, args.flower_app)
        args.workers = 1
    if args.calibrate:
        calibrate(args, tasks, backend)
        return

    out_path = args.out or (f"{os.path.splitext(args.contexts)[0]}.pred.{model_slug(args.model)}.jsonl"
                            if args.contexts else None)
    if not out_path:
        p.error("--closed-book needs --out")
    cache_path = args.cache or os.path.join(ROOT, "results", "m9", "raw", "answer_cache",
                                            f"{backend.name}__{model_slug(args.model)}.jsonl")
    done = done_keys(out_path)
    todo = [t for t in tasks if (t[0]["qid"], t[4]) not in done]
    log(f"{len(tasks)} rows, {len(tasks) - len(todo)} already answered in {out_path}")
    progress = {"n": 0, "started": time.time()}
    lock = threading.Lock()

    def report(row):
        with lock:
            progress["n"] += 1
            n = progress["n"]
            if n % 50 == 0 or n == len(todo):
                rate = (time.time() - progress["started"]) / n
                log(f"{n}/{len(todo)} answered, {rate:.2f} s each, about {(len(todo) - n) * rate / 60:.0f} min left")

    cache = Cache(cache_path)
    with open_append(out_path) as out:
        rows = answer_all(todo, backend, args.model, cache, args.workers, out, report)
    truncated = sum(r["truncated"] for r in rows)
    if truncated:
        log(f"WARNING: {truncated} prompts filled the {OLLAMA_OPTIONS['num_ctx']}-token window and may be cut")
    log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
