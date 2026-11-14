"""Shared helpers for the M9 scripts (Slakshna vs Flower FedRAG evaluation).

Every system's retrieval output is a "contexts file": JSONL, one line per
question and run, with the fields made by `context_record`. The answering and
evaluation scripts read any of them the same way.
"""

import hashlib
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# The 18 books of MedRAG Textbooks, as named in huggingface.co/datasets/MedRAG/textbooks
# (the misspellings are theirs).
BOOKS = [
    "Anatomy_Gray", "Biochemistry_Lippincott", "Cell_Biology_Alberts", "First_Aid_Step1",
    "First_Aid_Step2", "Gynecology_Novak", "Histology_Ross", "Immunology_Janeway",
    "InternalMed_Harrison", "Neurology_Adams", "Obstentrics_Williams", "Pathology_Robbins",
    "Pathoma_Husain", "Pediatrics_Nelson", "Pharmacology_Katzung", "Physiology_Levy",
    "Psichiatry_DSM-5", "Surgery_Schwartz",
]

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"


def log(*parts):
    print(*parts, file=sys.stderr, flush=True)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    """Write all rows at once, through a temporary file, so a crash leaves no half file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def write_json(path, value):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def text_hash(text):
    """Identifies a snippet by its text. Flower returns text without ids, so
    snippets from different systems are matched this way."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def model_slug(model):
    return model.replace("/", "__").replace(":", "_")


def load_questions(path, qids_file=None):
    """Questions in file order; only those listed in `qids_file` (a JSON list) if given."""
    questions = read_jsonl(path)
    if qids_file:
        with open(qids_file, "r", encoding="utf-8") as f:
            wanted = set(json.load(f))
        questions = [q for q in questions if q["qid"] in wanted]
    return questions


def context_record(system, scheme, run, question, contexts, retrieval_s, extra=None):
    """One question's retrieval result.

    `contexts`: best first, each `{"text", "id", "node", "score"}`; `id` is the
    MedRAG snippet id when the system reports it, `node` the shard it came
    from when known, `score` the system's own score (its meaning differs
    between systems).
    """
    return {
        "system": system,
        "scheme": scheme,
        "run": run,
        "qid": question["qid"],
        "dataset": question["dataset"],
        "contexts": contexts,
        "retrieval_s": round(retrieval_s, 4),
        "extra": extra or {},
    }


def open_append(path):
    """Open a JSONL file for appending, first dropping a last line cut short by a crash."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if os.path.exists(path):
        with open(path, "rb+") as f:
            data = f.read()
            if data and not data.endswith(b"\n"):
                f.truncate(data.rfind(b"\n") + 1)
    return open(path, "a", encoding="utf-8")


def done_keys(path):
    """(qid, run) pairs already in an output file, so an interrupted run can resume."""
    if not os.path.exists(path):
        return set()
    keys = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a line cut short by a crash
                keys.add((row["qid"], row["run"]))
    return keys
