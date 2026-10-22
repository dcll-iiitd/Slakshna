"""
State of the tokenized training dataset.

Tokenization is a separate step that runs once, before training
(``scripts/tokenize_dataset.py``). ``ml_engine.py`` only consumes the result, and
uses this module to refuse a cache that is missing, truncated, or built with a
different tokenizer: token ids from another model train without any error and
silently produce garbage.

A cache is stamped with ``tokenizer_model.txt`` (the ``model.name`` it was built
for) when the tokenize step finishes.
"""
from __future__ import annotations

import os

MARKER = "tokenizer_model.txt"
_MIN_PARQUET_BYTES = 100  # a valid parquet file is larger than its 8 magic bytes

READY = "ready"
MISSING = "missing"
CORRUPTED = "corrupted"
UNMARKED = "unmarked"
WRONG_MODEL = "wrong_model"


def marker_path(path: str) -> str:
    return os.path.join(path, MARKER)


def cache_status(path: str, model_name: str) -> str:
    """One of READY, MISSING, CORRUPTED, UNMARKED, WRONG_MODEL."""
    parquets = []
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            parquets += [os.path.join(root, f) for f in files if f.endswith(".parquet")]
    if not parquets:
        return MISSING
    if any(os.path.getsize(p) < _MIN_PARQUET_BYTES for p in parquets):
        return CORRUPTED
    marker = marker_path(path)
    if not os.path.isfile(marker):
        return UNMARKED
    with open(marker, encoding="utf-8") as f:
        return READY if f.read().strip() == model_name else WRONG_MODEL


def built_for(path: str) -> str | None:
    try:
        with open(marker_path(path), encoding="utf-8") as f:
            return f.read().strip() or None
    except OSError:
        return None


def write_marker(path: str, model_name: str) -> None:
    os.makedirs(path, exist_ok=True)
    with open(marker_path(path), "w", encoding="utf-8") as f:
        f.write(model_name)


def explain(status: str, path: str, model_name: str) -> str:
    """Why training cannot start, and what to run."""
    fix = "python scripts/tokenize_dataset.py"
    if status == MISSING:
        return f"No tokenized dataset at {path}. Tokenize it once before training: {fix}"
    if status == CORRUPTED:
        return (f"The tokenized dataset at {path} has truncated parquet files (an aborted tokenize run?). "
                f"Re-run: {fix} --force  (the old directory is moved aside, not deleted)")
    if status == UNMARKED:
        return (f"The tokenized dataset at {path} has no {MARKER}, so it cannot be verified against "
                f"model.name = {model_name}. If it was built for that model, stamp it with "
                f"`python scripts/tokenize_dataset.py --stamp`; otherwise re-tokenize: {fix} --force")
    if status == WRONG_MODEL:
        return (f"The tokenized dataset at {path} was built for {built_for(path)}, but model.name is "
                f"{model_name}. Training on it would silently use the wrong token ids. "
                f"Re-tokenize: {fix} --force  (the old directory is moved aside, not deleted)")
    return "ok"
