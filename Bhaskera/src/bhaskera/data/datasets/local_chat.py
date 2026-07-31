"""
bhaskera.data.datasets.local_chat
=================================
Generic dataset builder for *local* files (JSONL / JSON / Parquet).

Reads from a path you specify in YAML and forwards everything else to the
existing tokenizer pipeline. The ``format`` column determines how rows are
rendered into a single text string before tokenisation.

YAML usage (tokenize CLI — needs raw paths):

    data:
      name: local
      format: chatml
      train_path: /data/customer_svc/train.jsonl
      val_path:   /data/customer_svc/val.jsonl
      seq_len:    2048
      cache_dir:  /scratch/cache

YAML usage (training — only needs the cache):

    data:
      name: local
      format: chatml
      tokenized_path:     /scratch/cache/local_train_<hash>
      val_tokenized_path: /scratch/cache/local_val_<hash>


"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import ray.data

from bhaskera.data.registry import register, register_raw
from bhaskera.data.tokenize import load_tokenized, tokenize_dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

_JSON_EXTS    = (".jsonl", ".json")
_PARQUET_EXTS = (".parquet",)


def _resolve_files(path: str) -> List[str]:
    """Expand ``path`` (file, directory, or glob) into a sorted list of files."""
    p = Path(path)

    if p.is_file():
        return [str(p)]

    if p.is_dir():
        files = []
        for ext in _JSON_EXTS + _PARQUET_EXTS:
            files.extend(sorted(p.rglob(f"*{ext}")))
        return [str(f) for f in files]

    # Treat as glob (e.g. "/data/shards/*.jsonl")
    parent = p.parent if p.parent.exists() else Path(".")
    matches = sorted(parent.glob(p.name))
    if not matches:
        raise FileNotFoundError(
            f"local dataset path '{path}' matched no files. "
            "Provide a file, a directory, or a glob pattern."
        )
    return [str(m) for m in matches]


def _read_files(files: List[str]) -> ray.data.Dataset:
    """Dispatch to ray.data.read_parquet / read_json based on extension."""
    parquet = [f for f in files if f.lower().endswith(_PARQUET_EXTS)]
    json_   = [f for f in files if f.lower().endswith(_JSON_EXTS)]

    if parquet and json_:
        raise ValueError(
            f"Mixed parquet and json files in local dataset path. "
            f"Keep them separated.  parquet={parquet[:3]} json={json_[:3]}"
        )
    if parquet:
        return ray.data.read_parquet(parquet)
    if json_:
        # ray.data.read_json handles both .json and .jsonl (line-delimited)
        return ray.data.read_json(json_)

    raise ValueError(
        f"No supported files found. Expected one of {_JSON_EXTS + _PARQUET_EXTS}."
    )


# ---------------------------------------------------------------------------
# Path resolution per split
# ---------------------------------------------------------------------------

def _path_for_split(cfg, split: Optional[str]) -> str:
    """
    Pick which YAML field to read for a given split.

    Only called on the *raw* path (tokenize CLI, or live tokenisation in
    training). When training from a pre-tokenized cache, we never get here.
    """
    data = cfg.data
    train_path = getattr(data, "train_path", None)
    val_path   = getattr(data, "val_path",   None)
    plain_path = getattr(data, "path",       None)

    if split == "val":
        if not val_path:
            raise ValueError(
                "split='val' requested but data.val_path is not set in config."
            )
        return val_path

    chosen = train_path or plain_path
    if not chosen:
        raise ValueError(
            "data.train_path (or data.path) must be set for the local dataset. "
            "Point it at a JSONL file, a directory, or a glob."
        )
    return chosen


# ---------------------------------------------------------------------------
# Registered builders
# ---------------------------------------------------------------------------

@register_raw("local", text_col="text")
def _build_raw(cfg, split: Optional[str] = None) -> ray.data.Dataset:
    """Return the raw (un-tokenized) local dataset for a given split."""
    path = _path_for_split(cfg, split)
    files = _resolve_files(path)

    if not files:
        raise FileNotFoundError(f"No files found for path '{path}'.")

    logger.info(
        f"local dataset: split={split!r} path={path!r} "
        f"-> {len(files)} file(s)"
    )
    return _read_files(files)


@register("local")
def build(cfg, world_size: int = 1) -> ray.data.Dataset:
    """
    Build the tokenized local dataset for the trainer.

    Two paths:
      * cfg.data.tokenized_path set  → load pre-tokenized parquet from disk
                                       (the normal training case). We never
                                       touch train_path here.
      * cfg.data.tokenized_path None → live tokenisation from train_path
                                       (slow; usually you want the
                                       bhaskera-tokenize CLI instead).
    """
    
    if cfg.data.tokenized_path:
        logger.info(
            f"local dataset: loading pre-tokenized cache "
            f"{cfg.data.tokenized_path!r} (world_size={world_size})"
        )
        return load_tokenized(cfg.data.tokenized_path, cfg, world_size)

    # ── Live tokenisation path: needs raw inputs. ────────────────────────
    return tokenize_dataset(_build_raw(cfg), cfg, "text", world_size=world_size)




def build_val_ray_dataset(cfg, world_size: int = 1) -> Optional[ray.data.Dataset]:
    """
    Return the tokenized validation dataset, or None if no val cache is
    configured. Mirrors build() but reads cfg.data.val_tokenized_path.
    """
    val_path = getattr(cfg.data, "val_tokenized_path", None)
    if not val_path:
        logger.info("No data.val_tokenized_path configured — skipping val dataset.")
        return None

    logger.info(
        f"local dataset (val): loading pre-tokenized cache "
        f"{val_path!r} (world_size={world_size})"
    )
    return load_tokenized(val_path, cfg, world_size)
