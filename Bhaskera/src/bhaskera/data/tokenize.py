"""
bhaskera.data.tokenize
======================
Stateful Ray-Data tokeniser with persistent caching, CPT continuous packing,
and SFT Multipack (First-Fit Decreasing) support.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import ray.data
import torch.distributed as dist

logger = logging.getLogger(__name__)

_BHASKERA_VERSION = "2.3.0"

def _cache_version_hash(
    model_name: str,
    seq_len: int,
    dataset_name: str,
    format_name: Optional[str] = None,
    format_options: Optional[dict] = None,
    is_cpt: bool = False,
    pack_sequences: bool = False,
    train_on_inputs: bool = False,
) -> str:
    parts = [model_name, str(seq_len), dataset_name]
    if format_name:
        parts.append(f"fmt:{format_name}")
    if format_options:
        parts.append(f"opts:{json.dumps(format_options, sort_keys=True, default=str)}")
    if is_cpt:
        parts.append("cpt:true")
    if pack_sequences:
        parts.append("pack:true")
    if train_on_inputs:
        parts.append("toi:true")

    key = "|".join(parts)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _write_metadata(
    cache_path: str,
    model_name: str,
    seq_len: int,
    dataset_name: str,
    num_rows: int,
    format_name: Optional[str] = None,
    format_options: Optional[dict] = None,
    is_cpt: bool = False,
    pack_sequences: bool = False,
    train_on_inputs: bool = False,
) -> None:
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return

    meta = {
        "model_name":       model_name,
        "seq_len":          seq_len,
        "dataset_name":     dataset_name,
        "num_rows":         num_rows,
        # Updated schema for block-diagonal masking and RoPE alignment
        "schema":           ["input_ids", "attention_mask", "labels", "position_ids", "seq_idx"],
        "created_at":       datetime.datetime.utcnow().isoformat() + "Z",
        "bhaskera_version": _BHASKERA_VERSION,
        "format_name":      format_name,
        "format_options":   format_options or {},
        "is_cpt":           is_cpt,
        "pack_sequences":   pack_sequences,
        "train_on_inputs":  train_on_inputs,
    }
    meta_path = os.path.join(cache_path, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info(f"Tokenizer cache metadata written -> {meta_path}")


def _verify_cache(
    cache_path: str,
    model_name: str,
    seq_len: int,
    dataset_name: str,
    format_name: Optional[str] = None,
    is_cpt: bool = False,
    pack_sequences: bool = False,
    train_on_inputs: bool = False,
) -> bool:
    meta_path = os.path.join(cache_path, "metadata.json")
    if not os.path.isfile(meta_path):
        return False

    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False

    for key, expected in (
        ("model_name",   model_name),
        ("seq_len",      seq_len),
        ("dataset_name", dataset_name),
    ):
        if meta.get(key) != expected:
            return False

    if meta.get("is_cpt", False) != is_cpt:
        return False

    if meta.get("pack_sequences", False) != pack_sequences:
        return False

    if meta.get("train_on_inputs", False) != train_on_inputs:
        return False

    if format_name is not None and meta.get("format_name") != format_name:
        return False

    parquet_files = list(Path(cache_path).glob("*.parquet"))
    if not parquet_files:
        return False

    return True


def _compute_num_partitions(cfg, world_size: int) -> int:
    base = max(world_size * 4, cfg.data.num_workers * 4, 16)
    return ((base + world_size - 1) // world_size) * world_size


def persist_tokenized(
    ds: ray.data.Dataset,
    cfg,
    text_col: str,
    dataset_name: str,
) -> str:
    if not cfg.data.cache_dir:
        raise ValueError("cfg.data.cache_dir must be set to use persist_tokenized().")

    model_name     = cfg.model.name
    seq_len        = cfg.data.seq_len
    format_name    = getattr(cfg.data, "format", None)
    format_options = dict(getattr(cfg.data, "format_options", None) or {})
    is_cpt         = getattr(cfg.data, "is_cpt", False)
    pack_sequences = getattr(cfg.data, "pack_sequences", False)

    train_on_inputs = getattr(cfg.data, "train_on_inputs", None)
    if train_on_inputs is None:
        train_on_inputs = is_cpt

    version = _cache_version_hash(
        model_name, seq_len, dataset_name, format_name, format_options, is_cpt, pack_sequences, train_on_inputs
    )
    cache_path = os.path.join(cfg.data.cache_dir, f"{dataset_name}_{version}")

    if (_verify_cache(cache_path, model_name, seq_len, dataset_name, format_name, is_cpt, pack_sequences, train_on_inputs)
            and not cfg.data.overwrite_cache):
        logger.info(f"Tokenizer cache hit -> {cache_path}")
        return cache_path

    if cfg.data.overwrite_cache and os.path.exists(cache_path):
        import shutil
        shutil.rmtree(cache_path)

    tokenized_ds = _apply_map_batches(ds, cfg, text_col)

    Path(cache_path).mkdir(parents=True, exist_ok=True)
    tokenized_ds.write_parquet(
        cache_path,
        compression=cfg.data.tokenize_compression,
        num_rows_per_file=50_000,
    )

    try:
        num_rows = tokenized_ds.count()
    except Exception:
        num_rows = -1

    _write_metadata(
        cache_path, model_name, seq_len, dataset_name, num_rows,
        format_name, format_options, is_cpt, pack_sequences, train_on_inputs
    )

    logger.info(f"Tokenization complete -> {cache_path}")
    return cache_path


def load_tokenized(tokenized_path: str, cfg, world_size: int) -> ray.data.Dataset:
    if not os.path.isdir(tokenized_path):
        raise FileNotFoundError(f"tokenized_path='{tokenized_path}' does not exist.")

    parquet_files = list(Path(tokenized_path).glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in cache.")

    ds = ray.data.read_parquet(tokenized_path)
    num_partitions = _compute_num_partitions(cfg, world_size)
    ds = ds.repartition(num_partitions)
    return ds


def tokenize_dataset(
    ds: ray.data.Dataset,
    cfg,
    text_col: str,
    world_size: int = 1,
) -> ray.data.Dataset:
    if cfg.data.tokenized_path:
        return load_tokenized(cfg.data.tokenized_path, cfg, world_size)
    return _apply_map_batches(ds, cfg, text_col)


class TokenizerActor:
    def __init__(
        self,
        model_name: str,
        seq_len: int,
        text_col: str = "text",
        trust_remote_code: bool = False,
        format_name: Optional[str] = None,
        format_options: Optional[dict] = None,
        is_cpt: bool = False,
        pack_sequences: bool = False,
        train_on_inputs: bool = False,
    ):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.seq_len = seq_len
        self.text_col = text_col
        self.format_name = format_name
        self.format_options = format_options or {}

        self.is_cpt = is_cpt
        self.pack_sequences = pack_sequences
        self.train_on_inputs = train_on_inputs

        # State kept exclusively for continuous pre-training (cross-boundary padding)
        self._remainder_ids: list[int] = []
        self._remainder_labels: list[int] = []

        if self.format_name:
            from bhaskera.data.formats import _ensure_builtins_loaded
            _ensure_builtins_loaded()

    def _render_batch(self, batch: dict) -> list[Any]:
        from bhaskera.data.formats import render_with_format
        any_col = next(iter(batch.values()))
        n = len(any_col)
        items = []
        for i in range(n):
            row = {k: v[i] for k, v in batch.items()}
            items.append(render_with_format(self.format_name, row, self.tokenizer, self.format_options))
        return items

    def __call__(self, batch: dict) -> dict:
        if self.format_name:
            texts_or_msgs = self._render_batch(batch)
        else:
            texts_or_msgs = batch[self.text_col]
            if hasattr(texts_or_msgs, "tolist"):
                texts_or_msgs = texts_or_msgs.tolist()

        # Ray Data numpy conversion can yield arrays of structs instead of lists of dicts
        _clean = []
        for x in texts_or_msgs:
            if hasattr(x, "tolist"):
                x = x.tolist()
            if isinstance(x, list) and len(x) > 0 and not isinstance(x[0], dict):
                try:
                    x = [dict(i) if not isinstance(i, dict) else i for i in x]
                except Exception:
                    pass
            _clean.append(x)
        texts_or_msgs = _clean

        has_template = hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None)
        from bhaskera.data.formats.builtins import _manual_chatml_tokenize, _apply_chat_template_safe

        # -- Continuous Packing / CPT Mode (Cross-boundary truncation) ------------------------
        if self.is_cpt:
            stream_ids = self._remainder_ids
            stream_labels = self._remainder_labels
            eos = self.tokenizer.eos_token_id

            for item in texts_or_msgs:
                if not item:
                    continue
                if isinstance(item, list):
                    if self.train_on_inputs:
                        text = _apply_chat_template_safe(self.tokenizer, item)
                        ids = self.tokenizer.encode(text, add_special_tokens=False)
                        lbls = list(ids)
                    else:
                        if has_template:
                            try:
                                enc = self.tokenizer.apply_chat_template(
                                    item, tokenize=True, return_dict=True, return_assistant_tokens_mask=True
                                )
                                ids = enc["input_ids"]
                                if "assistant_masks" in enc:
                                    lbls = [t if m else -100 for t, m in zip(ids, enc["assistant_masks"])]
                                else:
                                    ids, lbls = _manual_chatml_tokenize(self.tokenizer, item)
                            except Exception:
                                ids, lbls = _manual_chatml_tokenize(self.tokenizer, item)
                        else:
                            ids, lbls = _manual_chatml_tokenize(self.tokenizer, item)
                else:
                    text = str(item)
                    ids = self.tokenizer.encode(text, add_special_tokens=False)
                    lbls = list(ids) if self.train_on_inputs else [-100]*len(ids)

                if not ids:
                    continue

                stream_ids.extend(ids)
                stream_labels.extend(lbls)

                if eos is not None:
                    stream_ids.append(eos)
                    stream_labels.append(eos if self.train_on_inputs else -100)

            n_chunks = len(stream_ids) // self.seq_len
            valid_len = n_chunks * self.seq_len

            self._remainder_ids = stream_ids[valid_len:]
            self._remainder_labels = stream_labels[valid_len:]

            if n_chunks == 0:
                dummy = np.zeros((1, self.seq_len), dtype=np.int32)
                return {
                    "input_ids": dummy,
                    "attention_mask": np.zeros((1, self.seq_len), dtype=np.int32),
                    "labels": np.full((1, self.seq_len), -100, dtype=np.int32),
                    "position_ids": dummy,
                    "seq_idx": dummy,
                }

            reshaped_ids = np.array(stream_ids[:valid_len], dtype=np.int32).reshape(n_chunks, self.seq_len)
            reshaped_lbls = np.array(stream_labels[:valid_len], dtype=np.int32).reshape(n_chunks, self.seq_len)

            # CPT is treated as one massive continuous timeline.
            reshaped_pos_ids = np.tile(np.arange(self.seq_len, dtype=np.int32), (n_chunks, 1))
            reshaped_seq_idx = np.ones((n_chunks, self.seq_len), dtype=np.int32)

            return {
                "input_ids": reshaped_ids,
                "attention_mask": np.ones_like(reshaped_ids),
                "labels": reshaped_lbls,
                "position_ids": reshaped_pos_ids,
                "seq_idx": reshaped_seq_idx,
            }

        # -- SFT Multipack Mode (FFD, Boundaries Preserved) -----------------------------------
        elif self.pack_sequences:
            tokenized_items = []

            for item in texts_or_msgs:
                if not item: continue
                if isinstance(item, list):
                    if self.train_on_inputs:
                        # _apply_chat_template_safe renders the STRING form
                        # of the chat template, which for many families
                        # (Llama-style templates especially) already
                        # contains the literal BOS token text. Re-encoding
                        # with add_special_tokens=True would then prepend a
                        # second BOS, so this must be False here (mirrors
                        # the CPT branch above, which already gets this
                        # right).
                        text = _apply_chat_template_safe(self.tokenizer, item)
                        ids = self.tokenizer.encode(text, add_special_tokens=False)
                        lbls = list(ids)
                    else:
                        if has_template:
                            try:
                                enc = self.tokenizer.apply_chat_template(
                                    item, tokenize=True, return_dict=True, return_assistant_tokens_mask=True
                                )
                                ids = enc["input_ids"]
                                if "assistant_masks" in enc:
                                    lbls = [t if m else -100 for t, m in zip(ids, enc["assistant_masks"])]
                                else:
                                    ids, lbls = _manual_chatml_tokenize(self.tokenizer, item)
                            except Exception:
                                ids, lbls = _manual_chatml_tokenize(self.tokenizer, item)
                        else:
                            ids, lbls = _manual_chatml_tokenize(self.tokenizer, item)
                else:
                    text = str(item)
                    ids = self.tokenizer.encode(text, add_special_tokens=True)
                    lbls = list(ids) if self.train_on_inputs else [-100]*len(ids)

                if not ids: continue

                # Strict Truncation per-sequence to guarantee it fits inside a single bucket
                if len(ids) > self.seq_len:
                    ids = ids[:self.seq_len]
                    lbls = lbls[:self.seq_len]

                tokenized_items.append((ids, lbls))

            # First-Fit Decreasing Bin Packing
            tokenized_items.sort(key=lambda x: len(x[0]), reverse=True)

            bins_ids, bins_labels, bins_pos_ids, bins_seq_idx = [], [], [], []

            for ids, lbls in tokenized_items:
                seq_len_current = len(ids)
                pos_ids = list(range(seq_len_current))  # Reset position_ids to 0 for every sample

                placed = False
                for b_idx in range(len(bins_ids)):
                    if len(bins_ids[b_idx]) + seq_len_current <= self.seq_len:
                        bins_ids[b_idx].extend(ids)
                        bins_labels[b_idx].extend(lbls)
                        bins_pos_ids[b_idx].extend(pos_ids)

                        # Increment document index so collator can block-diagonal mask
                        doc_id = bins_seq_idx[b_idx][-1] + 1 if bins_seq_idx[b_idx] else 1
                        bins_seq_idx[b_idx].extend([doc_id] * seq_len_current)

                        placed = True
                        break

                if not placed:
                    bins_ids.append(list(ids))
                    bins_labels.append(list(lbls))
                    bins_pos_ids.append(list(pos_ids))
                    bins_seq_idx.append([1] * seq_len_current)

            batch_input_ids, batch_attention_mask, batch_labels = [], [], []
            batch_position_ids, batch_seq_idx = [], []

            pad_tok = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

            for b_ids, b_lbls, b_pos, b_seq in zip(bins_ids, bins_labels, bins_pos_ids, bins_seq_idx):
                pad_len = self.seq_len - len(b_ids)

                att_mask = [1] * len(b_ids) + [0] * pad_len
                b_ids = b_ids + [pad_tok] * pad_len
                b_lbls = b_lbls + [-100] * pad_len
                b_pos = b_pos + [0] * pad_len
                b_seq = b_seq + [0] * pad_len

                batch_input_ids.append(b_ids)
                batch_attention_mask.append(att_mask) # 1D valid vs pad mask
                batch_labels.append(b_lbls)
                batch_position_ids.append(b_pos)
                batch_seq_idx.append(b_seq)

            if len(batch_input_ids) == 0:
                dummy = np.zeros((1, self.seq_len), dtype=np.int32)
                return {
                    "input_ids": dummy, "attention_mask": dummy, "labels": np.full_like(dummy, -100),
                    "position_ids": dummy, "seq_idx": dummy
                }

            return {
                "input_ids": np.array(batch_input_ids, dtype=np.int32),
                "attention_mask": np.array(batch_attention_mask, dtype=np.int32),
                "labels": np.array(batch_labels, dtype=np.int32),
                "position_ids": np.array(batch_position_ids, dtype=np.int32),
                "seq_idx": np.array(batch_seq_idx, dtype=np.int32),
            }

        # -- Standard SFT Truncation / Padding Path -------------------------------------------
        batch_input_ids, batch_attention_mask, batch_labels = [], [], []
        batch_position_ids, batch_seq_idx = [], []

        for item in texts_or_msgs:
            if not item:
                continue
            if isinstance(item, list):
                if self.train_on_inputs:
                    # See the matching comment in the multipack branch above:
                    # the rendered template string can already embed a
                    # literal BOS, so add_special_tokens must be False here
                    # to avoid a duplicate leading BOS token.
                    text = _apply_chat_template_safe(self.tokenizer, item)
                    ids = self.tokenizer.encode(text, add_special_tokens=False)
                    lbls = list(ids)
                else:
                    if has_template:
                        try:
                            enc = self.tokenizer.apply_chat_template(
                                item, tokenize=True, return_dict=True, return_assistant_tokens_mask=True
                            )
                            ids = enc["input_ids"]
                            if "assistant_masks" in enc and any(enc["assistant_masks"]):
                                lbls = [t if m else -100 for t, m in zip(ids, enc["assistant_masks"])]
                            else:
                                ids, lbls = _manual_chatml_tokenize(self.tokenizer, item)
                        except Exception:
                            ids, lbls = _manual_chatml_tokenize(self.tokenizer, item)
                    else:
                        ids, lbls = _manual_chatml_tokenize(self.tokenizer, item)
            else:
                text = str(item)
                ids = self.tokenizer.encode(text, add_special_tokens=True)
                lbls = list(ids) if self.train_on_inputs else [-100]*len(ids)

            if not ids: continue

            # Determine whether this conversation has any trainable
            # (non -100) label *before* truncating to seq_len. A
            # conversation that has real assistant content beyond the
            # truncation boundary is still legitimate data -- it just
            # loses its label supervision for this particular window --
            # and should be kept as a real (if fully-masked) row, not
            # silently dropped as if it never had an assistant turn at
            # all (that case, e.g. test_conversation_without_assistant,
            # is genuinely degenerate and should still be filtered out).
            has_label = any(l != -100 for l in lbls)

            if len(ids) > self.seq_len:
                ids = ids[:self.seq_len]
                lbls = lbls[:self.seq_len]

            pad_len = self.seq_len - len(ids)
            pos_ids = list(range(len(ids)))
            seq_idx = [1] * len(ids)

            if pad_len > 0:
                pad_tok = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                ids = ids + [pad_tok] * pad_len
                lbls = lbls + [-100] * pad_len
                att_mask = [1] * (self.seq_len - pad_len) + [0] * pad_len
                pos_ids = pos_ids + [0] * pad_len
                seq_idx = seq_idx + [0] * pad_len
            else:
                att_mask = [1] * self.seq_len

            if has_label:
                batch_input_ids.append(ids)
                batch_attention_mask.append(att_mask)
                batch_labels.append(lbls)
                batch_position_ids.append(pos_ids)
                batch_seq_idx.append(seq_idx)

        if len(batch_input_ids) == 0:
            dummy = np.zeros((1, self.seq_len), dtype=np.int32)
            return {
                "input_ids": dummy, "attention_mask": dummy, "labels": np.full_like(dummy, -100),
                "position_ids": dummy, "seq_idx": dummy
            }

        return {
            "input_ids": np.array(batch_input_ids, dtype=np.int32),
            "attention_mask": np.array(batch_attention_mask, dtype=np.int32),
            "labels": np.array(batch_labels, dtype=np.int32),
            "position_ids": np.array(batch_position_ids, dtype=np.int32),
            "seq_idx": np.array(batch_seq_idx, dtype=np.int32),
        }


class _TokenizerActorFactory:
    def __init__(
        self,
        model_name: str,
        seq_len: int,
        text_col: str,
        trust_remote_code: bool = False,
        format_name: Optional[str] = None,
        format_options: Optional[dict] = None,
        is_cpt: bool = False,
        pack_sequences: bool = False,
        train_on_inputs: bool = False,
    ):
        self.model_name = model_name
        self.seq_len = seq_len
        self.text_col = text_col
        self.trust_remote_code = trust_remote_code
        self.format_name = format_name
        self.format_options = format_options
        self.is_cpt = is_cpt
        self.pack_sequences = pack_sequences
        self.train_on_inputs = train_on_inputs
        self._actor: Optional[TokenizerActor] = None

    def __call__(self, batch: dict) -> dict:
        if self._actor is None:
            self._actor = TokenizerActor(
                model_name=self.model_name,
                seq_len=self.seq_len,
                text_col=self.text_col,
                trust_remote_code=self.trust_remote_code,
                format_name=self.format_name,
                format_options=self.format_options,
                is_cpt=self.is_cpt,
                pack_sequences=self.pack_sequences,
                train_on_inputs=self.train_on_inputs,
            )
        return self._actor(batch)


def _apply_map_batches(
    ds: ray.data.Dataset,
    cfg,
    text_col: str,
) -> ray.data.Dataset:

    model_name = cfg.model.name
    seq_len = cfg.data.seq_len
    trust_remote_code = getattr(cfg.model, "trust_remote_code", False)
    num_workers = getattr(cfg.data, "num_workers", 4)
    batch_size = getattr(cfg.data, "tokenize_batch_size", 128)
    format_name = getattr(cfg.data, "format", None)
    format_options = dict(getattr(cfg.data, "format_options", None) or {})

    is_cpt = getattr(cfg.data, "is_cpt", False)
    pack_sequences = getattr(cfg.data, "pack_sequences", False)

    train_on_inputs = getattr(cfg.data, "train_on_inputs", None)
    if train_on_inputs is None:
        train_on_inputs = getattr(cfg.data, "is_cpt", False)

    factory = _TokenizerActorFactory(
        model_name=model_name,
        seq_len=seq_len,
        text_col=text_col,
        trust_remote_code=trust_remote_code,
        format_name=format_name,
        format_options=format_options,
        is_cpt=is_cpt,
        pack_sequences=pack_sequences,
        train_on_inputs=train_on_inputs,
    )

    ds = ds.repartition(max(num_workers * 2, 1))

    ds = ds.map_batches(
        factory,
        batch_format="numpy",
        batch_size=batch_size,
        num_cpus=1,
        concurrency=num_workers,
    )

    # Any mode's TokenizerActor can emit an all--100-labels / all-zero
    # attention_mask dummy row when a whole tokenize_batch_size-sized chunk
    # produces no unmasked label anywhere (e.g. a run of conversations with
    # no assistant turn) -- not just CPT/packed. If such a row reaches
    # training, CrossEntropyLoss(reduction='mean') sees zero non-ignored
    # tokens in that batch, i.e. a 0/0 -> NaN loss for that step. Filter it
    # out unconditionally rather than only for is_cpt/pack_sequences.
    ds = ds.filter(lambda row: bool(np.sum(row["attention_mask"]) > 0))

    return ds