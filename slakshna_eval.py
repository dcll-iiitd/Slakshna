"""Exact held-out language-model metrics for chat fine-tuning.

Definitions (these are the standard ones and are what the paper should state):

  scored tokens   the assistant-response tokens (including the closing <|im_end|> token); every other
                  token (system prompt, user turn, role headers) is context and is not scored. The same
                  rule produces the training labels (bhaskera.data.tokenize), so training and validation
                  measure the same thing.
  NLL             total negative log-likelihood of the scored tokens, in nats, divided by the NUMBER OF
                  SCORED TOKENS over the whole evaluation set (token-weighted corpus mean, not a mean of
                  per-batch or per-example means).
  perplexity      exp(NLL). Tokenizer-dependent, so only comparable between runs with the same tokenizer.
  bits per byte   total NLL / (ln 2 x number of UTF-8 bytes of the scored text). Tokenizer-independent.

Every example is scored on its own (no packing, right-padded batches), so the numbers do not depend on
batch size or packing. Dropout is off (eval mode).
"""
import json
import math

import torch
import torch.nn.functional as F


def tokenize_chat(tokenizer, messages):
    """(input_ids, labels) with labels == -100 everywhere except assistant tokens.

    Mirrors the branch logic of bhaskera.data.tokenize for ``train_on_inputs=False``.
    """
    from bhaskera.data.formats.builtins import _manual_chatml_tokenize

    if getattr(tokenizer, "chat_template", None):
        try:
            enc = tokenizer.apply_chat_template(
                messages, tokenize=True, return_dict=True, return_assistant_tokens_mask=True)
            ids = enc["input_ids"]
            if "assistant_masks" in enc and any(enc["assistant_masks"]):
                return list(ids), [t if m else -100 for t, m in zip(ids, enc["assistant_masks"])]
        except Exception:
            pass
    return _manual_chatml_tokenize(tokenizer, messages)


def load_examples(tokenizer, path, max_len=2048, limit=None):
    """Tokenize a chat JSONL file -> list of dicts with ids, labels and the byte length of scored text."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            ids, labels = tokenize_chat(tokenizer, row["messages"])
            ids, labels = ids[:max_len], labels[:max_len]            # same truncation as training
            scored = [t for t in labels[1:] if t != -100]            # label at position 0 is never predicted
            if not scored:
                continue
            n_bytes = len(tokenizer.decode(scored, skip_special_tokens=False).encode("utf-8"))
            out.append({"ids": ids, "labels": labels, "bytes": n_bytes, "n_scored": len(scored)})
            if limit and len(out) >= limit:
                break
    return out


@torch.no_grad()
def evaluate(model, examples, pad_token_id, batch_size=8, device=None):
    """Token-weighted corpus NLL / perplexity / bits-per-byte of ``model`` on ``examples``."""
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()
    order = sorted(range(len(examples)), key=lambda i: len(examples[i]["ids"]))   # less padding
    total_nll, total_tokens, total_bytes = 0.0, 0, 0
    for s in range(0, len(order), batch_size):
        chunk = [examples[i] for i in order[s:s + batch_size]]
        width = max(len(e["ids"]) for e in chunk)
        ids = torch.full((len(chunk), width), pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        labels = torch.full((len(chunk), width), -100, dtype=torch.long)
        for i, e in enumerate(chunk):
            n = len(e["ids"])
            ids[i, :n] = torch.tensor(e["ids"])
            mask[i, :n] = 1
            labels[i, :n] = torch.tensor(e["labels"])
        logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
        nll = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                              labels[:, 1:].to(device).reshape(-1), ignore_index=-100, reduction="sum")
        total_nll += float(nll.double().item())
        total_tokens += sum(e["n_scored"] for e in chunk)
        total_bytes += sum(e["bytes"] for e in chunk)
    if was_training:
        model.train()
    if total_tokens == 0:
        raise ValueError("no scored tokens in the evaluation set")
    mean_nll = total_nll / total_tokens
    return {
        "nll": mean_nll,
        "perplexity": math.exp(mean_nll) if mean_nll < 700 else float("inf"),
        "bits_per_byte": total_nll / (math.log(2) * total_bytes),
        "tokens": total_tokens,
        "bytes": total_bytes,
        "examples": len(examples),
    }
