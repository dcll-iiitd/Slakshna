"""
bhaskera.data.formats.builtins
==============================
Built-in renderers for the most common SFT data layouts.

Imported lazily by ``formats.__init__._ensure_builtins_loaded`` so the
registry stays cheap to import.
"""
from __future__ import annotations

from typing import Any, List

from . import register_format


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_list(x: Any) -> List:
    """Normalise numpy arrays / pyarrow lists to a plain Python list."""
    if x is None:
        return []
    if hasattr(x, "tolist"):
        return x.tolist()
    return list(x)


def _to_dict(x: Any) -> dict:
    """Normalise a row entry that may arrive as numpy.void / dict-like."""
    if isinstance(x, dict):
        return x
    # numpy structured array element / pyarrow struct: fall back to dict()
    try:
        return dict(x)
    except Exception:
        return {"role": "user", "content": str(x)}


def _manual_chatml(messages: List[dict]) -> str:
    """
    Fallback ChatML rendering when the tokenizer has no chat_template.

    Format:
        <|im_start|>role
        content<|im_end|>
        <|im_start|>role
        content<|im_end|>
    """
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(parts)


def _manual_chatml_tokenize(tokenizer: Any, messages: List[dict]) -> tuple[list[int], list[int]]:
    """
    Fallback manual ChatML tokenisation.

    The previous implementation tokenized each message's role-header and
    content separately, then concatenated the resulting id lists. BPE
    merges that would normally happen *across* a role/content seam (e.g.
    the "\n" right after "<|im_start|>user") aren't guaranteed to match
    what you'd get tokenizing the whole rendered string in one call, so
    the emitted token sequence could silently diverge from what
    apply_chat_template would produce at inference time.

    Fix: build the full rendered ChatML string first, then tokenize it in
    a *single* call, and recover per-token assistant/non-assistant labels
    by aligning against the character spans of each message's content.
    This guarantees input_ids always matches tokenizing the whole string
    at once. Fast tokenizers use offset_mapping directly; slow tokenizers
    fall back to incremental re-tokenization of growing (real) prefixes
    of the same full string, so cross-boundary merges are still captured
    correctly.
    """
    full_text = ""
    spans: list[tuple[str, int, int]] = []  # (role, content_start_char, content_end_char)
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")

        role_str = f"<|im_start|>{role}\n"
        content_str = f"{content}<|im_end|>\n"

        full_text += role_str
        content_start = len(full_text)
        full_text += content_str
        content_end = len(full_text)
        spans.append((role, content_start, content_end))

    if not full_text:
        return [], []

    if getattr(tokenizer, "is_fast", False):
        enc = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
        input_ids = list(enc["input_ids"])
        offsets = enc["offset_mapping"]
        labels = [-100] * len(input_ids)
        for role, start, end in spans:
            if role != "assistant":
                continue
            for i, (tok_start, tok_end) in enumerate(offsets):
                if tok_start == tok_end:
                    continue  # special/empty tokens
                if tok_start >= start and tok_end <= end:
                    labels[i] = input_ids[i]
    else:
        # Slow-tokenizer fallback: no offset_mapping available, so
        # incrementally re-tokenize growing (real) prefixes of full_text
        # and diff token counts at each boundary. The final input_ids is
        # still exactly tokenizer.encode(full_text, add_special_tokens=False)
        # -- only label attribution near a boundary could rarely shift by
        # a token if a later merge reaches backward across it.
        input_ids, labels = [], []
        prev_len = 0
        for role, start, end in spans:
            header_ids = tokenizer.encode(full_text[:start], add_special_tokens=False)
            ids_so_far = tokenizer.encode(full_text[:end], add_special_tokens=False)

            n_header_new = max(len(header_ids) - prev_len, 0)
            role_new_ids = ids_so_far[prev_len:prev_len + n_header_new]
            content_new_ids = ids_so_far[prev_len + n_header_new:]

            input_ids.extend(role_new_ids)
            labels.extend([-100] * len(role_new_ids))
            input_ids.extend(content_new_ids)
            labels.extend(content_new_ids if role == "assistant" else [-100] * len(content_new_ids))

            prev_len = len(ids_so_far)

    # Match the primary tokenize=True path: prepend BOS if the tokenizer
    # defines one and it isn't already the first emitted token. The old
    # fallback never inserted a leading special token at all, which could
    # silently disagree with the primary apply_chat_template path.
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is not None and (not input_ids or input_ids[0] != bos_id):
        input_ids = [bos_id] + input_ids
        labels = [-100] + labels

    return input_ids, labels

def _apply_chat_template_safe(tokenizer: Any, messages: List[dict]) -> str:
    has_template = (
        hasattr(tokenizer, "apply_chat_template")
        and getattr(tokenizer, "chat_template", None)
    )
    if not has_template:
        return _manual_chatml(messages)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


@register_format("chatml")
def render_chatml(row: dict, tokenizer: Any, options: dict) -> list[dict]:
    messages_field = options.get("messages_field", "messages")
    messages = _to_list(row.get(messages_field, []))
    return [_to_dict(m) for m in messages]


# ---------------------------------------------------------------------------
# Alpaca
# ---------------------------------------------------------------------------

_ALPACA_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately "
    "completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)

_ALPACA_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{output}"
)


@register_format("alpaca")
def render_alpaca(row: dict, tokenizer: Any, options: dict) -> list[dict]:
    instruction = str(row.get("instruction", "") or "")
    inp         = str(row.get("input", "") or "")
    output      = str(row.get("output", "") or "")

    user_turn = instruction if not inp else f"{instruction}\n\n{inp}"

    # Render Alpaca directly to messages so the trainer masks instruction tokens natively.
    return [
        {"role": "user", "content": user_turn},
        {"role": "assistant", "content": output},
    ]


# ---------------------------------------------------------------------------
# ShareGPT
# ---------------------------------------------------------------------------

_SHAREGPT_ROLE_MAP = {
    "human":     "user",
    "user":      "user",
    "gpt":       "assistant",
    "assistant": "assistant",
    "chatgpt":   "assistant",
    "bard":      "assistant",
    "system":    "system",
    "tool":      "tool",
    "function":  "tool",
}


@register_format("sharegpt")
def render_sharegpt(row: dict, tokenizer: Any, options: dict) -> list[dict]:
    field    = options.get("conversations_field", "conversations")
    role_map = {**_SHAREGPT_ROLE_MAP, **(options.get("role_map") or {})}

    convs = _to_list(row.get(field, []))
    messages = []
    for c in convs:
        c = _to_dict(c)
        sender = str(c.get("from", "user")).lower()
        role = role_map.get(sender, "user")
        messages.append({"role": role, "content": c.get("value", "") or ""})

    return messages
