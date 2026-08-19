# src/bhaskera/data/

Ray Data dataset pipeline with persistent tokenisation cache and pluggable chat-format renderers.

## Module layout

```
data/
├── __init__.py          # re-exports REGISTRY, register, build_ray_dataset
├── registry.py          # decorator registries for raw + tokenised builders
├── tokenize.py          # TokenizerActor, persistent Parquet cache
├── datasets/
│   ├── __init__.py      # imports trigger @register side-effects
│   ├── ultrachat.py     # HuggingFaceH4/ultrachat_200k
│   ├── openassistant.py # timdettmers/openassistant-guanaco
│   ├── redpajama.py     # togethercomputer/RedPajama-Data-1T-Sample
│   └── local_chat.py    # generic JSONL / JSON / Parquet loader
└── formats/
    ├── __init__.py      # FORMAT_REGISTRY, @register_format, render_with_format
    └── builtins.py      # chatml, alpaca, sharegpt renderers
```

## How datasets are registered

Two decorators in `registry.py`:

- `@register_raw(name, text_col="...")` — registers an un-tokenised builder that returns a `ray.data.Dataset`. The `text_col` argument is recorded in the `TEXT_COL` dict so the tokenise CLI knows which column carries the text.
- `@register(name)` — registers the tokenised builder that the training launcher calls. Typically composes `@register_raw` + `tokenize_dataset`.

Importing `bhaskera.data.datasets` triggers the decorators for all built-ins. Adding a new dataset means dropping a module that imports both decorators and applying them — no registry edits required.

`build_ray_dataset(cfg, world_size)` is the single entry point used by `launcher/train.py`. It dispatches by `cfg.data.name` and forwards `world_size` to the builder so partitioning is cluster-aware.

`call_raw_builder(name, cfg, split=None)` is used by the tokenise CLI. It inspects the builder signature and only forwards `split` if the builder accepts it (so generic HF builders that don't know about splits keep working).

## Built-in datasets

| Name | Source | Text column | Notes |
|---|---|---|---|
| `ultrachat` | `HuggingFaceH4/ultrachat_200k`, `train_sft` split | `prompt` | |
| `openassistant` | `timdettmers/openassistant-guanaco`, `train` | `text` | |
| `redpajama` | `togethercomputer/RedPajama-Data-1T-Sample`, `train` | `text` | |
| `local` | Local JSONL / JSON / Parquet | `text` (via format renderer) | Accepts file, directory, or glob; train/val split via `train_path`/`val_path` |

For `local`, `_resolve_files()` handles three input shapes:
- a single file → returned as-is
- a directory → recursive `rglob` for `.jsonl`, `.json`, `.parquet`
- otherwise treated as a glob pattern relative to its parent

Mixed JSON and Parquet under one path raises rather than guessing.

## Tokenisation pipeline (`tokenize.py`)

Stateful Ray Data tokeniser with persistent on-disk cache.

### Cache layout

```
<cache_dir>/<dataset_name>_<16-char-hash>/
├── part-0.parquet
├── part-1.parquet
├── ...
└── metadata.json
```

`metadata.json` records `model_name`, `seq_len`, `dataset_name`, `num_rows`, schema, creation timestamp, Bhaskera version, format name, and format options.

### Cache key

`_cache_version_hash()` builds a SHA-256 over `(model_name | seq_len | dataset_name | format_name | format_options)` and truncates to 16 hex chars. SHA-256 (not Python `hash()`) — stable across processes and machines. Cache keys written by older Bhaskera versions (no format component) still validate when the new code is asked for "no format".

### Cache verification

`_verify_cache()` requires all of: `metadata.json` present and parseable; `model_name`, `seq_len`, `dataset_name` match; format matches when the caller supplied one; at least one `.parquet` file exists.

### Public functions

- `persist_tokenized(ds, cfg, text_col, dataset_name)` — write the cache. Reads `cfg.data.cache_dir`, `cfg.model.name`, `cfg.data.seq_len`, `cfg.data.format`, `cfg.data.format_options`. Skips work on cache hit unless `cfg.data.overwrite_cache=True`. Returns the absolute cache path.
- `load_tokenized(path, cfg, world_size)` — load a pre-tokenised cache as a `ray.data.Dataset`.
- `tokenize_dataset(ds, cfg, text_col, world_size)` — dispatch: load cache if `cfg.data.tokenized_path` is set, else tokenise on the fly via `_apply_map_batches`.

### `TokenizerActor`

Loaded once per Ray worker via the `_TokenizerActorFactory` lazy initialiser (so Ray pickles a small stateless factory rather than a live tokenizer). Each call:

1. Renders the batch through the configured format (if any) or reads `text_col` directly.
2. Calls `tokenizer(..., max_length=seq_len, truncation=True, padding="max_length", return_tensors="np")`.
3. Builds `labels = input_ids.copy()` and masks pad positions with `-100`.
4. Drops empty rows (text was empty or rendered to empty).
5. If the batch is entirely empty after filtering, returns a single dummy row with all-`-100` labels to keep the pipeline stable.

`pad_token` is set to `eos_token` when the tokenizer has none — required for Falcon, GPT-2, Llama-2, Param2.

## Format renderers (`formats/`)

A format renderer maps one raw row to one rendered string ready for tokenisation:

```python
fn(row: dict, tokenizer, options: dict) -> str
```

`register_format(name)` adds the function to `FORMAT_REGISTRY`. `render_with_format(name, row, tokenizer, options)` looks up and calls it. Built-ins are imported lazily by `_ensure_builtins_loaded()` so the registry stays cheap.

### Built-ins (`formats/builtins.py`)

- **`chatml`** — rows with `messages: [{role, content}, ...]`. Calls `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)` when the tokenizer has a `chat_template`; falls back to manual ChatML (`<|im_start|>role\n...<|im_end|>`) when it doesn't. Portable across Llama-3, Qwen, Mistral, Param2. Options: `messages_field` (default `"messages"`).
- **`alpaca`** — rows with `instruction`, optional `input`, and `output`. Uses the classic Alpaca prose template; pass `use_chat_template: true` to render as a 2-turn chat instead.
- **`sharegpt`** — rows with `conversations: [{from, value}, ...]`. Maps `from` values to chat roles via `_SHAREGPT_ROLE_MAP` (overridable through `role_map`). Options: `conversations_field` (default `"conversations"`), `role_map`.

### Adding a custom format

```python
from bhaskera.data.formats import register_format

@register_format("my_custom")
def render(row, tokenizer, options):
    return f"### Q: {row['question']}\n### A: {row['answer']}"
```

Then set `data.format: my_custom` in the YAML config.

## Partitioning

`_compute_num_partitions(cfg, world_size)` rounds up to the nearest multiple of `world_size` with a floor of `max(world_size * 4, num_workers * 4, 16)`. This prevents empty or unequal shards on small datasets.

## Public API

Re-exported from `bhaskera.data`:

- `REGISTRY` — name → tokenised builder
- `register(name)` — decorator
- `build_ray_dataset(cfg, world_size=1)` — returns the tokenised dataset for `cfg.data.name`
