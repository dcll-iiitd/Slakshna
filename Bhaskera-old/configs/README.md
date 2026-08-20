# configs/

YAML run configurations consumed by the Bhaskera CLIs. Each file is a complete, self-contained specification — model, data, distributed strategy, LoRA, MoE, checkpointing, logging, and monitoring.

Files are loaded via `bhaskera.config.load_config(path)`, which parses YAML and populates the dataclass tree defined in `src/bhaskera/config.py`. Unknown keys are ignored; missing keys fall back to dataclass defaults.

## Files

| File | Scenario | Strategy | Model | Data |
|---|---|---|---|---|
| `2node.yaml` | 4 GPUs across 2 nodes (2 × 2) | FSDP2 `FULL_SHARD` | Param2-17B-A2.4B-Thinking (MoE) | UltraChat 200k |
| `finetune_param_local_data.yaml` | 4 × A100, 2 nodes × 2 | FSDP2 `FULL_SHARD` | Param2-17B-A2.4B-Thinking (MoE) | Local ChatML JSONL |
| `qwen.yaml` | 4 GPUs across 2 nodes (2 × 2) | FSDP2 `FULL_SHARD` | Qwen2.5-14B (dense) | UltraChat 200k |
| `qwen_ddp.yaml` | DDP variant of `qwen.yaml` | DDP | Qwen2.5-14B (dense) | UltraChat 200k |
| `qwen_hybrid_shard.yaml` | Hybrid-shard variant | FSDP2 `HYBRID_SHARD` | Qwen2.5-14B (dense) | UltraChat 200k |
| `tokenize.yaml` | Tokenisation job for UltraChat | — | Param2 tokenizer | UltraChat 200k |
| `tokenize_qwen.yaml` | Tokenisation job for Qwen | — | Qwen tokenizer | UltraChat 200k |

## Schema reference

The keys below mirror the dataclasses in `bhaskera.config`. See the dataclass definitions for full defaults and types.

### `model`
- `name` — HF repo id or local path
- `dtype` — `bfloat16` (default), `float16`, `float32`, or `auto`
- `attn_impl` — `flash_attention_2`, `sdpa`, `eager`, or `null` (let HF pick)
- `trust_remote_code` — required `true` for Param2 and similar custom-code models
- `use_liger_kernel` — enable Triton-fused kernel patching at load time

### `data`
- `name` — registered dataset key (`ultrachat`, `openassistant`, `redpajama`, `local`)
- `seq_len`, `num_workers`, `prefetch_batches`, `local_shuffle_buffer_multiplier`, `pack_sequences`
- `tokenized_path` — pre-tokenised cache to load directly
- `val_tokenized_path` — optional validation cache
- `cache_dir`, `overwrite_cache`, `tokenize_batch_size`, `tokenize_compression` (`snappy` / `zstd` / `none`)
- `format` — registered format renderer name (`chatml`, `alpaca`, `sharegpt`, custom)
- `format_options` — free-form dict passed to the renderer; hashed into the cache key
- `path`, `train_path`, `val_path` — file / directory / glob for local sources

### `lora`
- `enabled`, `r`, `alpha`, `dropout`
- `target_modules` — `["auto"]` triggers introspection; otherwise an explicit list
- `include_experts` — also LoRA expert FFN linears on MoE models
- `freeze_router` — freeze MoE gate/router weights (recommended `true`)
- `modules_to_save` — modules to fully train alongside LoRA (e.g. `embed_tokens`)

### `moe`
- `aux_loss_weight`, `router_z_loss_weight`
- `freeze_router`, `log_expert_utilization`, `log_every_n_steps`

### `training`
- `batch_size`, `grad_accum`, `lr`, `weight_decay`
- `max_steps`, `num_epochs`, `warmup_steps`
- `max_grad_norm`, `grad_clip`, `max_grad_skip_steps`
- `seed`, `deterministic`
- `distributed.strategy` — `fsdp` or `ddp`
- `distributed.fsdp` — `sharding_strategy` (`FULL_SHARD` / `HYBRID_SHARD` / etc.), `transformer_layer_cls` (empty list = auto-detect), `param_dtype`, `reduce_dtype`, `buffer_dtype`, `activation_checkpointing`, `cpu_offload`, `shard_experts_individually`
- `distributed.ddp` — `find_unused_parameters`, `gradient_as_bucket_view`, `broadcast_buffers`, `activation_checkpointing`, `static_graph`

### `checkpoint`
- `enabled`, `save_dir`, `save_interval` (epochs), `keep_last_n` (best-by-loss retention)

### `logging`
- `tracker` — string, list, or `null`; recognised values `wandb`, `mlflow`, `ray`
- `project`, `run_name`, `tags`, `group`
- `mlflow_tracking_uri` — leave empty to default to `~/mlflow-runs`
- `log_gpu_every_n_steps`

### `monitoring`
- `dashboard`, `dashboard_host`, `dashboard_port` (Ray Dashboard)
- `metrics_export_port` (Prometheus pull endpoint Ray exposes)
- `metrics.*` — per-step metric toggles: `system_every_n_steps`, `cuda_every_n_steps`, `gpu`, `cpu`, `cuda_memory`, `throughput`, `peak_tflops_per_gpu`, `throughput_window`, `throughput_warmup`

### `inference`
- `max_new_tokens`, `temperature`, `top_p`, `top_k`, `do_sample`, `batch_size`
- `kv_cache` — `static` / `dynamic`
- `device`, `torch_compile`
- `turboquant.{enabled, key_bits, value_bits, residual_window, protected_layers}`
- `speculative.{enabled, draft_model_name, num_draft_tokens}`

## Usage

```bash
# Tokenise once
bhaskera-tokenize --config configs/tokenize.yaml --split both

# Train
bhaskera-train --config configs/qwen.yaml --num-workers 4

# DDP variant
bhaskera-train --config configs/qwen_ddp.yaml --num-workers 4

# SLURM (handled by scripts/submit.sh)
sbatch scripts/submit.sh --config configs/2node.yaml
```

The headers of `2node.yaml`, `finetune_param_local_data.yaml`, and `qwen.yaml` contain inline memory plans and effective batch arithmetic for their respective targets — useful as templates when adapting to a new GPU layout.
