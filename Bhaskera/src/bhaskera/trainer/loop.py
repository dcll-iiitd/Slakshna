"""
bhaskera.trainer.loop
=====================
Pure training loop with pluggable evaluation and throughput clock protection.

Changes vs previous version
----------------------------
1. Integrated EvaluationLifecycle for production-grade GPU memory reclamation
   before evaluation. The old inline evaluator.run_validation() / run_benchmarks()
   calls are replaced by a lifecycle context manager that:
     - zeros gradients and destroys the Ray Data iterator before eval
     - offloads optimizer state to CPU (configurable via cfg.evaluation.offload_optimizer)
     - runs evaluation inside torch.inference_mode()
     - restores optimizer from CPU and rebuilds the iterator after eval
     - resumes from exactly the same sample position

2. DatasetCursor tracks samples_consumed / tokens_consumed per epoch so the
   iterator can be fast-forwarded after a checkpoint resume via dataset.skip(n).

3. maybe_resume now returns (step, meta_dict); the meta_dict carries
   eval_lifecycle/* keys that reconstruct the cursor.

4. save_and_prune now receives cursor_meta so the cursor is embedded in
   meta.json at every checkpoint.

5. Non-finite grad_norm skip now resets the throughput clock so the skipped
   step's wall-time does not inflate the next valid step's dt.

6. ray_dataset_shard param added to train() and _run_epoch() so the lifecycle
   can rebuild the Ray Data iterator independently of the epoch-level dataset arg.
"""
from __future__ import annotations

import contextlib
import logging
import math
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from bhaskera.introspect import ModelProfile
from bhaskera.utils import ThroughputTracker
from bhaskera.utils.system_stats import system_stats, cuda_memory_stats
from .checkpointing import maybe_resume, save_and_prune
from .eval_lifecycle import (
    DatasetCursor,
    EvaluationLifecycle,
    TrainingPipelineState,
    cursor_from_checkpoint_metadata,
    cursor_to_checkpoint_metadata,
    _skip_torch_batches,
)
from .moe import compute_expert_utilization, extract_aux_loss
from .optim import build_optimizer, build_scheduler
from .precision import resolve_autocast_dtype
from bhaskera.evaluation import Evaluator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FSDP2 + DDP gradient-sync helper
# ---------------------------------------------------------------------------

def _set_grad_sync(model: torch.nn.Module, enabled: bool) -> None:
    """
    Toggle gradient all-reduce for the wrapped model.
    Dispatches by wrapper type: FSDP2 → set_requires_gradient_sync(model, enabled)
    DDP → model.require_backward_grad_sync = enabled
    """
    # ── FSDP2 path ──────────────────────────────────────────────────
    try:
        from torch.distributed._composable.fsdp import (
            FSDPModule,
            set_requires_gradient_sync,
        )
        if isinstance(model, FSDPModule) or any(
            isinstance(m, FSDPModule) for m in model.modules()
        ):
            set_requires_gradient_sync(model, enabled)
            return
    except ImportError:
        # torch < 2.4 — FSDP2 unavailable, fall through to DDP / no-op.
        pass

    # ── DDP path ────────────────────────────────────────────────────
    if isinstance(model, DDP):
        if getattr(model, "_bhaskera_static_graph", False) or getattr(
            model, "static_graph", False
        ):
            return
        model.require_backward_grad_sync = enabled
        return

    # ── Non-distributed: nothing to do ──────────────────────────────


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def train(
    *,
    model: torch.nn.Module,
    dataset,
    val_dataset=None,
    cfg,
    profile: ModelProfile,
    rank: int,
    local_rank: int,
    tracker=None,
    world_size: int = 1,
    ray_dataset_shard=None,
) -> None:
    """
    Run the training loop.

    Args:
        ray_dataset_shard:  The raw Ray Dataset shard for this rank, obtained
                            via ray.train.get_dataset_shard("train") in worker.py.
                            Passed through to _run_epoch so the EvaluationLifecycle
                            can rebuild the iterator via dataset.skip(n) after eval.
                            If None (e.g. single-GPU debug without Ray), the old
                            in-line eval path is used as a fallback.
    """
    device = torch.device(f"cuda:{local_rank}")
    train_cfg = cfg.training
    ckpt_cfg = cfg.checkpoint

    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)

    # Instantiate the Evaluator Orchestrator
    evaluator = Evaluator(cfg, model, profile, rank, world_size)

    model.train()

    step = 0
    _resume_meta: dict = {}
    if ckpt_cfg.enabled:
        step, _resume_meta = maybe_resume(model, optimizer, ckpt_cfg.save_dir)
        model.train()

    # Restore dataset cursor from checkpoint so the iterator can be
    # fast-forwarded to the exact position the run was interrupted at.
    _resume_cursor = cursor_from_checkpoint_metadata(_resume_meta)
    _samples_consumed: int = _resume_cursor.samples_consumed
    _tokens_consumed: int  = _resume_cursor.tokens_consumed

    best_ckpts: list[tuple[float, str]] = []

    # ── Throughput / MFU tracker ────────────────────────────────────
    metrics_cfg = getattr(getattr(cfg, "monitoring", None), "metrics", None)
    throughput_on = bool(getattr(metrics_cfg, "throughput", True)) if metrics_cfg else True
    peak_tflops = float(getattr(metrics_cfg, "peak_tflops_per_gpu", 312.0)) if metrics_cfg else 312.0

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_for_flops = total_params or trainable_params

    throughput = ThroughputTracker(
        params_for_flops=params_for_flops,
        world_size=max(1, int(world_size)),
        peak_flops_per_gpu=peak_tflops * 1e12,
        window=int(getattr(metrics_cfg, "throughput_window", 50)) if metrics_cfg else 50,
        warmup_steps=int(getattr(metrics_cfg, "throughput_warmup", 5)) if metrics_cfg else 5,
        is_peft=getattr(cfg.lora, "enabled", False),
        activation_checkpointing=getattr(train_cfg, "gradient_checkpointing", False),
        num_layers=getattr(profile, "num_hidden_layers", 0),
        hidden_size=getattr(profile, "hidden_size", 0),
    ) if throughput_on else None

    if tracker:
        tracker.log({
            "model/total_params": float(total_params),
            "model/trainable_params": float(trainable_params),
            "model/world_size": float(world_size),
        }, step=0)

    for epoch in range(train_cfg.num_epochs):
        step, best_ckpts, _samples_consumed, _tokens_consumed = _run_epoch(
            model=model,
            dataset=dataset,
            val_dataset=val_dataset,
            evaluator=evaluator,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            profile=profile,
            rank=rank,
            local_rank=local_rank,
            device=device,
            epoch=epoch,
            step=step,
            tracker=tracker,
            best_ckpts=best_ckpts,
            throughput=throughput,
            world_size=world_size,
            ray_dataset_shard=ray_dataset_shard,
            samples_consumed=_samples_consumed,
            tokens_consumed=_tokens_consumed,
        )
        # After each epoch the position resets to 0 (new epoch starts from
        # the beginning of the dataset). The cursor from a checkpoint only
        # applies to the first epoch after resume.
        _samples_consumed = 0
        _tokens_consumed  = 0
        if step >= train_cfg.max_steps:
            break

    if tracker:
        tracker.finish()
    if rank == 0:
        logger.info("Training complete.")


# ---------------------------------------------------------------------------
# Single epoch
# ---------------------------------------------------------------------------

def _run_epoch(
    *,
    model,
    dataset,
    val_dataset,
    evaluator,
    optimizer,
    scheduler,
    cfg,
    profile,
    rank,
    local_rank,
    device,
    epoch,
    step,
    tracker,
    best_ckpts,
    throughput: Optional[ThroughputTracker],
    world_size: int,
    ray_dataset_shard=None,
    samples_consumed: int = 0,
    tokens_consumed: int = 0,
):
    train_cfg = cfg.training
    ckpt_cfg = cfg.checkpoint
    grad_accum = train_cfg.grad_accum
    strategy = cfg.training.distributed.strategy.lower()

    autocast_dtype = resolve_autocast_dtype(cfg, profile)
    use_autocast = (strategy == "ddp" and device.type == "cuda")

    moe_cfg = getattr(cfg, "moe", None)
    aux_loss_weight = getattr(moe_cfg, "aux_loss_weight", 0.01) if moe_cfg else 0.01
    log_expert_util = (
        profile.is_moe
        and moe_cfg is not None
        and getattr(moe_cfg, "log_expert_utilization", True)
    )
    expert_log_every = getattr(moe_cfg, "log_every_n_steps", 10) if moe_cfg else 10

    metrics_cfg = getattr(getattr(cfg, "monitoring", None), "metrics", None)
    sys_every = int(getattr(metrics_cfg, "system_every_n_steps", 10)) if metrics_cfg else 10
    cuda_every = int(getattr(metrics_cfg, "cuda_every_n_steps", 10)) if metrics_cfg else 10
    sys_on = bool(getattr(metrics_cfg, "enabled", True)) if metrics_cfg else True

    # ── Data loader ─────────────────────────────────────────────────
    # On checkpoint resume, fast-forward the iterator past already-consumed
    # samples using dataset.skip(n). This is O(num_parquet_files), not
    # O(samples_consumed), so it is cheap even for large positions.
    _loader_kwargs = dict(
        batch_size=train_cfg.batch_size,
        local_shuffle_buffer_size=max(
            train_cfg.batch_size * cfg.data.local_shuffle_buffer_multiplier,
            1000
        ),
        local_shuffle_seed=train_cfg.seed + rank,
        prefetch_batches=cfg.data.prefetch_batches,
        drop_last=True,
        dtypes={
            "input_ids": torch.long,
            "attention_mask": torch.long,
            "labels": torch.long,
        },
        device=device,
    )

    if samples_consumed > 0 and ray_dataset_shard is not None:
        if rank == 0:
            logger.info(
                f"[loop] Checkpoint resume: skipping {samples_consumed} "
                f"already-consumed samples (manual drain — DataIterator has no .skip())"
            )
        loader = None
        loader_iter = _skip_torch_batches(ray_dataset_shard, samples_consumed, **_loader_kwargs)
    else:
        loader = dataset.iter_torch_batches(**_loader_kwargs)
        loader_iter = None  # set below via iter(loader), unchanged from current flow

    epoch_loss = 0.0
    epoch_aux_loss = 0.0
    epoch_steps = 0

    loss_ema: Optional[float] = None
    loss_ema_alpha = 0.05

    # Running position counters for this epoch.
    # Initialised from the checkpoint cursor (non-zero only on resume).
    _step_samples_consumed: int = samples_consumed
    _step_tokens_consumed: int  = tokens_consumed

    # Pre-declare window accumulators so they exist if the while loop
    # body never executes (e.g. dataset is empty on this rank).
    window_hardware_tokens = 0
    window_tokens = 0
    window_samples = 0
    window_seq_len = 0

    optimizer.zero_grad(set_to_none=True)

    if throughput is not None:
        throughput.reset_step_clock()

    if loader_iter is None:
        loader_iter = iter(loader)
    while step < train_cfg.max_steps:
        micro_losses: list[torch.Tensor] = []
        micro_aux_losses: list[torch.Tensor] = []

        # Reset window accumulators at the start of each optimizer step.
        window_hardware_tokens = 0
        window_tokens = 0
        window_samples = 0
        window_seq_len = 0

        # ── Gradient accumulation loop ───────────────────────────────
        for micro_step in range(grad_accum):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = None  # type: ignore[assignment]
                break

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            try:
                _bs  = int(input_ids.size(0))
                _seq = int(input_ids.size(1))
                window_hardware_tokens += int(input_ids.numel())
                window_tokens += int(attention_mask.sum().item())
                window_samples += _bs
                window_seq_len = _seq
                # Increment BEFORE the forward pass so that if the process
                # is interrupted mid-step, the cursor points to the start
                # of the interrupted step and those samples are re-processed
                # on resume (they did not complete an optimizer step).
                _step_samples_consumed += _bs
                _step_tokens_consumed  += _bs * _seq
            except Exception:
                pass

            is_last = (micro_step == grad_accum - 1)
            _set_grad_sync(model, enabled=is_last)

            autocast_ctx = (
                torch.autocast("cuda", dtype=autocast_dtype)
                if use_autocast
                else contextlib.nullcontext()
            )

            with autocast_ctx:
                forward_kwargs = dict(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False,
                )
                if profile.is_moe and profile.has_aux_loss:
                    forward_kwargs["output_router_logits"] = True

                out = model(**forward_kwargs)
                main_loss = out.loss
                aux_loss = extract_aux_loss(out, profile)

                if aux_loss is not None:
                    total_loss = (main_loss + aux_loss_weight * aux_loss) / grad_accum
                else:
                    total_loss = main_loss / grad_accum

                total_loss.backward()

            micro_losses.append(main_loss.detach())
            if aux_loss is not None:
                micro_aux_losses.append(aux_loss.detach())

        if loader_iter is None:
            break

        _set_grad_sync(model, enabled=True)

        # ── Optimizer step ──────────────────────────────────────────
        grad_clip = getattr(train_cfg, "grad_clip", None) or getattr(train_cfg, "max_grad_norm", 1.0)

        if hasattr(model, "clip_grad_norm_"):
            grad_norm = model.clip_grad_norm_(grad_clip).item()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad),
                grad_clip,
            ).item()

        if not math.isfinite(grad_norm):
            logger.warning(
                f"[rank {rank}][epoch {epoch}][step {step}] "
                f"Non-finite grad_norm={grad_norm} — skipping optimizer step"
            )
            optimizer.zero_grad(set_to_none=True)
            if tracker:
                tracker.log({"train/non_finite_grad": 1.0}, step=step)
            # Reset the throughput clock so this skipped step's wall-time
            # (which includes the full backward) does not inflate the next
            # valid step's dt and crash the MFU calculation.
            if throughput is not None:
                throughput.reset_step_clock()
            continue

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        window_loss = torch.stack(micro_losses).mean().item()
        window_aux = (
            torch.stack(micro_aux_losses).mean().item()
            if micro_aux_losses
            else 0.0
        )

        if loss_ema is None:
            loss_ema = window_loss
        else:
            loss_ema = (1 - loss_ema_alpha) * loss_ema + loss_ema_alpha * window_loss

        loss_spike = (window_loss / loss_ema) if loss_ema > 0 else 1.0
        lr = scheduler.get_last_lr()[0]

        epoch_loss += window_loss
        epoch_aux_loss += window_aux
        epoch_steps += 1
        step += 1

        # ── Throughput ──────────────────────────────────────────────
        throughput_metrics: dict[str, float] = {}
        if throughput is not None:
            throughput_metrics = throughput.step(
                local_tokens_in_step=window_hardware_tokens,
                local_samples_in_step=window_samples,
                seq_len=window_seq_len,
            )

            if "throughput/tokens_per_sec_global" in throughput_metrics:
                global_tps = throughput_metrics.get("throughput/tokens_per_sec_global", 0.0)
                mfu = throughput_metrics.get("throughput/mfu_pct", 0.0)

                ratio = window_tokens / window_hardware_tokens if window_hardware_tokens > 0 else 1.0
                useful_global_tps = global_tps * ratio
                padding_pct = (1.0 - ratio) * 100.0

                print(
                    f"Step {step} | HW Tok/s: {global_tps:,.0f} | "
                    f"Useful Tok/s: {useful_global_tps:,.0f} | "
                    f"Pad Waste: {padding_pct:.1f}% | HW MFU: {mfu:.2f}%"
                )

        # ── Logging ────────────────────────────────────────────────
        if rank == 0:
            msg = (
                f"[epoch {epoch}][step {step}] loss={window_loss:.4f} "
                f"lr={lr:.2e} grad_norm={grad_norm:.4f}"
            )
            if "throughput/tokens_per_sec_global" in throughput_metrics:
                msg += f" tok/s={throughput_metrics['throughput/tokens_per_sec_global']:.0f}"
            if "throughput/mfu_pct" in throughput_metrics:
                msg += f" MFU={throughput_metrics['throughput/mfu_pct']:.1f}%"
            logger.info(msg)

        metrics: dict[str, float] = {
            "loss": window_loss,
            "lr": lr,
            "grad_norm": grad_norm,
            "epoch": float(epoch),
            "loss_running_avg": loss_ema,
            "loss_spike_ratio": loss_spike,
        }
        if profile.is_moe:
            metrics["aux_loss"] = window_aux
            metrics["total_loss"] = window_loss + aux_loss_weight * window_aux

        metrics.update(throughput_metrics)

        if tracker:
            if log_expert_util and step % expert_log_every == 0:
                metrics.update(compute_expert_utilization(out, profile))
            tracker.log(metrics, step=step)

            if sys_on and sys_every > 0 and step % sys_every == 0:
                sysm: dict[str, float] = {}
                sysm.update(system_stats(
                    gpu=bool(getattr(metrics_cfg, "gpu", True)) if metrics_cfg else True,
                    cpu=bool(getattr(metrics_cfg, "cpu", True)) if metrics_cfg else True,
                ))
                if cuda_every > 0 and step % cuda_every == 0:
                    if not metrics_cfg or getattr(metrics_cfg, "cuda_memory", True):
                        sysm.update(cuda_memory_stats(device))
                if sysm:
                    tracker.log(sysm, step=step)

        # ── Evaluation & Benchmarking ──────────────────────────────
        ran_eval = False
        _needs_val   = evaluator.should_run_validation(step)
        _needs_bench = evaluator.should_run_benchmark(step)

        if (_needs_val or _needs_bench) and ray_dataset_shard is not None:
            # ── Production path: full lifecycle teardown + rebuild ──────
            # Tears down all training-only GPU allocations before eval,
            # runs eval on the same model instance, then rebuilds the
            # training pipeline and resumes from the exact same position.
            _offload = getattr(
                getattr(cfg, "evaluation", None), "offload_optimizer", True
            )
            _cursor = DatasetCursor(
                samples_consumed=_step_samples_consumed,
                tokens_consumed=_step_tokens_consumed,
                global_step=step,
                epoch=epoch,
            )
            _pipeline_state = TrainingPipelineState(
                optimizer=optimizer,
                scheduler=scheduler,
                cursor=_cursor,
                grad_accum_steps=grad_accum,
                autocast_ctx_fn=contextlib.nullcontext,
            )
            _lifecycle = EvaluationLifecycle(
                model=model,
                pipeline_state=_pipeline_state,
                device=device,
                cfg=cfg,
                ray_dataset_shard=ray_dataset_shard,
                rank=rank,
                offload_optimizer=_offload,
            )
            with _lifecycle.run(loader_iter=loader_iter, loader=loader):
                # Inside this block:
                #   - model is in eval mode inside torch.inference_mode()
                #   - optimizer state is on CPU (if offload_optimizer=True)
                #   - loader_iter has been destroyed and its memory freed
                if _needs_val:
                    val_metrics = evaluator.run_validation(val_dataset)
                    if tracker and val_metrics:
                        tracker.log(val_metrics, step=step)
                    if rank == 0 and val_metrics:
                        logger.info(f"\033[1;32m[Validation @ Step {step}] {val_metrics}\033[0m")
                if _needs_bench:
                    bench_metrics = evaluator.run_benchmarks(tokenizer=None)
                    if tracker and bench_metrics:
                        tracker.log(bench_metrics, step=step)
                    if rank == 0 and bench_metrics:
                        logger.info(f"\033[1;34m[Benchmarks @ Step {step}] {bench_metrics}\033[0m")

            # After the context exits: model is back in train mode, optimizer
            # is back on GPU, and the iterator has been rebuilt at the cursor
            # position. Replace the (now-destroyed) iterator reference.
            loader_iter = _lifecycle.rebuilt_iterator
            ran_eval = True

        elif _needs_val or _needs_bench:
            # ── Fallback: no Ray shard (single-GPU debug without Ray) ──
            # Uses the old in-line path without lifecycle teardown.
            if _needs_val:
                val_metrics = evaluator.run_validation(val_dataset, optimizer=optimizer)
                if tracker and val_metrics:
                    tracker.log(val_metrics, step=step)
                if rank == 0 and val_metrics:
                    logger.info(f"\033[1;32m[Validation @ Step {step}] {val_metrics}\033[0m")
            if _needs_bench:
                bench_metrics = evaluator.run_benchmarks(tokenizer=None)
                if tracker and bench_metrics:
                    tracker.log(bench_metrics, step=step)
                if rank == 0 and bench_metrics:
                    logger.info(f"\033[1;34m[Benchmarks @ Step {step}] {bench_metrics}\033[0m")
            ran_eval = True

        # Clock protection: Reset throughput step timers if evaluations
        # halted execution so the eval wall-time is not charged to the
        # first training step after eval.
        if ran_eval and throughput is not None:
            throughput.reset_step_clock()

    if epoch_steps == 0:
        return step, best_ckpts, _step_samples_consumed, _step_tokens_consumed

    avg_loss = epoch_loss / epoch_steps
    if rank == 0:
        epoch_msg = f"[epoch {epoch}] avg_loss={avg_loss:.4f}"
        epoch_metrics = {"epoch_avg_loss": avg_loss, "epoch": epoch}
        if profile.is_moe:
            avg_aux = epoch_aux_loss / epoch_steps
            epoch_msg += f" avg_aux_loss={avg_aux:.4f}"
            epoch_metrics["epoch_avg_aux_loss"] = avg_aux
        logger.info(epoch_msg)
        if tracker:
            tracker.log(epoch_metrics, step=step)

    # ── Checkpoint ──────────────────────────────────────────────────
    if ckpt_cfg.enabled and (epoch + 1) % ckpt_cfg.save_interval == 0:
        _ckpt_cursor = DatasetCursor(
            samples_consumed=_step_samples_consumed,
            tokens_consumed=_step_tokens_consumed,
            global_step=step,
            epoch=epoch,
        )
        best_ckpts = save_and_prune(
            model=model,
            optimizer=optimizer,
            step=step,
            avg_loss=avg_loss,
            ckpt_cfg=ckpt_cfg,
            rank=rank,
            best_ckpts=best_ckpts,
            cursor_meta=cursor_to_checkpoint_metadata(_ckpt_cursor),
        )

    return step, best_ckpts, _step_samples_consumed, _step_tokens_consumed
