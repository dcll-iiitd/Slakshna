"""
bhaskera.evaluation.validation
===============================
Distributed validation loop.

Design notes:
    1. Chunked loss (see _chunked_causal_lm_loss): fused CE kernels (e.g.
       Liger) avoid materializing full [batch, seq_len, vocab] logits by
       computing loss in chunks internally, but that fusion is generally
       only active in training mode. In eval mode most implementations
       fall back to full logits + fp32 upcast, which can exceed GPU
       memory for large-vocab models regardless of batch size. We chunk
       the loss computation manually so this is bounded independent of
       vocab size, sequence length, or which kernel is active.

    2. Optimizer offload (see _maybe_offload_optimizer /
       _maybe_restore_optimizer): mirrors eval_lifecycle.py's memory
       reclamation — if an optimizer reference is passed in, its CUDA
       state is moved to CPU before validation runs and restored after,
       freeing GPU memory for the eval forward pass. This makes
       run_distributed_validation self-sufficient: it applies the same
       protection whether it's called from inside a full
       EvaluationLifecycle window (which already offloaded the optimizer,
       in which case this is a safe no-op — see _optimizer_offloaded
       check) or from a path that skips the lifecycle wrapper entirely
       (e.g. the no-Ray-shard fallback in loop.py, or a standalone script).
"""
import gc
import logging
import torch
import torch.distributed as dist
import torch.nn.functional as F
from typing import Optional

from bhaskera.evaluation.registry import get_metric
from bhaskera.trainer.eval_lifecycle import (
    _offload_optimizer_to_cpu,
    _restore_optimizer_to_gpu,
)

logger = logging.getLogger(__name__)

VALIDATION_BATCH_SIZE = 2
LOSS_CHUNK_SEQ_LEN = 512
MEMORY_CLEANUP_EVERY_N_BATCHES = 5


def _memory_snapshot(device: torch.device, rank: int, label: str) -> None:
    if not torch.cuda.is_available() or rank != 0:
        return
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    logger.info(
        f"[Validation][Memory] {label}: "
        f"allocated={allocated / 1024**2:.1f} MiB  reserved={reserved / 1024**2:.1f} MiB"
    )


def _maybe_offload_optimizer(optimizer, rank: int) -> bool:
    """
    Offload optimizer state to CPU if an optimizer was provided and it
    isn't already offloaded (e.g. by an enclosing EvaluationLifecycle).

    Returns True if this call performed the offload (and is therefore
    responsible for restoring it), False otherwise.
    """
    if optimizer is None:
        return False
    already_offloaded = getattr(optimizer, "_bhaskera_offloaded", False)
    if already_offloaded:
        return False
    if rank == 0:
        logger.info("[Validation] Offloading optimizer state to CPU")
    _offload_optimizer_to_cpu(optimizer)
    optimizer._bhaskera_offloaded = True
    return True


def _maybe_restore_optimizer(optimizer, device: torch.device, rank: int, owns_offload: bool) -> None:
    """Restore optimizer state to GPU only if this call was the one that offloaded it."""
    if optimizer is None or not owns_offload:
        return
    if rank == 0:
        logger.info("[Validation] Restoring optimizer state to GPU")
    _restore_optimizer_to_gpu(optimizer, device)
    optimizer._bhaskera_offloaded = False


def _chunked_causal_lm_loss(logits_fn, labels: torch.Tensor, chunk_seq_len: int) -> torch.Tensor:
    """Compute causal LM loss in sequence chunks to bound peak memory."""
    seq_len = labels.size(1)
    total_loss = torch.zeros((), device=labels.device, dtype=torch.float64)
    total_count = torch.zeros((), device=labels.device, dtype=torch.float64)

    for start in range(0, seq_len, chunk_seq_len):
        end = min(start + chunk_seq_len, seq_len)
        logits_chunk = logits_fn(start, end).float()
        labels_chunk = labels[:, start:end]

        loss_chunk = F.cross_entropy(
            logits_chunk.reshape(-1, logits_chunk.size(-1)),
            labels_chunk.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        count_chunk = (labels_chunk != -100).sum()

        total_loss += loss_chunk.double()
        total_count += count_chunk.double()

        del logits_chunk, loss_chunk

    return (total_loss / total_count.clamp(min=1)).float()


def run_distributed_validation(
    cfg,
    model,
    val_dataset,
    profile,
    rank: int,
    world_size: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict:
    """
    Runs validation distributed, aggregates metrics, broadcasts results.

    Args:
        optimizer: Optional live optimizer. If provided and not already
                   offloaded by an enclosing EvaluationLifecycle, its CUDA
                   state is offloaded to CPU before validation and restored
                   after — freeing GPU memory for the eval forward pass.
                   Pass None (default) if the caller already guarantees
                   offload elsewhere, or has no optimizer reference handy.
    """
    if not val_dataset:
        logger.warning("Validation enabled but val_dataset is None. Skipping.")
        return {}

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    was_training = model.training
    model.eval()

    metric_names = list(cfg.evaluation.validation.metrics or [])
    active_metrics = []
    for m_name in metric_names:
        m_cls = get_metric(m_name)
        if m_cls:
            active_metrics.append(m_cls())
        else:
            logger.warning(f"Metric '{m_name}' not found.")

    needs_preds = any(m_name in ["token_accuracy"] for m_name in metric_names)

    local_losses: list[float] = []
    local_preds: list[torch.Tensor] = []
    local_labels: list[torch.Tensor] = []

    _memory_snapshot(device, rank, "pre-validation")

    # ── Free GPU memory before running eval ─────────────────────────
    owns_offload = _maybe_offload_optimizer(optimizer, rank)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _memory_snapshot(device, rank, "post-teardown")

    loader = (
        val_dataset.iter_torch_batches(
            batch_size=VALIDATION_BATCH_SIZE,
            dtypes={"input_ids": torch.long, "attention_mask": torch.long, "labels": torch.long},
            device=device,
        )
        if hasattr(val_dataset, "iter_torch_batches")
        else val_dataset
    )

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                forward_kwargs = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                    "use_cache": False,
                    "output_hidden_states": False,
                }
                labels = batch["labels"]

                out = model(**forward_kwargs)
                full_logits = out.logits

                # ── SHIFT FOR NEXT-TOKEN PREDICTION ─────────────────────────
                shift_logits = full_logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                loss = _chunked_causal_lm_loss(
                    logits_fn=lambda s, e: shift_logits[:, s:e, :],
                    labels=shift_labels,
                    chunk_seq_len=LOSS_CHUNK_SEQ_LEN,
                )
                local_losses.append(loss.item())

                if needs_preds:
                    local_preds.append(shift_logits.argmax(dim=-1).cpu())
                    local_labels.append(shift_labels.cpu())

                del out, full_logits, shift_logits, loss

                if (batch_idx + 1) % MEMORY_CLEANUP_EVERY_N_BATCHES == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    _memory_snapshot(device, rank, f"post-batch-{batch_idx + 1}")

    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _memory_snapshot(device, rank, "post-validation-cleanup")

        # ── Restore optimizer after eval, before returning ──────────
        _maybe_restore_optimizer(optimizer, device, rank, owns_offload)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _memory_snapshot(device, rank, "post-optimizer-restore")

    if dist.is_available() and dist.is_initialized():
        sum_loss = torch.tensor(
            [sum(local_losses), len(local_losses)],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(sum_loss, op=dist.ReduceOp.SUM)
        global_loss_sum, global_count = sum_loss.tolist()
        global_losses = [global_loss_sum / max(1, global_count)] * int(global_count)
    else:
        global_losses = local_losses

    results = {}
    if rank == 0:
        for metric in active_metrics:
            results.update(metric.compute(local_preds, local_labels, global_losses))

    if dist.is_available() and dist.is_initialized():
        obj_list = [results]
        dist.broadcast_object_list(obj_list, src=0)
        results = obj_list[0]

    if was_training:
        model.train()

    return results
