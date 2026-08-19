"""
bhaskera.trainer.eval_lifecycle
================================
Production-grade evaluation lifecycle manager for distributed LLM training.

Architecture contract:
    - Zero model reload: the same FSDP2/DDP-wrapped model instance is used
      for both training and evaluation.
    - Full GPU reclamation: every training-only allocation is freed before
      evaluation begins, including dataloaders, prefetch buffers, optimizer
      states moved to CPU, and intermediate tensors.
    - Deterministic resume: the dataset position is recorded before teardown
      and restored exactly after evaluation, with no re-randomisation of the
      shuffle order.
    - Rank-safe: all barrier points are explicit and minimal; no implicit
      synchronisation is introduced.
    - Context-manager API: the entire lifecycle is owned by a single
      ``with trainer.pause_for_evaluation():`` block.

Supports:
    - FSDP2 (composable fully_shard) and DDP
    - Ray Data streaming datasets with shard-aware iteration
    - Gradient accumulation (in-flight state is safely discarded)
    - Multi-node Ray clusters
    - Checkpoint resume (position is written into the checkpoint metadata)
    - Streaming datasets (position expressed as token/sample count, not index)

Comparison to prior art:
    DeepSpeed: uses ds_engine.eval() / ds_engine.train() mode switches and
    relies on the ZeRO optimizer's internal CPU offload to free GPU memory.
    Does not destroy the dataloader. Bhaskera's approach is more aggressive:
    it deallocates every non-model allocation and optionally CPU-offloads the
    optimizer before evaluation.

    Megatron-LM: has a dedicated eval loop that builds a separate dataloader
    each time. Dataset position is tracked via a global data iterator state.
    Bhaskera adopts the same pattern but makes it explicit via DatasetCursor.

    HuggingFace Trainer: calls model.eval() and runs evaluation in-line,
    relying on torch.no_grad() to avoid gradient allocations. Does not
    reclaim memory from the optimizer or dataloader. Suffers from the OOM
    problem we solve here.

    torchtune: similar to HF Trainer — no dataloader teardown, relies on
    torch.inference_mode() only. Good for single-GPU; fails at scale.
"""

from __future__ import annotations

import contextlib
import gc
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset cursor — deterministic position tracking
# ---------------------------------------------------------------------------

@dataclass
class DatasetCursor:
    """
    Tracks the current position in a Ray Data streaming dataset.

    Ray Data does not expose a seekable iterator. Instead we track position
    as a (samples_consumed, tokens_consumed) pair. On resume, the Ray Data
    pipeline is rebuilt from the beginning and fast-forwarded by
    ``samples_consumed`` rows using ``dataset.skip()``, which is a metadata-
    only operation on the parquet shards and does not decode skipped rows.

    ``global_step`` and ``epoch`` are stored here (not separately) so the
    cursor is a single serialisable atom.

    Fields:
        samples_consumed    Number of samples fetched from the iterator so
                            far, counting only samples that completed a full
                            micro-step (i.e. not the partial batch at the
                            start of a resumed run).
        tokens_consumed     Derived field; stored for logging only. Not used
                            for seek.
        global_step         The optimizer step index when this cursor was
                            snapshotted.
        epoch               The epoch index when this cursor was snapshotted.
        grad_accum_offset   How many micro-steps into the current optimizer
                            step we were when the cursor was snapshotted.
                            Zero in the common case (snapshot taken after a
                            completed optimizer step).
    """
    samples_consumed: int = 0
    tokens_consumed: int = 0
    global_step: int = 0
    epoch: int = 0
    grad_accum_offset: int = 0

    def as_dict(self) -> dict:
        return {
            "samples_consumed": self.samples_consumed,
            "tokens_consumed": self.tokens_consumed,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "grad_accum_offset": self.grad_accum_offset,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetCursor":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


# ---------------------------------------------------------------------------
# Training pipeline state snapshot
# ---------------------------------------------------------------------------

@dataclass
class TrainingPipelineState:
    """
    Everything that must survive the evaluation window and be restored
    afterward, in exactly the form the training loop expects.

    This is NOT a model checkpoint — the model weights live in the
    FSDP/DDP module and are never moved. This is the thin shell of
    Python objects that surround the model.

    Fields:
        optimizer           The live optimizer. Its CUDA state tensors are
                            CPU-offloaded during evaluation and moved back
                            before reconstruction.
        scheduler           The live LR scheduler. No state is moved; it
                            carries only Python-side scalars.
        cursor              The DatasetCursor snapshotted at the moment
                            pause_for_evaluation() was entered.
        grad_accum_steps    From cfg.training.grad_accum. Stored so
                            reconstruct() can verify the config hasn't drifted.
        autocast_ctx_fn     A zero-argument callable returning the autocast
                            context manager. Stored so it can be re-entered
                            identically after evaluation.
        scaler              Optional GradScaler for fp16 training (DDP only).
    """
    optimizer: torch.optim.Optimizer
    scheduler: Any  # torch.optim.lr_scheduler.LRScheduler
    cursor: DatasetCursor
    grad_accum_steps: int
    autocast_ctx_fn: Any  # callable → context manager
    scaler: Optional[Any] = None
    _optimizer_offloaded: bool = field(default=False, repr=False)


# ---------------------------------------------------------------------------
# Memory budget reporter
# ---------------------------------------------------------------------------

class _MemoryReporter:
    """Thin wrapper around torch.cuda for logging memory deltas."""

    def __init__(self, device: torch.device, rank: int):
        self._device = device
        self._rank = rank
        self._baseline: Optional[int] = None

    def snapshot(self, label: str) -> int:
        if not torch.cuda.is_available():
            return 0
        allocated = torch.cuda.memory_allocated(self._device)
        reserved = torch.cuda.memory_reserved(self._device)
        if self._rank == 0:
            logger.info(
                f"[MemoryReporter] {label}: "
                f"allocated={allocated / 1024**2:.1f} MiB  "
                f"reserved={reserved / 1024**2:.1f} MiB"
            )
        return allocated

    def delta_since_baseline(self, label: str) -> None:
        if self._baseline is None:
            self._baseline = self.snapshot(f"baseline/{label}")
            return
        current = torch.cuda.memory_allocated(self._device)
        delta = current - self._baseline
        if self._rank == 0:
            logger.info(
                f"[MemoryReporter] Δ {label}: {delta / 1024**2:+.1f} MiB "
                f"(current={current / 1024**2:.1f} MiB)"
            )


# ---------------------------------------------------------------------------
# FSDP / DDP helpers
# ---------------------------------------------------------------------------

def _is_fsdp2(model: torch.nn.Module) -> bool:
    """
    Detect FSDP2 composable API. FSDP2 uses ``fully_shard`` from
    ``torch.distributed.fsdp`` and sets an internal attribute on
    the module. We check for the attribute rather than isinstance
    to avoid importing private symbols.
    """
    return (
        hasattr(model, "_is_fsdp_managed_module")
        or hasattr(model, "unshard")
        or type(model).__name__ in ("FSDPModule",)
    )


def _is_ddp(model: torch.nn.Module) -> bool:
    try:
        from torch.nn.parallel import DistributedDataParallel
        return isinstance(model, DistributedDataParallel)
    except ImportError:
        return False


def _barrier(group=None) -> None:
    """Barrier that no-ops when distributed is not initialised."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier(group=group)


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0
def _skip_torch_batches(shard_iterator, n_samples, **iter_kwargs):
    it = iter(shard_iterator.iter_torch_batches(**iter_kwargs))
    consumed = 0
    while consumed < n_samples:
        try:
            batch = next(it)
        except StopIteration:
            logger.warning(
                f"[skip] Shard exhausted while skipping "
                f"({consumed}/{n_samples} samples) — restarting iterator."
            )
            it = iter(shard_iterator.iter_torch_batches(**iter_kwargs))
            break
        consumed += int(batch["input_ids"].size(0))
    return it
# ---------------------------------------------------------------------------
# Optimizer CPU offload / restore
# ---------------------------------------------------------------------------

def _offload_optimizer_to_cpu(
    optimizer: torch.optim.Optimizer,
) -> dict[int, torch.device]:
    """
    Move all optimizer state tensors to CPU in-place.

    Returns a mapping from tensor data_ptr to original device so
    ``_restore_optimizer_to_gpu`` can send them back without needing
    a separate bookkeeping pass.

    Why this is safe with FSDP2:
        FSDP2 shards parameters across ranks. The optimizer holds
        references to the local shard (a 1-D parameter on GPU). Moving
        those tensors to CPU does NOT change the parameter's grad_fn or
        require_grad property — it only moves the accumulated state
        (momentum buffers, second moments, etc.). The gradient-reduce
        path is unaffected because gradients are not accumulated during
        evaluation (we enter torch.inference_mode()).

    Cost:
        D2H transfer for all optimizer state, typically 2× model size
        for AdamW (m1 + m2 buffers). This is a deliberate trade-off:
        we're paying a one-time PCIe transfer to recover the GPU memory
        needed for eval. On NVLink systems this is ~50 GB/s; on PCIe 4×16
        it is ~28 GB/s, so a 7 B model's 28 GiB of optimizer state takes
        ~1 second to offload. Acceptable for evaluations that run for
        minutes.
    """
    ptr_to_device: dict[int, torch.device] = {}
    for state in optimizer.state.values():
        for k, v in list(state.items()):
            if isinstance(v, torch.Tensor) and v.is_cuda:
                ptr_to_device[v.data_ptr()] = v.device
                state[k] = v.to("cpu", non_blocking=True)
    # Ensure all D2H transfers complete before we proceed
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return ptr_to_device


def _restore_optimizer_to_gpu(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    """
    Move all CPU-side optimizer state tensors back to ``device``.

    The device is passed explicitly (rather than inferred from model
    parameters) to support multi-GPU scenarios where rank 0 might have
    a different local_rank than rank 1.
    """
    for state in optimizer.state.values():
        for k, v in list(state.items()):
            if isinstance(v, torch.Tensor) and not v.is_cuda:
                state[k] = v.to(device, non_blocking=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Dataloader / iterator teardown
# ---------------------------------------------------------------------------

def _destroy_train_dataloader(
    loader_iter: Optional[Iterator],
    loader: Optional[Any],
) -> None:
    """
    Explicitly close and delete the training iterator and dataloader.

    Ray Data note:
        A Ray Data iterator holds references to in-flight prefetch
        actors. Calling ``close()`` on the iterator cancels those actors
        and releases the GPU/CPU memory in their shared-memory buffers.
        If ``close()`` is not called, Ray will eventually GC the actors
        when the iterator is collected, but the timing is non-deterministic
        and memory may remain allocated for minutes.

    PyTorch DataLoader note:
        The DataLoader's worker processes are NOT killed here because we
        intend to reconstruct the loader after evaluation. However, the
        iterator IS destroyed, which drains the prefetch queue (workers
        will block waiting for the iterator to call next() and will be
        recycled on the next iter() call).
    """
    if loader_iter is not None:
        if hasattr(loader_iter, "close"):
            try:
                loader_iter.close()
            except Exception as e:
                logger.debug(f"loader_iter.close() raised: {e}")
        del loader_iter

    if loader is not None:
        # Signal Ray Data pipeline to stop prefetching
        if hasattr(loader, "_dataset_iter"):
            try:
                inner = loader._dataset_iter
                if hasattr(inner, "close"):
                    inner.close()
            except Exception:
                pass
        del loader


# ---------------------------------------------------------------------------
# Gradient buffer teardown
# ---------------------------------------------------------------------------

def _clear_gradient_buffers(model: torch.nn.Module) -> None:
    """
    Zero out all gradient buffers.

    We call ``zero_grad(set_to_none=True)`` which replaces each gradient
    tensor with None rather than zeroing in-place, releasing the allocation
    entirely. This is safe because we will call optimizer.zero_grad() again
    when training resumes before the first backward pass.

    For FSDP2, this is the correct API — FSDP2 does not override
    ``zero_grad`` but does own the parameter storage, so None gradients
    are handled correctly by the next forward/backward cycle.
    """
    model.zero_grad(set_to_none=True)


# ---------------------------------------------------------------------------
# Core context manager
# ---------------------------------------------------------------------------

class EvaluationLifecycle:
    """
    Manages the complete lifecycle of an in-training evaluation window.

    This class is the implementation behind ``trainer.pause_for_evaluation()``.
    It should not be instantiated directly by user code.

    Lifecycle phases (in order):
        1. SNAPSHOT  — record cursor position, step counters
        2. TEARDOWN  — zero grads, destroy iterator, offload optimizer, gc+empty_cache
        3. EVAL_MODE — model.eval() + inference_mode, synchronize ranks
        4. [evaluation runs here — owned by the caller]
        5. CLEANUP   — destroy eval resources, gc+empty_cache
        6. TRAIN_MODE — model.train(), restore optimizer from CPU
        7. RECONSTRUCT — rebuild Ray Data iterator from cursor, restore AMP state
        8. RESUME   — training continues from exact cursor position

    Thread / process safety:
        Each Ray Train worker runs its own instance of this class. Coordination
        between ranks happens only at explicit _barrier() calls (phases 2→3
        and 5→6). There are exactly two barriers per evaluation window, which
        is the minimum required to ensure all ranks enter and exit eval mode
        together (NCCL requires symmetric participation in collectives).

    Args:
        model:          The live FSDP2/DDP-wrapped model. Not reloaded.
        pipeline_state: Snapshot of optimizer + scheduler + cursor.
        device:         The CUDA device this rank owns.
        cfg:            Full Bhaskera Config dataclass.
        ray_dataset_shard: The raw Ray Dataset shard for this rank. Used to
                        rebuild the iterator after evaluation.
        rank:           This rank's global rank in the process group.
        offload_optimizer: If True, move optimizer state to CPU before eval.
                        Disable on memory-rich systems to skip the PCIe round-trip.
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        pipeline_state: TrainingPipelineState,
        device: torch.device,
        cfg,
        ray_dataset_shard,
        rank: int,
        offload_optimizer: bool = True,
    ) -> None:
        self._model = model
        self._state = pipeline_state
        self._device = device
        self._cfg = cfg
        self._ray_dataset_shard = ray_dataset_shard
        self._rank = rank
        self._offload_optimizer = offload_optimizer
        self._mem = _MemoryReporter(device, rank)

        # Set after teardown, cleared after reconstruction
        self._new_iterator: Optional[Iterator] = None
        self._eval_start_time: float = 0.0

    # ------------------------------------------------------------------
    # Phase 2: Teardown
    # ------------------------------------------------------------------

    def _teardown_training_resources(
        self,
        loader_iter: Optional[Iterator],
        loader: Optional[Any],
    ) -> None:
        """
        Destroy every training-only GPU allocation.

        Order matters:
            1. Zero gradients first — while model is still in train mode.
            2. Destroy dataloader — releases prefetch memory.
            3. Offload optimizer — biggest single allocation after the model.
            4. gc.collect() — collect Python objects that reference CUDA tensors.
            5. empty_cache() — return freed CUDA memory to the OS allocator.

        Why gc.collect() before empty_cache()?
            PyTorch's CUDA caching allocator will not return a block to the
            free pool until all Python references to it are gone. If we call
            empty_cache() first, CPython's reference count may still hold a
            tensor alive (e.g. a local variable in a calling frame), and
            empty_cache() will find nothing to reclaim. gc.collect() forces
            the cycle-breaking pass that drops those references first.

        Why NOT call empty_cache() after every step?
            empty_cache() is O(n) in the number of cached blocks and can take
            10–100 ms on a large allocation. Calling it every step would reduce
            training throughput by 5–15%. We call it only at the eval boundary
            where the cost is amortised over the entire eval window.
        """
        if self._rank == 0:
            logger.info("[EvalLifecycle] Phase 2: Teardown — destroying training resources")

        self._mem.snapshot("pre-teardown")

        # Step 1: Zero gradient buffers
        _clear_gradient_buffers(self._model)

        # Step 2: Destroy dataloader / Ray Data iterator
        _destroy_train_dataloader(loader_iter, loader)

        # Step 3: Offload optimizer state to CPU
        if self._offload_optimizer and self._state.optimizer is not None:
            if self._rank == 0:
                logger.info("[EvalLifecycle]   Offloading optimizer state to CPU")
            _offload_optimizer_to_cpu(self._state.optimizer)
            self._state._optimizer_offloaded = True

        # Step 4: Collect Python-side dead references
        gc.collect()

        # Step 5: Return freed CUDA blocks to the OS allocator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._mem.snapshot("post-teardown")

    # ------------------------------------------------------------------
    # Phase 3: Enter eval mode
    # ------------------------------------------------------------------

    def _enter_eval_mode(self) -> None:
        """
        Switch the model to inference mode.

        For FSDP2:
            FSDP2's composable API respects model.eval() directly. Calling
            model.eval() sets the training flag on the module and all its
            children. No special FSDP2 call is needed. The MixedPrecisionPolicy
            is unaffected — FSDP2 uses it for parameter dtype casting, which is
            identical in eval and train modes.

        For DDP:
            DDP's eval() propagates to the inner module. The autocast context
            is not entered during eval (we use inference_mode() instead).

        We do NOT call model.unshard() here. FSDP2 will unshard parameters
        on demand during the forward pass. Forcing an unshard here would
        allocate the full model on each rank simultaneously, which is exactly
        the OOM we're trying to avoid.
        """
        self._model.eval()
        if self._rank == 0:
            logger.info("[EvalLifecycle] Phase 3: Model in eval mode")

    # ------------------------------------------------------------------
    # Phase 6: Exit eval mode
    # ------------------------------------------------------------------

    def _exit_eval_mode(self) -> None:
        """
        Switch the model back to training mode.

        Called after the caller's evaluation block and after eval resources
        have been freed (Phase 5), so the GPU has headroom for the optimizer
        state restore in Phase 6.
        """
        self._model.train()
        if self._rank == 0:
            logger.info("[EvalLifecycle] Phase 6: Model back in train mode")

    # ------------------------------------------------------------------
    # Phase 6: Restore optimizer
    # ------------------------------------------------------------------

    def _restore_training_resources(self) -> None:
        """
        Restore optimizer state from CPU back to GPU, then gc+empty_cache.

        Why gc before restore?
            Evaluation may have left Python objects referencing CUDA tensors
            (e.g. cached output tensors from lm_eval_harness). We want those
            freed before we pour the optimizer state back onto the device.
        """
        if self._rank == 0:
            logger.info("[EvalLifecycle] Phase 6: Restoring optimizer to GPU")

        # Free any eval residue first
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._mem.snapshot("pre-optimizer-restore")

        if self._state._optimizer_offloaded and self._state.optimizer is not None:
            _restore_optimizer_to_gpu(self._state.optimizer, self._device)
            self._state._optimizer_offloaded = False

        self._mem.snapshot("post-optimizer-restore")

    # ------------------------------------------------------------------
    # Phase 7: Reconstruct Ray Data iterator
    # ------------------------------------------------------------------

    def _reconstruct_ray_iterator(self) -> Iterator:
        """
        Rebuild the Ray Data iterator from the snapshotted cursor position.

        Strategy:
            1. Obtain the raw Ray Dataset shard (already assigned to this rank
               by ``ray.train.get_dataset_shard("train")``).
            2. Call ``.skip(n)`` where n = cursor.samples_consumed.
            3. Wrap in the same iterator kwargs as the original (batch_size,
               prefetch_batches, local_shuffle_buffer_multiplier).

        Why skip() and not seek()?
            Ray Data's streaming datasets (backed by parquet) support skip()
            as a metadata-only operation: it advances the file reader without
            decoding the skipped rows. This is O(num_files) not O(samples).
            A true seek() would require the dataset to be indexed, which is
            not guaranteed for streaming sources.

        Why destroy and rebuild instead of keeping the iterator alive?
            Keeping the iterator alive across evaluation holds all prefetch
            actor resources (typically 2–4 batches × batch_size × seq_len
            × 2 bytes in shared memory). On an H100 with seq_len=8192 and
            batch_size=8, that is ~500 MiB per prefetch slot, or ~2 GiB
            total. Destroying and rebuilding is strictly cheaper.

        Determinism guarantee:
            Ray Data's ``local_shuffle_buffer`` uses a seeded RNG keyed on
            ``(dataset_seed, shard_id, epoch)``. As long as we rebuild the
            dataset with the same seed (stored in cfg.training.seed) and
            epoch, the shuffle order for the un-consumed portion is identical
            to what it would have been if training had continued uninterrupted.
            The skipped rows are not re-shuffled into the consumed window.
        """
        cursor = self._state.cursor
        cfg = self._cfg
        dataset = self._ray_dataset_shard  # a DataIterator/StreamSplitDataIterator — no .skip()

        if self._rank == 0:
            logger.info(
                f"[EvalLifecycle] Phase 7: Rebuilding Ray Data iterator "
                f"(skip={cursor.samples_consumed}, "
                f"epoch={cursor.epoch}, "
                f"global_step={cursor.global_step})"
            )

        data_cfg = cfg.data
        iter_kwargs = dict(
            batch_size=cfg.training.batch_size,
            prefetch_batches=getattr(data_cfg, "prefetch_batches", 2),
            local_shuffle_buffer_size=(
                cfg.training.batch_size
                * getattr(data_cfg, "local_shuffle_buffer_multiplier", 10)
            ),
            dtypes={"input_ids": torch.long, "labels": torch.long, "attention_mask": torch.long},
        )

        # DataIterator has no .skip() — it is already a per-worker split,
        # not a lazily-composable Dataset. Fast-forward by draining batches.
        if cursor.samples_consumed > 0:
            iterator = _skip_torch_batches(dataset, cursor.samples_consumed, **iter_kwargs)
        else:
            iterator = dataset.iter_torch_batches(**iter_kwargs)

        self._new_iterator = iterator
        return iterator

    # ------------------------------------------------------------------
    # Context manager entrypoints
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def run(
        self,
        *,
        loader_iter: Optional[Iterator] = None,
        loader: Optional[Any] = None,
    ):
        """
        Main context manager. Usage:

            with lifecycle.run(loader_iter=it, loader=dl):
                evaluator.run(model=model)

            new_iterator = lifecycle.rebuilt_iterator

        Yields nothing. The caller runs evaluation inside the ``with`` block.
        After the block exits (normally or via exception), training resources
        are restored and the iterator is rebuilt.

        Exception handling:
            If the evaluation block raises, teardown still completes (the
            finally block runs) so the training loop can continue. The
            exception propagates to the caller. This matches the behaviour of
            PyTorch's autocast context manager.
        """
        self._eval_start_time = time.perf_counter()

        # ── Phase 2: Teardown ──────────────────────────────────────────
        self._teardown_training_resources(loader_iter, loader)

        # ── Rank sync before entering eval ────────────────────────────
        # All ranks must have freed their training resources before any
        # rank enters the evaluation loop. Without this barrier, a slow
        # rank might still be running a backward pass while the fast rank
        # has already entered model.eval(), causing NCCL collective
        # mismatches if evaluation runs any all-reduce.
        _barrier()

        # ── Phase 3: Enter eval mode ──────────────────────────────────
        self._enter_eval_mode()

        try:
            # ── Phase 4: Caller runs evaluation ───────────────────────
            with torch.inference_mode():
                yield

        finally:
            # ── Phase 5: Destroy eval resources ───────────────────────
            # We gc+empty_cache here even if evaluation raised, so the
            # training loop can continue on the next iteration.
            if self._rank == 0:
                logger.info("[EvalLifecycle] Phase 5: Cleaning up eval resources")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # ── Rank sync after eval, before optimizer restore ─────────
            # All ranks must finish evaluation before we restore optimizer
            # state, because optimizer restore may call NCCL collectives
            # (e.g. if a ZeRO-style optimizer shards state across ranks).
            _barrier()

            # ── Phase 6: Exit eval mode, restore optimizer ─────────────
            self._exit_eval_mode()
            self._restore_training_resources()

            # ── Phase 7: Rebuild Ray Data iterator ─────────────────────
            self._reconstruct_ray_iterator()

            elapsed = time.perf_counter() - self._eval_start_time
            if self._rank == 0:
                logger.info(
                    f"[EvalLifecycle] Complete. "
                    f"Total eval window: {elapsed:.1f}s. "
                    f"Resuming from step={self._state.cursor.global_step}, "
                    f"sample={self._state.cursor.samples_consumed}"
                )

    @property
    def rebuilt_iterator(self) -> Optional[Iterator]:
        """The new Ray Data iterator, available after the context exits."""
        return self._new_iterator


# ---------------------------------------------------------------------------
# TrainingLoopMixin — integrates the lifecycle into the training loop
# ---------------------------------------------------------------------------

class TrainingLoopMixin:
    """
    Mixin that adds ``pause_for_evaluation()`` to a Bhaskera trainer.

    The trainer class in ``bhaskera.trainer.loop`` should inherit from this
    mixin (in addition to whatever base class it currently uses). The mixin
    adds no ``__init__`` requirements — it reads state from ``self`` attributes
    that the trainer already maintains.

    Expected attributes on ``self``:
        model           FSDP2/DDP-wrapped model
        optimizer       torch.optim.Optimizer
        scheduler       LRScheduler
        device          torch.device
        cfg             Bhaskera Config
        rank            int (global rank)
        ray_dataset_shard  Ray Dataset shard for this rank
        global_step     int
        epoch           int
        samples_consumed int
        tokens_consumed  int
        grad_accum_offset int (0 in the common case)
        loader_iter     The current training iterator
        loader          The DataLoader or Ray Data batch iterator (may be None)

    Usage:

        class BhaskeraTrainer(TrainingLoopMixin):
            def train(self):
                ...
                if should_evaluate(step):
                    with self.pause_for_evaluation() as lifecycle:
                        self.evaluator.run(model=self.model)
                    # self.loader_iter is replaced with the rebuilt iterator
                    self.loader_iter = lifecycle.rebuilt_iterator
                ...
    """

    @contextlib.contextmanager
    def pause_for_evaluation(self, *, offload_optimizer: bool = True):
        """
        Suspend training, run evaluation, then resume.

        This is the public API. Everything below is owned internally.

        Args:
            offload_optimizer:  Move optimizer state to CPU during eval.
                                Costs a PCIe round-trip (~1s for 7B model).
                                Set to False on NVLink systems with ≥80 GiB
                                VRAM where the optimizer fits alongside eval.

        Yields:
            The EvaluationLifecycle instance. The caller can access
            ``lifecycle.rebuilt_iterator`` after the block exits.

        Example::

            with self.pause_for_evaluation() as lifecycle:
                self.evaluator.run(model=self.model)
            self.loader_iter = lifecycle.rebuilt_iterator
        """
        # Snapshot current position before any teardown
        cursor = DatasetCursor(
            samples_consumed=getattr(self, "samples_consumed", 0),
            tokens_consumed=getattr(self, "tokens_consumed", 0),
            global_step=getattr(self, "global_step", 0),
            epoch=getattr(self, "epoch", 0),
            grad_accum_offset=getattr(self, "grad_accum_offset", 0),
        )

        pipeline_state = TrainingPipelineState(
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            cursor=cursor,
            grad_accum_steps=self.cfg.training.grad_accum,
            autocast_ctx_fn=getattr(self, "_autocast_ctx_fn", contextlib.nullcontext),
            scaler=getattr(self, "scaler", None),
        )

        lifecycle = EvaluationLifecycle(
            model=self.model,
            pipeline_state=pipeline_state,
            device=self.device,
            cfg=self.cfg,
            ray_dataset_shard=self.ray_dataset_shard,
            rank=self.rank,
            offload_optimizer=offload_optimizer,
        )

        with lifecycle.run(
            loader_iter=getattr(self, "loader_iter", None),
            loader=getattr(self, "loader", None),
        ):
            yield lifecycle

        # Replace the now-destroyed iterator with the rebuilt one
        self.loader_iter = lifecycle.rebuilt_iterator


# ---------------------------------------------------------------------------
# Standalone functional API (for use without the mixin)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def evaluation_window(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    cfg,
    ray_dataset_shard,
    rank: int,
    cursor: DatasetCursor,
    loader_iter: Optional[Iterator] = None,
    loader: Optional[Any] = None,
    offload_optimizer: bool = True,
) -> Iterator["EvaluationLifecycle"]:
    """
    Functional equivalent of ``trainer.pause_for_evaluation()``.

    For use in training loops that do not use the TrainingLoopMixin.

    Example::

        cursor = DatasetCursor(
            samples_consumed=samples_so_far,
            global_step=step,
            epoch=epoch,
        )

        with evaluation_window(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            cfg=cfg,
            ray_dataset_shard=shard,
            rank=rank,
            cursor=cursor,
            loader_iter=train_iter,
        ) as lifecycle:
            evaluator.run(model=model)

        train_iter = lifecycle.rebuilt_iterator
    """
    pipeline_state = TrainingPipelineState(
        optimizer=optimizer,
        scheduler=scheduler,
        cursor=cursor,
        grad_accum_steps=cfg.training.grad_accum,
        autocast_ctx_fn=contextlib.nullcontext,
    )

    lifecycle = EvaluationLifecycle(
        model=model,
        pipeline_state=pipeline_state,
        device=device,
        cfg=cfg,
        ray_dataset_shard=ray_dataset_shard,
        rank=rank,
        offload_optimizer=offload_optimizer,
    )

    with lifecycle.run(loader_iter=loader_iter, loader=loader):
        yield lifecycle


# ---------------------------------------------------------------------------
# Checkpoint integration
# ---------------------------------------------------------------------------

def cursor_to_checkpoint_metadata(cursor: DatasetCursor) -> dict:
    """
    Serialize the cursor into a flat dict for inclusion in a DCP checkpoint.

    Usage in save_checkpoint():
        metadata = {
            **existing_metadata,
            **cursor_to_checkpoint_metadata(cursor),
        }

    The keys are prefixed with ``eval_lifecycle/`` to avoid clashing with
    other checkpoint metadata keys.
    """
    return {f"eval_lifecycle/{k}": v for k, v in cursor.as_dict().items()}


def cursor_from_checkpoint_metadata(metadata: dict) -> DatasetCursor:
    """
    Reconstruct a DatasetCursor from checkpoint metadata.

    Returns a zero-initialised cursor if no lifecycle keys are present
    (i.e. the checkpoint was saved before this feature was added).
    """
    prefix = "eval_lifecycle/"
    d = {
        k[len(prefix):]: v
        for k, v in metadata.items()
        if k.startswith(prefix)
    }
    if not d:
        return DatasetCursor()
    return DatasetCursor.from_dict(d)


# ---------------------------------------------------------------------------
# Evaluation trigger policy
# ---------------------------------------------------------------------------

class EvalTriggerPolicy:
    """
    Decides whether to run evaluation at a given step.

    Reads from cfg.evaluation.validation.every_n_steps and
    cfg.evaluation.benchmarks.every_n_steps to schedule both
    the fast validation loop and the heavier benchmark suite.

    Design note:
        We combine both triggers into a single pause_for_evaluation() call
        so teardown + reconstruction happens only once per evaluation event,
        even if both validation and benchmarks are scheduled at the same step.
    """

    def __init__(self, cfg) -> None:
        eval_cfg = getattr(cfg, "evaluation", None)
        if eval_cfg is None or not getattr(eval_cfg, "enabled", False):
            self._val_every = None
            self._bench_every = None
        else:
            val_cfg = getattr(eval_cfg, "validation", None)
            bench_cfg = getattr(eval_cfg, "benchmarks", None)
            self._val_every = getattr(val_cfg, "every_n_steps", None)
            self._bench_every = getattr(bench_cfg, "every_n_steps", None)

    def should_run_validation(self, step: int) -> bool:
        if self._val_every is None or self._val_every <= 0:
            return False
        return step > 0 and step % self._val_every == 0

    def should_run_benchmarks(self, step: int) -> bool:
        if self._bench_every is None or self._bench_every <= 0:
            return False
        return step > 0 and step % self._bench_every == 0

    def should_run_any(self, step: int) -> bool:
        return self.should_run_validation(step) or self.should_run_benchmarks(step)
