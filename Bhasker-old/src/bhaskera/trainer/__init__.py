
"""
bhaskera.trainer
================
Pure training loop.  No distributed init, no Ray, no SLURM logic — all of
that lives upstream in launcher/.
 
Public API:
    train(model, dataset, cfg, profile, rank, local_rank, tracker) -> None
    EvalTriggerPolicy
    DatasetCursor
    evaluation_window
"""
from __future__ import annotations
 
from .loop import train
from .eval_lifecycle import (       # ← ADD THESE FOUR LINES
    DatasetCursor,
    EvalTriggerPolicy,
    evaluation_window,
)
 
__all__ = [
    "train",
    "DatasetCursor",
    "EvalTriggerPolicy",
    "evaluation_window",
]
