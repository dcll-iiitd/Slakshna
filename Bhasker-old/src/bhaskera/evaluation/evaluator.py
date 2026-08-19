import logging
from typing import Dict, Any

from .validation import run_distributed_validation
from .benchmark_runner import run_benchmarks

logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, cfg, model, profile, rank: int, world_size: int):
        self.cfg = cfg
        self.model = model
        self.profile = profile
        self.rank = rank
        self.world_size = world_size
        self.eval_cfg = getattr(cfg, "evaluation", None)
        self._tokenizer = None  # Cache for lazy loading

    @property
    def tokenizer(self):
        """Lazy-loads the tokenizer only when the first benchmark runs."""
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
                trust_remote_code = getattr(self.cfg.model, "trust_remote_code", False)
                logger.info(f"Evaluator loading tokenizer for {self.cfg.model.name}...")
                
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.cfg.model.name, 
                    trust_remote_code=trust_remote_code
                )
                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
            except Exception as e:
                logger.error(f"Failed to load tokenizer for evaluation: {e}")
        return self._tokenizer

    def should_run_validation(self, step: int) -> bool:
        if not self.eval_cfg or not self.eval_cfg.enabled:
            return False
        every_n = self.eval_cfg.validation.every_n_steps
        return every_n > 0 and step > 0 and step % every_n == 0

    def should_run_benchmark(self, step: int) -> bool:
        if not self.eval_cfg or not self.eval_cfg.enabled:
            return False
        every_n = self.eval_cfg.benchmarks.every_n_steps
        return every_n > 0 and step > 0 and step % every_n == 0

    def run_validation(self, val_dataset,optimizer=None) -> Dict[str, Any]:
        """Runs validation distributed, aggregates metrics, broadcasts results."""
        if not val_dataset:
            logger.warning("Validation enabled but val_dataset is None. Skipping.")
            return {}
        logger.info("Running Validation...")
        return run_distributed_validation(
            self.cfg, self.model, val_dataset, self.profile, self.rank, self.world_size
        )

    def run_benchmarks(self, tokenizer=None) -> Dict[str, Any]:
        """Runs benchmarks symmetrically for FSDP, logs on rank 0, broadcasts."""
        logger.info("Running Benchmarks...")
        
        # Use provided tokenizer, or fallback to the lazy-loaded one
        tok = tokenizer or self.tokenizer
        
        return run_benchmarks(
            self.cfg, self.model, tok, self.profile, self.rank, self.world_size
        )
