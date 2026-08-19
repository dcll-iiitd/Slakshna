import logging
import torch
import torch.distributed as dist
from bhaskera.evaluation.registry import get_benchmark

logger = logging.getLogger(__name__)

def run_benchmarks(cfg, model, tokenizer, profile, rank: int, world_size: int) -> dict:
    was_training = model.training
    model.eval()
    results = {}
    
    with torch.no_grad():
        for task_name in cfg.evaluation.benchmarks.tasks:
            benchmark_cls = get_benchmark(task_name)
            if not benchmark_cls:
                logger.warning(f"Benchmark '{task_name}' not found.")
                continue
            
            benchmark = benchmark_cls()
            
            task_results = benchmark.run(model, tokenizer, cfg)
            
            if rank == 0 and task_results:
                results.update(task_results)

    if dist.is_available() and dist.is_initialized():
        obj_list = [results]
        dist.broadcast_object_list(obj_list, src=0)
        results = obj_list[0]

    if was_training:
        model.train()
        
    return results
