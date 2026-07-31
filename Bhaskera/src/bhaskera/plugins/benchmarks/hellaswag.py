import torch
import torch.distributed as dist
import logging
from tqdm import tqdm
from bhaskera.evaluation.registry import register_benchmark

logger = logging.getLogger(__name__)

def chunk_iterable(iterable, batch_size):
    """Yield successive n-sized chunks from iterable."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

@register_benchmark("hellaswag")
class HellaSwagBenchmark:
    def __init__(self):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")
        self.load_dataset = load_dataset
        self.batch_size = 16  # Process 16 questions (64 choices) at once

    def run(self, model, tokenizer, cfg) -> dict:
        if tokenizer is None:
            logger.warning("HellaSwag requires a tokenizer. Skipping.")
            return {}

        # Ensure right-padding for proper indexing math
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        ds = self.load_dataset("Rowan/hellaswag", split="validation")
        local_ds = ds.shard(num_shards=world_size, index=rank)
        
        local_correct = 0
        local_total = len(local_ds)

        if rank == 0:
            logger.info(f"Running Batched HellaSwag on {local_total} examples...")

        device = model.device
        
        # Disable gradient and standard dropout
        with torch.no_grad():
            for batch in tqdm(chunk_iterable(local_ds, self.batch_size), 
                              total=(local_total + self.batch_size - 1) // self.batch_size, 
                              disable=(rank != 0), desc="HellaSwag Eval"):
                
                flat_texts = []
                flat_ctx_lens = []
                labels = []
                slices = [] # Keep track of which flat texts belong to which batch item
                
                current_idx = 0
                for item in batch:
                    ctx = item["ctx"]
                    endings = item["endings"]
                    labels.append(int(item["label"]))
                    
                    num_choices = len(endings)
                    slices.append((current_idx, current_idx + num_choices))
                    current_idx += num_choices
                    
                    for ending in endings:
                        full_text = f"{ctx} {ending}"
                        # Calculate length of context to know where the ending starts
                        ctx_len = len(tokenizer(ctx, add_special_tokens=True).input_ids)
                        flat_texts.append(full_text)
                        flat_ctx_lens.append(ctx_len)
                
                # Tokenize the entire batch of choices with padding
                encodings = tokenizer(flat_texts, padding=True, return_tensors="pt", add_special_tokens=True)
                input_ids = encodings.input_ids.to(device)
                attention_mask = encodings.attention_mask.to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Shift logits and labels for next-token prediction
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                shift_mask = attention_mask[:, 1:].contiguous()
                
                # Calculate log probabilities
                log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                gathered_log_probs = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
                
                # Create a mask to zero-out padding tokens AND context tokens
                seq_len = shift_labels.size(1)
                token_indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(len(flat_texts), -1)
                ctx_lengths_tensor = torch.tensor(flat_ctx_lens, device=device).unsqueeze(1)
                
                # tokens indices >= (ctx_len - 1) belong to the ending
                is_ending_mask = (token_indices >= (ctx_lengths_tensor - 1)).float()
                final_mask = shift_mask.float() * is_ending_mask
                
                # Sum log probs for the ending and normalize by length
                ending_log_probs = (gathered_log_probs * final_mask).sum(dim=1)
                ending_lengths = final_mask.sum(dim=1)
                scores = ending_log_probs / torch.clamp(ending_lengths, min=1.0)
                
                # Group scores back by question and calculate accuracy
                for i, (start, end) in enumerate(slices):
                    item_scores = scores[start:end]
                    prediction = torch.argmax(item_scores).item()
                    if prediction == labels[i]:
                        local_correct += 1

        # Distributed Aggregation
        if dist.is_initialized():
            local_stats = {"correct": local_correct, "total": local_total}
            gathered_stats = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_stats, local_stats)
            
            if rank == 0:
                global_correct = sum(stat["correct"] for stat in gathered_stats)
                global_total = sum(stat["total"] for stat in gathered_stats)
                accuracy = global_correct / global_total if global_total > 0 else 0.0
                return {"benchmark/hellaswag_accuracy": accuracy}
        else:
            accuracy = local_correct / local_total if local_total > 0 else 0.0
            return {"benchmark/hellaswag_accuracy": accuracy}
            
        return {}
