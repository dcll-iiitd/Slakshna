"""
bhaskera.trainer.packing
========================
Utilities for First-Fit Decreasing (FFD) sequence packing,
including 4D block-diagonal mask generation and Flash Attention 2 helpers.
"""
import torch

def build_4d_attention_mask(seq_idx: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """
    Builds a 4D block-diagonal causal attention mask from sequence IDs.

    Returns:
        Tensor of shape (batch_size, 1, seq_len, seq_len) with 0.0 for valid
        attention and -inf for masked attention.
    """
    batch_size, seq_len = seq_idx.shape
    device = seq_idx.device

    # 1. Base Causal Mask: (seq_len, seq_len)
    causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))

    # 2. Document Boundaries: (batch_size, seq_len, seq_len)
    seq_idx_exp = seq_idx.unsqueeze(2)   # (batch_size, seq_len, 1)
    seq_idx_trans = seq_idx.unsqueeze(1) # (batch_size, 1, seq_len)
    doc_mask = (seq_idx_exp == seq_idx_trans) & (seq_idx_exp != 0)

    # 3. Combine masks
    valid_mask = doc_mask & causal_mask.unsqueeze(0)

    # 4. Convert to HF Additive Mask
    attn_mask = torch.zeros((batch_size, 1, seq_len, seq_len), dtype=dtype, device=device)
    attn_mask.masked_fill_(~valid_mask.unsqueeze(1), torch.finfo(dtype).min)

    return attn_mask


def prepare_flash_attention_varlen(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    position_ids: torch.Tensor,
    seq_idx: torch.Tensor
):
    """
    Optional helper: Flattens a packed batch and computes cu_seqlens
    for direct usage with flash_attn_varlen_func in custom architectures.
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    valid_mask = (seq_idx != 0)

    flat_input_ids = input_ids[valid_mask]
    flat_labels = labels[valid_mask]
    flat_position_ids = position_ids[valid_mask]

    flat_seq_idx = seq_idx[valid_mask]
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, seq_len)
    flat_batch_indices = batch_indices[valid_mask]

    # Boundary triggering
    boundaries = torch.zeros(flat_seq_idx.size(0), dtype=torch.bool, device=device)
    boundaries[0] = True
    boundaries[1:] = (flat_batch_indices[1:] != flat_batch_indices[:-1]) | \
                     (flat_seq_idx[1:] != flat_seq_idx[:-1])

    boundary_indices = torch.nonzero(boundaries).squeeze(-1).to(torch.int32)
    total_tokens = torch.tensor([flat_input_ids.size(0)], dtype=torch.int32, device=device)
    cu_seqlens = torch.cat([boundary_indices, total_tokens])

    max_seqlen_in_batch = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

    return flat_input_ids, flat_labels, flat_position_ids, cu_seqlens, max_seqlen_in_batch
