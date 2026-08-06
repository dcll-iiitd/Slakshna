import torch

def aggregate_deltas(available_deltas, w_i):
    """
    Multiply each received peer's delta by its local trust score (reputation), 
    then sum them together to form the global model update.
    """
    delta_agg = {}
    for j, d_j in available_deltas.items():
        weight = w_i.get(j, 0.0)
        for k in d_j:
            if k not in delta_agg:
                delta_agg[k] = torch.zeros_like(d_j[k])
            if (
                torch.is_tensor(d_j[k]) and d_j[k].shape == delta_agg[k].shape
            ):  # Safety check
                delta_agg[k] += weight * d_j[k]
    return delta_agg
