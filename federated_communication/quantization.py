import torch

def quantize_symmetric_int8(values):
    values = values.detach().float().cpu()
    if not torch.isfinite(values).all():
        raise ValueError("cannot quantize NaN or Inf values")
    max_abs = values.abs().max().item() if values.numel() else 0.0
    scale = max_abs / 127.0 if max_abs else 1.0
    return torch.round(values / scale).clamp(-127, 127).to(torch.int8), float(scale)
