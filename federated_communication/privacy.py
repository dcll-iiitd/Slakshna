import sys
import torch

def apply_differential_privacy_and_clipping(delta_dict, max_norm=1.0, noise_multiplier=0.01):
    """
    Applies L2 norm clipping and Gaussian noise to model deltas for Differential Privacy (DP-SGD).
    Prevents gradient inversion attacks and caps Byzantine norm anomalies.
    """
    if not delta_dict:
        return delta_dict

    total_norm_sq = 0.0
    for k, v in delta_dict.items():
        if torch.is_tensor(v) and v.numel() > 0:
            total_norm_sq += torch.sum(v.float() ** 2).item()
    total_norm = (total_norm_sq ** 0.5)

    clip_factor = max(1.0, total_norm / max_norm)
    dp_delta = {}
    for k, v in delta_dict.items():
        if torch.is_tensor(v):
            v_clipped = v / clip_factor
            if noise_multiplier > 0:
                noise = torch.randn_like(v_clipped, device=v_clipped.device) * (max_norm * noise_multiplier)
                v_clipped = v_clipped + noise
            dp_delta[k] = v_clipped
        else:
            dp_delta[k] = v

    print(f"[SECURITY] 🛡️ DP & Norm Clipping Applied: Norm={total_norm:.4f}, Max Allowed={max_norm:.4f}, Noise Mult={noise_multiplier}", file=sys.stderr)
    return dp_delta



def validate_peer_delta(delta_dict, max_allowed_norm=10.0):
    """
    Validates peer deltas to prevent NaN/Inf injection or extreme norm poisoning.
    Returns True if valid, False if rejected.
    """
    total_norm_sq = 0.0
    for k, v in delta_dict.items():
        if torch.is_tensor(v):
            if torch.isnan(v).any() or torch.isinf(v).any():
                print(f"[SECURITY] ⚠️ REJECTED PEER DELTA: Contains NaN or Inf values!", file=sys.stderr)
                return False
            total_norm_sq += torch.sum(v.float() ** 2).item()
    
    total_norm = total_norm_sq ** 0.5
    if total_norm > max_allowed_norm:
        print(f"[SECURITY] ⚠️ REJECTED PEER DELTA: Total Norm {total_norm:.2f} exceeds max threshold {max_allowed_norm}", file=sys.stderr)
        return False

    return True
