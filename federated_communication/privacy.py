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



def delta_norm(delta_dict):
    """L2 norm over every floating tensor in a delta."""
    total = 0.0
    for v in delta_dict.values():
        if torch.is_tensor(v):
            total += torch.sum(v.float() ** 2).item()
    return total ** 0.5


def validate_peer_delta(delta_dict, max_allowed_norm=10.0, reference_norm=None, max_ratio=None):
    """
    Validates peer deltas to prevent NaN/Inf injection or extreme norm poisoning.
    Returns True if valid, False if rejected.

    Besides the absolute limit, a delta is rejected when its norm exceeds ``max_ratio`` times
    ``reference_norm`` (normally this node's own delta). The relative check is skipped when the
    reference is zero, e.g. a round in which this node took no training step.
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

    if max_ratio and reference_norm and total_norm > max_ratio * reference_norm:
        print(f"[SECURITY] ⚠️ REJECTED PEER DELTA: Norm {total_norm:.2f} is more than {max_ratio}x "
              f"this node's own delta norm {reference_norm:.2f}", file=sys.stderr)
        return False

    return True
