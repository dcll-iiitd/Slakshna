import io
import math
import base64
import torch

from .config import (
    DELTA_SPARSITY, DELTA_QUANTIZATION, QUANTIZATION, DELTA_FORMAT, 
    DELTA_VERSION, MAX_DELTA_PAYLOAD_BYTES, MAX_DELTA_TENSORS, 
    MAX_DELTA_TENSOR_ELEMENTS, ALLOW_LEGACY_DELTA_FORMAT
)
from .quantization import quantize_symmetric_int8


def topk_indices_and_values(tensor, sparsity=0.01):
    """Return deterministic flattened top-k indices and their float32 values."""
    if not tensor.is_floating_point():
        raise ValueError("only floating-point tensors may be encoded as deltas")
    flat = tensor.detach().contiguous().reshape(-1).float().cpu()
    if flat.numel() == 0:
        return torch.empty(0, dtype=torch.int64), flat
    if not 0.0 < sparsity <= 1.0:
        raise ValueError("delta sparsity must be in (0, 1]")
    k = max(1, int(math.floor(flat.numel() * sparsity)))
    # stable=True makes equal magnitudes deterministic: lower original index wins.
    indices = torch.argsort(flat.abs(), descending=True, stable=True)[:k]
    return indices, flat[indices]


def sparsify_tensor(tensor, sparsity=0.01):
    """Compatibility helper for the disabled-compression migration path."""
    indices, values = topk_indices_and_values(tensor, sparsity)
    dense = torch.zeros(tensor.numel(), dtype=tensor.dtype, device=tensor.device)
    dense.scatter_(0, indices.to(tensor.device), values.to(tensor.device, tensor.dtype))
    return dense.reshape(tensor.shape)


def encode_delta_envelope(delta_dict, sparsity=DELTA_SPARSITY, sender=None, round_number=None):
    """Encode sparse int8 records and return (base64 payload, reconstructed delta, metrics)."""
    if DELTA_QUANTIZATION != QUANTIZATION:
        raise ValueError(f"unsupported delta quantization: {DELTA_QUANTIZATION}")
    tensors, reconstructed = {}, {}
    selected_count = index_bytes = value_bytes = dense_bytes = 0
    for name, tensor in delta_dict.items():
        if not torch.is_tensor(tensor) or not tensor.is_floating_point():
            continue
        source = tensor.detach().float().cpu().contiguous()
        indices, values = topk_indices_and_values(source, sparsity)
        quantized, scale = quantize_symmetric_int8(values)
        dequantized = quantized.float() * scale
        sparse = torch.zeros(source.numel(), dtype=torch.float32)
        sparse.scatter_(0, indices, dequantized)
        reconstructed[name] = sparse.reshape(source.shape)
        tensors[name] = {
            "shape": list(source.shape), "numel": source.numel(),
            "indices": indices.to(torch.int32), "values": quantized,
            "scale": scale, "zero_point": 0,
        }
        selected_count += indices.numel()
        index_bytes += indices.numel() * 4
        value_bytes += quantized.numel()
        dense_bytes += source.numel() * source.element_size()
    envelope = {
        "format": DELTA_FORMAT, "version": DELTA_VERSION,
        "quantization": QUANTIZATION, "sender": sender, "round": round_number,
        "tensors": tensors,
    }
    buffer = io.BytesIO()
    torch.save(envelope, buffer)
    raw = buffer.getvalue()
    if len(raw) > MAX_DELTA_PAYLOAD_BYTES:
        raise ValueError(f"encoded delta is {len(raw)} bytes; limit is {MAX_DELTA_PAYLOAD_BYTES}")
    return base64.b64encode(raw).decode("ascii"), reconstructed, {
        "dense_bytes": dense_bytes, "selected_count": selected_count,
        "index_bytes": index_bytes, "quantized_value_bytes": value_bytes,
        "serialized_bytes": len(raw), "base64_bytes": len(base64.b64encode(raw)),
    }


def decode_delta_envelope(payload_b64, device, allow_legacy=ALLOW_LEGACY_DELTA_FORMAT):
    """Strictly decode v1 payload before allocating its reconstructed tensors."""
    if not isinstance(payload_b64, str) or len(payload_b64) > ((MAX_DELTA_PAYLOAD_BYTES + 2) // 3) * 4:
        raise ValueError("invalid or oversized base64 delta")
    try:
        raw = base64.b64decode(payload_b64.encode("ascii"), validate=True)
    except Exception as exc:
        raise ValueError("invalid base64 delta") from exc
    if len(raw) > MAX_DELTA_PAYLOAD_BYTES:
        raise ValueError("decoded delta exceeds payload limit")
    loaded = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict) or loaded.get("format") != DELTA_FORMAT:
        if not allow_legacy:
            raise ValueError("unsupported legacy or malformed delta format")
        if not isinstance(loaded, dict) or any(not isinstance(k, str) or not torch.is_tensor(v) for k, v in loaded.items()):
            raise ValueError("invalid legacy delta")
        return {k: v.float().to(device) for k, v in loaded.items()}
    if loaded.get("version") != DELTA_VERSION or loaded.get("quantization") != QUANTIZATION:
        raise ValueError("unsupported delta format version or quantizer")
    records = loaded.get("tensors")
    if not isinstance(records, dict) or len(records) > MAX_DELTA_TENSORS:
        raise ValueError("invalid tensor records")
    result, total_elements = {}, 0
    for name, record in records.items():
        if not isinstance(name, str) or not isinstance(record, dict):
            raise ValueError("invalid tensor record")
        shape, numel = record.get("shape"), record.get("numel")
        indices, values, scale, zero_point = (record.get("indices"), record.get("values"),
                                              record.get("scale"), record.get("zero_point"))
        if (not isinstance(shape, list) or any(not isinstance(d, int) or d < 0 for d in shape)
                or not isinstance(numel, int) or numel < 0 or not isinstance(scale, (float, int))
                or not math.isfinite(scale) or scale <= 0 or zero_point != 0):
            raise ValueError(f"invalid metadata for tensor {name}")
        calculated_numel = math.prod(shape)
        if calculated_numel != numel or numel > MAX_DELTA_TENSOR_ELEMENTS:
            raise ValueError(f"invalid shape or element limit for tensor {name}")
        total_elements += numel
        if total_elements > MAX_DELTA_TENSOR_ELEMENTS:
            raise ValueError("global tensor element limit exceeded")
        if (not torch.is_tensor(indices) or not torch.is_tensor(values)
                or indices.dtype not in (torch.int32, torch.int64) or values.dtype != torch.int8
                or indices.dim() != 1 or values.dim() != 1 or indices.numel() != values.numel()):
            raise ValueError(f"invalid sparse values for tensor {name}")
        indices = indices.cpu().to(torch.int64)
        if indices.numel() and (indices.min().item() < 0 or indices.max().item() >= numel):
            raise ValueError(f"out-of-range index for tensor {name}")
        if indices.numel() != torch.unique(indices).numel():
            raise ValueError(f"duplicate index for tensor {name}")
        dense = torch.zeros(numel, dtype=torch.float32)
        dense.scatter_(0, indices, values.cpu().float() * float(scale))
        result[name] = dense.reshape(shape).to(device)
    return result
