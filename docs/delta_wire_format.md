# Sparse quantized delta wire format (v1)

`compressed_delta` remains a base64 string to Rust and gossip. Its decoded
bytes are a `torch.save` envelope with `format = "slakshna.sparse-quantized"`,
`version = 1`, and `quantization = "symmetric_int8"`.

Each entry in `tensors` contains `shape`, `numel`, int32 flattened `indices`,
int8 `values`, a finite positive float32-compatible `scale`, and zero-point
zero. Values reconstruct as `int8_value * scale`; missing elements are zero.
Indices use normal PyTorch integer semantics (host-native tensor serialization,
not a raw byte buffer), so no receiver depends on a machine byte order.

The decoder accepts only v1 by default. It base64-validates, bounds raw payload
bytes (7 MiB by default), validates every record and index before allocating a
dense output, and rejects duplicates, invalid shapes/scales, unsupported dtypes,
and excessive tensor counts/elements. Legacy dense state dictionaries are only
accepted when `allow_legacy_delta_format` is explicitly enabled during rollout.

Configuration lives in `[compression]`; all nodes in a federation should use
the same sparsity, quantizer, and size limits. The 7 MiB raw default expands to
about 9.34 MiB in base64, leaving room beneath the 10 MiB gossip message ceiling.
