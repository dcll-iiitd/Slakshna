import os

# Delta transport wire format. `torch.save` is retained only as a container;
# receivers validate this schema before materializing any dense tensors.
DELTA_FORMAT = "slakshna.sparse-quantized"
DELTA_VERSION = 1
QUANTIZATION = "symmetric_int8"

def _env_bool(name, default=False):
    return os.environ.get(name, str(default)).lower() in ("1", "true", "yes", "on")

# Largest L2 norm a delta may have. A sender clips its own delta to this value, and receivers
# accept up to it (plus headroom, below), so an honest peer's delta is never rejected for size.
# (They used to differ: senders clipped at 100 but receivers rejected above 10, so peers
# silently dropped each other.)
MAX_DELTA_NORM = float(os.environ.get("SLAKSHNA_MAX_DELTA_NORM", "100.0"))
# Receivers allow 5% headroom: a delta clipped to exactly MAX_DELTA_NORM can measure slightly
# higher after float rounding and int8 quantization, and must not be rejected for that.
MAX_PEER_DELTA_NORM = MAX_DELTA_NORM * 1.05
# A fixed limit cannot tell a poisoned delta from an honest one: an attacker simply sends a delta
# clipped to the limit. So a peer's delta is also rejected when its norm exceeds this multiple of
# this node's own delta norm (honest peers training on similar models have similar norms).
MAX_PEER_NORM_RATIO = float(os.environ.get("SLAKSHNA_MAX_PEER_NORM_RATIO", "5.0"))
COMPRESSION_ENABLED = _env_bool("SLAKSHNA_COMPRESSION_ENABLED", True)
DELTA_SPARSITY = float(os.environ.get("SLAKSHNA_DELTA_SPARSITY", "0.1"))
DELTA_QUANTIZATION = os.environ.get("SLAKSHNA_DELTA_QUANTIZATION", QUANTIZATION)
ALLOW_LEGACY_DELTA_FORMAT = _env_bool("SLAKSHNA_ALLOW_LEGACY_DELTA_FORMAT", False)
MAX_DELTA_PAYLOAD_BYTES = int(os.environ.get("SLAKSHNA_MAX_DELTA_PAYLOAD_BYTES", str(7 * 1024 * 1024)))
MAX_DELTA_TENSOR_ELEMENTS = int(os.environ.get("SLAKSHNA_MAX_DELTA_TENSOR_ELEMENTS", "10000000"))
MAX_DELTA_TENSORS = 100000
