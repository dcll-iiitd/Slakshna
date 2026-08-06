import os

# Delta transport wire format. `torch.save` is retained only as a container;
# receivers validate this schema before materializing any dense tensors.
DELTA_FORMAT = "slakshna.sparse-quantized"
DELTA_VERSION = 1
QUANTIZATION = "symmetric_int8"

def _env_bool(name, default=False):
    return os.environ.get(name, str(default)).lower() in ("1", "true", "yes", "on")

COMPRESSION_ENABLED = _env_bool("SLAKSHNA_COMPRESSION_ENABLED", True)
DELTA_SPARSITY = float(os.environ.get("SLAKSHNA_DELTA_SPARSITY", "0.1"))
DELTA_QUANTIZATION = os.environ.get("SLAKSHNA_DELTA_QUANTIZATION", QUANTIZATION)
ALLOW_LEGACY_DELTA_FORMAT = _env_bool("SLAKSHNA_ALLOW_LEGACY_DELTA_FORMAT", False)
MAX_DELTA_PAYLOAD_BYTES = int(os.environ.get("SLAKSHNA_MAX_DELTA_PAYLOAD_BYTES", str(7 * 1024 * 1024)))
MAX_DELTA_TENSOR_ELEMENTS = int(os.environ.get("SLAKSHNA_MAX_DELTA_TENSOR_ELEMENTS", "10000000"))
MAX_DELTA_TENSORS = 100000
