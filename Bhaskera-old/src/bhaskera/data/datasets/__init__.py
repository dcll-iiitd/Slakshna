"""
bhaskera.data.datasets
======================
Importing this package triggers `@register` decorators for all built-ins.
New datasets should be added as new modules and imported here.
"""
from __future__ import annotations

# Each import has a module-level @register side-effect; do not remove.
from . import ultrachat       # noqa: F401
from . import openassistant   # noqa: F401
from . import redpajama       # noqa: F401
from . import local_chat      # noqa: F401  — generic JSONL/JSON/Parquet loader
from . import local_cpt       # noqa: F401  — generic CPT continuous packing loader
