# syntax=docker/dockerfile:1.7
#
# SLAKSHNA node image: Rust P2P node (iiitd) + Python ML engine (ml_engine.py + Bhaskera)
#
# Build from the repository ROOT (context = repo root):
#   git submodule update --init            # Bhaskera must be present
#   docker buildx build -f docker/Dockerfile \
#     --build-arg UID=$(id -u) --build-arg GID=$(id -g) \
#     --build-arg GIT_COMMIT=$(git rev-parse --short HEAD) \
#     --build-arg BHASKERA_COMMIT=$(git -C Bhaskera rev-parse HEAD) \
#     -t slakshna:$(git rev-parse --short HEAD) -t slakshna:dev .
#
# Base images are pinned by digest so the build is reproducible.
ARG RUST_IMAGE=rust:1.93-bookworm@sha256:7c4ae649a84014c467d79319bbf17ce2632ae8b8be123ac2fb2ea5be46823f31
ARG PYTHON_IMAGE=python:3.11-slim-bookworm@sha256:0a310eeecf4e1f5a0743f9a6520c90c88d089c903ca5fd283f501e3a805f5f89

# ── Stage 1: Rust node ──────────────────────────────────────────────────────
FROM ${RUST_IMAGE} AS rust-builder

RUN apt-get update \
 && apt-get install -y --no-install-recommends cmake clang libclang-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src ./src

# --locked: fail if Cargo.lock is out of date instead of silently re-resolving.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/target \
    cargo build --release --locked \
 && install -m 0755 target/release/iiitd /usr/local/bin/iiitd

# ── Stage 2: Python environment (same pins as the lab training venv) ────────
FROM ${PYTHON_IMAGE} AS python-deps

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# Exact versions from the lab venv. --no-deps because the lock is already complete;
# 'pip check' below proves it is self-consistent.
COPY docker/requirements.lock.txt /tmp/requirements.lock.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps -r /tmp/requirements.lock.txt

# flash-attn: the lab venv has it, so the image does too. Prebuilt wheel for
# torch 2.5 / cu12 / cp311 (no compilation). A missing wheel must not be silent.
ARG FLASH_ATTN_WHEEL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps "${FLASH_ATTN_WHEEL}" \
 || { echo "WARNING: flash-attn wheel could not be installed (image continues without it)"; touch /opt/venv/.flash_attn_missing; }

# Bhaskera comes from the git submodule, never from the network.
COPY Bhaskera/ /src/Bhaskera/
RUN test -f /src/Bhaskera/pyproject.toml || { \
      echo "ERROR: Bhaskera/pyproject.toml is missing - the Bhaskera submodule is empty." >&2; \
      echo "       Run: git submodule update --init --recursive   (or clone with --recursive)" >&2; \
      exit 1; }
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps /src/Bhaskera \
 && pip check \
 && python -c "import torch, ray, transformers, peft, accelerate, bitsandbytes, bhaskera; print('imports ok, torch', torch.__version__)"

# ── Stage 3: runtime ────────────────────────────────────────────────────────
FROM ${PYTHON_IMAGE} AS runtime

ARG UID=1000
ARG GID=1000
ARG GIT_COMMIT=unknown
ARG BHASKERA_COMMIT=unknown

# gcc + libc6-dev: Triton (used by liger-kernel) JIT-compiles small launcher stubs.
# procps: ps/pkill for debugging inside the container.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates gcc libc6-dev procps \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --gid ${GID} app \
 && useradd --create-home --uid ${UID} --gid ${GID} --shell /bin/bash app

COPY --from=rust-builder /usr/local/bin/iiitd /usr/local/bin/iiitd
COPY --from=python-deps /opt/venv /opt/venv

WORKDIR /app
COPY ml_engine.py node_template.yaml ./
COPY federated_communication ./federated_communication
COPY docker/entrypoint.sh docker/preflight.py docker/healthcheck.py /usr/local/bin/

# Every directory the node writes to is created AND owned by the runtime user here,
# so fresh named volumes mounted on them inherit the right ownership.
# (ml_engine.py uses BASE_DIR=/app, node data_dir is /app/node-data in our configs.)
RUN chmod 0755 /usr/local/bin/entrypoint.sh /usr/local/bin/preflight.py /usr/local/bin/healthcheck.py \
 && mkdir -p /app/ml_models /app/ml_states /app/data /app/logs /app/hf_cache /app/cache /app/node-data \
 && chown -R app:app /app

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    WANDB_MODE=offline \
    SLAKSHNA_CONFIG=/app/config.toml

LABEL org.opencontainers.image.title="slakshna" \
      org.opencontainers.image.description="SLAKSHNA P2P federated learning node (Rust + Bhaskera)" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.slakshna.commit="${GIT_COMMIT}" \
      org.slakshna.bhaskera_commit="${BHASKERA_COMMIT}"

USER app

# Ports are informational: compose uses host networking. Real values come from the node config.
EXPOSE 8545 8546 9000/udp

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD ["python", "/usr/local/bin/healthcheck.py"]

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
