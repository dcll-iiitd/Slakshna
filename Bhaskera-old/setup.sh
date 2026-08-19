#!/usr/bin/env bash
# =============================================================================
# Bhaskera — fully automatic setup
# Works on: local workstation, HPC login node (module or spack), compute node
#
# Usage:
#   bash setup.sh
#
# Optional overrides:
#   BHASKERA_CUDA=12.4   bash setup.sh
#   BHASKERA_VENV=/path  bash setup.sh
# =============================================================================
set -euo pipefail

PROBE_TIMEOUT=120

# Log helpers. All four write to stderr so that functions like detect_cuda
# and spack_load_cuda — which return a value by echoing to stdout — don't
# pollute their return value with progress messages when called inside
# `$(...)` command substitution.
die()  { echo "❌  $*" >&2; exit 1; }
info() { echo "ℹ️   $*" >&2; }
ok()   { echo "✅  $*" >&2; }
warn() { echo "⚠️   $*" >&2; }

# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION: pure-bash version comparison (no bc needed)
# ver_to_int "12.4" → 1204
# ─────────────────────────────────────────────────────────────────────────────
ver_to_int() {
    local major minor
    major=$(echo "$1" | cut -d. -f1)
    minor=$(echo "$1" | cut -d. -f2)
    printf "%d%02d" "${major:-0}" "${minor:-0}"
}

ver_lt() { [[ $(ver_to_int "$1") -lt $(ver_to_int "$2") ]]; }

# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION: map CUDA version → PyTorch wheel tag
# ─────────────────────────────────────────────────────────────────────────────
cuda_to_torch_tag() {
    local ver="$1"
    [[ "$ver" == "cpu" ]] && echo "cpu" && return

    local f; f=$(echo "$ver" | grep -oP '^\d+\.\d+' || echo "0.0")

    if   ver_lt "$f" "12.0"; then echo "cu118"
    elif ver_lt "$f" "12.3"; then echo "cu121"
    else                          echo "cu124"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION: try to load CUDA via Spack and return version
# Picks the newest stable CUDA (12.4.x with gcc) automatically
# ─────────────────────────────────────────────────────────────────────────────
spack_load_cuda() {
    # Check if spack is available (either already on PATH or via setup script)
    local spack_setup="/home/apps/SPACK/spack/share/spack/setup-env.sh"

    if ! command -v spack &>/dev/null; then
        if [[ -f "$spack_setup" ]]; then
            info "Sourcing Spack from $spack_setup"
            # shellcheck disable=SC1090
            source "$spack_setup"
        else
            return 1
        fi
    fi

    command -v spack &>/dev/null || return 1

    info "Spack found — finding best CUDA..."

    # List all available cuda specs, pick newest 12.x compiled with gcc
    # Output format: hash cuda@version ...%compiler...
    local best_hash best_ver
    best_hash=""
    best_ver="0.0"

    while IFS= read -r line; do
        local hash ver
        hash=$(echo "$line" | grep -oP '^\s*\K[a-z0-9]+')
        ver=$(echo  "$line" | grep -oP 'cuda@\K[0-9]+\.[0-9]+\.[0-9]+')
        [[ -z "$hash" || -z "$ver" ]] && continue

        # Only consider CUDA 12.x built with gcc (most compatible for PyTorch)
        echo "$line" | grep -q '%gcc' || continue
        echo "$ver"  | grep -q '^12\.' || continue

        local major_minor; major_minor=$(echo "$ver" | grep -oP '^\d+\.\d+')
        if ver_lt "$best_ver" "$major_minor"; then
            best_ver="$major_minor"
            best_hash="$hash"
        fi
    done < <(spack find --format "{hash:7} cuda@{version} %{compiler}" 2>/dev/null || true)

    # Fall back: any 12.x if no gcc build found
    if [[ -z "$best_hash" ]]; then
        while IFS= read -r line; do
            local hash ver
            hash=$(echo "$line" | grep -oP '^\s*\K[a-z0-9]+')
            ver=$(echo  "$line" | grep -oP 'cuda@\K[0-9]+\.[0-9]+\.[0-9]+')
            [[ -z "$hash" || -z "$ver" ]] && continue
            echo "$ver" | grep -q '^12\.' || continue

            local major_minor; major_minor=$(echo "$ver" | grep -oP '^\d+\.\d+')
            if ver_lt "$best_ver" "$major_minor"; then
                best_ver="$major_minor"
                best_hash="$hash"
            fi
        done < <(spack find --format "{hash:7} cuda@{version} %{compiler}" 2>/dev/null || true)
    fi

    if [[ -z "$best_hash" ]]; then
        warn "No suitable CUDA 12.x found in Spack"
        return 1
    fi

    info "Loading Spack CUDA $best_ver (hash /$best_hash)..."
    spack load "/$best_hash" 2>/dev/null || {
        warn "spack load /$best_hash failed"
        return 1
    }

    echo "$best_ver"
}

# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION: SLURM probe — fallback if everything else fails
# ─────────────────────────────────────────────────────────────────────────────
slurm_probe_cuda() {
    local probe_dir; probe_dir=$(mktemp -d)
    local out="$probe_dir/cuda_ver.txt"

    cat > "$probe_dir/probe.sh" <<PROBE
#!/usr/bin/env bash
#SBATCH --job-name=bhaskera-probe
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:02:00
#SBATCH --output=$out
nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \\K[0-9]+\\.[0-9]+" | head -1 || echo "cpu"
PROBE

    local jid
    jid=$(sbatch "$probe_dir/probe.sh" 2>/dev/null | grep -oP '\d+$' || true)

    if [[ -z "$jid" ]]; then
        # Try to find a GPU partition and retry
        local gpu_partition
        gpu_partition=$(sinfo -h -o "%P %G" 2>/dev/null \
            | grep -v "null" | grep "gpu" | head -1 \
            | awk '{print $1}' | tr -d '*' || true)

        if [[ -n "$gpu_partition" ]]; then
            info "Retrying probe on partition: $gpu_partition"
            sed -i "s|#SBATCH --gres=gpu:1|#SBATCH --gres=gpu:1\n#SBATCH --partition=$gpu_partition|" \
                "$probe_dir/probe.sh"
            jid=$(sbatch "$probe_dir/probe.sh" 2>/dev/null | grep -oP '\d+$' || true)
        fi

        if [[ -z "$jid" ]]; then
            warn "Could not submit SLURM probe job"
            rm -rf "$probe_dir"
            echo "cpu"; return
        fi
    fi

    info "Probe job $jid submitted — waiting up to ${PROBE_TIMEOUT}s..."
    local elapsed=0
    while [[ ! -f "$out" ]]; do
        sleep 5; elapsed=$(( elapsed + 5 ))
        if (( elapsed >= PROBE_TIMEOUT )); then
            warn "Probe job timed out"
            scancel "$jid" 2>/dev/null || true
            rm -rf "$probe_dir"
            echo "cpu"; return
        fi
        local state
        state=$(squeue -j "$jid" -h -o "%T" 2>/dev/null || echo "DONE")
        if [[ "$state" == "FAILED" || "$state" == "CANCELLED" ]]; then
            rm -rf "$probe_dir"; echo "cpu"; return
        fi
    done

    local result; result=$(cat "$out" | tr -d '[:space:]')
    rm -rf "$probe_dir"
    echo "$result"
}

# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION: detect CUDA — full layered strategy
# ─────────────────────────────────────────────────────────────────────────────
detect_cuda() {
    # Layer 0: manual override
    if [[ -n "${BHASKERA_CUDA:-}" ]]; then
        info "Manual override: CUDA $BHASKERA_CUDA"
        echo "$BHASKERA_CUDA"; return
    fi

    # Layer 1: env vars set by module load / spack load already done by user
    for var in CUDA_VERSION CUDA_VER; do
        if [[ -n "${!var:-}" ]]; then
            info "Found \$$var = ${!var}"
            echo "${!var}"; return
        fi
    done

    # Layer 2: CUDA_HOME path encodes version
    if [[ -n "${CUDA_HOME:-}" ]]; then
        local v; v=$(echo "$CUDA_HOME" | grep -oP '(?<=cuda-)\d+\.\d+' | head -1 || true)
        if [[ -n "$v" ]]; then
            info "Parsed CUDA $v from CUDA_HOME=$CUDA_HOME"
            echo "$v"; return
        fi
    fi

    # Layer 3: nvcc already on PATH (spack/module already loaded)
    if command -v nvcc &>/dev/null; then
        local v; v=$(nvcc --version 2>/dev/null \
            | grep -oP "release \K[0-9]+\.[0-9]+" | head -1 || true)
        if [[ -n "$v" ]]; then
            info "nvcc reports CUDA $v"
            echo "$v"; return
        fi
    fi

    # Layer 4: nvidia-smi (works on local machines and compute nodes)
    if command -v nvidia-smi &>/dev/null; then
        local v; v=$(nvidia-smi 2>/dev/null \
            | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1 || true)
        if [[ -n "$v" ]]; then
            info "nvidia-smi reports CUDA $v"
            echo "$v"; return
        fi
    fi

    # Layer 5: Spack — auto-load best available CUDA 12.x
    if command -v spack &>/dev/null || \
       [[ -f "/home/apps/SPACK/spack/share/spack/setup-env.sh" ]]; then
        local v; v=$(spack_load_cuda || true)
        if [[ -n "$v" && "$v" != "cpu" ]]; then
            # After spack load, nvcc is now on PATH — confirm
            if command -v nvcc &>/dev/null; then
                local confirmed
                confirmed=$(nvcc --version 2>/dev/null \
                    | grep -oP "release \K[0-9]+\.[0-9]+" | head -1 || true)
                [[ -n "$confirmed" ]] && { echo "$confirmed"; return; }
            fi
            echo "$v"; return
        fi
    fi

    # Layer 6: SLURM probe (last resort — submits a mini job)
    if command -v sbatch &>/dev/null && [[ -z "${SLURM_JOB_ID:-}" ]]; then
        info "Falling back to SLURM compute node probe..."
        local v; v=$(slurm_probe_cuda)
        if [[ -n "$v" && "$v" != "cpu" ]]; then
            echo "$v"; return
        fi
    fi

    warn "No CUDA detected anywhere — falling back to CPU-only"
    echo "cpu"
}

# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION: pick best venv path
# ─────────────────────────────────────────────────────────────────────────────
pick_venv_path() {
    [[ -n "${BHASKERA_VENV:-}" ]] && echo "$BHASKERA_VENV" && return

    echo "$PWD/.venv"
}

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════"
echo "  Bhaskera Setup"
echo "══════════════════════════════════════════"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CUDA_VER=$(detect_cuda)
TORCH_TAG=$(cuda_to_torch_tag "$CUDA_VER")
TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_TAG}"
VENV=$(pick_venv_path)

echo ""
info "CUDA version  : $CUDA_VER"
info "PyTorch tag   : $TORCH_TAG"
info "Index URL     : $TORCH_INDEX"
info "Venv location : $VENV"
echo ""

# ── ensure uv is available ───────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    info "uv not found — installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# ── create venv pinned to Python 3.11 ───────────────────────────────────────
uv venv "$VENV" --python 3.11

source "$VENV/bin/activate"

# ── install torch with correct CUDA flavor FIRST ────────────────────────────
uv pip install torch torchvision torchaudio \
    --index-url "$TORCH_INDEX" \
    --quiet

# ── install bhaskera ────────────────────────────────────────────────────────
# This pulls in liger-kernel (declared in pyproject.toml) too — it's a pure
# Python package on top of the torch+triton we just installed, so it slots
# in without any CUDA-compiler dance.
uv pip install -e "$SCRIPT_DIR[wandb,mlflow]" --quiet

# ── install flash-attn (GPU only, post-torch, out-of-band) ──────────────────
# flash-attn ships a CUDA extension whose setup.py imports torch at build
# time. PEP-517 build isolation hides the torch we just installed, so we
# must pass --no-build-isolation. We also pre-install its build-time deps
# (packaging, ninja, wheel) into the same env so the build sees them.
#
# This step is intentionally:
#   * skipped on CPU-only setups (nothing to build against)
#   * capped at MAX_JOBS=4 to avoid OOM-ing the login node during compile
#   * non-fatal — if the build fails (e.g. nvcc missing on a login node,
#     or an unsupported GPU arch), the rest of the framework still works
#     with attn_impl=sdpa or eager. We surface a clear message so the user
#     can retry on a compute node.
if [[ "$CUDA_VER" != "cpu" ]]; then
    echo ""
    info "Installing flash-attn (CUDA extension — may take several minutes on first build)..."

    # Build-time prerequisites for flash-attn's setup.py.
    # setuptools is explicit: `uv venv` creates minimal environments without
    # it, and `--no-build-isolation` means the build sees exactly this env
    # (no isolated build backend to pull setuptools in behind the scenes).
    uv pip install setuptools packaging ninja wheel --quiet

    # Cap build parallelism — flash-attn's nvcc jobs are memory-hungry
    export MAX_JOBS="${MAX_JOBS:-4}"

    if uv pip install flash-attn --no-build-isolation --quiet; then
        ok "flash-attn installed (MAX_JOBS=$MAX_JOBS)"
    else
        warn "flash-attn install failed — training will still work with"
        warn "  attn_impl: sdpa  (or null/eager) in your config."
        warn "To retry manually on a GPU/compute node with nvcc on PATH:"
        warn "  MAX_JOBS=4 uv pip install flash-attn --no-build-isolation"
    fi
else
    info "Skipping flash-attn (CPU-only setup — nothing to build against)"
fi

# ── smoke test ───────────────────────────────────────────────────────────────
echo ""
echo "🔬 Smoke test:"
python - <<EOF
import importlib, sys, torch
print(f"   python       : {sys.version.split()[0]}")
print(f"   torch        : {torch.__version__}")
print(f"   CUDA built   : {torch.version.cuda}")
print(f"   CUDA visible : {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("   (no GPU on login node — expected, will work on compute nodes)")

# Report optional accelerators — absence is not a failure.
try:
    lk = importlib.import_module("liger_kernel")
    print(f"   liger-kernel : {getattr(lk, '__version__', 'installed')}")
except Exception as e:
    print(f"   liger-kernel : NOT available ({type(e).__name__})")

try:
    fa = importlib.import_module("flash_attn")
    print(f"   flash-attn   : {getattr(fa, '__version__', 'installed')}")
except Exception as e:
    print(f"   flash-attn   : NOT available ({type(e).__name__})")
EOF

# ── write portable activate helper ───────────────────────────────────────────
ACTIVATE_SCRIPT="$PWD/bhaskera-activate.sh"
CUDA_MAJOR_MINOR=$(echo "$CUDA_VER" | grep -oP '^\d+\.\d+' || echo "")
SPACK_SETUP="/home/apps/SPACK/spack/share/spack/setup-env.sh"

# Capture the spack hash that was loaded (if any) for reproducibility
SPACK_CUDA_HASH=""
if command -v spack &>/dev/null; then
    SPACK_CUDA_HASH=$(spack find --loaded --format "/{hash:7}" cuda 2>/dev/null \
        | head -1 || true)
fi

cat > "$ACTIVATE_SCRIPT" <<ACTIVATE
#!/usr/bin/env bash
# Auto-generated by bhaskera/setup.sh - do not edit manually
# Source this at the top of your SLURM job scripts

# Load CUDA via spack or module
if [[ -f "$SPACK_SETUP" ]] && [[ -n "$SPACK_CUDA_HASH" ]]; then
    source "$SPACK_SETUP" 2>/dev/null || true
    spack load $SPACK_CUDA_HASH 2>/dev/null || true
elif command -v module &>/dev/null && [[ -n "$CUDA_MAJOR_MINOR" ]]; then
    module load cuda/${CUDA_MAJOR_MINOR} 2>/dev/null || true
fi

source "${VENV}/bin/activate"
ACTIVATE

chmod +x "$ACTIVATE_SCRIPT"

echo ""
ok "Setup complete!"
echo ""
echo "  Local / interactive:"
echo "    source $VENV/bin/activate"
echo ""
echo "  In every SLURM job script — just one line:"
echo "    source $ACTIVATE_SCRIPT"
echo ""
