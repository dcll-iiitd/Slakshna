# SLAKSHNA - CPU-only setup for Windows (PowerShell).
#
#   powershell -ExecutionPolicy Bypass -File .\setup_cpu.ps1
#   powershell -ExecutionPolicy Bypass -File .\setup_cpu.ps1 -SkipPrepare   # skip model download + tokenization
#
# Creates .venv, installs CPU-only PyTorch + the Python deps (no CUDA, no Ray),
# installs the Bhaskera submodule, then downloads the model and tokenizes the dataset once.
param(
    [switch]$SkipPrepare
)
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

if (-not (Test-Path "Bhaskera\pyproject.toml")) {
    Write-Host "Bhaskera submodule is empty - fetching it..." -ForegroundColor Yellow
    git submodule update --init --recursive
}

# Python >= 3.11 required
$py = (Get-Command python -ErrorAction SilentlyContinue)
if (-not $py) { throw "python not found on PATH. Install Python 3.11+ from python.org." }
$ver = & python -c "import sys; print('%d.%d' % sys.version_info[:2])"
if ([version]$ver -lt [version]"3.11") { throw "Python $ver found; 3.11 or newer is required." }

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating virtual environment (.venv)..." -ForegroundColor Cyan
    python -m venv .venv
}
$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

Write-Host "Installing CPU-only PyTorch..." -ForegroundColor Cyan
& $venvPy -m pip install --upgrade pip
& $venvPy -m pip install --index-url https://download.pytorch.org/whl/cpu torch
Write-Host "Installing Python dependencies..." -ForegroundColor Cyan
& $venvPy -m pip install -r requirements-cpu.txt
& $venvPy -m pip install -e Bhaskera --no-deps

if (-not $SkipPrepare) {
    Write-Host "Downloading the model and tokenizing the dataset (one-time)..." -ForegroundColor Cyan
    & $venvPy prepare_cpu.py --smoke
}

Write-Host ""
if (Get-Command cargo -ErrorAction SilentlyContinue) {
    Write-Host "Rust found. Build the node with:  cargo build --release" -ForegroundColor Green
} else {
    Write-Host "Rust/cargo not found. To run the real P2P nodes install https://rustup.rs," -ForegroundColor Yellow
    Write-Host "plus the 'Desktop development with C++' workload (VS Build Tools) and LLVM (for RocksDB)." -ForegroundColor Yellow
    Write-Host "You can still run the full ML pipeline without Rust:" -ForegroundColor Yellow
    Write-Host "  .\.venv\Scripts\python.exe scripts\simulate_cpu_cluster.py --config-dir examples\cpu\2node --rounds 3"
}
Write-Host "Setup complete. See CPU_QUICKSTART.md" -ForegroundColor Green
