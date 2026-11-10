# Start a local SLAKSHNA cluster on Windows (PowerShell). CPU configs by default.
#
#   .\start_cluster.ps1 -Nodes 4            # examples\cpu\4node  (laptop / CPU)
#   .\start_cluster.ps1 -Nodes 2 -Gpu       # examples\2node      (original GPU configs)
#   .\stop_cluster.ps1                      # stop everything
param(
    [Parameter(Mandatory = $true)][ValidateRange(2, 6)][int]$Nodes,
    [switch]$Gpu
)
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$configDir = if ($Gpu) { "examples\${Nodes}node" } else { "examples\cpu\${Nodes}node" }
if (-not (Test-Path $configDir)) { throw "Config directory $configDir not found." }

$exe = "target\release\iiitd.exe"
if (-not (Test-Path $exe)) {
    if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
        throw "No $exe and no cargo. Install Rust (rustup.rs) and run 'cargo build --release', or run the ML-only simulation: .\.venv\Scripts\python.exe scripts\simulate_cpu_cluster.py --config-dir $configDir"
    }
    Write-Host "Rust binary not found. Building (first build takes a while)..."
    cargo build --release
}

# The node launches `python ml_engine.py`; point it at the venv that has torch/peft/bhaskera.
$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (Test-Path $venvPy) { $env:SLAKSHNA_PYTHON = $venvPy }
$env:PYTHONUTF8 = "1"

New-Item -ItemType Directory -Force -Path logs | Out-Null
if (Test-Path cluster.pids) { Remove-Item cluster.pids }

Write-Host "Starting $Nodes node cluster from $configDir ..."
for ($i = 1; $i -le $Nodes; $i++) {
    $cfg = "$configDir\node$i.toml"
    if (-not (Test-Path $cfg)) { Write-Warning "$cfg not found, skipping."; continue }
    $log = "logs\node${i}_cluster.log"
    Write-Host "  node $i -> $cfg (logs: $log, $log.err)"
    $p = Start-Process -FilePath $exe -ArgumentList @("--config", $cfg) `
        -RedirectStandardOutput $log -RedirectStandardError "$log.err" `
        -WindowStyle Hidden -PassThru
    $p.Id | Add-Content cluster.pids
}

Write-Host ""
Write-Host "All $Nodes nodes started. Follow a node with:  Get-Content logs\node1_cluster.log -Wait"
Write-Host "Windows Firewall may ask to allow iiitd.exe on private networks - allow it (mDNS / QUIC need it)."
