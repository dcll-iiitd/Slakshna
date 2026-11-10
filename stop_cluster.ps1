# Stop a cluster started with start_cluster.ps1 and clean up the Python ML engines.
$ErrorActionPreference = "SilentlyContinue"
Set-Location -Path $PSScriptRoot

if (Test-Path cluster.pids) {
    Get-Content cluster.pids | ForEach-Object {
        $p = Get-Process -Id ([int]$_) -ErrorAction SilentlyContinue
        if ($p) { Stop-Process -Id $p.Id -Force; Write-Host "Stopped iiitd process $($p.Id)" }
    }
    Remove-Item cluster.pids
} else {
    Get-Process iiitd -ErrorAction SilentlyContinue | Stop-Process -Force
}

# Child ML engines / trainers launched by the nodes.
Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -match "ml_engine\.py|bhaskera\.launcher" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force; Write-Host "Stopped python process $($_.ProcessId)" }

Write-Host "Cluster stopped."
