# Fix "Docker Desktop encountered an unexpected error ... remove ...\*.sock /
# dockerInference: The file cannot be accessed by the system."
#
# Cause: Docker Desktop was not shut down cleanly (laptop turned off / crashed while
# it was running) and left Unix socket files behind that Windows cannot delete.
# Fix: stop Docker, delete those runtime files through WSL (Docker recreates them),
# start Docker again.
#
# Usage (PowerShell):  .\tools\fix-docker.ps1
# Tip: quit Docker Desktop from the tray icon before shutting down the laptop.

$ErrorActionPreference = "Continue"
$local = $env:LOCALAPPDATA -replace '\\', '/'
$wslLocal = "/mnt/" + $local.Substring(0, 1).ToLower() + $local.Substring(2)

Write-Host "Stopping Docker Desktop..."
Get-Process "Docker Desktop", "com.docker.backend", "com.docker.build" -ErrorAction SilentlyContinue |
    Stop-Process -Force -Confirm:$false
Start-Sleep -Seconds 3

Write-Host "Removing stale socket files..."
wsl -- sh -c "rm -fv '$wslLocal/Docker/run/'* '$wslLocal/docker-secrets-engine/engine.sock' 2>/dev/null; true"

Write-Host "Starting Docker Desktop..."
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
Write-Host "Done. Wait ~1 minute for 'Engine running' in Docker Desktop."
