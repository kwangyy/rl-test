# Start the backend and the tunnel in hidden windows, detached from this terminal.
# install-autostart.ps1 runs this at every Windows logon. Safe to run twice: it skips what is already up.
#
#   powershell -ExecutionPolicy Bypass -File deploy\start-all.ps1
$ErrorActionPreference = "Stop"
$ps = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"

function Running($pattern) {
    @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
        Where-Object { $_.CommandLine -like "*$pattern*" }).Count -gt 0
}

if (Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue) {
    Write-Host "backend already listening on 8000"
} elseif (Running "start-backend.ps1") {
    Write-Host "backend already starting"
} else {
    Start-Process $ps -WindowStyle Hidden -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSScriptRoot\start-backend.ps1`""
    Write-Host "backend started"
}

if (Get-Process cloudflared -ErrorAction SilentlyContinue) {
    Write-Host "tunnel already running"
} else {
    Start-Process $ps -WindowStyle Hidden -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSScriptRoot\start-tunnel.ps1`""
    Write-Host "tunnel started"
}
