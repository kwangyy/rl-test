# Stop the backend (and its restart loop), the tunnel and the auto-updater started by start-all.ps1.
#   powershell -ExecutionPolicy Bypass -File deploy\stop-all.ps1
# -KeepUpdater leaves auto-update.ps1 running (it uses this to restart everything else).
param([switch]$KeepUpdater)
$scripts = @("start-backend.ps1", "start-tunnel.ps1") + $(if ($KeepUpdater) { @() } else { @("auto-update.ps1") })
Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
    Where-Object { $cl = $_.CommandLine; $_.ProcessId -ne $PID -and ($scripts | Where-Object { $cl -like "*$_*" }) } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -Confirm:$false }
Get-Process cloudflared -ErrorAction SilentlyContinue | Stop-Process -Force -Confirm:$false
$p = (Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue).OwningProcess
if ($p) { Stop-Process -Id $p -Force -Confirm:$false }
Write-Host "stopped backend and tunnel$(if (-not $KeepUpdater) { ' and auto-updater' })"
