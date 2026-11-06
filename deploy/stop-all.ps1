# Stop the backend (and its restart loop) and the tunnel started by start-all.ps1.
#   powershell -ExecutionPolicy Bypass -File deploy\stop-all.ps1
Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
    Where-Object { $_.CommandLine -like "*start-backend.ps1*" -or $_.CommandLine -like "*start-tunnel.ps1*" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -Confirm:$false }
Get-Process cloudflared -ErrorAction SilentlyContinue | Stop-Process -Force -Confirm:$false
$p = (Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue).OwningProcess
if ($p) { Stop-Process -Id $p -Force -Confirm:$false }
Write-Host "stopped backend and tunnel"
