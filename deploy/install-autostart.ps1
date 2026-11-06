# Run start-all.ps1 at every logon of this Windows user (Startup folder; no admin needed).
#   powershell -ExecutionPolicy Bypass -File deploy\install-autostart.ps1            # install
#   powershell -ExecutionPolicy Bypass -File deploy\install-autostart.ps1 -Remove    # uninstall
param([switch]$Remove)
$startup = [Environment]::GetFolderPath("Startup")
$launcher = Join-Path $startup "rl-project-demo.cmd"
if ($Remove) {
    if (Test-Path $launcher) { Remove-Item $launcher -Confirm:$false }
    Write-Host "removed $launcher"
    return
}
$target = Join-Path $PSScriptRoot "start-all.ps1"
Set-Content -Path $launcher -Encoding ascii -Value @(
    "@echo off",
    "rem Starts the rl-project demo backend and Cloudflare tunnel (deploy/install-autostart.ps1)",
    "start `"`" /min powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$target`""
)
Write-Host "installed $launcher"
