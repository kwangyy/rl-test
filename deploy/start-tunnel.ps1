# Connect this machine to Cloudflare so https://rl-project.kwangyyinc.com reaches the backend.
# One-time setup is in deploy/README.md (cloudflared login, tunnel create, route dns).
#
#   powershell -ExecutionPolicy Bypass -File deploy\start-tunnel.ps1
param([string]$Config = "$PSScriptRoot\cloudflared.yml")
$ErrorActionPreference = "Stop"
$cf = (Get-Command cloudflared -ErrorAction SilentlyContinue).Source
if (-not $cf -and (Test-Path "$HOME\bin\cloudflared.exe")) { $cf = "$HOME\bin\cloudflared.exe" }
if (-not $cf) {
    throw "cloudflared not found; winget install --id Cloudflare.cloudflared, or put cloudflared.exe in $HOME\bin"
}
if (-not (Test-Path $Config)) {
    throw "Missing $Config; copy deploy\cloudflared.example.yml to it and fill in the tunnel id"
}
New-Item -ItemType Directory -Force "$PSScriptRoot\logs" | Out-Null
& $cf tunnel --no-autoupdate --logfile "$PSScriptRoot\logs\tunnel.log" --config $Config run
