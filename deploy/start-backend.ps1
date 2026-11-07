# Serve the demo UI and the live 2B verifier from this machine on http://127.0.0.1:8000.
# The Cloudflare tunnel (start-tunnel.ps1) is the only way in from outside.
#
#   powershell -ExecutionPolicy Bypass -File deploy\start-backend.ps1
#
# Needs in the repo-root .env (git-ignored): DEMO_PASSWORD, and an OpenRouter key for the
# `prompted` policy. Restarts the server if it exits; stop with Ctrl+C.
param(
    [int]$Port = 8000,
    [string]$Adapter = "verifier/experiments/sft-direct-from-base-qwen3p5-2b-5895-samples/runs/run1/artifacts/checkpoints",
    # the 2B imitation controller (ties flash on dev 900) for policy=controller; "" turns it off
    [string]$ControllerAdapter = "verifier/experiments/sft-controller-from-base-qwen3p5-2b-flash-follow-18291-samples/runs/run1/artifacts/checkpoints",
    [string]$Origin = "https://rl-project.kwangyyinc.com",
    [string]$SaveLive = "public-live",
    [string]$KeyEnv = "OPENROUTER_API_KEY_TEAM",
    [string]$Python = ""
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not $Python) {
    $Python = @("$root\.venv\Scripts\python.exe", "$root\..\rl-test\.venv\Scripts\python.exe") |
        Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $Python) { throw "No virtualenv found; run 'uv sync' here or pass -Python <path to python.exe>" }

$envFile = Get-Content "$root\.env" -ErrorAction SilentlyContinue
if (-not ($envFile -match '^\s*DEMO_PASSWORD\s*=\s*\S') -and -not $env:DEMO_PASSWORD) {
    throw "Set DEMO_PASSWORD in $root\.env first; the site is public through the tunnel"
}
if (-not (Test-Path "$Adapter\adapter_model.safetensors")) { throw "Adapter weights missing: $Adapter" }
$ctl = ""
if ($ControllerAdapter) {
    if (-not (Test-Path "$ControllerAdapter\adapter_model.safetensors")) { throw "Controller weights missing: $ControllerAdapter" }
    $ctl = "--controller-adapter `"$ControllerAdapter`""
}
if (-not (Test-Path "verifier\data\fever\pages.sqlite")) { throw "FEVER index missing; see verifier/docs/data.md" }

$env:CONTROLLER_API_KEY_ENV = $KeyEnv   # which .env key the prompted controller uses
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
New-Item -ItemType Directory -Force "$root\deploy\logs" | Out-Null
# uvicorn logs to stderr; under "Stop", Windows PowerShell 5.1 would treat that as a failure
$ErrorActionPreference = "Continue"

while ($true) {
    $log = "$root\deploy\logs\backend-$(Get-Date -Format yyyyMMdd).log"
    Write-Host "$(Get-Date -Format s) starting backend on 127.0.0.1:$Port (log: $log)"
    # through cmd so the log stays plain UTF-8 text (PowerShell 5.1 redirection writes UTF-16)
    cmd /c "`"$Python`" -m agent.server --host 127.0.0.1 --port $Port --adapter `"$Adapter`" $ctl --cors $Origin --save-live $SaveLive >> `"$log`" 2>&1"
    Write-Host "$(Get-Date -Format s) backend exited with code $LASTEXITCODE; restarting in 5 s"
    Start-Sleep -Seconds 5
}
