# Keep this checkout on the newest origin/deploy/self-host and restart what changed.
# start-all.ps1 starts it in a hidden window; stop-all.ps1 stops it. Log: deploy/logs/update.log
#
# Only fast-forwards: if local commits or uncommitted changes to tracked files exist, it logs
# and skips, so nothing here is ever overwritten. Whatever is pushed to this branch runs on this
# PC as this user, so only push (or merge into) deploy/self-host what you want live.
param([int]$Minutes = 5, [string]$Branch = "deploy/self-host", [switch]$Once)
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
$log = "$PSScriptRoot\logs\update.log"
New-Item -ItemType Directory -Force "$PSScriptRoot\logs" | Out-Null
function Log($msg) { Add-Content -Path $log -Value "$(Get-Date -Format s) $msg" -Encoding utf8 }

function Check {
    Set-Location $root
    if ((git rev-parse --abbrev-ref HEAD) -ne $Branch) { Log "not on $Branch; skipped"; return }
    git fetch --quiet origin $Branch 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { Log "fetch failed (offline or no credentials); will retry"; return }
    $old = git rev-parse HEAD
    $new = git rev-parse "origin/$Branch"
    if ($old -eq $new) { return }
    git merge-base --is-ancestor $old $new
    if ($LASTEXITCODE -ne 0) { Log "local commits not on origin/$Branch; skipped (push or reset them)"; return }
    if (git status --porcelain --untracked-files=no) { Log "uncommitted changes to tracked files; skipped"; return }

    $changed = @(git diff --name-only $old $new)
    git merge --ff-only --quiet $new 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { Log "fast-forward failed; skipped"; return }
    Log "updated $($old.Substring(0,7)) -> $($new.Substring(0,7)): $($changed.Count) files"

    if ($changed | Where-Object { $_ -like "deploy/*.ps1" -or $_ -like "deploy/*.yml" }) {
        Log "deploy scripts changed; full restart of backend and tunnel"
        & "$PSScriptRoot\stop-all.ps1" -KeepUpdater | Out-Null
        Start-Sleep -Seconds 3
        & "$PSScriptRoot\start-all.ps1" | Out-Null
        if ($changed -contains "deploy/auto-update.ps1") {
            Log "auto-update.ps1 changed; handing over to the new version"
            Start-Process "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -WindowStyle Hidden `
                -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Minutes $Minutes -Branch $Branch"
            exit
        }
    } elseif ($changed | Where-Object { $_ -notlike "verifier/data/traces/*" -and $_ -notlike "*.md" }) {
        # start-backend.ps1's loop starts the server again within 5 s, on the new code
        $p = (Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue).OwningProcess
        if ($p) { Stop-Process -Id $p -Force -Confirm:$false; Log "restarted backend" }
    } else {
        Log "only traces or docs changed; no restart needed"
    }
}

Log "auto-update started (every $Minutes min, $Branch)"
do {
    try { Check } catch { Log "error: $_" }
    if (-not $Once) { Start-Sleep -Seconds ($Minutes * 60) }
} while (-not $Once)
