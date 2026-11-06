# Self-hosting the demo at rl-project.kwangyyinc.com

This branch (`deploy/self-host`) runs the demo from one Windows PC with a GPU, without renting a VM.

```
browser ──https──> Cloudflare (rl-project.kwangyyinc.com) ──tunnel──> this PC 127.0.0.1:8000
                                                                      agent.server (FastAPI)
                                                                      ├─ /        built UI (demo/dist)
                                                                      └─ /api/*   traces, live 2B verifier
```

- **One origin.** FastAPI already serves the built UI from `demo/dist/`, and the UI calls relative `/api/...` URLs. Serving both from the same address means no CORS, and live runs stream over server-sent events with no separate frontend host. There is nothing to deploy to Vercel. If the PC is off, the site is down, but the UI can do nothing without the backend anyway.
- **No open ports.** `cloudflared` makes an outbound connection to Cloudflare. The server listens only on `127.0.0.1`, so nothing on the router or Windows firewall is opened.
- **Password.** With `DEMO_PASSWORD` set, `agent.server` requires HTTP Basic auth on every route: the UI, the API and the live stream. The browser shows its own login box, the user is `demo`, and `/docs` is switched off. Cloudflare Access (below) is an optional second layer.

## What runs

- **Verifier:** Qwen3.5-2B with `sft-direct-from-base-qwen3p5-2b-5895-samples/run1`, the same verifier as the controller experiments. It uses about 5 GB of VRAM, and a typed claim takes about 20 s on the first request (loading the index and model), then a few seconds.
- **Policies:** `fixed` (local only) and `prompted`. `prompted` calls `qwen/qwen3.8-flash` on OpenRouter, paid for with the key named by `-KeyEnv` (default `OPENROUTER_API_KEY_TEAM`).
- **Browse, Compare and Replay:** every `verifier/data/traces/*.jsonl`, including the controller GRPO, imitation and flash traces.
- **Live runs** are saved to `verifier/data/traces/fever_dev_public-live.jsonl` (git-ignored, because it holds visitors' text), so they show up in Browse.
- **Limits:** one live run at a time on the GPU, and others queue. Holdout and calibration claim IDs are refused.

## One-time setup

1. **Python environment.** In this checkout run `uv sync`. `start-backend.ps1` also finds `..\rl-test\.venv`.
2. **Data.** These are ignored by git, so link or copy them from the main checkout:
   ```powershell
   New-Item -ItemType Junction -Path verifier\data\fever -Target ..\rl-test\verifier\data\fever
   Copy-Item -Recurse ..\rl-test\verifier\experiments\sft-direct-from-base-qwen3p5-2b-5895-samples\runs\run1\artifacts\checkpoints\* `
     verifier\experiments\sft-direct-from-base-qwen3p5-2b-5895-samples\runs\run1\artifacts\checkpoints\
   ```
3. **`.env`** in the repo root (git-ignored) needs two settings:
   ```
   DEMO_PASSWORD=<long random string>
   OPENROUTER_API_KEY_TEAM=<key>      # or set -KeyEnv to another variable name
   ```
4. **DNS on Cloudflare.** `kwangyyinc.com` currently uses Namecheap's nameservers (`dns1/dns2.registrar-servers.com`). A named Cloudflare tunnel needs the zone on Cloudflare, which is free:
   1. Go to dash.cloudflare.com, then Add a domain, enter `kwangyyinc.com` and choose the Free plan.
   2. Check that the imported DNS records match what Namecheap has now (mail MX/TXT records, other subdomains).
   3. In Namecheap, under Domain, then Nameservers, choose Custom DNS and enter the two Cloudflare nameservers. Registration stays with Namecheap.
   4. Wait until Cloudflare shows the zone as Active, which can take minutes or hours.
5. **Tunnel.**
   ```powershell
   # cloudflared is already at ~\bin\cloudflared.exe (standalone; winget install needs admin)
   ~\bin\cloudflared.exe tunnel login                       # browser: pick kwangyyinc.com
   ~\bin\cloudflared.exe tunnel create rl-project           # prints the tunnel id, writes ~/.cloudflared/<id>.json
   ~\bin\cloudflared.exe tunnel route dns rl-project rl-project.kwangyyinc.com
   Copy-Item deploy\cloudflared.example.yml deploy\cloudflared.yml   # then put the id in it
   ```
6. **Optional Cloudflare Access.** In Zero Trust, add a self-hosted application for `rl-project.kwangyyinc.com` with an "emails" or "email ends in" policy. Visitors then get a one-time code by email before they reach the Basic auth prompt. It is free for up to 50 users.

## Running it

```powershell
powershell -ExecutionPolicy Bypass -File deploy\start-all.ps1          # backend, tunnel and auto-updater
powershell -ExecutionPolicy Bypass -File deploy\stop-all.ps1           # stop all three
powershell -ExecutionPolicy Bypass -File deploy\install-autostart.ps1  # run start-all at every logon
powershell -ExecutionPolicy Bypass -File deploy\install-autostart.ps1 -Remove
```

Then open https://rl-project.kwangyyinc.com and log in as `demo` with the `DEMO_PASSWORD` from `.env`.

**Auto-update.** `start-all.ps1` also starts `auto-update.ps1`. Every 5 minutes it fetches `origin/deploy/self-host`, fast-forwards this checkout and restarts what changed:
- **Code changes:** the backend restarts, which takes about 20 s.
- **Changes to `deploy/*.ps1` or `deploy/*.yml`:** the backend and tunnel both restart.
- **Only traces or `.md` files:** nothing restarts.

It never overwrites anything. If this checkout has local commits or uncommitted edits to tracked files, it logs that and skips the update. The log is `deploy/logs/update.log`.

Whatever is pushed to `deploy/self-host` runs on this PC, so treat that branch as the release branch. Merge `demo` or other branches into it when you want them live:
`git checkout deploy/self-host && git merge origin/demo && git push`.

To debug, run `start-backend.ps1` or `start-tunnel.ps1` on its own in a terminal instead. `start-backend.ps1` restarts the server if it exits.

- **Autostart** puts `rl-project-demo.cmd` in this user's Startup folder, so the site comes back after a reboot once you log in. Running before logon needs an admin-installed service (`cloudflared service install`) and a Task Scheduler task.
- **Sleep:** the PC must not sleep, or the site goes down. Set "Sleep: never" when plugged in, under Settings, then System, then Power.
- **A new zone's HTTPS certificate** can take from about 15 minutes to 24 hours after the domain becomes active on Cloudflare. Until then the browser shows a TLS or handshake error even though the tunnel is connected.

## Checks

```powershell
curl.exe -s -u demo:<password> http://127.0.0.1:8000/api/health       # local
curl.exe -s -o NUL -w "%{http_code}" https://rl-project.kwangyyinc.com/api/health   # expect 401 without a password
```

Logs are written to `deploy/logs/backend-<date>.log` and `deploy/logs/tunnel.log`.

## Without moving DNS

- **Quick tunnel:** `cloudflared tunnel --url http://127.0.0.1:8000` gives a random `*.trycloudflare.com` URL with no account or DNS change. It works for a one-off showing and changes on every restart. Keep `DEMO_PASSWORD` set.
- **Tailscale Funnel:** Tailscale is already installed here. `tailscale funnel 8000` gives `https://<pc>.<tailnet>.ts.net`, which is stable but cannot use your own domain.
- **Port forwarding and Caddy:** point an A record at your home IP and open port 443. This is not recommended, because it exposes your home IP and many ISPs block or change it.
