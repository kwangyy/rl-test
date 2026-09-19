# rl-test

Two independent RL experiments. They share one venv and `requirements.txt`
at the repo root and nothing else; neither imports from the other.

| Folder | What | Status |
|---|---|---|
| [`calendar-negotiation/`](calendar-negotiation/README.md) | PPO vs heuristics on toy calendar-slot negotiation envs, plus a CalBench integration probe | Feasibility spike, iterations 1-4 done |
| [`factcheck/`](factcheck/README.md) | Claim verification with a cost-aware controller and a calibration-aware reward | Scaffold only, no model, nothing trained |

## Setup

```
python -m venv .venv
./.venv/Scripts/pip install -r requirements.txt   # Windows
```

Secrets (API keys) go in `.env` at the repo root, which is gitignored.
