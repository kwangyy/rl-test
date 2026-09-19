# Calendar negotiation RL — MVP feasibility spike

This repo currently holds a **3-day feasibility spike**, not the full
CalBench-based project described in the proposal. See
[`docs/mvp-plan.md`](docs/mvp-plan.md) for why, what's in/out of scope,
and the go/no-go rule.

## Setup

```
python -m venv .venv
./.venv/Scripts/pip install -r requirements.txt   # Windows
```

## Run

```
cd src
python train_ppo.py
```

Trains PPO on `SlotNegotiationEnv` for ~100k steps and prints its reward,
deal rate, mean rounds-to-agreement, and mean accepted-slot cost
side-by-side with the first-fit baseline.

## Files

- `src/slot_negotiation_env.py` — the hand-rolled Gymnasium POMDP env.
- `src/first_fit.py` — baseline: earliest free slot, à la Calendly.
- `src/evaluate.py` — shared eval harness (reward, deal rate, rounds, cost).
- `src/train_ppo.py` — trains PPO, prints the go/no-go comparison.
- `src/concession_env.py` — Iteration 4: responsive (Boulware/Conceder/rigid) counterparts.
- `src/concede_first.py` — baseline: offer your costliest slot first, step down.
- `src/stakes_switch.py` — baseline: let low-stakes requests fail, concede otherwise.
- `src/train_ppo_concession.py` — Iteration 4 gate: PPO vs best baseline, 3 seeds.
- `docs/mvp-plan.md` — why this spike exists, scope, decision rule.
- `docs/calbench-notes.md` — CalBench access check (Task 0) — resolved
  (real repo cloned and verified working; license/reuse terms still open).
