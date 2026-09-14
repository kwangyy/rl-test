"""Feasibility test #2 -- results in docs/mvp-plan.md ("Iteration 3").

NOT runnable standalone from this repo's own venv -- this imports
`calendar_game`, which only exists inside a CalBench checkout. To
reproduce: `git clone https://github.com/bosonphoton/calbench2026.git`,
`uv sync` inside it, drop this file in that repo's root, then
`uv run python calbench_rlclient_probe.py`. Kept here as a record of what
was tested and how, not as a script this repo can execute.

A minimal custom agent ("RLClient") plugged into the real CalBench engine
via its public testing seam (`CalendarGame._run_with_agents`), so we
don't touch game.py at all.

Scope, deliberately cut down for an MVP test:
- No cheap-talk / negotiation dialogue yet (turn() always passes). The
  agent goes straight to deciding, based only on its own calendar.
- Policy is a hand-coded heuristic (random / greedy-cheapest-slot), not a
  trained PPO model yet. The point of this script is to prove the plug-in
  seam and action format work against the REAL engine, and to see whether
  our toy-spike finding ("greedy-cost-first is a surprisingly strong
  baseline") replicates here -- not to train RL yet. That's the next,
  separate milestone (see docs/mvp-plan.md).
"""

import random
import statistics

from calendar_game.agents import Agent, BaseClient, DecideResult, TurnResult
from calendar_game.calendar import Calendar
from calendar_game.clients import DSMClient
from calendar_game.game import CalendarGame


class RLClient(BaseClient):
    """Minimal custom agent. `policy` is 'random' or 'greedy'."""

    def __init__(self, calendar: Calendar, policy: str = "greedy"):
        self.calendar = calendar  # live reference -- reflects moves/reschedules as they happen
        self.policy = policy

    def register(self, agent_id, game_config):
        self.agent_id = agent_id

    def start_round(self, meeting, calendar_render, round_num):
        pass

    def turn(self, messages, turn_index=None, max_turns_per_round=None):
        return TurnResult(tool_calls=[], text=None, thinking=None, usage=None, latency_ms=None, raw=None)

    def decide(self, meeting, calendar_render):
        free_slots = [i for i in range(self.calendar.num_slots) if self.calendar.is_free(i)]
        if not free_slots:
            return DecideResult(tool_calls=[], text=None, thinking=None, usage=None, latency_ms=None, raw=None)
        if self.policy == "random":
            slot = random.choice(free_slots)
        else:  # greedy: cheapest slot to occupy is meaningless here (free slots have no cost);
            # "greedy" for this decision is simply the earliest free slot -- there's no
            # per-slot cost to self until something occupies it, so ranking free slots
            # needs a different signal: prefer slots that don't sit next to other
            # occupied (costly) items is out of scope for this probe; use the
            # lowest-index free slot as a stand-in ("first-fit-among-free"),
            # since CalBench's own cost model charges cost only for *moving things*,
            # not for which free slot a new meeting lands in.
            slot = free_slots[0]
        return DecideResult(
            tool_calls=[{"type": "schedule", "meeting_id": meeting["id"], "slot": slot}],
            text=None, thinking=None, usage=None, latency_ms=None, raw=None,
        )


def make_calendar(num_slots, slots_data):
    cal = Calendar(num_slots=num_slots)
    cal.slots = list(slots_data)
    return cal


def run_one(seed, agent0_type, num_agents=2, num_slots=16, num_meetings=3, density=0.6):
    config = {
        "game_name": "calendar",
        "num_agents": num_agents,
        "num_slots": num_slots,
        "density": density,
        "pref_level": 1,
        "num_meetings": num_meetings,
        "num_participants": num_agents,
        "seed": seed,
        "agents": [{"type": "dsm"} for _ in range(num_agents)],  # placeholder, unused when we pass agents in
    }
    game = CalendarGame(config)
    scenario = game.generate_scenario()

    agents = []
    for agent_id in range(num_agents):
        cal = make_calendar(num_slots, scenario["calendars"][agent_id])
        if agent_id == 0 and agent0_type in ("random", "greedy"):
            client = RLClient(cal, policy=agent0_type)
        else:
            client = DSMClient()
        agent = Agent(client)
        agent.calendar = cal
        agents.append(agent)

    trace = game._run_with_agents(agents, scenario)
    return trace.metrics


def summarize(label, metrics_list):
    realized = [m["realized_cost"] for m in metrics_list]
    optimal = [m["optimal_cost"] for m in metrics_list]
    greedy_shipped = [m["greedy_cost"] for m in metrics_list]
    success = [m["success_score"] for m in metrics_list]
    headline = [m["headline_score"] for m in metrics_list]
    regret = [r - o for r, o in zip(realized, optimal)]
    print(f"\n{label}  (n={len(metrics_list)})")
    print(f"  mean realized cost      : {statistics.mean(realized):.2f}")
    print(f"  mean CP-SAT optimal cost: {statistics.mean(optimal):.2f}")
    print(f"  mean shipped-greedy cost: {statistics.mean(greedy_shipped):.2f}")
    print(f"  mean regret vs oracle   : {statistics.mean(regret):.2f}")
    print(f"  mean success_score      : {statistics.mean(success):.3f}")
    print(f"  mean headline_score     : {statistics.mean(headline):.3f}")


if __name__ == "__main__":
    seeds = list(range(100, 130))  # 30 real scenarios

    for label, agent0_type in [
        ("DSM vs DSM (both real baseline)", "dsm"),
        ("RLClient(random) vs DSM", "random"),
        ("RLClient(first-fit-among-free) vs DSM", "greedy"),
    ]:
        results = [run_one(seed, agent0_type) for seed in seeds]
        summarize(label, results)
