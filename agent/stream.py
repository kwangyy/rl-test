"""Step events for one claim (AG-2): the agent loops, yielding each step as it happens.

The live demo needs the episode as it unfolds, not one row at the end. Each
function here runs one claim through an existing policy and yields plain-dict
events; `agent/server.py` sends them to the browser. The last event's `row` is
exactly the trace row the batch loop writes (docs/trace-format.md), so a live
run can be saved and opened in the same viewer as an offline run.

    run     {claim, id, policy, run}                         first event
    action  {t, action, args[, controller]}                  chosen, about to execute
    step    {step}                                           one trace step, with its result
    final   {row}                                            the full trace row
    error   {message}                                        the run stopped

Events from `replay` have the same shape, read back from a saved trace.

A claim is a FEVER row ({id, claim, label, evidence}) or free text
({id: None, claim, label: None}). Free text has no gold, so its row has
`gold: null`, `gold_evidence: []`, `reward: null` and `dataset: "live"`.
"""

import time
from concurrent.futures import ThreadPoolExecutor

from agent import agent_loop, prompted
from verifier.model import K_PAGES, MAX_SENTS_PER_PAGE

K = K_PAGES * MAX_SENTS_PER_PAGE


def _scorable(c):
    """The claim as the row builders need it; free text gets a placeholder gold that _ungold removes."""
    return c if c.get("label") else {**c, "label": "NOT ENOUGH INFO", "evidence": []}


def _ungold(row, c):
    if not c.get("label"):
        row.update(dataset="live", split=None, gold=None, gold_evidence=[], reward=None)
    return row


def _start(c, run_info):
    return {"event": "run", "claim": c["claim"], "id": c.get("id"), "policy": run_info.get("policy"), "run": run_info}


def stream_fixed(c, search, verify, run_info, search_k=K):
    """agent_loop's fixed policy on one claim: search the claim, verify every hit, commit or abstain."""
    yield _start(c, run_info)
    yield {"event": "action", "t": 0, "action": "search", "args": {"query": c["claim"], "k": search_k}}
    t0 = time.time()
    hits = search.batch([c["claim"]], search_k)[0]
    s_search = time.time() - t0
    yield {"event": "step", "step": {"t": 0, "action": "search", "args": {"query": c["claim"], "k": search_k},
                                     "result": {"hits": hits}, "seconds": round(s_search, 3)}}
    ids = [f"{h['page']}:{h['line']}" for h in hits]
    yield {"event": "action", "t": 1, "action": "verify", "args": {"claim": c["claim"], "evidence": ids}}
    t0 = time.time()
    res = verify.batch([c["claim"]], [ids])[0]
    s_verify = time.time() - t0
    row = _ungold(agent_loop.to_row(_scorable(c), hits, ids, res, run_info, s_search, s_verify, search_k), c)
    for step in row["steps"][1:]:
        yield {"event": "step", "step": step}
    yield {"event": "final", "row": row}


def stream_prompted(c, search, verify, big, decomposer, run_info, search_k=K):
    """prompted.py's controller on one claim, one decision at a time, up to prompted.MAX_STEPS."""
    yield _start(c, run_info)
    ep = prompted.Episode(_scorable(c))
    ep.texts = {}
    with ThreadPoolExecutor(2) as pool:
        while ep.end is None:
            if len(ep.steps) >= prompted.MAX_STEPS:
                ep.end = "budget"
                break
            a, ctl = prompted.decide(ep, big)
            yield {"event": "action", "t": len(ep.steps), "action": a["action"],
                   "args": {k: v for k, v in a.items() if k != "action"}, "controller": ctl}
            prompted.act([ep], [(a, ctl)], search, verify, decomposer, pool, search_k)
            yield {"event": "step", "step": ep.steps[-1]}
    yield {"event": "final", "row": _ungold(prompted.to_row(ep, run_info), c)}


def replay(row, delay=0.0, sleep=time.sleep):
    """A saved trace row as the same events, `delay` seconds apart, for Browse mode and frontend work."""
    yield {"event": "run", "claim": row["claim"], "id": row.get("id"),
           "policy": row.get("run", {}).get("policy"), "run": row.get("run", {}), "replay": True}
    for step in row["steps"]:
        sleep(delay)
        action = {"event": "action", "t": step["t"], "action": step["action"], "args": step["args"]}
        if "controller" in step:
            action["controller"] = step["controller"]
        yield action
        sleep(delay)
        yield {"event": "step", "step": step}
    yield {"event": "final", "row": row}
