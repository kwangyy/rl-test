"""The streamed loops must produce the batch loops' trace rows, and the server must stream them.

Fake search, verifier and controller, so no index, GPU or API key is needed:

    python -m pytest verifier/tests/test_agent_stream.py -q
"""

import json
import threading

import pytest

from agent import agent_loop, prompted, stream

CLAIM = {"id": 7, "claim": "Ada Lovelace wrote an algorithm.", "label": "SUPPORTS",
         "evidence": [[[0, 0, "Ada_Lovelace", 1]]]}
NEI_CLAIM = {"id": 8, "claim": "Ada Lovelace owned a cat.", "label": "NOT ENOUGH INFO", "evidence": [[[0, 0, None, None]]]}


class FakeSearch:
    def batch(self, queries, k):
        return [[{"page": "Ada_Lovelace", "line": i, "text": f"{q} sentence {i}", "score": 10.0 - i}
                 for i in range(min(k, 3))] for q in queries]


class FakeVerify:
    name = "fake-verifier"

    def __init__(self, fail=False):
        self.fail = fail

    def batch(self, claims, evidences):
        if self.fail:
            raise RuntimeError("verifier exploded")
        return [{"verdict": "SUPPORTED" if "owned" not in c else "NOT_ENOUGH_INFO",
                 "evidence": ev[1:2] if "owned" not in c else [],
                 "confidence": 0.9,
                 "token_prob": {"SUPPORTED": 0.9, "REFUTED": 0.05, "NOT_ENOUGH_INFO": 0.05},
                 "_tokens": (100, 5)} for c, ev in zip(claims, evidences)]


class ScriptedController:
    """Search, decompose, verify a sub-claim, verify the claim, commit: chosen from the observation
    alone, so the batch and streamed runs make the same decisions."""
    name = "fake-controller"

    def __init__(self, always=None):
        self.always = always

    def __call__(self, system, user):
        if self.always:
            obj = self.always
        elif "search '" not in user:
            obj = {"action": "search", "query": "Ada Lovelace algorithm"}
        elif "decompose ->" not in user:
            obj = {"action": "decompose"}
        elif "verify sub-claim" not in user:
            obj = {"action": "verify", "subclaim": 0}
        elif "verify claim" not in user:
            obj = {"action": "verify", "subclaim": None}
        else:
            obj = {"action": "commit"}
        return {"text": json.dumps(obj), "in_tokens": 50, "out_tokens": 4}


class FakeDecomposer:
    def run(self, claim):
        return {"subclaims": [claim, "Ada Lovelace was a writer."], "tokens_in": 30, "tokens_out": 8, "cached": False}


def timeless(x):
    """Drop wall-clock fields, which differ between any two runs."""
    if isinstance(x, dict):
        return {k: timeless(v) for k, v in x.items() if k != "seconds"}
    if isinstance(x, list):
        return [timeless(v) for v in x]
    return x


RUN = {"policy": "fixed-bm25-p5-k3", "verifier": "fake-verifier", "reward_version": None, "seed": 0}


@pytest.mark.parametrize("claim", [CLAIM, NEI_CLAIM])
def test_fixed_stream_matches_batch_row(claim):
    batch = agent_loop.run_chunk([claim], FakeSearch(), FakeVerify(), RUN, search_k=3)[0]
    events = list(stream.stream_fixed(claim, FakeSearch(), FakeVerify(), RUN, search_k=3))
    assert [e["event"] for e in events] == ["run", "action", "step", "action", "step", "action", "step", "final"]
    row = events[-1]["row"]
    assert timeless(row) == timeless(batch)
    assert [e["step"] for e in events if e["event"] == "step"] == row["steps"]
    # same action/step sequence as a replay of the saved row
    replay = list(stream.replay(row))
    assert [e["event"] for e in replay] == ["run", "action", "step", "action", "step", "action", "step", "final"]
    assert [(e["t"], e["action"], e["args"]) for e in events if e["event"] == "action"] == \
           [(e["t"], e["action"], e["args"]) for e in replay if e["event"] == "action"]
    assert row["final"]["ended_by"] == ("commit" if claim is CLAIM else "abstain")


def test_prompted_stream_matches_batch_row():
    run = {**RUN, "policy": "prompted-bm25-p5-k3", "controller": "fake-controller"}
    batch = prompted.run_chunk([CLAIM], FakeSearch(), FakeVerify(), ScriptedController(), FakeDecomposer(),
                               run, search_k=3)[0]
    events = list(stream.stream_prompted(CLAIM, FakeSearch(), FakeVerify(), ScriptedController(),
                                         FakeDecomposer(), run, search_k=3))
    row = events[-1]["row"]
    assert [s["action"] for s in row["steps"]] == ["search", "decompose", "verify", "verify", "commit"]
    assert timeless(row) == timeless(batch)
    steps = [e["step"] for e in events if e["event"] == "step"]
    actions = [e for e in events if e["event"] == "action"]
    assert steps == row["steps"]
    assert [a["action"] for a in actions] == [s["action"] for s in steps]
    assert all(a["t"] == s["t"] for a, s in zip(actions, steps))


def test_prompted_stream_stops_at_budget():
    run = {**RUN, "policy": "prompted"}
    events = list(stream.stream_prompted(CLAIM, FakeSearch(), FakeVerify(),
                                         ScriptedController(always={"action": "search", "query": "again"}),
                                         FakeDecomposer(), run, search_k=3))
    row = events[-1]["row"]
    assert len(row["steps"]) == prompted.MAX_STEPS
    assert row["final"]["ended_by"] == "budget"
    assert row["final"]["verdict"] is None


def test_free_text_claim_has_no_gold_or_reward():
    c = {"id": None, "claim": "Ada Lovelace wrote an algorithm.", "label": None}
    row = list(stream.stream_fixed(c, FakeSearch(), FakeVerify(), RUN, search_k=3))[-1]["row"]
    assert (row["dataset"], row["split"], row["gold"], row["gold_evidence"], row["reward"]) == \
        ("live", None, None, [], None)
    assert row["final"]["verdict"] == "SUPPORTED"


def test_replay_rebuilds_the_row():
    row = agent_loop.run_chunk([CLAIM], FakeSearch(), FakeVerify(), RUN, search_k=3)[0]
    events = list(stream.replay(row, delay=0.0))
    assert events[0]["event"] == "run" and events[0]["replay"] is True
    assert [e["step"] for e in events if e["event"] == "step"] == row["steps"]
    assert events[-1] == {"event": "final", "row": row}


# --- server -------------------------------------------------------------------------------------

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from agent.server import Backends, create_app  # noqa: E402


def parse_sse(text):
    out = []
    for block in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        e = json.loads(fields["data"])
        assert fields["event"] == e["event"]
        out.append(e)
    return out


def client(tmp_path, verify=None, **kw):
    backends = Backends(search=FakeSearch(), verify=verify or FakeVerify(), big=ScriptedController(),
                        decomposer=FakeDecomposer(), claims={"7": CLAIM, "8": NEI_CLAIM})
    return TestClient(create_app(backends, trace_dir=str(tmp_path), **kw))


def test_live_by_id_streams_a_trace_row(tmp_path):
    r = client(tmp_path).get("/api/live", params={"id": 7, "policy": "fixed", "search_k": 3})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    assert events[0]["event"] == "run" and events[-1]["event"] == "final"
    row = events[-1]["row"]
    assert row["id"] == 7 and row["gold"] == "SUPPORTED" and row["reward"] is not None
    assert row["run"]["policy"] == "fixed-bm25-p5-k3"


def test_live_prompted_and_free_text(tmp_path):
    r = client(tmp_path).get("/api/live", params={"claim": "Ada Lovelace wrote an algorithm.", "policy": "prompted"})
    row = parse_sse(r.text)[-1]["row"]
    assert row["gold"] is None and row["final"]["ended_by"] == "commit"
    assert row["run"]["controller"] == "fake-controller"


def test_live_rejects_bad_requests(tmp_path):
    c = client(tmp_path)
    assert c.get("/api/live").status_code == 422
    assert c.get("/api/live", params={"id": 7, "claim": "x"}).status_code == 422
    assert c.get("/api/live", params={"claim": "   "}).status_code == 422
    assert c.get("/api/live", params={"id": 7, "policy": "random"}).status_code == 422
    assert c.get("/api/live", params={"id": 999}).status_code == 404
    assert client(tmp_path, live=False).get("/api/live", params={"id": 7}).status_code == 503


def test_live_refuses_reserved_claims(tmp_path):
    from utils.fever_eval import reserved_ids
    held = sorted(reserved_ids())
    if not held:
        pytest.skip("no reserved manifests in this checkout")
    assert client(tmp_path).get("/api/live", params={"id": held[0]}).status_code == 403


def test_live_reports_errors_as_events(tmp_path):
    events = parse_sse(client(tmp_path, verify=FakeVerify(fail=True)).get("/api/live", params={"id": 7}).text)
    assert events[-1] == {"event": "error", "message": "RuntimeError: verifier exploded"}


def test_live_runs_queue_one_at_a_time(tmp_path):
    c = client(tmp_path)
    gpu = c.app.state.gpu
    gpu.acquire()  # another claim is running
    released = threading.Timer(1.0, gpu.release)
    released.start()
    with c.stream("GET", "/api/live", params={"id": 7}) as r:
        events = parse_sse(r.read().decode())
    released.join()
    assert events[0]["event"] == "queued" and events[-1]["event"] == "final"
    assert not gpu.locked()


def test_trace_endpoints(tmp_path):
    rows = agent_loop.run_chunk([CLAIM, NEI_CLAIM], FakeSearch(), FakeVerify(), RUN, search_k=3)
    (tmp_path / "fever_dev_fake_n2.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    c = client(tmp_path)
    assert c.get("/api/traces").json() == [{"name": "fever_dev_fake_n2.jsonl", "rows": 2}]
    items = c.get("/api/traces/fever_dev_fake_n2.jsonl").json()["items"]
    assert [(i["id"], i["correct"]) for i in items] == [(7, True), (8, True)]
    assert c.get("/api/traces/fever_dev_fake_n2.jsonl/8").json() == rows[1]
    events = parse_sse(c.get("/api/traces/fever_dev_fake_n2.jsonl/7/replay", params={"delay": 0}).text)
    assert events[-1]["row"] == rows[0]
    assert c.get("/api/traces/fever_dev_fake_n2.jsonl/99").status_code == 404
    assert c.get("/api/traces/..%2Fsecret.jsonl").status_code == 404
    assert c.get("/api/traces/nope.jsonl").status_code == 404


def test_built_demo_shares_server_with_api(tmp_path, monkeypatch):
    from agent import server

    demo = tmp_path / "demo"
    demo.mkdir()
    (demo / "index.html").write_text('<script src="/app.js"></script>', encoding="utf-8")
    (demo / "app.js").write_text('console.log("demo");', encoding="utf-8")
    monkeypatch.setattr(server, "DEMO_DIR", demo)
    c = client(tmp_path)

    assert c.get("/").text == '<script src="/app.js"></script>'
    asset = c.get("/app.js")
    assert asset.status_code == 200 and "javascript" in asset.headers["content-type"]
    assert c.get("/api/health").json()["status"] == "ok"
    events = parse_sse(c.get("/api/live", params={"id": 7}).text)
    assert events[-1]["event"] == "final"
    assert c.get("/docs").status_code == 200
    assert c.get("/api/missing").status_code == 404
    assert c.get("/missing.js").status_code == 404


@pytest.mark.parametrize("empty_directory", [False, True])
def test_api_runs_without_built_demo(tmp_path, monkeypatch, empty_directory):
    from agent import server

    demo = tmp_path / "demo"
    if empty_directory:
        demo.mkdir()
    monkeypatch.setattr(server, "DEMO_DIR", demo)
    c = client(tmp_path, live=False)
    assert c.get("/api/health").json()["live"] is False
    assert c.get("/").status_code == 404
