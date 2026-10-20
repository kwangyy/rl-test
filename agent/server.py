"""Event-stream backend for the demo (AG-2): live agent runs and saved-trace replay over SSE.

    python -m agent.server                                     # base 2B verifier, http://127.0.0.1:8000
    python -m agent.server --adapter verifier/experiments/sft-direct-from-base-qwen3p5-2b-5895-samples/runs/run1/artifacts/checkpoints
    python -m agent.server --no-live                           # replay saved traces only; no GPU, index or API key

Endpoints (spec and event schema: verifier/docs/live-backend.md):

    GET /api/health                              loaded backends and settings
    GET /api/live?id=<fever dev id>&policy=fixed|prompted
    GET /api/live?claim=<free text>&policy=...   SSE: run, action, step, ..., final
    GET /api/traces                              saved trace files under verifier/data/traces/
    GET /api/traces/{name}                       one summary per claim
    GET /api/traces/{name}/{id}                  one full trace row
    GET /api/traces/{name}/{id}/replay?delay=    SSE: the saved row as live events

The verifier and the FEVER index load on the first live request. Live runs share
one GPU, so they run one at a time; a request that has to wait gets a `queued`
event first. Claims in the reserved calibration and holdout sets are refused
unless the server is started with --allow-reserved.
"""

import argparse
import json
import os
import threading

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from agent import stream
from agent.agent_loop import TRACE_DIR, reward_version
from verifier.model import BIG_MODEL, K_PAGES, LOCAL_MODEL, SEED

POLICIES = ("fixed", "prompted")


class Backends:
    """Search, verifier, controller and FEVER dev claims, each built on first use.

    Tests pass ready-made objects instead; anything passed is used as is."""

    def __init__(self, model=LOCAL_MODEL, adapter=None, method="bm25", candidate_pages=K_PAGES,
                 controller=BIG_MODEL, search=None, verify=None, big=None, decomposer=None, claims=None):
        self.model, self.adapter, self.method, self.candidate_pages = model, adapter, method, candidate_pages
        self.controller = controller
        self._search, self._verify, self._big, self._decomposer, self._claims = search, verify, big, decomposer, claims
        self.build = threading.Lock()

    @property
    def loaded(self):
        return {"search": self._search is not None, "verify": self._verify is not None,
                "controller": self._big is not None, "claims": self._claims is not None}

    def search(self):
        with self.build:
            if self._search is None:
                from agent.agent_loop import FeverSearch
                self._search = FeverSearch(method=self.method, candidate_pages=self.candidate_pages)
            return self._search

    def verify(self):
        search = self.search()
        with self.build:
            if self._verify is None:
                from agent.agent_loop import LocalVerify
                self._verify = LocalVerify(search, self.model, self.adapter)
            return self._verify

    def controller_pair(self):
        with self.build:
            if self._big is None:
                from agent.decompose import BigModel
                self._big = BigModel(self.controller)
            if self._decomposer is None:
                from agent.decompose import Decomposer, cache_path
                self._decomposer = Decomposer(self._big, cache_path(self._big.name, 900))
            return self._big, self._decomposer

    def claim(self, claim_id):
        with self.build:
            if self._claims is None:
                from verifier.model import DATA, SPLIT_FILE
                with open(os.path.join(DATA, SPLIT_FILE["dev"]), encoding="utf-8") as f:
                    self._claims = {str(r["id"]): r for r in map(json.loads, f)}
            return self._claims.get(str(claim_id))


def sse(events):
    """Server-sent events: one `event:` / `data:` block per dict, numbered from 0."""
    for i, e in enumerate(events):
        yield f"id: {i}\nevent: {e['event']}\ndata: {json.dumps(e, ensure_ascii=False)}\n\n"


def guarded(events):
    """Turn an exception inside a run into a final `error` event instead of a dropped connection."""
    try:
        yield from events
    except Exception as exc:  # noqa: BLE001 -- reported to the client, which has no other channel
        yield {"event": "error", "message": f"{type(exc).__name__}: {exc}"}


def trace_path(trace_dir, name):
    """A trace file by base name; nothing outside the trace directory."""
    if os.path.basename(name) != name or not name.endswith(".jsonl"):
        raise HTTPException(404, "unknown trace")
    path = os.path.join(trace_dir, name)
    if not os.path.isfile(path):
        raise HTTPException(404, "unknown trace")
    return path


def read_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# normpath: TRACE_DIR runs through data/fever/.., which does not resolve before FEVER is downloaded
def create_app(backends=None, *, live=True, allow_reserved=False, cors=("*",),
               trace_dir=os.path.normpath(TRACE_DIR)):
    backends = backends or Backends()
    gpu = threading.Lock()  # one live run at a time on the one GPU
    app = FastAPI(title="Fact-check agent stream")
    app.state.gpu = gpu
    app.add_middleware(CORSMiddleware, allow_origins=list(cors), allow_methods=["GET"], allow_headers=["*"])

    @app.get("/api/health")
    def health():
        return {"status": "ok", "live": live, "policies": list(POLICIES) if live else [],
                "loaded": backends.loaded, "model": backends.model, "adapter": backends.adapter,
                "method": backends.method, "candidate_pages": backends.candidate_pages,
                "controller": backends.controller}

    @app.get("/api/live")
    def run_live(claim: str | None = None, id: str | None = None,
                 policy: str = Query("fixed", pattern="^(fixed|prompted)$"),
                 search_k: int = Query(stream.K, ge=1, le=stream.K)):
        if not live:
            raise HTTPException(503, "server started with --no-live")
        if (claim is None) == (id is None):
            raise HTTPException(422, "give exactly one of claim or id")
        if id is not None:
            if not allow_reserved:
                from utils.fever_eval import reserved_ids
                if str(id) in reserved_ids():
                    raise HTTPException(403, "claim is in the reserved calibration/holdout sets")
            try:
                c = backends.claim(id)
            except FileNotFoundError:
                raise HTTPException(503, "FEVER dev file not found; see verifier/docs/data.md")
            if c is None:
                raise HTTPException(404, "no FEVER dev claim with that id")
        else:
            if not claim.strip():
                raise HTTPException(422, "empty claim")
            c = {"id": None, "claim": claim.strip(), "label": None}

        def events():
            if not gpu.acquire(blocking=False):
                yield {"event": "queued", "message": "waiting for the running claim to finish"}
                gpu.acquire()
            try:
                if not backends.loaded["verify"]:
                    yield {"event": "status", "message": "loading search index and verifier"}
                search, verify = backends.search(), backends.verify()
                tag = f"{backends.method}-p{backends.candidate_pages}-k{search_k}"
                run_info = {"policy": f"fixed-{tag}", "verifier": verify.name,
                            "reward_version": reward_version(), "seed": SEED}
                if policy == "fixed":
                    yield from stream.stream_fixed(c, search, verify, run_info, search_k)
                else:
                    big, decomposer = backends.controller_pair()
                    run_info = {"policy": f"prompted-{tag}", "controller": big.name, "verifier": verify.name,
                                "reward_version": reward_version(), "seed": SEED,
                                "max_steps": stream.prompted.MAX_STEPS,
                                "controller_prompt": stream.prompted.CONTROLLER_PROMPT}
                    yield from stream.stream_prompted(c, search, verify, big, decomposer, run_info, search_k)
            finally:
                gpu.release()

        return StreamingResponse(sse(guarded(events())), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/traces")
    def traces():
        if not os.path.isdir(trace_dir):
            return []
        out = []
        for name in sorted(os.listdir(trace_dir)):
            if name.endswith(".jsonl"):
                with open(os.path.join(trace_dir, name), encoding="utf-8") as f:
                    out.append({"name": name, "rows": sum(1 for line in f if line.strip())})
        return out

    @app.get("/api/traces/{name}")
    def trace_summary(name: str, offset: int = Query(0, ge=0), limit: int = Query(1000, ge=1, le=10000)):
        rows = read_rows(trace_path(trace_dir, name))
        return {"name": name, "rows": len(rows), "items": [
            {"id": r["id"], "claim": r["claim"], "gold": r["gold"], "verdict": r["final"]["verdict"],
             "confidence": r["final"]["confidence"], "ended_by": r["final"]["ended_by"],
             "correct": r["gold"] is not None and r["final"]["verdict"] == r["gold"],
             "n_steps": len(r["steps"]), "n_ops": r["cost"]["n_ops"]}
            for r in rows[offset:offset + limit]]}

    def find_row(name, claim_id):
        for r in read_rows(trace_path(trace_dir, name)):
            if str(r["id"]) == str(claim_id):
                return r
        raise HTTPException(404, "no row with that id in this trace")

    @app.get("/api/traces/{name}/{claim_id}")
    def trace_row(name: str, claim_id: str):
        return find_row(name, claim_id)

    @app.get("/api/traces/{name}/{claim_id}/replay")
    def trace_replay(name: str, claim_id: str, delay: float = Query(0.5, ge=0, le=10)):
        row = find_row(name, claim_id)
        return StreamingResponse(sse(guarded(stream.replay(row, delay))), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=LOCAL_MODEL)
    ap.add_argument("--adapter", default=None, help="verifier LoRA dir(s), as eval_verifier.py --adapter")
    ap.add_argument("--method", choices=("bm25", "hybrid"), default="bm25")
    ap.add_argument("--candidate-pages", type=int, default=K_PAGES)
    ap.add_argument("--controller", default=BIG_MODEL, help="big model for policy=prompted (OpenRouter)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--cors", nargs="*", default=["*"], help="allowed browser origins")
    ap.add_argument("--no-live", action="store_true", help="serve saved traces only")
    ap.add_argument("--allow-reserved", action="store_true", help="allow calibration/holdout claim ids")
    a = ap.parse_args()
    if a.candidate_pages < 1:
        ap.error("candidate-pages must be positive")
    import uvicorn
    app = create_app(Backends(a.model, a.adapter, a.method, a.candidate_pages, a.controller),
                     live=not a.no_live, allow_reserved=a.allow_reserved, cors=a.cors)
    uvicorn.run(app, host=a.host, port=a.port)
