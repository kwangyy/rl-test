"""Event-stream backend for the demo (AG-2): live agent runs and saved-trace replay over SSE.

    python -m agent.server                                     # base 2B verifier, http://127.0.0.1:8000
    python -m agent.server --adapter verifier/experiments/sft-direct-from-base-qwen3p5-2b-5895-samples/runs/run1/artifacts/checkpoints
    python -m agent.server --no-live                           # built UI + replay; no Bun, GPU, index or API key
    python -m agent.server --prompt-style cot_brief             # verifier explains itself; the UI shows it

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
import re
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from agent import stream
from agent.agent_loop import TRACE_DIR, reward_version
from verifier.model import BIG_MODEL, DEFAULT_MAX_NEW_TOKENS, K_PAGES, LOCAL_MODEL, PROMPT_STYLES, SEED

POLICIES = ("fixed", "prompted")
DEMO_DIR = Path(__file__).resolve().parent.parent / "demo" / "dist"


class Backends:
    """Search, verifier, controller and FEVER dev claims, each built on first use.

    Tests pass ready-made objects instead; anything passed is used as is.
    `prompt_style` picks the verifier prompt (and whether native thinking is on);
    `max_new_tokens` defaults to that style's budget."""

    def __init__(self, model=LOCAL_MODEL, adapter=None, method="bm25", candidate_pages=K_PAGES,
                 controller=BIG_MODEL, search=None, verify=None, big=None, decomposer=None, claims=None,
                 *, max_new_tokens=None, prompt_style="direct"):
        self.model, self.adapter, self.method, self.candidate_pages = model, adapter, method, candidate_pages
        self.controller = controller
        self.prompt_style = prompt_style
        self.max_new_tokens = DEFAULT_MAX_NEW_TOKENS[prompt_style] if max_new_tokens is None else max_new_tokens
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
                self._verify = LocalVerify(search, self.model, self.adapter,
                                           system_prompt=PROMPT_STYLES[self.prompt_style],
                                           max_new_tokens=self.max_new_tokens,
                                           enable_thinking=self.prompt_style == "cot_native")
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


def live_trace_path(trace_dir, name):
    """Where `--save-live` appends rows: a bare slug, never a path, so it stays in the trace directory."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name or ""):
        raise ValueError("save-live name must be a bare slug, e.g. reasoning-demo")
    return os.path.join(trace_dir, f"fever_dev_{name}.jsonl")


def live_id(path):
    """`live-<n>` for a saved free-text run: its position in the file, so Browse can key and open it."""
    try:
        with open(path, encoding="utf-8") as f:
            n = sum(1 for line in f if line.strip())
    except FileNotFoundError:
        return "live-1"
    return f"live-{n + 1}"


def saving(events, path):
    """Pass live events through, appending each finished run's row to `path`.

    The saved row is `final.row` (docs/trace-format.md), so a live run can be browsed and
    replayed like an offline one. A free-text run has no FEVER id, so it gets a `live-<n>`
    id first and keeps `dataset: "live"`, `gold: null`, `reward: null`; Browse keys rows by
    claim id, and the streamed `final` row carries the same id as the saved one.
    """
    for e in events:
        if e["event"] == "final":
            row = e["row"]
            if row.get("id") is None:
                row["id"] = live_id(path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        yield e


# normpath: TRACE_DIR runs through data/fever/.., which does not resolve before FEVER is downloaded
def create_app(backends=None, *, live=True, allow_reserved=False, cors=("*",),
               trace_dir=os.path.normpath(TRACE_DIR), save_live=None):
    backends = backends or Backends()
    live_path = live_trace_path(trace_dir, save_live) if save_live else None
    gpu = threading.Lock()  # one live run at a time on the one GPU
    app = FastAPI(title="Fact-check agent stream")
    app.state.gpu = gpu
    app.add_middleware(CORSMiddleware, allow_origins=list(cors), allow_methods=["GET"], allow_headers=["*"])

    @app.get("/api/health")
    def health():
        return {"status": "ok", "live": live, "policies": list(POLICIES) if live else [],
                "loaded": backends.loaded, "model": backends.model, "adapter": backends.adapter,
                "method": backends.method, "candidate_pages": backends.candidate_pages,
                "controller": backends.controller, "max_new_tokens": backends.max_new_tokens,
                "prompt_style": backends.prompt_style}

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
                            "reward_version": reward_version(), "seed": SEED,
                            "max_new_tokens": backends.max_new_tokens,
                            "prompt_style": backends.prompt_style}
                if policy == "fixed":
                    yield from stream.stream_fixed(c, search, verify, run_info, search_k)
                else:
                    big, decomposer = backends.controller_pair()
                    run_info = {"policy": f"prompted-{tag}", "controller": big.name, "verifier": verify.name,
                                "max_new_tokens": backends.max_new_tokens,
                                "prompt_style": backends.prompt_style,
                                "reward_version": reward_version(), "seed": SEED,
                                "max_steps": stream.prompted.MAX_STEPS,
                                "controller_prompt": stream.prompted.CONTROLLER_PROMPT}
                    yield from stream.stream_prompted(c, search, verify, big, decomposer, run_info, search_k)
            finally:
                gpu.release()

        # errors become events first, so a failed run is still saved as an error, not a truncated file
        live_events = guarded(events())
        return StreamingResponse(sse(saving(live_events, live_path) if live_path else live_events),
                                 media_type="text/event-stream",
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

    # Mount last so the built UI cannot shadow the API or interactive API docs.
    if (DEMO_DIR / "index.html").is_file():
        app.mount("/", StaticFiles(directory=DEMO_DIR, html=True), name="demo")

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=LOCAL_MODEL)
    ap.add_argument("--adapter", default=None, help="verifier LoRA dir(s), as eval_verifier.py --adapter")
    ap.add_argument("--method", choices=("bm25", "hybrid"), default="bm25")
    ap.add_argument("--candidate-pages", type=int, default=K_PAGES)
    ap.add_argument("--controller", default=BIG_MODEL, help="big model for policy=prompted (OpenRouter)")
    ap.add_argument("--prompt-style", choices=tuple(PROMPT_STYLES), default="direct",
                    help="verifier prompt: direct (JSON only) or a reasoning style (cot_native = Qwen thinking)")
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="verifier output limit (default: the prompt style's budget, e.g. 512 for cot_chain)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--cors", nargs="*", default=["*"], help="allowed browser origins")
    ap.add_argument("--no-live", action="store_true", help="serve saved traces only")
    ap.add_argument("--save-live", metavar="NAME", default=None,
                    help="append each live row to <traces>/fever_dev_<NAME>.jsonl, so Browse and replay show it")
    ap.add_argument("--allow-reserved", action="store_true", help="allow calibration/holdout claim ids")
    a = ap.parse_args()
    if a.candidate_pages < 1:
        ap.error("candidate-pages must be positive")
    if a.max_new_tokens is not None and a.max_new_tokens < 1:
        ap.error("max-new-tokens must be positive")
    if a.save_live:
        try:
            live_trace_path(os.path.normpath(TRACE_DIR), a.save_live)
        except ValueError as exc:
            ap.error(str(exc))
    import uvicorn
    app = create_app(Backends(a.model, a.adapter, a.method, a.candidate_pages, a.controller,
                             max_new_tokens=a.max_new_tokens, prompt_style=a.prompt_style),
                     live=not a.no_live, allow_reserved=a.allow_reserved, cors=a.cors,
                     save_live=a.save_live)
    uvicorn.run(app, host=a.host, port=a.port)
