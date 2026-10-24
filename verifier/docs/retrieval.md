# Retrieval: tools and TODO

Goal: collect evidence beyond local search. Experiment designs, commands and
results are in [retrieval-extension-results.md](retrieval-extension-results.md);
historical page/sentence baselines remain in [retrieval-results.md](retrieval-results.md).

## Code responsibilities

| Module | Responsibility |
|---|---|
| `fever_retrieval.py` | FEVER corpus/index access, page resolution, search and sentence BM25/hybrid ranking. |
| `follow.py` | Observe sentence anchors and open known targets; per-claim visits, budgets and trace. |
| `pipeline.py` | Optional expansion policy: select links, call tools and rerank combined candidates. |
| `exa.py` | Exa search/contents, cache, request budgets, provenance and web passage selection. |
| `metrics.py` | Shared exact evidence scoring, paired statistics and optional similarity diagnostics. |

Follow is a tool; deciding which links to open is a policy. Keep those separate.
Evaluation stays separate: `eval_retrieval.py` for page/sentence baselines,
`eval_retrieval_extension.py` for FEVER follow/Exa, and `eval_averitec.py` for the
AVeriTeC web pilot. Commands and metric interpretation are in the results document;
`evaluate_averitec` documents its callable input format in its docstring.

## Use FEVER search and follow

```python
from retrieval.fever_retrieval import FeverRetriever
from retrieval.follow import FeverFollow
from retrieval.pipeline import collect_evidence

query = "Nikolaj Coster-Waldau is a Danish actor."
r = FeverRetriever(method="bm25", candidate_pages=5)  # or method="hybrid"
try:
    hits = r.search([query], k=10)[0]  # k counts sentences.
    tool = FeverFollow(r, max_actions=5)  # One instance per claim.
    links = tool.observe(hits, pages={h["page"] for h in hits})
    if links:
        opened = tool.follow(links[0]["id"], query, k=10)
    # Alternatively, run the fixed local+follow policy:
    expanded = collect_evidence(query, r, expansion="follow", final_k=10)
finally:
    r.db.close()
```

Follow reads anchors from observed Wikipedia sentences, then returns ranked target
sentences and newly exposed links. It does not search for URLs in claim text.
Unknown, missing, visited and exhausted-budget outcomes have explicit statuses.
The fixed policy uses one hop; callers can explicitly follow additional observed
links within the tool's budget. `expansion="none"` retains the local baseline.

## Where the pipeline fits

| Call | Who decides the next action? | Current use |
|---|---|---|
| `r.search(...)` | Caller | Basic retrieval, including the existing agent search adapter. |
| `FeverFollow.observe/follow(...)` | Agent/caller chooses an observed link | Standalone tool; agent action integration remains TODO. |
| `collect_evidence(..., expansion="follow")` | Fixed FEVER policy | Used by FEVER experiments; other callers can import it directly. |

`pipeline.py` composes **search → optional expansion → merge/deduplicate → rerank
→ top-k**. `expansion="none"` is baseline; `"follow"` adds up to the link budget;
`"local"` widens local search; `"exa"` maps Exa Wikipedia URLs to snapshot pages.
It returns `retained` hits, candidate/ranking details and `trace`. It does not
automatically judge when expansion helps, and it is not wired into the agent loop.
For agent-chosen actions, use the tools directly. For actual live web text, use
`search_web` below. This pipeline is FEVER-specific, not an AVeriTeC/SciFact adapter.

## Use live search

Install the project dependencies (`exa-py` is listed in `requirements.txt`), then
configure the key before making live requests. Either export it in your shell:

```bash
export EXA_API_KEY="your-api-key"
```

Or put it in a local, Git-ignored `.env` file:

```dotenv
EXA_API_KEY=your-api-key
```

The evaluation CLIs load that file only when explicitly passed
`--allow-network --env-file .env`; also set `--max-api-requests 50` for a bounded
pilot. Direct Python callers must export the key or explicitly load `.env` before
creating requests; the Exa client does not automatically load it. Never commit the
real key. Local search, follow and offline cached analysis need no Exa key.

```python
from retrieval.exa import ExaClient, search_web

client = ExaClient("verifier/data/exa-cache", allow_network=True, max_requests=50)
web = search_web(query, client, n=5, k=10, context_characters=4000)
```

Exa reads `EXA_API_KEY` from the environment. Search is unrestricted unless the
caller supplies domain/date filters. Network access is opt-in; cached responses
consume no new requests, and HTTP 402 stops further attempts. Web passages carry
URLs, offsets and fetch hashes, not FEVER sentence IDs. `fuse_rankings` combines
caller-normalized passage lists; `pipeline`'s `expansion="exa"` instead maps
Wikipedia URLs into the frozen FEVER corpus.

## Runtime retrieval versus offline evaluation

| Component | Agent/demo use |
|---|---|
| `search_web(query, client, ...)` | Existing reusable live retrieval; returns passages/provenance without gold annotations. |
| AVeriTeC pilot `collect(...)` | Experiment collector: builds all three comparison conditions, not a production agent tool. |
| `evaluate_averitec(rows, gold_by_id, ...)` | Offline scoring after retrieval; the agent must not call it or receive its gold annotations. |

No duplicate Exa search implementation is needed. Before AVeriTeC agent use,
extract its allowed-input/date/site policy into a thin runtime adapter around
`search_web`; preserve the generic Exa client for demo/open claims. This integration
is TODO. The current agent uses `FeverSearch` and does not call `pipeline.py`.

## Retrieval TODO (canonical)

### FEVER

- [ ] Choose the smallest useful evidence budget: k=3/5/10/20/30.
- [ ] Test baseline-preserving fusion to reduce evidence displaced by expansion.
- [ ] Evaluate end-to-end hybrid follow and local + follow + Exa.
- [ ] Compare original versus rewritten queries from saved traces/rollouts.
- [ ] Test title + text versus text-only; finalize retained evidence metadata.

### Live search

- [ ] Audit cached AVeriTeC search misses, extraction failures and retention losses.
- [ ] Re-evaluate local versus local + Exa with improved evidence selection.
- [ ] Freeze the policy and validate on untouched claims; cap new pilots at 50 claims / 50 requests.

### Dataset extensions

- [ ] Extract a reusable AVeriTeC search adapter with date/source policies.
- [ ] Implement AVeriTeC follow for allowed original URLs and observed web links.
- [ ] Evaluate AVeriTeC search with/without follow and original-URL starts.
- [ ] Implement SciFact abstract search and document opening with sentence IDs.
- [ ] Implement SciFact follow using real citation/DOI links where available; do not invent a graph.
- [ ] Evaluate SciFact document/rationale coverage and follow gains.

### Finish

- [ ] Synchronize dependencies and test setup, caching, budgets and failures.
- [ ] Update results, conclusions and callable examples; keep raw outputs local.
