# Retrieval results

## How search works

```text
Query → title matching + page BM25 → candidate pages
      → all numbered sentences → sentence BM25 or hybrid → top-k hits
```

**BM25** ranks keyword matches. **Hybrid** combines sentence BM25 with MiniLM
semantic similarity using reciprocal-rank fusion (combining rank positions).
Both score **title + sentence text** and rank the combined sentence pool;
hybrid does not add candidate pages. **k is the total returned sentence budget.**

`FeverRetriever` in `retrieval/fever_retrieval.py` implements retrieval.
`FeverSearch` in `agent/agent_loop.py` adapts it to the agent's single-query tool.
Hits keep `{page, line, text, score}`.

### Function reference

| Function | Purpose / return value | Meaning of k |
|---|---|---|
| `FeverRetriever.search_pages(queries, k)` | Plain page BM25; page IDs for each query. Used by the page baseline and historical verifier mode. | Pages per query |
| `FeverRetriever.search_with_titles(queries, k)` | Title matching + page BM25; page IDs for each query. Supplies candidates for complete search. | Pages per query |
| `FeverRetriever.sentences(page_id)` | Read all numbered `(line, text)` pairs from one page; no ranking. | — |
| `FeverRetriever.search(queries, k)` | Complete pipeline: select pages, collect sentences, rank using configured BM25/hybrid; a hit list per query. | Sentences per query |
| `FeverSearch.__call__(query, k)` | Agent tool, called as `tool(query, k)`; returns one hit list through `batch()`. | Sentences |
| `FeverSearch.batch(queries, k)` | Delegates multiple queries to `FeverRetriever.search()`. | Sentences per query |

## Main results — k counts sentences

| Split | Method | @1 | @3 | @5 | @10 | @20 | @30 | ms/claim |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Train | Sentence BM25 | 0.362 | 0.573 | 0.677 | 0.777 | **0.830** | 0.843 | 41 |
| Train | Hybrid | 0.484 | 0.699 | 0.761 | **0.823** | 0.843 | 0.847 | 189 |
| Dev | Sentence BM25 | 0.377 | 0.590 | 0.697 | 0.795 | **0.852** | 0.867 | 38 |
| Dev | Hybrid | 0.558 | 0.735 | 0.795 | **0.848** | 0.865 | 0.873 | 177 |

**Recall@k is evidence coverage, not verifier accuracy.** A claim counts once if
its top-k results contain every sentence in at least one complete gold evidence
set, matched by `(page, line)`. For example, dev hybrid @5 = 0.795 means **477 of
600 claims** have complete annotated evidence among five returned sentences.

- **Train:** 1,000 non-NEI claims, seed 1.
- **Dev:** 600 non-NEI claims from the frozen 900-ID development manifest; the other 300 are excluded because NEI has no gold evidence for this metric.
- **ms/claim:** average retrieval time, excluding title-index initialization; hybrid includes its first model load. Timing varies by run.

Methods use the same claims within each split and the same five-page candidate
pool. **Dev is already used development data, not an untouched final test.**
Calibration and holdout were not evaluated.

## Observations

- **Keep title matching + page BM25 with five candidate pages.** It consistently beats plain page BM25.
- **Test sentence BM25 at k=20 as the practical choice.** It returns one-third fewer sentences than k=30, losing only 1.3–1.5 percentage points of evidence coverage.
- **Test hybrid at k=10 when shorter context matters.** It returns half as many sentences as BM25 k=20 with coverage within 0.7 points, but retrieval took about **4.6× longer**.
- **Do not adopt k=3 yet:** even hybrid covers complete evidence for only about 70–74% of claims.

**Next:** compare BM25 k=20 and hybrid k=10 with the same verifier, claims and
prompt, keeping BM25 k=30 as the reference. Measure **verdict accuracy and total
cost** before changing tool defaults. Retrieval coverage alone does not establish
the best k for verification.

## Run the tests

Run from the **repository root** with the project environment (`venv/bin/python`
locally, or `python` below). Requires the local FEVER data/index; see
[data setup](data.md). Hybrid loads cached MiniLM weights or downloads them on
first use. **CPU is sufficient; verifier weights are not required.**

```bash
# Page comparison: train, then frozen dev
python -m retrieval.eval_retrieval --split train --n 1000
python -m retrieval.eval_retrieval --claim-ids verifier/data/eval/fever-dev-900-ids.json

# Sentence BM25 vs hybrid: train, then the same frozen dev set
python -m retrieval.eval_sentence_retrieval --split train --n 1000 --hybrid
python -m retrieval.eval_sentence_retrieval --claim-ids verifier/data/eval/fever-dev-900-ids.json --hybrid
```

- Omit `--hybrid` for BM25 only; use `--ks 3 5 10` to change tested sentence budgets.
- **Keep `--candidate-pages 5` fixed when comparing k.** This is the default. A frozen `--claim-ids` manifest overrides `--n`.
- Add `--output retrieval/results/my-check/run1/metrics.json` to a sentence command to save metrics and companion `.jsonl` predictions. **Use a new path for each run.** The page evaluator prints its table only.

## Supporting page results — k counts pages

These diagnose candidate-page selection, rather than final sentence selection.
`titles(max N, longest)` prefers up to N title-matched pages, then fills the page
budget with BM25. `longest` drops nested shorter title matches; `all` allows them.
The **max 3, longest** rows are the configuration used by sentence retrieval.

**Page @5 and sentence @5 are different metrics:** finding an evidence page does
not guarantee returning its correct evidence lines. The page evaluator retains
its original logic with only the `search_pages()` API correction.

### Train — 1,000 claims

| Method | @1 | @3 | @5 | @10 | ms/claim |
|---|---:|---:|---:|---:|---:|
| bm25 | 0.218 | 0.379 | 0.464 | 0.546 | 28 |
| titles(max 2, longest) | 0.662 | 0.807 | 0.833 | 0.854 | 31 |
| titles(max 2, all) | 0.587 | 0.781 | 0.814 | 0.840 | 36 |
| **titles(max 3, longest)** | 0.662 | 0.817 | 0.847 | 0.866 | 45 |
| titles(max 3, all) | 0.587 | 0.780 | 0.830 | 0.857 | 39 |
| titles(max 5, longest) | 0.662 | 0.817 | 0.853 | 0.876 | 33 |
| titles(max 5, all) | 0.587 | 0.780 | 0.816 | 0.863 | 30 |

### Dev — 600 claims

| Method | @1 | @3 | @5 | @10 | ms/claim |
|---|---:|---:|---:|---:|---:|
| bm25 | 0.280 | 0.468 | 0.533 | 0.620 | 18 |
| titles(max 2, longest) | 0.707 | 0.837 | 0.862 | 0.890 | 20 |
| titles(max 2, all) | 0.622 | 0.812 | 0.838 | 0.870 | 26 |
| **titles(max 3, longest)** | 0.707 | 0.837 | 0.873 | 0.898 | 32 |
| titles(max 3, all) | 0.622 | 0.800 | 0.847 | 0.882 | 33 |
| titles(max 5, longest) | 0.707 | 0.837 | 0.872 | 0.907 | 29 |
| titles(max 5, all) | 0.622 | 0.800 | 0.837 | 0.888 | 20 |
