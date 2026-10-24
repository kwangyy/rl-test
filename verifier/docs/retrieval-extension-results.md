# Retrieval extensions: experiments and findings

[Code usage and TODO](retrieval.md) · [Original page/sentence retrieval results](retrieval-results.md)

**Main finding:** FEVER follow improves retained evidence when paired with hybrid
ranking. Exa finds additional FEVER evidence pages, but the tested live-text
selection policy displaces useful local evidence. On AVeriTeC, Exa improves
reference-relatedness over a transferred Wikipedia baseline; evidence coverage
remains low. These are retrieval experiments; no verifier accuracy was measured.

## 1. FEVER: local search + hyperlink follow

### Question and setup

**Does following links from retrieved Wikipedia sentences improve evidence recall?**

- **Sample:** all 600 gold-eligible claims in the frozen 900-ID development
  manifest. The 300 NEI claims are excluded from evidence recall.
- **Baseline:** original claim query; five initial Wikipedia pages; up to three
  longest title matches.
- **Extension:** inspect links in the first ten observed sentences, try up to five
  new targets, then rerank the combined sentence pool. Links come from Wikipedia
  annotations, rather than URLs extracted from claim text.
- **Ranking:** BM25, or equal-weight reciprocal rank fusion of BM25 and pinned
  MiniLM ranks (RRF constant 60), using title + text. Hybrid reuses the saved BM25
  pages and follow links: this tests **sentence ranking**, not hybrid traversal.

**Gold-page coverage** measures evidence-page discovery. **Recall@k** requires
all sentences from at least one acceptable gold evidence set within the final
*k* retained sentences. Finding a page does not guarantee retaining its evidence.

### Results

| Ranking / extension | Gold-page coverage | Recall @3 | Recall @10 | Recall @30 |
|---|---:|---:|---:|---:|
| BM25 baseline | 87.33% | 59.00% | 79.50% | 86.67% |
| BM25 + follow | 90.33% | 61.00% | 79.50% | 88.67% |
| Hybrid baseline | 87.33% | **73.50%** | 84.83% | 87.33% |
| Hybrid + follow | **90.33%** | 73.00% | **86.50%** | **89.33%** |

| Follow effect at @10, against matching baseline | Recovered / lost claims | Change | Paired 95% interval |
|---|---:|---:|---|
| BM25 + follow | 16 / 16 | 0.00 pp | [−1.83, +1.83] pp |
| Hybrid + follow | 14 / 4 | **+1.67 pp** | [+0.33, +3.00] pp |

Follow opened **2,491 / 2,960 attempted targets**; 469 targets were unavailable in
the frozen snapshot. Hybrid ranking recovers 13 of the 16 BM25-follow losses.
Intervals use 2,000 paired claim-bootstrap resamples, seed 0. Comparisons across
budgets are exploratory development results.

### Conclusion

**Follow improves discovery, but ranking determines whether that gain survives
selection.** BM25 gains and losses cancel at @10. Hybrid + follow retains evidence
for ten more claims than hybrid alone, although it still loses four claims and
does not improve @3.

## 2. FEVER: Exa pilot — 50 claims

### Question and setup

**Does adding Exa to local search improve discovery or retained evidence?**

The pilot uses 50 seeded claims (seed 20261024), original claim queries and up to
five **unrestricted web results** per claim, subject to benchmark-domain
exclusions. It separates two experiments:

| Experiment | What Exa contributes | Evidence evaluated |
|---|---|---|
| **Snapshot discovery** | Wikipedia URLs found through web search | Corresponding sentences from the frozen FEVER corpus |
| **Live-text extension** | Current text returned by Exa, including other domains | Local passages + live web passages |

Only the second experiment tests retained **live content**. Both reuse saved Exa
responses; the analysis made no new requests.

### A. Snapshot discovery results

| Method | Gold-page coverage | Recall @10 | Recall @30 |
|---|---:|---:|---:|
| Local baseline | 92.00% | 86.00% | 88.00% |
| Local + Exa-discovered URLs | **96.00%** | **88.00%** | **92.00%** |

At @10, Exa recovers two claims and loses one: **+2 pp**, paired 95% interval
**[−4, +8] pp**. This is a small positive discovery signal, with substantial
uncertainty on 50 claims.

### B. Live-text results

Both conditions use the available local-page sentences, **600-character
sentence-aware passages**, pooled title/text BM25 ranking and a **4,000-character
retained-text cap**. This is a passage-budget comparison; its percentages are
**not sentence Recall@k** and should not be compared directly with section 1.

| Outcome | Local baseline | Local + live search | Interpretation |
|---|---:|---:|---|
| Complete gold-page URL coverage in retained context | **92.00%** | 76.00% | Fewer claims retain all required evidence-page URLs |
| Literal complete gold-text retention | **92.00%** | 58.00% | Fewer claims retain an entire gold evidence set verbatim |
| Gold-reference cosine similarity | **0.971** | 0.896 | Retained text is less similar to required reference evidence |
| Claim-text cosine similarity | 0.752 | **0.777** | Text is more related to the claim topic |
| Mean retained characters | 3,193 | 3,943 | Same cap, different actual context lengths |

Literal retention misses paraphrases. Gold-reference cosine uses the best
acceptable evidence set and its weakest required sentence match. Neither cosine
nor literal matching establishes semantic evidence sufficiency.

| Diagnostic | Result | What it explains |
|---|---:|---|
| URL overlap with baseline candidates, mean Jaccard | 8.9% | Exa mostly discovers different sources |
| Novel live URL fraction | 84.4% | Source diversity increases |
| Baseline passages retained after expansion | 45.2% | Much local context is displaced |
| Live-text share of combined context | 66.4% | Live passages dominate selection |

Literal complete retention has **0 gains / 17 losses**, a **−34 pp** change
(interval [−48, −20] pp). All lost gold evidence is still in the candidate pool:
the failure occurs during ranking/retention. Gold-reference cosine drops by
0.075 (interval [−0.116, −0.037]), despite higher claim similarity.

### Conclusion

**Exa shows discovery potential, but this live-text merging policy hurts FEVER
evidence retention.** More novel or topic-related text does not necessarily
provide better evidence. Test protected baseline slots or rank fusion using the
cached responses before spending on a larger run.

This does not establish that live search is generally unhelpful: FEVER references
come from a frozen 2017 snapshot and favor historical wording. The matched
comparison also changes passage construction and fusion relative to the older
mixed-size diagnostic, so differences between those runs cannot be attributed to
one change alone.

**Collection:** 50 requests; 250/250 sources returned text; zero failed searches
and zero exact URL/text duplicates. Stored cost estimates total **USD 0.35**;
mean original request time was **1.87 seconds**. Extraction is provider-reported;
site availability and historical content were not independently verified.

## 3. AVeriTeC: local baseline + live search

### Question and setup

**Does Exa add useful retrieval material for open-web claims?**

- **Sample:** 50 seeded AVeriTeC development claims.
- **Baseline:** five pages retrieved from the FEVER-2017 Wikipedia corpus.
  This is a transferred local baseline, not a native AVeriTeC web baseline.
- **Extension:** add up to five current-web Exa sources per claim.
- **Selection:** 600-character passages, pooled BM25 and a 4,000-character cap.
- **Allowed inputs:** claim and date only. Gold QAs, answers, labels and source
  URLs are used only for scoring. Claim-date publication cutoffs and fixed
  fact-check/benchmark-site exclusions apply.

### Metrics and results

| Metric | What it measures / limit |
|---|---|
| **Annotated-source coverage** | Retained matches to allowed reference URLs; annotations do not list every useful source |
| **Literal answer coverage** | Fraction of eligible answers found verbatim, averaged per eligible claim; misses paraphrases |
| **QA-reference cosine** | Similarity of retained passages to reference answers, averaged within each claim; relatedness rather than sufficiency |
| **Overlap / novelty / retention** | Source diversity and context displacement; diagnostic rather than evidence-quality measures |

| Outcome | Local baseline | Local + live search |
|---|---:|---:|
| Claims retaining an annotated allowed source | 0 / 50 | **2 / 50** |
| Mean annotated-source URL recall | 0.00% | **1.50%** |
| Mean literal answer coverage, 30 eligible claims | 0.00% | **3.33%** |
| Claim-weighted QA-reference cosine | 0.475 | **0.668** |
| Mean retained characters | 3,261 | 3,928 |

QA-reference cosine increases **0.193** (paired interval [0.157, 0.229]). The two
additional source-hit claims give **+4 pp** ([0, 10] pp). Source recall falls from
2.0% in the candidate pool to 1.5% after selection. Literal answers require at
least five normalized words. Negated-reference controls also score highly,
confirming that cosine cannot be reported as semantic recall.

| Diagnostic | Result |
|---|---:|
| URL overlap / novel live URL fraction | 0.0% / 100.0% |
| Baseline passages retained after expansion | 34.4% |
| Live-text share of combined context | 77.1% |
| Failed searches / sources returning text | 0 / 242 of 242 |
| Duplicate returned URLs / texts | 0 / 1 |
| Original requests / estimated cost | 50 / USD 0.35 |
| Mean original request time | 2.07 seconds |

### Conclusion

**Exa adds more reference-related material and a few annotated-source matches,
but strong evidence coverage has not been demonstrated.** The gain is relative
to a historical Wikipedia baseline with limited coverage of these open-web
claims. It does not establish superiority over another web retriever.

Audit cached source/query misses and displaced passages before increasing the
request budget. Date handling also needs attention: **95/242 source dates are
unknown and were retained**; publication filtering cannot prove that the current
page text existed at claim time.

The earlier wider-local control had cosine 0.480 versus 0.475 for baseline. It
remains in historical artifacts and is outside this two-condition comparison.

## 4. Overall conclusion and priorities

| Extension | What the experiments support | Decision / next step |
|---|---|---|
| **FEVER follow** | More evidence pages; hybrid improves @10 by 1.67 pp on the development cohort | Keep optional; test retention protection and full hybrid traversal, then validate |
| **FEVER Exa discovery** | Small positive discovery signal in 50 claims; interval includes no gain | Preserve as a separate discovery condition; avoid a general benefit claim |
| **FEVER live text** | Tested pooled ranking displaces gold evidence despite novel, topic-related sources | Fix selection on cached data before expanding collection |
| **AVeriTeC live search** | Higher QA-relatedness and limited exact source/answer gains over the transferred baseline | Audit evidence coverage and date policy before a larger pilot |

**Keep local search as the baseline and enable extensions explicitly.** Evidence
coverage is the outcome of interest; similarity, overlap, novelty, extraction,
cost and latency help explain it. FEVER sentence evidence and AVeriTeC QAs use
different scoring units, so their cosine values are not comparable across datasets.

These findings are development/pilot results, not independent holdout validation.

## 5. Reproduce the comparisons

Run from the repository root with local datasets/indexes and cached MiniLM
weights. Use new output directories. These commands rescore cached responses and
make **zero Exa requests**.

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# FEVER: all 600 claims, saved local/follow BM25 and hybrid rankings.
venv/bin/python -m retrieval.eval_retrieval_extension snapshot \
  --analyze retrieval/results/fever-follow-hybrid-dev600-20261024-run1 \
  --output retrieval/results/NEW-FOLLOW-ANALYSIS

# FEVER: 50-claim snapshot discovery and matched live-text comparison.
venv/bin/python -m retrieval.eval_retrieval_extension web \
  --analyze retrieval/results/fever-web50-20261024-final \
  --matched-passages --semantic-similarity --output retrieval/results/NEW-FEVER-LIVE

# AVeriTeC: 50 claims, baseline versus baseline + live search.
venv/bin/python -m retrieval.eval_averitec \
  --analyze retrieval/results/averitec-web50-20261024-run1 \
  --conditions local local_exa --semantic-similarity \
  --output retrieval/results/NEW-AVERI-LIVE
```
