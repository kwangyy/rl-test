"""Query side of the FEVER BM25 index built by build_bm25_index.py.

    r = FeverRetriever(method="bm25")  # or "hybrid"
    hits = r.search(["claim one", "claim two"], k=5) # -> [[{"page": ..., "line": ..., "text": ..., "score": ...}, ...], ...]
    pages = r.search_pages(["claim one"], k=5)  # -> [[page_id, ...]]
    r.sentences("Nikolaj_Coster-Waldau")        # -> [(line, "text"), ...]

Search k counts sentences; search_pages k counts pages.
"""

from copy import copy
from functools import lru_cache
import json
import math
import os
import re
import sqlite3

import bm25s
import Stemmer
from bm25s.tokenization import Tokenizer

from core.interfaces import Hit
from retrieval.build_bm25_index import DATA, unescape

MAX_TITLE_WORDS = 8
PAREN = re.compile(r"^(.*) \([^()]*\)$")
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DENSE_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"


def _norm(title, *, lower=True):
    """Decode a page title; lowercase for matching, preserve case for context."""
    text = unescape(title.replace("_", " "))
    return text.lower() if lower else text


def _unique_hits(hits: list[Hit]) -> list[Hit]:
    """Keep the first occurrence of each stable sentence id, in input order."""
    seen = set()
    out = []
    for h in hits:
        key = (h["page"], h["line"])
        if key not in seen and h["text"].strip():
            seen.add(key)
            out.append(dict(h))
    return out


class FeverRetriever:
    """Retrieve candidate pages and return top-k sentences using BM25 or hybrid."""

    def __init__(self, data_dir=DATA, *, method="bm25", candidate_pages=5):
        """Load the saved index/database and set the method and candidate-page limit."""
        if method not in ("bm25", "hybrid"):
            raise ValueError("method must be bm25 or hybrid")
        if candidate_pages < 1:
            raise ValueError("candidate_pages must be positive")
        self.method = method
        self.candidate_pages = candidate_pages

        index = os.path.join(data_dir, "bm25_index")
        self.page_bm25 = bm25s.BM25.load(index, mmap=True)
        self.page_tokenizer = Tokenizer(stemmer=Stemmer.Stemmer("english"), stopwords="en")
        self.page_tokenizer.load_vocab(index)
        self.page_tokenizer.load_stopwords(index)
        # Reuse page preprocessing rules, but never share its saved token ids.
        self.sentence_tokenizer = copy(self.page_tokenizer)
        self.sentence_tokenizer.reset_vocab()
        with open(os.path.join(index, "doc_ids.json"), encoding="utf-8") as f:
            self.doc_ids = json.load(f)
        self.db = sqlite3.connect(os.path.join(data_dir, "pages.sqlite"), check_same_thread=False)

    def search(self, queries: list[str], k: int) -> list[list[Hit]]:
        """Retrieve up to k evidence sentences per query, preserving query order.

        This is the shared pipeline for agent tools, GRPO and evaluation.
        It accepts a list even for one query; callers own their tool interfaces.
        """
        if k < 0:
            raise ValueError("k must be nonnegative")
        if self.method not in ("bm25", "hybrid"):
            raise ValueError("method must be bm25 or hybrid")
        out = [[] for _ in queries]
        active = []
        for i, query in enumerate(queries):
            if not k:
                continue
            if self.method == "bm25" and not self._tokenize_sentences([query])[0]:
                continue
            if self.method == "hybrid" and not query.strip():
                continue
            active.append(i)
        if not active:
            return out

        # 1. Candidate pages: title matching plus the saved page BM25 index.
        pages_per_query = self.search_with_titles([queries[i] for i in active], k=self.candidate_pages)
        # 2. Candidate sentences: all numbered lines from the selected pages.
        for i, pages in zip(active, pages_per_query):
            hits = self._collect_sentences(pages)
            # 3. Rank the combined pool, then apply the evidence budget once.
            if self.method == "hybrid":
                hits = self._rank_hybrid(queries[i], hits)
            elif self.method == "bm25":
                hits = self._rank_bm25(queries[i], hits)
            out[i] = hits[:k]
        return out

    def search_pages(self, queries, k=5):
        """Page BM25 candidates; k counts pages. search() returns sentence hits."""
        q = self.page_tokenizer.tokenize(queries, update_vocab=False, show_progress=False)
        idx, _ = self.page_bm25.retrieve(q, k=k, show_progress=False, n_threads=8)
        return [[self.doc_ids[i] for i in row] for row in idx]

    def _build_title_index(self):
        """Lower-cased title -> doc index, and base title (parenthetical
        disambiguator stripped, e.g. "Titanic (1997 film)") -> doc indices."""
        self.exact, self.base, self.disambig = {}, {}, set()
        for i, pid in enumerate(self.doc_ids):
            t = _norm(pid)
            self.exact.setdefault(t, i)
            if t.endswith("(disambiguation)"):
                self.disambig.add(i)
            m = PAREN.match(t)
            if m:
                self.base.setdefault(m.group(1), []).append(i)

    def title_matches(self, claim, longest_only=True):
        """(exact, variant) doc indices whose title appears verbatim in the claim.

        exact: the title itself ("Titanic"). variant: same title with a
        disambiguator ("Titanic (1997 film)"); "(disambiguation)" pages are
        never returned since they hold no evidence. Only spans starting with
        a capital or digit are tried. With longest_only, a span inside a
        longer matching span is dropped ("Harry Potter" loses to "Harry
        Potter and the Goblet of Fire")."""
        if not hasattr(self, "exact"):
            self._build_title_index()
        raw = claim.split()
        spans = []
        for i, w in enumerate(raw):
            if not (w[:1].isupper() or w[:1].isdigit()):
                continue
            for j in range(min(len(raw), i + MAX_TITLE_WORDS), i, -1):
                text = " ".join(raw[i:j])
                exact, variant = [], []
                # as written ("Robert Downey Jr."), then punctuation and possessive stripped
                for span in dict.fromkeys([_norm(text), _norm(text.rstrip(",.;:!?\"")),
                                           _norm(text.rstrip(",.;:!?\"").removesuffix("'s"))]):
                    if span in self.exact:
                        exact.append(self.exact[span])
                    variant += [d for d in self.base.get(span, []) if d not in self.disambig]
                if exact or variant:
                    spans.append((i, j, exact, variant))
                    if longest_only:
                        break
        if longest_only:
            spans = [s for s in spans if not any(o[0] <= s[0] and s[1] <= o[1] and o[:2] != s[:2] for o in spans)]
        exact = list(dict.fromkeys(d for s in spans for d in s[2] if d not in self.disambig))
        variant = list(dict.fromkeys(d for s in spans for d in s[3]))
        return exact, variant

    def search_with_titles(self, queries, k=5, max_title=3, longest_only=True):
        """Up to max_title title-matched pages first -- exact titles before
        disambiguated variants, each group by BM25 score -- then fill to k
        with plain BM25 results."""
        q = self.page_tokenizer.tokenize(queries, update_vocab=False, show_progress=False)
        idx, _ = self.page_bm25.retrieve(q, k=k, show_progress=False, n_threads=8)
        out = []
        for claim, toks, row in zip(queries, q, idx):
            exact, variant = self.title_matches(claim, longest_only)
            if len(exact) + len(variant) > 1 and toks:
                scores = self.page_bm25.get_scores(toks)
                exact.sort(key=lambda i: -scores[i])
                variant.sort(key=lambda i: -scores[i])
            picked = list(dict.fromkeys((exact + variant)[:max_title] + list(row)))[:k]
            out.append([self.doc_ids[i] for i in picked])
        return out

    def sentences(self, page_id):
        """Numbered sentences of a page, PTB escapes undone. Line numbers are
        FEVER's, so (page_id, line) matches gold evidence directly."""
        row = self.db.execute("SELECT lines FROM pages WHERE id = ?", (page_id,)).fetchone()
        if row is None:
            return []
        out = []
        for line in row[0].split("\n"):
            fields = line.split("\t")
            if len(fields) >= 2 and fields[0].isdigit() and fields[1].strip():
                out.append((int(fields[0]), unescape(fields[1])))
        return out

    def _collect_sentences(self, pages):
        """Load numbered lines from selected pages; scores are assigned by ranking."""
        return _unique_hits([
            {"page": page, "line": line, "text": text, "score": 0.0}
            for page in pages
            for line, text in self.sentences(page)
        ])

    def _tokenize_sentences(self, texts):
        """Use the shared page/sentence policy with isolated sentence token ids."""
        tokenizer = copy(self.sentence_tokenizer)
        tokenizer.reset_vocab()  # copies/queries must not share mutable token ids
        return tokenizer.tokenize(texts, return_as="string", allow_empty=False, show_progress=False)

    def _rank_bm25(self, query, hits):
        """Rank all candidate sentences with bm25s; final k is applied by search()."""
        terms = sorted(set(self._tokenize_sentences([query])[0]))
        candidates = _unique_hits(hits)
        if not terms or not candidates:
            return []
        texts = [_norm(h["page"], lower=False) + " " + h["text"]
                 for h in candidates]
        docs = self._tokenize_sentences(texts)
        if any(docs):
            index = bm25s.BM25(k1=1.5, b=0.75, method="atire", idf_method="lucene", dtype="float64")
            index.index(docs, show_progress=False)
            scores = index.get_scores(terms)
        else:
            scores = [0.0] * len(candidates)
        scored = [{**hit, "score": float(score)} for hit, score in zip(candidates, scores)]
        return sorted(scored, key=lambda h: -h["score"])

    def _rank_hybrid(self, query, hits):
        """Fuse lexical and dense sentence rankings via equal-weight RRF.
        Both rankers see the same candidate sentences with their page titles.
        Stable ids/ties are retained.
        """
        rrf_c = 60
        pool = _unique_hits(hits)
        if not query.strip() or not pool:
            return []
        documents = [_norm(h["page"], lower=False) + " " + h["text"] for h in pool]
        scores = self._dense_scores(query, documents)
        if len(scores) != len(pool) or not all(math.isfinite(float(s)) for s in scores):
            raise ValueError("dense scorer must return one finite score per candidate")
        lexical = self._rank_bm25(query, pool)
        key = lambda h: (h["page"], h["line"])
        fused = {key(h): 0.0 for h in pool}
        for position, h in enumerate([h for h in lexical if h["score"] > 0], 1):
            fused[key(h)] += 1 / (rrf_c + position)
        for position, i in enumerate(sorted(range(len(pool)), key=lambda i: -scores[i]), 1):
            fused[key(pool[i])] += 1 / (rrf_c + position)
        return sorted([{**h, "score": fused[key(h)]} for h in pool], key=lambda h: -h["score"])

    @staticmethod
    @lru_cache(maxsize=1)
    def _dense_model():
        """Lazy optional MiniLM model; lexical search never loads weights."""
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(DENSE_MODEL, revision=DENSE_REVISION)
        model = AutoModel.from_pretrained(DENSE_MODEL, revision=DENSE_REVISION).eval()
        return tokenizer, model

    def _dense_scores(self, query, documents):
        """cosine scores: MiniLM masked mean pooling, L2 norm, 256-token cap."""
        import torch
        import torch.nn.functional as F
        tokenizer, model = self._dense_model()
        vectors, texts = [], [query] + documents
        with torch.inference_mode():
            for start in range(0, len(texts), 32):
                batch = tokenizer(texts[start:start + 32], padding=True, truncation=True,
                                  max_length=256, return_tensors="pt")
                tokens = model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).to(tokens.dtype)
                pooled = (tokens * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                vectors.append(F.normalize(pooled, p=2, dim=1))
        embeddings = torch.cat(vectors)
        return (embeddings[1:] @ embeddings[0]).tolist()
