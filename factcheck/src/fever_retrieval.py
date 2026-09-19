"""Query side of the FEVER BM25 index built by build_bm25_index.py.

    r = FeverRetriever()
    pages = r.search(["claim one", "claim two"], k=5)   # -> [[page_id, ...], ...]
    r.sentences("Nikolaj_Coster-Waldau")                 # -> [(0, "text"), ...]
"""

import json
import os
import re
import sqlite3

import bm25s
import Stemmer
from bm25s.tokenization import Tokenizer

from build_bm25_index import DATA, unescape

MAX_TITLE_WORDS = 8
PAREN = re.compile(r"^(.*) \([^()]*\)$")


def _norm(title):
    return unescape(title.replace("_", " ")).lower()


class FeverRetriever:
    def __init__(self, data_dir=DATA):
        index = os.path.join(data_dir, "bm25_index")
        self.bm25 = bm25s.BM25.load(index, mmap=True)
        self.tokenizer = Tokenizer(stemmer=Stemmer.Stemmer("english"), stopwords="en")
        self.tokenizer.load_vocab(index)
        self.tokenizer.load_stopwords(index)
        with open(os.path.join(index, "doc_ids.json"), encoding="utf-8") as f:
            self.doc_ids = json.load(f)
        self.db = sqlite3.connect(os.path.join(data_dir, "pages.sqlite"), check_same_thread=False)

    def search(self, queries, k=5):
        q = self.tokenizer.tokenize(queries, update_vocab=False, show_progress=False)
        idx, _ = self.bm25.retrieve(q, k=k, show_progress=False, n_threads=8)
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
        q = self.tokenizer.tokenize(queries, update_vocab=False, show_progress=False)
        idx, _ = self.bm25.retrieve(q, k=k, show_progress=False, n_threads=8)
        out = []
        for claim, toks, row in zip(queries, q, idx):
            exact, variant = self.title_matches(claim, longest_only)
            if len(exact) + len(variant) > 1 and toks:
                scores = self.bm25.get_scores(toks)
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
