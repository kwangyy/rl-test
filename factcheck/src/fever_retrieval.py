"""Query side of the FEVER BM25 index built by build_bm25_index.py.

    r = FeverRetriever()
    pages = r.search(["claim one", "claim two"], k=5)   # -> [[page_id, ...], ...]
    r.sentences("Nikolaj_Coster-Waldau")                 # -> [(0, "text"), ...]
"""

import json
import os
import sqlite3

import bm25s
import Stemmer
from bm25s.tokenization import Tokenizer

from build_bm25_index import DATA, unescape


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
