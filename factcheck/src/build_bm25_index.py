"""Build a document-level BM25 index over the FEVER 2017 Wikipedia dump.

Writes to factcheck/data/fever/:
  bm25_index/  - bm25s index plus the tokenizer vocab and stopwords
  pages.sqlite - page id -> numbered sentences, for evidence lookup after retrieval

Each document is the page title followed by its intro text, with FEVER's
PTB escapes (-LRB- etc.) undone. The title is included because most FEVER
claims name the page they are about.

Run from factcheck/src:  python build_bm25_index.py
"""

import glob
import json
import os
import sqlite3
import time

import bm25s
import Stemmer
from bm25s.tokenization import Tokenized, Tokenizer

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "fever")
PTB = {"-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]",
       "-LCB-": "{", "-RCB-": "}", "-COLON-": ":"}


def unescape(s):
    for k, v in PTB.items():
        s = s.replace(k, v)
    return s


def iter_pages():
    for path in sorted(glob.glob(os.path.join(DATA, "wiki-pages", "wiki-*.jsonl"))):
        with open(path, encoding="utf-8") as f:
            for line in f:
                page = json.loads(line)
                if page["id"] and page["text"]:
                    yield page


def main():
    t0 = time.time()
    db_path = os.path.join(DATA, "pages.sqlite")
    if os.path.exists(db_path):
        os.remove(db_path)
    db = sqlite3.connect(db_path)
    db.execute("CREATE TABLE pages (id TEXT PRIMARY KEY, lines TEXT)")

    tokenizer = Tokenizer(stemmer=Stemmer.Stemmer("english"), stopwords="en")
    doc_ids, token_ids, batch, rows = [], [], [], []

    def flush():
        token_ids.extend(tokenizer.tokenize(batch, update_vocab=True, show_progress=False))
        db.executemany("INSERT INTO pages VALUES (?, ?)", rows)
        batch.clear()
        rows.clear()

    for page in iter_pages():
        title = unescape(page["id"].replace("_", " "))
        doc_ids.append(page["id"])
        batch.append(title + " . " + unescape(page["text"]))
        rows.append((page["id"], page["lines"]))
        if len(batch) == 50_000:
            flush()
            print(f"{len(doc_ids):>9,} pages  {time.time() - t0:6.0f}s", flush=True)
    flush()
    db.commit()
    db.close()

    corpus = Tokenized(ids=token_ids, vocab=tokenizer.get_vocab_dict())
    retriever = bm25s.BM25()
    retriever.index(corpus)
    out = os.path.join(DATA, "bm25_index")
    retriever.save(out)
    tokenizer.save_vocab(out)
    tokenizer.save_stopwords(out)
    with open(os.path.join(out, "doc_ids.json"), "w", encoding="utf-8") as f:
        json.dump(doc_ids, f)
    print(f"done: {len(doc_ids):,} pages, {len(corpus.vocab):,} terms, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
