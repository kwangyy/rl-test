"""Retrieval tools the prompted controller can call besides search.

    follow     {"link": "L<n>"}    open the page a listed link points to (retrieval.follow.FeverFollow)
    open_page  {"title": str}      open a Wikipedia page by title
    find_pages {"query": str}      list matching page titles; the verifier sees nothing new
    web_pages  {"query": str}      Exa web search limited to en.wikipedia.org, mapped to the
                                   FEVER snapshot (retrieval.pipeline.discover_exa), then
                                   sentences ranked as in search

`--tools` presets in agent/prompted.py pick which of these the controller gets.
One ToolKit per claim holds its links, pages seen and follow budget. follow and
open_page put the opened page's sentences first in the verifier's input;
web_pages replaces the latest results, like search.

web_pages needs an ExaClient (retrieval/exa.py). Without network access it only
replays cached responses; live requests need EXA_API_KEY and the exa-py package.
"""

import time

from retrieval.fever_retrieval import _norm
from retrieval.follow import FeverFollow
from retrieval.pipeline import discover_exa
from verifier.model import MAX_SENTS_PER_PAGE

LINK_HITS = 10  # top-ranked sentences per search or follow whose links the controller sees
MAX_LINKS = 20  # most recent unfollowed link targets shown
OPEN_K = MAX_SENTS_PER_PAGE  # sentences taken from a followed or opened page
MAX_FOUND = 10  # titles find_pages lists
WEB_PAGES = 5  # snapshot pages web_pages keeps, as search's candidate pages

ARGS = {"follow": "link", "open_page": "title", "find_pages": "query", "web_pages": "query"}
FIRST = ("follow", "open_page")  # tools whose sentences go before the latest results


def title(page):
    return _norm(page, lower=False)


class ToolKit:
    """One claim's retrieval tools beyond search."""

    def __init__(self, retriever, names, exa=None):
        unknown = set(names) - set(ARGS)
        if unknown:
            raise ValueError(f"unknown tools {sorted(unknown)}")
        if "web_pages" in names and exa is None:
            raise ValueError("web_pages needs an ExaClient")
        self.r, self.names, self.exa = retriever, tuple(names), exa
        self.follower = FeverFollow(retriever, max_actions=64)
        self.links = {}  # "L<n>" -> link from FeverFollow plus its resolved page, in the order first seen
        self.followed = set()
        self.seen = set()  # pages any search or tool has returned sentences from

    def observe(self, hits, links=True):
        """Record pages returned; with follow, list the outgoing links of the top sentences."""
        self.seen.update(h["page"] for h in hits)
        if links and "follow" in self.names:
            self._add_links(self.follower.observe(hits[:LINK_HITS]))

    def observe_cited(self, evidence):
        """List links from sentences the verifier cited, which may sit below the top LINK_HITS."""
        if "follow" in self.names and evidence:
            cited = [{"page": e.rsplit(":", 1)[0], "line": int(e.rsplit(":", 1)[1])} for e in evidence]
            self._add_links(self.follower.observe(cited))

    def _add_links(self, links):
        pages = {link["page"] for link in self.links.values()}
        for link in links:
            page = self.r.resolve_page(link["target"])
            if page is not None and page not in pages:
                pages.add(page)
                self.links[f"L{len(self.links) + 1}"] = {**link, "page": page}

    def open_links(self):
        """Links the controller may follow: not yet followed, to a page nothing has returned."""
        return [(lid, l) for lid, l in self.links.items()
                if lid not in self.followed and l["page"] not in self.seen][-MAX_LINKS:]

    def valid(self, action, obj):
        if action not in self.names:
            return False
        value = obj.get(ARGS[action])
        if action == "follow":
            return value in dict(self.open_links())
        return isinstance(value, str) and bool(value.strip())

    def run(self, action, obj, claim):
        """Carry out one tool call; returns (args, result, seconds). result["hits"] go to the verifier."""
        t0 = time.time()
        value = obj[ARGS[action]].strip() if action != "follow" else obj["link"]
        if action == "follow":
            link = self.links[value]
            self.followed.add(value)
            res = self.follower.follow(link["id"], claim, OPEN_K)
            hits = res["hits"]
            status = res["status"] if res["status"] != "ok" or hits else "empty_page"
            args = {"link": value, "target": link["target"], "page": link["page"]}
            result = {"status": status, "source": link["source"], "hits": hits}
        elif action == "open_page":
            page = self.r.resolve_page(value)
            hits = self.r.rank_sentences(claim, self.r.collect_sentences([page]), OPEN_K) if page else []
            args = {"title": value, "page": page}
            result = {"status": "not_in_snapshot" if page is None else "ok" if hits else "empty_page",
                      "hits": hits}
        elif action == "find_pages":
            # every page titled with a name in the query ("Savages", "Savages (2012 film)", ...), then BM25
            pages = self.r.search_with_titles([value], k=MAX_FOUND, max_title=MAX_FOUND)[0]
            args = {"query": value, "k": MAX_FOUND}
            result = {"status": "ok", "pages": pages, "hits": []}
        else:  # web_pages
            d = discover_exa(self.r, self.exa, value, WEB_PAGES, self.seen)
            hits = (self.r.rank_sentences(value, self.r.collect_sentences(d["pages"]), OPEN_K * WEB_PAGES)
                    if d["pages"] else [])
            args = {"query": value, "pages": WEB_PAGES}
            call = self.exa.calls[-1] if self.exa.calls else {}
            status = (d["status"] if d["status"] != "ok" or hits
                      else "only_pages_already_seen" if any(m["status"] == "duplicate" for m in d["mappings"])
                      else "no_snapshot_pages")
            result = {"status": status,
                      "error": d.get("error"), "pages": d["pages"], "mappings": d["mappings"],
                      "cache_hit": call.get("cache_hit"), "cost_dollars": call.get("recorded_cost_dollars"),
                      "hits": hits}
        if result["hits"]:
            self.observe(result["hits"])
        return args, result, time.time() - t0


def describe(step):
    """One observation line for a tool step."""
    a, args, r = step["action"], step["args"], step["result"]
    pages = list(dict.fromkeys(title(h["page"]) for h in r["hits"]))
    got = (f"{len(r['hits'])} sentences from " + ", ".join(pages) if pages
           else r["status"].replace("_", " "))
    if a == "follow":
        return f"follow {args['link']} ({args['target']}) -> {got}"
    if a == "open_page":
        return f"open_page {args['title']!r} -> {got}"
    if a == "find_pages":
        return f"find_pages {args['query']!r} -> titles: " + ("; ".join(map(title, r["pages"])) or "none")
    return f"web_pages {args['query']!r} -> " + ("pages: " + ", ".join(pages) if pages
                                                 else r["status"].replace("_", " "))
