"""FEVER hyperlink tool: observe sentence anchors, then open a known target.

Corpus search stays in FeverRetriever; decisions about which links to follow stay
in pipeline.py. Create one tool per claim so visits and action budgets are isolated.
No Exa, annotations, network access or experiment orchestration belongs here.
"""
import hashlib
import time


def outgoing_links(retriever, hits):
    """Read anchor/target pairs only from observed sentences, not entire pages."""
    wanted = {}
    for hit in hits:
        wanted.setdefault(hit['page'], set()).add(hit['line'])
    out = {}
    for page, lines in wanted.items():
        for raw in retriever.raw_lines(page).splitlines():
            fields = raw.split('\t')
            if len(fields) < 2 or not fields[0].isdigit() or int(fields[0]) not in lines:
                continue
            for i in range(2, len(fields) - 1, 2):
                anchor, target = fields[i:i+2]
                if not anchor.strip() or not target.strip():
                    continue
                source = f'{page}:{fields[0]}'
                identity = '\0'.join((source, anchor, target))
                lid = 'link:' + hashlib.sha256(identity.encode()).hexdigest()[:24]
                out[lid] = {'id': lid, 'source': source, 'anchor': anchor, 'target': target}
    return list(out.values())


class FeverFollow:
    """One claim's observed links, visited pages and bounded follow history.

    The shared retriever owns corpus access, sentence decoding and ranking. A caller supplies
    observed results explicitly; this component cannot silently search for links.
    """
    def __init__(self, retriever, *, max_actions=8):
        if max_actions < 1:
            raise ValueError('max_actions must be positive')
        self.r = retriever
        self.max_actions = max_actions
        self.links = {}
        self.visited = set()
        self.trace = []

    def observe(self, hits, pages=()):
        links = outgoing_links(self.r, hits)
        self.links.update((link['id'], link) for link in links)
        self.visited.update(pages)
        return links

    def follow(self, link_id, query, k=10):
        if not query.strip() or k < 1:
            raise ValueError('nonempty query and positive sentence budget required')
        if len(self.trace) >= self.max_actions:
            return {'status': 'budget_exhausted', 'hits': [], 'pages': [], 'links': []}
        start = time.perf_counter()
        link = self.links.get(link_id)
        page = self.r.resolve_page(link['target']) if link else None
        status = ('unknown_link' if link is None else 'not_in_snapshot' if page is None
                  else 'already_visited' if page in self.visited else 'ok')
        result = {'status': status, 'hits': [], 'pages': [], 'links': []}
        if link:
            result['link'] = link
        if status == 'ok':
            result['pages'] = [page]
            result['hits'] = self.r.rank_sentences(query, self.r.collect_sentences([page]), k)
            result['links'] = self.observe(result['hits'], result['pages'])
        self.trace.append({'action': 'follow', 'args': {'link_id': link_id, 'query': query, 'k': k},
                           'result': result, 'seconds': time.perf_counter()-start})
        return result
