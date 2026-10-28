"""Caller-facing FEVER evidence collection. Tools are opt-in via expansion.

No gold annotations, verifier imports or evaluation dependencies. Follow owns
traversal state; the retriever owns corpus search; ExaClient owns API/cache access.
"""
import hashlib
import time
from urllib.parse import urlsplit

from retrieval.exa import ExaError, canonical_url
from retrieval.follow import FeverFollow, outgoing_links


def discover_exa(retriever, client, query, page_budget, initial_pages, *, include_domains=('en.wikipedia.org',)) :
    """Resolve provider pages without replacing missing/duplicate results by extra calls."""
    start = time.perf_counter()
    result = {'status': 'ok', 'pages': [], 'sources': [], 'mappings': []}
    try:
        response, fetched = client.search(query, page_budget, include_domains=list(include_domains))
        seen_urls, seen_pages = set(), set(initial_pages)
        for raw in response['results']:
            if not isinstance(raw, dict):
                raw = {}
            raw_url = raw.get('url') if isinstance(raw.get('url'), str) else ''
            try:
                url = canonical_url(raw_url)
            except ValueError:
                url = None
            page = retriever.resolve_page(url) if url else None
            duplicate_url = url is not None and url in seen_urls
            duplicate_page = page is not None and page in seen_pages
            mapping = 'invalid_url' if url is None else 'not_in_snapshot' if page is None else 'duplicate' if duplicate_page else 'resolved'
            text = raw.get('text') if isinstance(raw.get('text'), str) else ''
            result['sources'].append({'url': url, 'raw_url': raw_url, 'title': raw.get('title') or '',
                                      'text': text, 'fetched_at': fetched,
                                      'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                                      'published_date': raw.get('publishedDate'),
                                      'text_extracted': bool(text.strip()), 'duplicate_url': duplicate_url,
                                      'duplicate_page': duplicate_page, 'mapping_status': mapping})
            result['mappings'].append({'url': url, 'page': page, 'status': mapping})
            if url:
                seen_urls.add(url)
            if page is not None:
                seen_pages.add(page)
            if mapping == 'resolved' and len(result['pages']) < page_budget:
                result['pages'].append(page)
        result['provider_statuses'] = response.get('statuses', [])
    except ExaError as error:
        result.update(status='error', error=str(error))
    result['seconds'] = time.perf_counter() - start
    return result



def collect_live(query, discovery, retriever, max_k, api_calls, initial_pages, final_k, ks):
    """Rank bounded current-text chunks; URL/chunk locators are NOT FEVER IDs.

    Candidate acquisition and ranking see no gold annotations. Chunk positions refer
    to the cached extracted text, not offsets in an original HTML document.
    """
    sources = discovery.get('sources', [])
    pool, seen_text = [], set()
    for source in sources:
        if not source['url'] or source['duplicate_url'] or not source['text_extracted']:
            continue
        digest = source['text_sha256']
        source['duplicate_content'] = digest in seen_text
        if digest in seen_text:
            continue
        seen_text.add(digest)
        # Deterministic 1000-character chunks, at most 12000 returned chars/source.
        text = source['text'][:12000]
        for offset in range(0, len(text), 1000):
            chunk = text[offset:offset+1000]
            if chunk.strip():
                pool.append({'page': source['url'], 'line': offset//1000,
                             'text': chunk, 'title': source['title'],
                             'source_type': 'live-web', 'text_sha256': digest,
                             'fetched_at': source['fetched_at']})
    ranked = retriever.rank_sentences(query, pool, max_k) if pool else []
    def retain(k):
        # Cap live contexts independently; report observed snapshot text volume
        # before making any cross-source relevance comparison.
        budget, selected = 12000, []
        for hit in ranked[:k]:
            if budget <= 0:
                break
            selected.append(dict(hit, text=hit['text'][:budget]))
            budget -= len(selected[-1]['text'])
        return selected
    urls = {source['url'] for source in sources if source['url']}
    mappings = {m['page'] for m in discovery.get('mappings', []) if m['page']}
    return {'status': discovery['status'], 'error': discovery.get('error'),
            'sources': sources, 'mappings': discovery.get('mappings', []),
            'pool': pool, 'ranked': ranked, 'retained': retain(final_k),
            'by_k': {str(k): {'retained_chunks': len(retain(k)),
                              'retained_chars': sum(len(h['text']) for h in retain(k))}
                     for k in sorted(set((*ks, final_k)))},
            'unique_urls': len(urls), 'returned_sources': len(sources),
            'extracted_sources': sum(s['text_extracted'] for s in sources),
            'duplicate_urls': sum(s['duplicate_url'] for s in sources),
            'duplicate_contents': sum(s.get('duplicate_content', False) for s in sources),
            'overlap_initial_snapshot_pages': len(mappings & set(initial_pages)),
            'novel_mapped_pages': len(mappings - set(initial_pages)),
            'unique_domains': sorted({urlsplit(url).hostname for url in urls}),
            'retained_chars': sum(len(h['text']) for h in retain(final_k)),
            'seconds': discovery['seconds'], 'api_calls': api_calls}



def select_follow_links(retriever, links, query, limit, excluded=()):
    """Deterministic anchor overlap; canonical dedup, no gold selection."""
    import re
    terms = lambda s: set(re.findall(r'\w+', s.casefold()))
    ordered = sorted(links, key=lambda link: (-len(terms(query) & terms(link['anchor'] + ' ' + link['target'])), link['id']))
    seen, chosen = set(excluded), []
    for link in ordered:
        canonical = retriever.resolve_page(link['target']) or link['target']
        if canonical not in seen:
            chosen.append(link)
            seen.add(canonical)
        if len(chosen) >= limit:
            break
    return chosen if limit > 0 else []


def _local_search(query, retriever, k, page_budget, exclude_pages=()):
    """Compose corpus primitives for a fixed policy; not a second search API."""
    if not query.strip() or min(k, page_budget) < 1:
        raise ValueError('nonempty query and positive budgets required')
    count = min(len(retriever.doc_ids), page_budget + len(set(exclude_pages)))
    candidates = retriever.search_with_titles([query], k=count)[0]
    pages = [page for page in candidates if page not in exclude_pages][:page_budget]
    hits = retriever.rank_sentences(query, retriever.collect_sentences(pages), k)
    return {'status': 'ok', 'pages': pages, 'hits': hits,
            'links': outgoing_links(retriever, hits)}


def initial_search(query, retriever, *, initial_pages=5, initial_k=10):
    """Prepare one shared initial observation for matched experiment conditions."""
    start = time.perf_counter()
    result = _local_search(query, retriever, initial_k, initial_pages)
    return {'query': query, 'page_budget': initial_pages, 'k': initial_k,
            'result': result, 'seconds': time.perf_counter()-start}


def collect_evidence(query, retriever, *, expansion='none', initial_pages=5,
                     expansion_pages=5, initial_k=10, final_k=10,
                     ks=(3, 5, 10, 20, 30), retry_query=None, client=None, initial=None):
    """Collect evidence with an explicit fixed policy, never a usefulness gate.

    none/local/follow/exa return frozen sentence hits; exa-live returns current
    text chunks in `live`. Exa needs an explicitly supplied provider client.
    A follow context is created internally per call, never shared across claims.
    `initial` is an optional shared observation from initial_search for evaluation.
    """
    if expansion not in ('none', 'local', 'follow', 'exa', 'exa-live'):
        raise ValueError('unknown expansion mode')
    if min(initial_pages, expansion_pages, initial_k, final_k, *ks) < 1:
        raise ValueError('budgets must be positive')
    if expansion in ('exa', 'exa-live') and client is None:
        raise ValueError('Exa expansion requires an explicit client')
    if initial is None:
        initial = initial_search(query, retriever, initial_pages=initial_pages, initial_k=initial_k)
    if initial['query'] != query or initial['page_budget'] != initial_pages or initial['k'] != initial_k:
        raise ValueError('shared initial observation must match query and search budgets')
    observation = initial['result']
    base_pages = observation['pages']
    trace = [{'action': 'search', 'args': {'query': query, 'k': initial_k, 'page_budget': initial_pages},
              'result': observation, 'seconds': initial['seconds']}]
    start = time.perf_counter()
    before = len(client.calls) if client else 0
    retry = retry_query or query
    result = {'status': 'ok', 'pages': []}
    if expansion == 'local':
        result = _local_search(retry, retriever, initial_k, expansion_pages, base_pages)
        trace.append({'action': 'search', 'args': {'query': retry, 'page_budget': expansion_pages},
                      'result': result, 'seconds': time.perf_counter()-start})
    elif expansion == 'follow':
        follow = FeverFollow(retriever, max_actions=expansion_pages)
        follow.observe(observation['hits'], base_pages)
        for link in select_follow_links(retriever, observation['links'], query, expansion_pages, base_pages):
            result['pages'].extend(follow.follow(link['id'], query, initial_k)['pages'])
        trace.extend(follow.trace)
    elif expansion in ('exa', 'exa-live'):
        result = discover_exa(retriever, client, retry, expansion_pages, base_pages)
        if expansion == 'exa-live':
            live = collect_live(query, result, retriever, max((*ks, final_k)),
                                client.calls[before:], base_pages, final_k, ks)
            return {'initial': observation, 'live': live}
        trace.append({'action': 'exa', 'args': {'query': retry, 'page_budget': expansion_pages},
                      'result': result, 'seconds': result['seconds']})
    pages = list(dict.fromkeys(base_pages + result['pages']))
    pool = retriever.collect_sentences(pages)
    ranked = retriever.rank_sentences(query, pool, max((*ks, final_k)))
    return {'initial': observation, 'pages': pages, 'pool': pool, 'ranked': ranked,
            'retained': ranked[:final_k], 'expansion_pages': result['pages'],
            'trace': trace, 'sources': result.get('sources', []),
            'api_calls': client.calls[before:] if client else [],
            'seconds': initial['seconds']+time.perf_counter()-start}
