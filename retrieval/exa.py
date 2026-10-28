"""Exa SDK search/contents with bounded opt-in network access and cache replay."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from threading import Lock
from urllib.parse import urlsplit, urlunsplit


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def canonical_url(url):
    parts = urlsplit(url)
    if parts.scheme not in ('https', 'http') or not parts.hostname or parts.username or parts.password:
        raise ValueError('expected http(s) source URL without credentials')
    port = parts.port
    host = parts.hostname.lower()
    netloc = '[' + host + ']' if ':' in host else host
    if port and (parts.scheme, port) not in (('https', 443), ('http', 80)):
        netloc += ':' + str(port)
    return urlunsplit((parts.scheme, netloc, parts.path or '/', parts.query, ''))


class ExaError(RuntimeError):
    def __init__(self, message, *, http_status=None):
        super().__init__(message)
        self.http_status = http_status


class ExaClient:
    def __init__(self, cache_dir, *, allow_network=False, max_requests=100, timeout=30, transport=None):
        if max_requests < 0 or timeout <= 0:
            raise ValueError('invalid request budget or timeout')
        self.cache_dir = Path(cache_dir)
        self.allow_network, self.max_requests, self.timeout = allow_network, max_requests, timeout
        self.transport = transport or self._transport
        self.calls, self.network_requests = [], 0
        self._budget_lock = Lock()
        self._billing_blocked = False

    def _transport(self, endpoint, payload):
        key = os.environ.get('EXA_API_KEY')
        if not key:
            raise ExaError('EXA_API_KEY is required for network requests')
        try:
            from exa_py import Exa
        except ImportError:
            raise ExaError('Install project requirements (exa-py) for live requests') from None
        try:
            # Public SDK request preserves the complete provider JSON (statuses,
            # warnings and cost fields), rather than a lossy result conversion.
            return Exa(api_key=key, timeout=self.timeout).request('/' + endpoint, payload)
        except Exception as error:
            # SDK errors may contain provider bodies or request details.
            match = re.match(r'Request failed with status code (\d{3}):', str(error))
            status = int(match[1]) if match else None
            message = f'Exa HTTP {status}' if status else 'Exa SDK request failed'
            raise ExaError(message + '; inspect provider dashboard', http_status=status) from None

    def request(self, endpoint, payload):
        if endpoint not in ('search', 'contents'):
            raise ValueError('unsupported endpoint')
        identity = {'version': 1, 'endpoint': endpoint, 'payload': payload}
        path = self.cache_dir / (fingerprint(identity) + '.json')
        start = time.perf_counter()
        event = {'endpoint': endpoint, 'cache_key': path.stem, 'cache_hit': path.exists(),
                 'network_attempted': False, 'status': 'error'}
        try:
            if path.exists():
                record = json.loads(path.read_text())
                if record['request'] != identity:
                    raise ExaError('cache request identity mismatch')
            else:
                if not self.allow_network:
                    raise ExaError('offline cache miss')
                with self._budget_lock:
                    if self._billing_blocked:
                        raise ExaError('Exa network disabled after HTTP 402; cached responses remain usable', http_status=402)
                    if self.network_requests >= self.max_requests:
                        raise ExaError('API request budget exhausted')
                    self.network_requests += 1
                event['network_attempted'] = True
                response = self.transport(endpoint, payload)
                if not isinstance(response, dict) or not isinstance(response.get('results'), list) or response.get('error'):
                    raise ExaError('invalid Exa results response')
                record = {'request': identity, 'response': response, 'response_sha256': fingerprint(response),
                          'fetched_at': datetime.now(timezone.utc).isoformat(), 'network_seconds': time.perf_counter() - start}
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(mode='w', dir=self.cache_dir, delete=False) as stream:
                    json.dump(record, stream, ensure_ascii=False)
                    temporary = Path(stream.name)
                try:
                    os.link(temporary, path)
                except FileExistsError:
                    record = json.loads(path.read_text())
                finally:
                    temporary.unlink()
            response = record['response']
            if not isinstance(response, dict) or not isinstance(response.get('results'), list):
                raise ExaError('invalid cached Exa results')
            if fingerprint(response) != record['response_sha256']:
                raise ExaError('cache response integrity mismatch')
            event.update(status='ok', fetched_at=record['fetched_at'],
                         response_sha256=record['response_sha256'],
                         recorded_network_seconds=record['network_seconds'],
                         recorded_cost_dollars=(response.get('costDollars') or {}).get('total'),
                         provider_statuses=response.get('statuses', []),
                         warnings=response.get('warnings', []), notices=response.get('notices', []))
            return response, record['fetched_at']
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            event['error'] = 'invalid or unreadable response cache'
            raise ExaError(event['error']) from None
        except ExaError as error:
            event['error'] = str(error)
            if error.http_status is not None:
                event['http_status'] = error.http_status
            if error.http_status == 402:
                with self._budget_lock:
                    self._billing_blocked = True
            raise
        finally:
            event['seconds'] = time.perf_counter() - start
            self.calls.append(event)

    def search(self, query, n=5, *, max_characters=12000, include_domains=None, exclude_domains=None, end_published_date=None):
        if not query.strip() or not 1 <= n <= 100 or max_characters < 1:
            raise ValueError('invalid query or search budgets')
        payload = {'query': query, 'numResults': n, 'type': 'auto',
                   'contents': {'text': {'maxCharacters': max_characters}}}
        for name, value in (('includeDomains', include_domains),
                            ('excludeDomains', exclude_domains),
                            ('endPublishedDate', end_published_date)):
            if value:
                payload[name] = value
        return self.request('search', payload)

    def contents(self, url, *, max_characters=12000):
        if max_characters < 1:
            raise ValueError('max_characters must be positive')
        return self.request('contents', {'urls': [canonical_url(url)], 'text': {'maxCharacters': max_characters}})


def fuse_rankings(rankings, *, context_characters=4000, rrf_constant=60):
    """Fuse local/extension rank positions; never compare their raw BM25 scores.

    Equal weights, baseline first for ties. Exact same-source text is merged and
    all contributing positions are recorded. No annotations or similarity judge
    enter this selection. Context limits count text characters, not source count.
    """
    if context_characters < 1 or rrf_constant < 1 or not rankings:
        raise ValueError('positive context/RRF budgets and at least one ranking required')
    values = {}
    for source, ranking in enumerate(rankings):
        seen = set()
        for rank, hit in enumerate(ranking, 1):
            if not hit['text'].strip():
                continue
            key = (hit['source_url'], ' '.join(re.findall(r'\w+', hit['text'].casefold())))
            if key in seen:
                continue
            seen.add(key)
            if key not in values:
                values[key] = dict(hit, score=0.0, rank_sources=[])
            value = values[key]
            value['score'] += 1/(rrf_constant+rank)
            value['rank_sources'].append({'ranking': source, 'rank': rank, 'id': hit['id']})
    ranked = sorted(values.values(), key=lambda h: -h['score'])
    retained, remaining = [], context_characters
    for hit in ranked:
        if len(hit['text']) <= remaining:
            retained.append(hit)
            remaining -= len(hit['text'])
    return {'ranked': ranked, 'retained': retained,
            'retained_characters': context_characters-remaining,
            'policy': {'fusion': 'equal-weight RRF', 'constant': rrf_constant,
                       'context_characters': context_characters, 'tie_priority': 'ranking order'}}


def passages(source, max_characters=1000):
    """Sentence-aware spans into cached extracted text; long sentences use windows."""
    if max_characters <= 200:
        raise ValueError('passage character limit must exceed overlap (200)')
    text = source['text']
    boundaries = [0]+[m.end() for m in re.finditer(r'(?<=[.!?])\s+|\n\s*\n', text)]+[len(text)]
    spans = []
    start = 0
    for end in boundaries[1:]:
        if end-start <= max_characters:
            continue
        previous = max(b for b in boundaries if start < b <= end and b-start <= max_characters) if any(
            start < b <= end and b-start <= max_characters for b in boundaries) else None
        if previous:
            spans.append((start,previous));start=previous
        while end-start > max_characters:
            spans.append((start,start+max_characters));start += max_characters-200
    if start < len(text):
        spans.append((start,len(text)))
    out=[]
    for start,end in spans:
        if not text[start:end].strip():continue
        identity=f"{source['url']}\0{source['text_sha256']}\0{start}:{end}"
        out.append({'id':'web:'+hashlib.sha256(identity.encode()).hexdigest()[:24],
            'source_url':source['url'],'title':source['title'],'text':text[start:end],
            'start':start,'end':end,'source_text_sha256':source['text_sha256'],
            'fetched_at':source['fetched_at'],'published_date':source.get('published_date')})
    return out


def rank_passages(query, candidates):
    if not candidates:return []
    import bm25s
    import Stemmer
    from bm25s.tokenization import Tokenizer
    tokenizer=Tokenizer(stemmer=Stemmer.Stemmer('english'),stopwords='en')
    docs=tokenizer.tokenize([p['title']+' '+p['text'] for p in candidates],return_as='string',show_progress=False)
    terms=tokenizer.tokenize([query],return_as='string',show_progress=False)[0]
    if not terms or not any(docs):return []
    index=bm25s.BM25(k1=1.5,b=0.75,method='atire',idf_method='lucene',dtype='float64')
    index.index(docs,show_progress=False)
    scores=index.get_scores(terms)
    return sorted([dict(p,score=float(score)) for p,score in zip(candidates,scores)],
                  key=lambda p:(-p['score'],p['id']))


def search_web(query, client, *, n=5, k=10, context_characters=4000,
               include_domains=None, exclude_domains=None, end_published_date=None):
    """Direct query -> Exa -> sentence-aware passages -> local BM25 selection.

    No domain restriction unless the caller supplies one. No generated answer,
    verdict, snapshot mapping or evidence annotation is accepted/used.
    """
    if not query.strip() or min(n,k,context_characters)<1:
        raise ValueError('nonempty query and positive budgets required')
    begin=time.perf_counter();before=len(client.calls)
    sources=[];pool=[];seen_urls=set();seen_text=set();status='ok';error=None
    try:
        response,fetched=client.search(query,n,include_domains=include_domains,
            exclude_domains=exclude_domains,end_published_date=end_published_date)
        for raw in response['results']:
            if not isinstance(raw,dict):raw={}
            try:url=canonical_url(raw.get('url') if isinstance(raw.get('url'),str) else '')
            except (ValueError,TypeError):url=None
            text=raw.get('text') if isinstance(raw.get('text'),str) else ''
            digest=hashlib.sha256(text.encode()).hexdigest()
            source={'url':url,'title':str(raw.get('title') or ''),'text':text,
                'text_sha256':digest,'fetched_at':fetched,'published_date':raw.get('publishedDate'),
                'duplicate_url':url in seen_urls if url else False,
                'duplicate_text':digest in seen_text if text.strip() else False,
                'extracted':bool(text.strip())}
            sources.append(source)
            if url and text.strip() and url not in seen_urls and digest not in seen_text:
                pool.extend(passages(source))
            if url:seen_urls.add(url)
            if text.strip():seen_text.add(digest)
    except ExaError as exc:
        status='error';error=str(exc)
    ranked=rank_passages(query,pool)
    retained=[];remaining=context_characters
    for hit in ranked:
        if len(retained)>=k:break
        if len(hit['text'])<=remaining:
            retained.append(hit);remaining-=len(hit['text'])
    return {'query':query,'status':status,'error':error,'sources':sources,
            'passages':pool,'ranked':ranked,'retained':retained,
            'retained_characters':context_characters-remaining,
            'api_calls':client.calls[before:],'seconds':time.perf_counter()-begin,
            'policy':{'n':n,'k':k,'context_characters':context_characters,
                      'include_domains':include_domains,'exclude_domains':exclude_domains,
                      'end_published_date':end_published_date},
            'temporal_note':'Fetch time is collection time; publication date does not freeze page contents.'}
