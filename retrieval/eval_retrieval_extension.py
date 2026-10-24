"""FEVER extension experiments: snapshot, web pilot and cached sentence reranking.

Gold is read by scoring only. All exact recall uses original FEVER snapshot sentences.
exa-live ranks current text for extraction/source/relevance audits, never FEVER gold IDs.
"""
import argparse
import gzip
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import subprocess
import time
from datetime import datetime, timezone

from collections import Counter
import os
import random
from urllib.parse import quote, urlsplit

from retrieval.build_bm25_index import unescape
from retrieval.exa import canonical_url, search_web, fuse_rankings, passages, rank_passages
from retrieval.metrics import paired_outcomes, contains_complete_set, normalized
from retrieval.pipeline import collect_evidence, initial_search

from retrieval.exa import ExaClient
from retrieval.metrics import (annotations, evidence_metrics, lexical_cosine, summarize, analyze_saved, analyze, summarize_live, report)
from utils.fever_eval import (LEGACY_IDS, assert_not_reserved, load_manifest_claims,
                              manifest_metadata, sha256, FROZEN_ROOT)


def evaluate_claim(claim, retriever, *, arms, initial_pages, expansion_pages,
                   initial_k, final_k, retry_query=None, client=None, semantic=False, ks=(3,5,10,20,30)):
    from retrieval.pipeline import initial_search, collect_evidence
    query = claim['claim']
    retry = retry_query or query
    initial = initial_search(query, retriever, initial_pages=initial_pages, initial_k=initial_k)
    base_pages = initial['result']['pages']
    initial_pool = retriever.collect_sentences(base_pages)
    conditions = {}
    live = None
    for arm in arms:
        collected = collect_evidence(query, retriever, expansion='none' if arm == 'initial' else arm,
                                     initial_pages=initial_pages, expansion_pages=expansion_pages,
                                     initial_k=initial_k, final_k=final_k, ks=ks,
                                     retry_query=retry, client=client, initial=initial)
        if arm == 'exa-live':
            live = collected['live']
            live['gold_sentence_recall'] = None
            live['metric_note'] = 'Current text has no FEVER sentence IDs; similarity is diagnostic only.'
            selected = live['retained']
            live['query_lexical_cosine'] = (sum(lexical_cosine(query, h['text']) for h in selected)
                                             /len(selected) if selected else None)
            for source in live['sources']:
                source['query_lexical_cosine'] = lexical_cosine(query, source['text'])
            continue
        pages, pool, ranked, final = (collected[key] for key in ('pages', 'pool', 'ranked', 'retained'))
        trace = collected['trace']
        budgets = sorted(set((*ks, final_k)))
        diagnostic_start = time.perf_counter()
        similarities = [lexical_cosine(query, h['page'].replace('_', ' ') + ' ' + h['text']) for h in final]
        semantic_values = (retriever._dense_scores(query, [h['page'].replace('_',' ')+' '+h['text'] for h in final])
                           if semantic and final else None)
        sources = collected['sources']
        for source in sources:
            source['query_lexical_cosine'] = lexical_cosine(query, source['text'])
        if semantic:
            usable = [s for s in sources if s['text_extracted']]
            if usable:
                cosines = retriever._dense_scores(query, [s['title']+' '+s['text'] for s in usable])
                for source, score in zip(usable, cosines):
                    source['query_semantic_cosine'] = float(score)
        diagnostic_seconds = time.perf_counter() - diagnostic_start
        initial_keys = {(h['page'],h['line']) for h in initial_pool}
        final_keys = {(h['page'],h['line']) for h in final}
        api_calls = collected['api_calls']
        elapsed = collected['seconds']
        conditions[arm] = {
            'pages': pages, 'expansion_pages': collected['expansion_pages'], 'trace': trace,
            'ranked': ranked,
            'by_k': {str(k): evidence_metrics(claim,pages,pool,ranked[:k]) for k in budgets},
            'pool': pool, 'retained': final, **evidence_metrics(claim,pages,pool,final),
            'source_jaccard_vs_initial': len(set(base_pages)&set(pages))/len(set(pages)) if pages else 1.0,
            'new_sources': len(set(pages)-set(base_pages)),
            'new_candidate_sentences': sum((h['page'],h['line']) not in initial_keys for h in pool),
            'new_retained_sentences': len(final_keys-initial_keys),
            'retained_chars': sum(len(h['text'])+len(h['page']) for h in final),
            'mean_query_lexical_cosine': sum(similarities)/len(similarities) if similarities else None,
            'mean_query_semantic_cosine': sum(semantic_values)/len(semantic_values) if semantic_values else None,
            'seconds': elapsed, 'diagnostic_seconds': diagnostic_seconds,
            'failed_actions': sum(t['result']['status'] not in ('ok','already_visited') for t in trace),
            'empty_snapshot_pages': sum(not retriever.sentences(page) for page in pages),
            'api_calls': api_calls,
        }
    groups = annotations(claim)
    return {'id': claim['id'], 'claim': query, 'retry_query': retry,
            'requires_multiple_pages': all(len({p for p,line in group}) > 1 for group in groups) if groups else None,
            'conditions': conditions, **({'live': live} if live is not None else {})}



def select_evaluation_manifest(args, frozen_root=FROZEN_ROOT):
    """Only the original predeclared snapshot policy may evaluate reserved IDs."""
    if args.evaluation_split == 'development':
        return args.claim_ids
    target = frozen_root/'holdout-3000-ids.json'
    if args.claim_ids.resolve() not in (LEGACY_IDS.resolve(), target.resolve()):
        raise ValueError('holdout uses the exact official holdout manifest only')
    frozen = (args.method == 'bm25' and args.initial_pages == 5 and args.expansion_pages == 5
              and args.initial_k == 10 and args.final_k == 10
              and sorted(args.ks) == [3,5,10,20,30]
              and args.arms == ['initial','local','follow','exa']
              and args.limit is None and args.retry_queries is None
              and not args.semantic_diagnostics)
    if not frozen:
        raise ValueError('holdout must use the frozen original-query BM25 5/5-page, k10 policy and all four snapshot arms; no subset or tuning')
    import json
    record = json.loads(target.read_text())
    if record.get('role') != 'holdout' or len(record['ids']) != 3000:
        raise ValueError('invalid official holdout manifest')
    return target

def snapshot_main(argv=None):
    import sys
    argv = sys.argv[1:] if argv is None else argv
    if '--analyze' in argv:
        parser = argparse.ArgumentParser(description='Analyze a saved run without models or API access')
        parser.add_argument('--analyze', type=Path, required=True)
        parser.add_argument('--output', type=Path, required=True)
        args = parser.parse_args(argv)
        print(report(analyze_saved(args.analyze, args.output)))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-split', choices=('development','holdout'), default='development')
    parser.add_argument('--data-dir',type=Path,default=Path('verifier/data/fever'))
    parser.add_argument('--claim-ids',type=Path,default=LEGACY_IDS)
    parser.add_argument('--limit',type=int,help='explicit smoke subset, actual IDs recorded')
    parser.add_argument('--arms',nargs='+',choices=('initial','local','follow','exa','exa-live'),default=['initial','local','follow'])
    parser.add_argument('--method',choices=('bm25','hybrid'),default='bm25')
    parser.add_argument('--initial-pages',type=int,default=5)
    parser.add_argument('--expansion-pages',type=int,default=5)
    parser.add_argument('--initial-k',type=int,default=10,help='initial sentences that expose followable links')
    parser.add_argument('--final-k',type=int,default=10)
    parser.add_argument('--ks',type=int,nargs='+',default=[3,5,10,20,30],help='score saved ranking prefixes without extra search/API requests')
    parser.add_argument('--retry-queries',type=Path,help='JSON id -> fixed nonempty retry query for every selected claim')
    parser.add_argument('--cache-dir',type=Path,default=Path('verifier/data/exa-cache'))
    parser.add_argument('--allow-network',action='store_true')
    parser.add_argument('--env-file', type=Path, help='explicit dotenv file; loads only EXA_API_KEY for live requests')
    parser.add_argument('--max-api-requests',type=int,default=100)
    parser.add_argument('--exa-workers', type=int, default=1, help='prefetch provider requests concurrently, then score offline')
    parser.add_argument('--semantic-diagnostics',action='store_true',help='optional existing pinned MiniLM cosine; may load/download weights')
    parser.add_argument('--output',type=Path,required=True,help='new run directory; never overwritten')
    parser.add_argument('--compress-predictions', action='store_true', help='save lossless predictions.jsonl.gz to reduce disk use')
    args = parser.parse_args(argv)
    if min(args.initial_pages,args.expansion_pages,args.initial_k,args.final_k,*args.ks) < 1:
        parser.error('budgets must be positive')
    if args.limit is not None and args.limit < 1 or args.max_api_requests < 0:
        parser.error('invalid count')
    if not 1 <= args.exa_workers <= 8:
        parser.error('exa-workers must be 1..8')
    if args.expansion_pages > 100:
        parser.error('expansion page budget must be at most 100')
    if args.output.exists() or len(set(args.arms)) != len(args.arms):
        parser.error('use a new output directory and distinct arms')
    if 'exa-live' in args.arms and 'initial' not in args.arms:
        parser.error('exa-live requires initial as the paired snapshot baseline')
    try:
        args.claim_ids = select_evaluation_manifest(args)
    except ValueError as error:
        parser.error(str(error))
    claims = load_manifest_claims(args.data_dir/'shared_task_dev.jsonl',args.claim_ids)
    if args.evaluation_split == 'development':
        assert_not_reserved(claims,'retrieval expansion development')
    if args.limit:
        claims = claims[:args.limit]
    retries = json.loads(args.retry_queries.read_text()) if args.retry_queries else {}
    if args.retry_queries and (not isinstance(retries,dict) or any(
            not isinstance(retries.get(str(c['id'])),str) or not retries[str(c['id'])].strip() for c in claims)):
        parser.error('retry file must supply a nonempty query for every selected claim ID')
    if args.env_file:
        if not args.allow_network or not args.env_file.is_file():
            parser.error('--env-file requires --allow-network and an existing file')
        import os
        from dotenv import dotenv_values
        key = dotenv_values(args.env_file).get('EXA_API_KEY')
        if not key and not os.environ.get('EXA_API_KEY'):
            parser.error('EXA_API_KEY is not configured in environment or selected file')
        if key:
            os.environ['EXA_API_KEY'] = key
    from retrieval.fever_retrieval import FeverRetriever, DENSE_MODEL, DENSE_REVISION
    retriever = FeverRetriever(str(args.data_dir),method=args.method,candidate_pages=args.initial_pages)
    client = ExaClient(args.cache_dir,allow_network=args.allow_network,max_requests=args.max_api_requests) if any(a in args.arms for a in ('exa','exa-live')) else None
    sources = [Path(__file__),Path('retrieval/metrics.py'),Path('retrieval/pipeline.py'),Path('retrieval/follow.py'),Path('retrieval/exa.py'),Path('retrieval/fever_retrieval.py')]
    db = (args.data_dir/'pages.sqlite').stat()
    packages = {}
    for name in ('bm25s','PyStemmer','numpy','torch','transformers','exa-py'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    manifest = {'created_at_utc':datetime.now(timezone.utc).isoformat(),
                'git_revision':subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip(),
                'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                'scope':'snapshot discovery plus optional current-text diagnostics; no verdict evaluation or full-web recall',
                'selected_ids':[c['id'] for c in claims],**manifest_metadata(args.claim_ids),
                'source_sha256':{str(p):sha256(p) for p in sources},
                'input_sha256':{'shared_task_dev.jsonl':sha256(args.data_dir/'shared_task_dev.jsonl'),
                                'doc_ids.json':sha256(args.data_dir/'bm25_index/doc_ids.json')},
                'sqlite_stat':{'bytes':db.st_size,'mtime_ns':db.st_mtime_ns,'note':'not a full content checksum'},
                'retry_queries_sha256':sha256(args.retry_queries) if args.retry_queries else None,
                'retry_policy':'recorded-query' if retries else 'same-query-wider-local-control',
                'semantic_model':{'name':DENSE_MODEL,'revision':DENSE_REVISION} if args.semantic_diagnostics else None}
    manifest['environment'] = {'python':platform.python_version(),'platform':platform.platform(),
                               'packages':packages}
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    rows = []
    try:
        if client and args.allow_network and args.exa_workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            from retrieval.exa import ExaError
            def prefetch(claim):
                try:
                    client.search(retries.get(str(claim['id']), claim['claim']), args.expansion_pages,
                                  include_domains=['en.wikipedia.org'])
                    return True
                except ExaError:
                    return False
            with ThreadPoolExecutor(max_workers=args.exa_workers) as executor:
                outcomes = []
                for i, ok in enumerate(executor.map(prefetch, claims), 1):
                    outcomes.append(ok)
                    if i % 25 == 0:
                        print(f'prefetch {i}/{len(claims)}', flush=True)
            manifest['prefetch'] = {'successful': sum(outcomes), 'failed': len(outcomes)-sum(outcomes),
                                    'calls': list(client.calls), 'workers': args.exa_workers}
            client.allow_network = False
            (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        warm_start = time.perf_counter()
        retriever.title_matches('warm up')
        manifest['title_index_setup_seconds'] = time.perf_counter()-warm_start
        (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        prediction_path = args.output / ('predictions.jsonl.gz' if args.compress_predictions else 'predictions.jsonl')
        open_predictions = gzip.open if args.compress_predictions else open
        with open_predictions(prediction_path, 'xt', encoding='utf-8') as stream:
            for claim in claims:
                row = evaluate_claim(claim,retriever,arms=args.arms,initial_pages=args.initial_pages,
                                     expansion_pages=args.expansion_pages,initial_k=args.initial_k,final_k=args.final_k,
                                     retry_query=retries.get(str(claim['id'])),client=client,semantic=args.semantic_diagnostics,ks=args.ks)
                stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
                rows.append(row)
                print(f"{len(rows)}/{len(claims)} id={claim['id']}",flush=True)
        snapshot_arms = [a for a in args.arms if a != 'exa-live']
        metrics = summarize(rows,snapshot_arms)
        if any('live' in row for row in rows):
            metrics['live'] = summarize_live(rows)
        analysis = analyze(rows, manifest)
        if 'live' in metrics:
            analysis['live'] = metrics['live']
        (args.output/'analysis.json').write_text(json.dumps(analysis, indent=2)+'\n')
        (args.output/'report.md').write_text(report(analysis))
        (args.output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
        print(json.dumps(metrics,indent=2))
    finally:
        retriever.db.close()


# Bounded web collection and cached FEVER/Exa analysis.
BENCHMARK_DOMAINS = ['fever.ai', 'github.com', 'huggingface.co', 'kaggle.com', 'paperswithcode.com']
CHARACTER_CAP = 4000
KS = (3, 5, 10, 20, 30)


def wikipedia_url(page):
    return canonical_url('https://en.wikipedia.org/wiki/' + quote(unescape(page), safe='()_'))


def snapshot_context(value):
    """Keep whole sentences from the saved top-30 ranking under the shared cap."""
    remaining = CHARACTER_CAP
    retained = []
    for hit in value['ranked']:
        if len(hit['text']) <= remaining:
            retained.append(dict(hit, id=f"snapshot:{hit['page']}:{hit['line']}",
                                 title=unescape(hit['page'].replace('_', ' ')),
                                 source_url=wikipedia_url(hit['page']), source_type='frozen-snapshot'))
            remaining -= len(hit['text'])
    return dict(value, retained=retained, retained_characters=CHARACTER_CAP-remaining)


def summarize_pilot(rows, claims, retriever):
    """Score saved contexts, keeping ID recall and live-text proxies distinct."""
    if [r['id'] for r in rows] != [c['id'] for c in claims] or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('predictions must match the complete cohort and order')
    arms = ('local_baseline', 'follow', 'web_current')
    outcomes = {arm: [] for arm in arms}
    url_coverage = {arm: [] for arm in arms}
    per_claim = []
    for row, claim in zip(rows, claims):
        groups = annotations(claim)
        if not groups:
            raise ValueError('this pilot requires gold-eligible claims')
        gold = {(page, line): text for page, line in set().union(*groups)
                for number, text in retriever.sentences(page) if number == line}
        if len(gold) != len(set().union(*groups)):
            raise ValueError('gold snapshot sentence missing')
        item = {'id': row['id'], 'conditions': {}}
        for arm in arms:
            value = row['conditions'][arm]
            matched = contains_complete_set(groups, gold, [h['text'] for h in value['retained']])
            urls = ({s['url'] for s in value['sources'] if s['url']}
                    if arm == 'web_current' else {wikipedia_url(p) for p in value['pages']})
            found = any({wikipedia_url(p) for p, line in group} <= urls for group in groups)
            outcomes[arm].append(matched)
            url_coverage[arm].append(found)
            item['conditions'][arm] = {'verbatim_complete_set': matched, 'annotated_urls_complete': found}
        per_claim.append(item)
    metrics = {'claims': len(rows), 'character_cap': CHARACTER_CAP, 'conditions': {}, 'paired': {},
               'per_claim': per_claim,
               'metric_note': 'Verbatim gold-text retention is a conservative diagnostic, not semantic/full-web recall. URL annotations are incomplete.'}
    for arm in arms:
        values = [r['conditions'][arm] for r in rows]
        summary = {'verbatim_complete_sets': sum(outcomes[arm]),
                   'verbatim_complete_set_rate': sum(outcomes[arm])/len(rows),
                   'complete_annotated_url_rate': sum(url_coverage[arm])/len(rows),
                   'mean_context_characters': sum(v['retained_characters'] for v in values)/len(rows)}
        if arm != 'web_current':
            summary['sentence_id_recall'] = {str(k): sum(evidence_metrics(c, v['pages'], [], v['ranked'][:k])['complete_set_coverage']
                for c, v in zip(claims, values))/len(rows) for k in KS}
            summary['follow_statuses'] = dict(Counter(t['result']['status'] for v in values for t in v['trace']
                                                     if t['action'] == 'follow'))
        else:
            sources = [s for v in values for s in v['sources']]
            calls = [e for v in values for e in v['api_calls']]
            new_calls = [e for e in calls if e['network_attempted']]
            new_urls, overlaps = [], []
            for row, value in zip(rows, values):
                base = {wikipedia_url(p) for p in row['conditions']['local_baseline']['pages']}
                live = {s['url'] for s in value['sources'] if s['url']}
                new_urls.append(len(live-base))
                overlaps.append(len(live & base)/len(live | base) if live | base else 1)
            summary.update(returned_sources=len(sources), extracted_sources=sum(s['extracted'] for s in sources),
                usable_sources=sum(bool(s['url']) and s['extracted'] for s in sources),
                extraction_rate=sum(s['extracted'] for s in sources)/len(sources) if sources else None,
                failed_requests=sum(v['status'] != 'ok' for v in values), empty_contexts=sum(not v['retained'] for v in values),
                duplicate_urls=sum(s['duplicate_url'] for s in sources), duplicate_texts=sum(s['duplicate_text'] for s in sources),
                distinct_hostnames=len({urlsplit(s['url']).hostname for s in sources if s['url']}),
                mean_novel_urls_vs_local=sum(new_urls)/len(rows), mean_url_jaccard_vs_local=sum(overlaps)/len(rows),
                network_attempts=len(new_calls), cache_hits=sum(e['cache_hit'] for e in calls),
                new_response_cost_estimate=sum(e.get('recorded_cost_dollars') or 0 for e in new_calls),
                replayed_response_cost_estimate=sum(e.get('recorded_cost_dollars') or 0 for e in calls if not e['network_attempted']),
                mean_new_request_seconds=sum(e['seconds'] for e in new_calls)/len(new_calls) if new_calls else None,
                cost_note='Provider response estimates, not an invoice; unknown failed-request billing is excluded.')
        metrics['conditions'][arm] = summary
        if arm != 'local_baseline':
            metrics['paired'][arm+'-vs-local_baseline-verbatim'] = paired_outcomes(list(zip(outcomes['local_baseline'], outcomes[arm])))
    return metrics


def snapshot_ranking(hits):
    return [dict(h, id=f"snapshot:{h['page']}:{h['line']}",
                 title=unescape(h['page'].replace('_', ' ')),
                 source_url=wikipedia_url(h['page']), source_type='frozen-snapshot') for h in hits]


def extension_conditions(row, character_cap=CHARACTER_CAP):
    """Local is always the baseline; extensions add to its candidate ranking."""
    local = snapshot_ranking(row['conditions']['local_baseline']['ranked'])
    followed = snapshot_ranking(row['conditions']['follow']['ranked'])
    web = [dict(h, source_type='live-web') for h in row['conditions']['web_current']['ranked']]
    return {arm: fuse_rankings(rankings, context_characters=character_cap) for arm, rankings in (
        ('local', [local]), ('local_follow', [followed]),
        ('local_exa', [local, web]), ('local_follow_exa', [followed, web]))}


def matched_live_conditions(row, retriever):
    """Compare local versus local+web with identical 600-char passage construction.

    Uses only saved query/pages/provider text. No annotations, new search or Exa
    requests enter candidate construction; all local page sentences are available.
    """
    local = []
    for page in row['conditions']['local_baseline']['pages']:
        text = '\n'.join(text for _, text in retriever.sentences(page))
        source = {'url': wikipedia_url(page), 'title': unescape(page.replace('_', ' ')),
                  'text': text, 'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                  'fetched_at': 'FEVER-2017-snapshot', 'published_date': None}
        local.extend(dict(hit, source_type='frozen-snapshot')
                     for hit in passages(source, max_characters=600))
    web = [dict(hit, source_type='live-web')
           for source in row['conditions']['web_current']['sources']
           if source['url'] and source['extracted'] and not source.get('duplicate_url')
           and not source.get('duplicate_text')
           for hit in passages(source, max_characters=600)]
    return {arm: fuse_rankings([rank_passages(row['claim'], pool)],
                               context_characters=CHARACTER_CAP)
            for arm, pool in (('local', local), ('local_exa', local + web))}


def analyze_extensions(run, data_dir, output, *, semantic=False, matched_passages=False):
    """Cache-only extension study; gold is loaded after retention is fixed."""
    source_manifest = json.loads((run/'manifest.json').read_text())
    rows = [json.loads(l) for l in (run/'predictions.jsonl').read_text().splitlines() if l.strip()]
    ids = source_manifest['selected_ids']
    if not 1 <= len(ids) <= 50 or [r['id'] for r in rows] != ids or len(set(ids)) != len(ids):
        raise ValueError('need the complete saved development cohort of at most 50 distinct claims')
    all_claims = {c['id']: c for c in load_manifest_claims(data_dir/'shared_task_dev.jsonl', LEGACY_IDS)}
    claims = [all_claims[i] for i in ids]
    assert_not_reserved(claims, 'cached extension development')
    if any(r['claim'] != c['claim'] for r,c in zip(rows,claims)):
        raise ValueError('saved claim text differs from the frozen source')
    output.mkdir(parents=True, exist_ok=False)
    policy = {'source_run': str(run), 'selected_ids': ids, 'role': 'development',
        'source_prediction_sha256': sha256(run/'predictions.jsonl'), 'new_network_requests': 0,
        'context_characters': CHARACTER_CAP, 'fusion': 'equal-weight RRF, constant 60, baseline first for ties',
        'arms': ['local', 'local_exa'] if matched_passages else ['local','local_follow','local_exa','local_follow_exa'],
        'semantic_thresholds': [0.6,0.7,0.8], 'gold_for_scoring_only': True,
        'controlled_snapshot': 'Map existing Exa Wikipedia URLs into frozen corpus; BM25 pooled ranking, k=3/10/30',
        'source_sha256': {p: sha256(p) for p in ('retrieval/exa.py','retrieval/eval_retrieval_extension.py',
            'retrieval/metrics.py','retrieval/pipeline.py','retrieval/follow.py','retrieval/fever_retrieval.py')},
        'note': 'Retrospective fixed development ablation. Candidate budgets differ; retained context cap matches. No held-out validation.'}
    if matched_passages:
        policy.update(passage_characters=600, prediction_format='jsonl.gz', fusion='pooled title/text BM25 on matched passages',
                      saved_snapshot_top_k=None,
                      note='Fixed cached development comparison; same passage construction/ranker/text cap. No held-out validation.')
    (output/'manifest.json').write_text(json.dumps(policy, indent=2)+'\n')
    contexts = [{'id': r['id'], 'claim': r['claim'], 'conditions': extension_conditions(r)} for r in rows]
    from retrieval.fever_retrieval import FeverRetriever
    retriever = FeverRetriever(str(data_dir), method='bm25', candidate_pages=5)
    references, per_claim, snapshot_rows = [], [], []
    try:
        if matched_passages:
            contexts = [{'id': row['id'], 'claim': row['claim'],
                         'conditions': matched_live_conditions(row, retriever)} for row in rows]
        prediction_path = output / ('predictions.jsonl.gz' if matched_passages else 'predictions.jsonl')
        open_predictions = gzip.open if matched_passages else open
        with open_predictions(prediction_path, 'xt', encoding='utf-8') as f:
            for row in contexts: f.write(json.dumps(row)+'\n')
        for raw, row, claim in zip(rows, contexts, claims):
            groups = annotations(claim)
            if not groups: raise ValueError('gold-eligible source cohort required')
            hits = {(page,line): {'title': unescape(page.replace('_',' ')), 'text': text}
                    for page,line in sorted(set().union(*groups))
                    for number,text in retriever.sentences(page) if number == line}
            if len(hits) != len(set().union(*groups)): raise ValueError('missing gold snapshot sentence')
            references.append({'groups': groups, 'hits': hits})
            gold = {key: h['text'] for key,h in hits.items()}
            base_urls = {wikipedia_url(p) for p in raw['conditions']['local_baseline']['pages']}
            item = {'id': row['id'], 'conditions': {}}
            for arm, value in row['conditions'].items():
                retained = value['retained']
                urls = {h['source_url'] for h in retained}
                text = [h['text'] for h in retained]
                item['conditions'][arm] = {
                    'verbatim_complete_set': contains_complete_set(groups,gold,text),
                    'ranked_verbatim_complete_set': contains_complete_set(groups,gold,[h['text'] for h in value['ranked']]),
                    'retained_characters': value['retained_characters'],
                    'retained_sources': len(urls), 'novel_retained_urls_vs_local_pages': len(urls-base_urls),
                    'retained_web_characters': sum(len(h['text']) for h in retained if h['source_type']=='live-web'),
                    'url_jaccard_vs_local_pages': len(urls & base_urls)/len(urls | base_urls) if urls | base_urls else 1,
                    'duplicate_retained_texts': len(text)-len({normalized(t) for t in text})}
            if matched_passages:
                required_urls = [{wikipedia_url(page) for page, _ in group} for group in groups]
                gold_urls = set().union(*required_urls)
                item['reference_urls'] = {}
                for arm, value in row['conditions'].items():
                    retained_urls = {hit['source_url'] for hit in value['retained']}
                    item['reference_urls'][arm] = {
                        'complete_gold_page_urls': any(urls <= retained_urls for urls in required_urls),
                        'annotated_page_url_recall': len(gold_urls & retained_urls)/len(gold_urls)}
                web_urls = {source['url'] for source in raw['conditions']['web_current']['sources']
                            if source['url'] and source['extracted'] and not source.get('duplicate_url')
                            and not source.get('duplicate_text')}
                key = lambda hit: (hit['source_url'], normalized(hit['text']))
                baseline_keys = {key(hit) for hit in row['conditions']['local']['retained']}
                extended = row['conditions']['local_exa']
                selected_keys = {key(hit) for hit in extended['retained']}
                item['source_diagnostics'] = {
                    'candidate_url_jaccard': len(base_urls & web_urls)/len(base_urls | web_urls) if base_urls | web_urls else None,
                    'novel_web_url_fraction': len(web_urls - base_urls)/len(web_urls) if web_urls else None,
                    'baseline_passage_retention': len(baseline_keys & selected_keys)/len(baseline_keys) if baseline_keys else None,
                    'live_text_character_share': item['conditions']['local_exa']['retained_web_characters']/extended['retained_characters'] if extended['retained_characters'] else None}
            per_claim.append(item)
            mapped = list(dict.fromkeys(p for s in raw['conditions']['web_current']['sources']
                                        if s['url'] and (p := retriever.resolve_page(s['url'])) is not None))
            snapshot = {'id': row['id'], 'mapped_exa_pages': mapped, 'conditions': {}}
            for arm in policy['arms']:
                base = raw['conditions']['follow' if arm in ('local_follow','local_follow_exa') else 'local_baseline']
                pages = list(dict.fromkeys(base['pages']+(mapped if arm in ('local_exa','local_follow_exa') else [])))
                pool = retriever.collect_sentences(pages)
                ranked = retriever.rank_sentences(row['claim'],pool,30)
                snapshot['conditions'][arm] = {'pages': pages, 'ranked': ranked,
                    'page_coverage': evidence_metrics(claim,pages,pool,[])['complete_gold_page_coverage'],
                    'pool_coverage': evidence_metrics(claim,pages,pool,[])['pool_complete_set_coverage'],
                    'by_k': {str(k): evidence_metrics(claim,pages,pool,ranked[:k])['complete_set_coverage'] for k in (3,10,30)}}
            snapshot_rows.append(snapshot)
            print(f"scored {len(per_claim)}/{len(rows)}", flush=True)
    finally:
        retriever.db.close()
    result = {'claims':len(rows),'new_network_requests':0,'new_provider_cost':0,
              'conditions':{},'paired_verbatim':{},'controlled_snapshot':{},'per_claim':per_claim}
    for arm in policy['arms']:
        values = [r['conditions'][arm] for r in per_claim]
        mean = lambda key: sum(v[key] for v in values)/len(values)
        result['conditions'][arm] = {key: mean(key) for key in values[0]}
        result['controlled_snapshot'][arm] = {
            'page_coverage': sum(r['conditions'][arm]['page_coverage'] for r in snapshot_rows)/len(rows),
            'pool_coverage': sum(r['conditions'][arm]['pool_coverage'] for r in snapshot_rows)/len(rows),
            'recall_by_k': {str(k): sum(r['conditions'][arm]['by_k'][str(k)] for r in snapshot_rows)/len(rows) for k in (3,10,30)}}
        if arm != 'local':
            result['paired_verbatim'][arm] = paired_outcomes([
                (r['conditions']['local']['verbatim_complete_set'],r['conditions'][arm]['verbatim_complete_set']) for r in per_claim])
            result['controlled_snapshot'][arm]['paired_vs_local'] = {str(k): paired_outcomes([
                (r['conditions']['local']['by_k'][str(k)],r['conditions'][arm]['by_k'][str(k)]) for r in snapshot_rows]) for k in (3,10,30)}
    if matched_passages:
        result['reference_urls'] = {arm: {key: sum(row['reference_urls'][arm][key] for row in per_claim)/len(rows)
            for key in per_claim[0]['reference_urls'][arm]} for arm in policy['arms']}
        result['source_diagnostics'] = {}
        for key in per_claim[0]['source_diagnostics']:
            values = [row['source_diagnostics'][key] for row in per_claim if row['source_diagnostics'][key] is not None]
            result['source_diagnostics'][key] = sum(values)/len(values) if values else None
        sources = [source for row in rows for source in row['conditions']['web_current']['sources']]
        events = [event for row in rows for event in row['conditions']['web_current']['api_calls']]
        result['web_collection'] = {
            'returned_sources': len(sources), 'extracted_sources': sum(source['extracted'] for source in sources),
            'duplicate_urls': sum(source.get('duplicate_url', False) for source in sources),
            'duplicate_texts': sum(source.get('duplicate_text', False) for source in sources),
            'failed_searches': sum(row['conditions']['web_current']['status'] != 'ok' for row in rows),
            'stored_response_cost_estimate': sum(event.get('recorded_cost_dollars') or 0 for event in events),
            'mean_original_request_seconds': sum(event.get('recorded_network_seconds', 0) for event in events)/len(events) if events else None,
            'note': 'Original response telemetry, including cache replay; not new requests/billing. New scoring requests=0.'}
    if semantic:
        from retrieval.metrics import evaluate_similarity
        result['similarity'] = evaluate_similarity(contexts,references)
    snapshot_path = output / ('snapshot-predictions.jsonl.gz' if matched_passages else 'snapshot-predictions.jsonl')
    with open_predictions(snapshot_path, 'xt', encoding='utf-8') as f:
        for row in snapshot_rows: f.write(json.dumps(row)+'\n')
    result['prediction_sha256'] = sha256(prediction_path)
    (output/'metrics.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def web_main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=Path('verifier/data/fever'))
    p.add_argument('--limit', type=int, default=50)
    p.add_argument('--seed', type=int, default=20261024)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cache-dir', type=Path, default=Path('verifier/data/exa-cache'))
    p.add_argument('--allow-network', action='store_true')
    p.add_argument('--env-file', type=Path)
    p.add_argument('--max-api-requests', type=int, default=50)
    p.add_argument('--analyze', type=Path, help='saved pilot: evaluate extensions without provider access')
    p.add_argument('--semantic-similarity', action='store_true', help='offline cached MiniLM scoring, never a human/LLM judge')
    p.add_argument('--matched-passages', action='store_true', help='cached baseline/live comparison with matched 600-char passages')
    args = p.parse_args(argv)
    if args.matched_passages and not args.analyze: p.error('--matched-passages requires --analyze')
    if args.output.exists() or not 1 <= args.limit <= 50 or not 0 <= args.max_api_requests <= 50:
        p.error('new output required; pilot supports 1..50 claims and 0..50 new requests')
    if args.analyze:
        if args.allow_network or args.env_file: p.error('saved-run analysis does not accept network/key options')
        result = analyze_extensions(args.analyze,args.data_dir,args.output,semantic=args.semantic_similarity,matched_passages=args.matched_passages)
        print(json.dumps({k:v for k,v in result.items() if k not in ('per_claim','similarity')},indent=2))
        return
    if args.semantic_similarity: p.error('--semantic-similarity requires --analyze')
    if args.env_file:
        if not args.allow_network or not args.env_file.is_file():
            p.error('--env-file requires --allow-network and an existing file')
        from dotenv import dotenv_values
        key = dotenv_values(args.env_file).get('EXA_API_KEY')
        if key:
            os.environ['EXA_API_KEY'] = key  # Explicit selected file wins over stale shell values.
        if not os.environ.get('EXA_API_KEY'):
            p.error('EXA_API_KEY is not configured')
    claims = load_manifest_claims(args.data_dir/'shared_task_dev.jsonl', LEGACY_IDS)
    assert_not_reserved(claims, 'web retrieval development')
    claims = random.Random(args.seed).sample([c for c in claims if annotations(c)], args.limit)
    from retrieval.fever_retrieval import FeverRetriever
    retriever = FeverRetriever(str(args.data_dir), method='bm25', candidate_pages=5)
    client = ExaClient(args.cache_dir, allow_network=args.allow_network, max_requests=args.max_api_requests)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {'created_at': datetime.now(timezone.utc).isoformat(), 'dataset': 'FEVER', 'role': 'development',
        'selected_ids': [c['id'] for c in claims], 'seed': args.seed,
        'git_revision': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
        'settings': {'query': 'original claim', 'n': 5, 'web_k': 10, 'context_characters': CHARACTER_CAP,
            'snapshot_pages': 5, 'follow_attempts': 5, 'observed_sentences': 10, 'saved_snapshot_k': 30,
            'web_exclude_domains': BENCHMARK_DOMAINS, 'max_new_requests': args.max_api_requests,
            'retries': 0, 'arms': ['local_baseline', 'follow', 'web_current'],
            'evaluation': 'Use --analyze for local-baseline extension comparisons; automatic metrics only'},
        'claim_ids_sha256': sha256(LEGACY_IDS), 'claims_sha256': sha256(args.data_dir/'shared_task_dev.jsonl'),
        'source_sha256': {str(f): sha256(f) for f in Path('retrieval').glob('*.py')},
        'metric_note': 'Snapshot ID recall and verbatim retention scored after collection; semantic labels separate. No verdict evaluation.'}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    rows = []
    try:
        with (args.output/'predictions.jsonl').open('x') as out:
            for claim in claims:
                initial = initial_search(claim['claim'], retriever)
                conditions = {}
                for arm, mode in (('local_baseline', 'none'), ('follow', 'follow')):
                    conditions[arm] = snapshot_context(collect_evidence(claim['claim'], retriever, expansion=mode,
                        initial=initial, final_k=30))
                conditions['web_current'] = search_web(claim['claim'], client, exclude_domains=BENCHMARK_DOMAINS)
                row = {'id': claim['id'], 'claim': claim['claim'], 'conditions': conditions}
                rows.append(row)
                out.write(json.dumps(row)+'\n'); out.flush()
                print(f'{len(rows)}/{len(claims)}', flush=True)
        metrics = summarize_pilot(rows, claims, retriever)
        metrics['prediction_sha256'] = sha256(args.output/'predictions.jsonl')
        metrics['scorer_sha256'] = sha256(__file__)
        (args.output/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
        print(json.dumps({k: v for k, v in metrics.items() if k != 'per_claim'}, indent=2))
    finally:
        (args.output/'requests.json').write_text(json.dumps(client.calls, indent=2)+'\n')
        retriever.db.close()


# Fixed-candidate ablation: change sentence ranking, not search or link exposure.
def rerank_claim(claim, saved, retriever, ks=(3, 5, 10, 20, 30)):
    """Score cached BM25 and hybrid on identical local/follow candidate pools."""
    if claim['id'] != saved['id'] or claim['claim'] != saved['claim']:
        raise ValueError('cached identity/query mismatch')
    conditions = {}
    for source_arm, hybrid_arm in (('initial', 'hybrid_local'), ('follow', 'hybrid_follow')):
        source = saved['conditions'][source_arm]
        for arm in (source_arm, hybrid_arm):
            start = time.perf_counter()
            ranked = (source['ranked'] if arm == source_arm else
                      retriever.rank_sentences(saved['claim'], source['pool'], max(ks)))
            elapsed = None if arm == source_arm else time.perf_counter() - start
            conditions[arm] = {
                'pages': source['pages'], 'expansion_pages': source['expansion_pages'],
                'ranked': ranked, 'retained': ranked[:10],
                'by_k': {str(k): evidence_metrics(claim, source['pages'], source['pool'], ranked[:k]) for k in ks},
                **evidence_metrics(claim, source['pages'], source['pool'], ranked[:10]),
                'new_sources': source['new_sources'], 'seconds': elapsed or 0.0,
                'retained_chars': sum(len(h['text']) + len(h['page']) for h in ranked[:10]),
                'api_calls': [], 'candidate_source_arm': source_arm,
                'timing_note': 'Hybrid ranking only; cached BM25 is not re-timed.'}
    return {'id': saved['id'], 'claim': saved['claim'], 'conditions': conditions}


def rerank_main(argv=None):
    """Offline controlled reranking of a saved BM25 local/follow experiment."""
    parser = argparse.ArgumentParser(description=rerank_main.__doc__)
    parser.add_argument('--analyze', type=Path, required=True, help='saved BM25 snapshot run')
    parser.add_argument('--data-dir', type=Path, default=Path('verifier/data/fever'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('use a new output directory')
    source_manifest = json.loads((args.analyze / 'manifest.json').read_text())
    config = source_manifest['config']
    if config['method'] != 'bm25' or not {'initial', 'follow'} <= set(config['arms']):
        parser.error('need a saved BM25 baseline/follow run')
    ks = (3, 5, 10, 20, 30)
    if not set(ks) <= set(config['ks']):
        parser.error('source must contain saved ranking prefixes at 3/5/10/20/30')
    source_path = args.analyze / 'predictions.jsonl'
    if source_path.exists():
        rows = [json.loads(line) for line in source_path.read_text().splitlines() if line.strip()]
    else:
        source_path = args.analyze / 'predictions.jsonl.gz'
        with gzip.open(source_path, 'rt', encoding='utf-8') as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
    claims = load_manifest_claims(args.data_dir / 'shared_task_dev.jsonl', config['claim_ids'])
    lookup = {c['id']: c for c in claims}
    selected_ids = source_manifest['selected_ids']
    if [r['id'] for r in rows] != selected_ids or len(set(selected_ids)) != len(rows):
        parser.error('source predictions must match complete distinct manifest IDs in order')
    claims = [lookup[cid] for cid in selected_ids]
    assert_not_reserved(claims, 'cached sentence reranking development')
    from retrieval.fever_retrieval import FeverRetriever, DENSE_MODEL, DENSE_REVISION
    import torch
    torch.set_num_threads(4)
    retriever = FeverRetriever(str(args.data_dir), method='hybrid', candidate_pages=config['initial_pages'])
    # Load pinned cached weights before creating outputs; never download a model.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    retriever._dense_model()
    manifest = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'role': 'development',
        'selected_ids': selected_ids, 'source_run': str(args.analyze),
        'source_manifest_sha256': sha256(args.analyze / 'manifest.json'),
        'source_prediction_sha256': sha256(source_path),
        'config': {'arms': ['initial', 'follow', 'hybrid_local', 'hybrid_follow'],
                   'ks': list(ks), 'final_k': 10},
        'ranking': {'model': DENSE_MODEL, 'revision': DENSE_REVISION,
                    'fusion': 'equal-weight sentence BM25 + MiniLM RRF', 'rrf_constant': 60},
        'policy': 'Frozen BM25 initial pages, observed links and follow pages; only sentence ranking changes.',
        'new_network_requests': 0, 'bootstrap_seed': 0,
        'source_sha256': {str(p): sha256(p) for p in Path('retrieval').glob('*.py')},
        'note': 'Retrospective development ablation, not end-to-end hybrid traversal or holdout validation.'}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    output_rows = []
    try:
        with gzip.open(args.output / 'predictions.jsonl.gz', 'xt', encoding='utf-8') as stream:
            for i, (claim, row) in enumerate(zip(claims, rows), 1):
                scored = rerank_claim(claim, row, retriever, ks)
                output_rows.append(scored)
                stream.write(json.dumps(scored, ensure_ascii=False) + '\n')
                stream.flush()
                if i % 25 == 0:
                    print(f'rerank {i}/{len(rows)}', flush=True)
        result = analyze(output_rows, manifest)
        (args.output / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
        (args.output / 'report.md').write_text(report(result))
        print(report(result))
    finally:
        retriever.db.close()


def main(argv=None):
    import sys
    argv = sys.argv[1:] if argv is None else argv
    modes = {'snapshot': snapshot_main, 'web': web_main, 'rerank': rerank_main}
    if argv and argv[0] in modes:
        return modes[argv[0]](argv[1:])
    parser = argparse.ArgumentParser(description='FEVER retrieval extension experiments; no verifier inference.')
    parser.add_argument('experiment', choices=tuple(modes), help='snapshot expansion, live-web pilot, or cached hybrid reranking')
    parser.parse_args(argv)


if __name__ == '__main__':
    main()
