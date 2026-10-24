"""AVeriTeC web-search pilot: Wikipedia controls versus Exa; gold QAs score only."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
from urllib.parse import urlsplit, unquote, quote, parse_qsl
from retrieval.build_bm25_index import unescape

from retrieval.exa import ExaClient, canonical_url
from retrieval.exa import search_web, passages, rank_passages
from retrieval.metrics import normalized, paired_outcomes
from utils.fever_eval import sha256

# Fixed policy, not derived from this sample's gold fact-check/source annotations.
EXCLUDED = ['snopes.com','politifact.com','factcheck.org','fullfact.org','checkyourfact.com',
 'factcheck.afp.com','factcheckni.org','leadstories.com','truthorfiction.com','africacheck.org',
 'altnews.in','boomlive.in','factly.in','logicallyfacts.com','aap.com.au','factcheck.kz',
 'github.com','huggingface.co','kaggle.com','averitec.github.io']
BLOCKED_PATHS = [('reuters.com','/fact-check'),('apnews.com','/hub/ap-fact-check'),
                 ('usatoday.com','/story/news/factcheck')]
ARMS = ('local','wider_local','local_exa')


def allowed_input(row):
    """Only explicitly allowed fields can cross into retrieval."""
    return {'claim':str(row['claim']), 'cutoff':datetime.strptime(row['claim_date'],'%d-%m-%Y').strftime('%Y-%m-%dT23:59:59Z')}


def original_url(url):
    url = (url or '').strip()
    for _ in range(3):
        match=re.match(r'https?://web\.archive\.org/web/[^/]+/(https?://.*)',url)
        if not match: break
        url=match[1]
    return url


def url_key(url):
    try:
        parts=urlsplit(canonical_url(original_url(url)))
        return (parts.hostname.removeprefix('www.'),unquote(parts.path).rstrip('/'),
                tuple(sorted((k,v) for k,v in parse_qsl(parts.query,keep_blank_values=True)
                             if not k.lower().startswith('utm_') and k.lower() not in ('fbclid','gclid'))))
    except (ValueError,TypeError):return None


def blocked(url):
    key=url_key(url)
    if key is None:return True
    host,path,_=key
    return any(host==d or host.endswith('.'+d) for d in EXCLUDED) or any(
        (host==d or host.endswith('.'+d)) and path.startswith(p) for d,p in BLOCKED_PATHS)


def source_allowed(source, cutoff):
    if blocked(source['url']):return False
    date=source.get('published_date')
    if date:
        try:
            published = datetime.fromisoformat(date.replace('Z', '+00:00'))
            # Preserve an explicit offset; interpret date-only/naive values as UTC.
            if published.tzinfo is None:
                published = published.replace(tzinfo=timezone.utc)
            if published > datetime.fromisoformat(cutoff.replace('Z', '+00:00')):
                return False
        except ValueError:
            return False
    return True  # Unknown publication dates are retained and counted explicitly.


def context(query, candidates, cap=4000):
    ranked=rank_passages(query,candidates)
    kept=[];remaining=cap
    for hit in ranked:
        if len(hit['text'])<=remaining:
            kept.append(hit);remaining-=len(hit['text'])
    return {'retained':kept,'retained_characters':cap-remaining,'candidate_passages':len(candidates)}


def collect(item, retriever, client):
    """No annotation object or source URL from gold is accepted here."""
    query=item['claim'];pages=retriever.search_with_titles([query],k=10)[0]
    pools=[]
    for page in pages:
        text='\n'.join(t for _,t in retriever.sentences(page))
        source={'url':'https://en.wikipedia.org/wiki/'+quote(unescape(page),safe='()_'),
            'title':unescape(page.replace('_',' ')),'text':text,'text_sha256':hashlib.sha256(text.encode()).hexdigest(),
            'fetched_at':'FEVER-2017-snapshot','published_date':None}
        pools.append(passages(source,max_characters=600))
    web=search_web(query,client,n=5,exclude_domains=EXCLUDED,end_published_date=item['cutoff'])
    usable=[s for s in web['sources'] if s['url'] and s['extracted'] and not s['duplicate_url']
            and not s['duplicate_text'] and source_allowed(s,item['cutoff'])]
    extra=[p for s in usable for p in passages(s,max_characters=600)]
    local=[h for pool in pools[:5] for h in pool]
    wider=[h for pool in pools for h in pool]
    conditions={'local':context(query,local),'wider_local':context(query,wider),
                'local_exa':context(query,local+extra)}
    return {'input':item,'conditions':conditions,'local_pages':pages,
            'web':web,'usable_web_urls':[s['url'] for s in usable]}


def _source_diagnostics(rows):
    """Source diversity/retention only; none of these quantities labels relevance."""
    values = []
    for row in rows:
        baseline = row['conditions']['local']['retained']
        local = {url_key(h['source_url']) for h in baseline}
        # Compare candidate sources when available, not only selected passages.
        if 'local_pages' in row:
            local = {url_key('https://en.wikipedia.org/wiki/' + quote(unescape(p), safe='()_'))
                     for p in row['local_pages'][:5]}
        web = {url_key(u) for u in row.get('usable_web_urls', [])}
        local.discard(None); web.discard(None)
        key = lambda h: (url_key(h['source_url']), normalized(h['text']))
        base_keys = {key(h) for h in baseline}
        extended = row['conditions'].get('local_exa', {}).get('retained', [])
        extended_keys = {key(h) for h in extended}
        total_chars = sum(len(h['text']) for h in extended)
        web_chars = sum(len(h['text']) for h in extended
                        if h.get('fetched_at') and h['fetched_at'] != 'FEVER-2017-snapshot')
        telemetry = 'usable_web_urls' in row
        values.append({'id': row['id'],
            'candidate_url_jaccard': len(local & web)/len(local | web) if telemetry and (local | web) else None,
            'novel_web_url_fraction': len(web - local)/len(web) if web else None,
            'baseline_passage_retention': len(base_keys & extended_keys)/len(base_keys)
                if base_keys and 'local_exa' in row['conditions'] else None,
            'live_text_character_share': web_chars/total_chars if total_chars and telemetry else None})
    mean = lambda name: (sum(v[name] for v in values if v[name] is not None) /
                         sum(v[name] is not None for v in values)) if any(v[name] is not None for v in values) else None
    sources = [s for row in rows for s in row.get('web', {}).get('sources', [])]
    return {'mean_candidate_url_jaccard': mean('candidate_url_jaccard'),
            'mean_novel_web_url_fraction': mean('novel_web_url_fraction'),
            'mean_baseline_passage_retention': mean('baseline_passage_retention'),
            'mean_live_text_character_share': mean('live_text_character_share'),
            'duplicate_returned_urls': sum(bool(s.get('duplicate_url')) for s in sources),
            'duplicate_returned_texts': sum(bool(s.get('duplicate_text')) for s in sources),
            'per_claim': values,
            'note': 'Macro claim means; source sets use canonical URLs. Novelty/overlap are diagnostics, not evidence recall. Text share uses fetch provenance.'}


def evaluate_averitec(rows, gold, *, semantic=False):
    """Evaluate saved passages without retrieval, network calls or file writes.

    rows: distinct {id, conditions: {name: {retained: [passage, ...]}}} records;
    each passage needs source_url/text (title for similarity). Include a `local`
    baseline and the same condition names on every row. Collection telemetry is
    optional; collect() supplies it. gold is a dict/list indexed by row id, used
    only for scoring. Returns coverage, paired outcomes and source diagnostics.
    semantic=True additionally loads the pinned offline MiniLM scorer.
    """
    if not rows or len({row['id'] for row in rows}) != len(rows):
        raise ValueError('need nonempty rows with distinct IDs')
    arms = tuple(rows[0]['conditions'])
    if 'local' not in arms or any(set(row['conditions']) != set(arms) for row in rows):
        raise ValueError('need a local baseline and matching condition names')
    per_claim=[];qa_rows=[];qa_refs=[];url_den=0;extractive_den=0
    for row in rows:
        annotation=gold[row['id']]
        refs={url_key(a.get('source_url')) for q in annotation.get('questions',[]) for a in q.get('answers',[])
              if a.get('source_url') and not blocked(a['source_url'])}
        refs.discard(None)
        extractive=[a['answer'] for q in annotation.get('questions',[]) for a in q.get('answers',[])
                    if a.get('answer_type')=='Extractive' and len(normalized(a.get('answer','')).split())>=5]
        url_den+=bool(refs);extractive_den+=bool(extractive)
        discovered={url_key(u) for u in row.get('usable_web_urls', [])}
        item={'id':row['id'],'eligible_gold_urls':len(refs),'extractive_answers':len(extractive),
              'discovered_gold_url_recall':len(refs & discovered)/len(refs) if refs and 'usable_web_urls' in row else None,'conditions':{}}
        for arm in arms:
            hits=row['conditions'][arm]['retained']
            urls={url_key(h['source_url']) for h in hits}
            texts=[' '+normalized(h['text'])+' ' for h in hits]
            found=sum(u in urls for u in refs)
            answers=sum(any(' '+normalized(a)+' ' in t for t in texts) for a in extractive)
            item['conditions'][arm]={'source_recall':found/len(refs) if refs else None,
                'any_annotated_source':bool(found) if refs else None,
                'extractive_answer_coverage':answers/len(extractive) if extractive else None,
                'chars':sum(len(h['text']) for h in hits)}
        per_claim.append(item)
        if semantic:
            for qi,q in enumerate(annotation.get('questions',[])):
                answers=[a['answer'] for a in q.get('answers',[]) if a.get('answer') and a.get('answer_type')!='Unanswerable']
                if not answers:continue
                qa_rows.append({'id':f"{row['id']}:{qi}",'claim':q['question'],'conditions':{arm: row['conditions'][arm] for arm in arms}})
                qa_refs.append({'groups':[{(qi,i)} for i in range(len(answers))],
                                'hits':{(qi,i):{'title':q['question'],'text':a} for i,a in enumerate(answers)}})
    discovered_scores = [r['discovered_gold_url_recall'] for r in per_claim if r['discovered_gold_url_recall'] is not None]
    metrics={'claims':len(rows),'source_eligible_claims':url_den,'extractive_answer_eligible_claims':extractive_den,
             'conditions':{},'paired_source_hits':{},'per_claim':per_claim,
             'mean_discovered_gold_url_recall':sum(discovered_scores)/len(discovered_scores) if discovered_scores else None}
    for arm in arms:
        vals=[r['conditions'][arm] for r in per_claim]
        metrics['conditions'][arm]={k:sum(v[k] for v in vals if v[k] is not None)/sum(v[k] is not None for v in vals)
           if any(v[k] is not None for v in vals) else None for k in vals[0]}
        if arm!='local':metrics['paired_source_hits'][arm]=paired_outcomes([
            (r['conditions']['local']['any_annotated_source'],r['conditions'][arm]['any_annotated_source'])
            for r in per_claim if r['conditions']['local']['any_annotated_source'] is not None])
    events=[e for r in rows for e in r.get('web', {}).get('api_calls', [])]
    sources=[s for r in rows for s in r.get('web', {}).get('sources', [])]
    metrics['operations']={'network_attempts':sum(e['network_attempted'] for e in events),
        'cache_hits':sum(e['cache_hit'] for e in events),
        'new_response_cost_estimate':sum(e.get('recorded_cost_dollars') or 0 for e in events if e['network_attempted']),
        'failed_searches':sum(r.get('web', {}).get('status', 'ok')!='ok' for r in rows),'returned_sources':len(sources),
        'extracted_sources':sum(s['extracted'] for s in sources),
        'policy_rejected_sources':sum(bool(s['url']) and not source_allowed(s,r['input']['cutoff']) for r in rows for s in r.get('web', {}).get('sources', [])),
        'unknown_publication_dates':sum(not s.get('published_date') for s in sources),
        'usable_sources':sum(len(r.get('usable_web_urls', [])) for r in rows),
        'gold_factcheck_article_url_matches':sum(url_key(s['url'])==url_key(gold[r['id']].get('fact_checking_article')) for r in rows for s in r.get('web', {}).get('sources', []) if s['url']),
        'mean_recorded_request_seconds':sum(e.get('recorded_network_seconds',0) for e in events)/len(events) if events else None}
    metrics['source_diagnostics'] = _source_diagnostics(rows)
    if semantic:
        from retrieval.metrics import evaluate_similarity
        metrics['qa_similarity']=evaluate_similarity(qa_rows,qa_refs)
        metrics['qa_similarity']['note']+=' QA-level diagnostic; questions within a claim are correlated, so QA-level bootstrap intervals are not claim-level inference.'
        # Aggregate QAs per claim before paired uncertainty, not as independent rows.
        import numpy as np
        by_claim={}
        for record in metrics['qa_similarity']['per_claim']:
            cid=record['id'].split(':')[0]
            by_claim.setdefault(cid,{arm:[] for arm in arms})
            for arm in arms:by_claim[cid][arm].append(record['conditions'][arm]['gold_similarity'])
        means={arm:np.asarray([np.mean(v[arm]) for v in by_claim.values()]) for arm in arms}
        metrics['qa_claim_weighted_cosine']={arm:float(v.mean()) for arm,v in means.items()}
        metrics['qa_claim_paired_delta']={}
        for arm in (a for a in arms if a != 'local'):
            differences=means[arm]-means['local']
            rng=np.random.default_rng(20261024)
            draws=rng.choice(differences,size=(2000,len(differences)),replace=True).mean(axis=1)
            metrics['qa_claim_paired_delta'][arm]={'mean_delta':float(differences.mean()),
                'paired_bootstrap_95':np.quantile(draws,[0.025,0.975]).tolist()}

    return metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('verifier/data/averitec/dev.json'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int,default=50);p.add_argument('--seed',type=int,default=20261024)
    p.add_argument('--allow-network',action='store_true');p.add_argument('--env-file',type=Path)
    p.add_argument('--max-api-requests',type=int,default=50)
    p.add_argument('--cache-dir',type=Path,default=Path('verifier/data/exa-cache/averitec-pilot50'))
    p.add_argument('--analyze',type=Path);p.add_argument('--semantic-similarity',action='store_true')
    p.add_argument('--conditions', nargs='+', choices=ARMS, default=list(ARMS), help='comparison conditions to score; collection records remain unchanged')
    args=p.parse_args()
    if 'local' not in args.conditions or len(set(args.conditions)) != len(args.conditions): p.error('conditions need local and distinct names')
    if args.output.exists() or not 1<=args.limit<=50 or not 0<=args.max_api_requests<=50:p.error('new output and 1..50 claims / 0..50 requests required')
    if args.analyze and (args.allow_network or args.env_file):p.error('analysis is offline')
    data=json.loads(args.data.read_text())
    ids=json.loads((args.analyze/'manifest.json').read_text())['selected_ids'] if args.analyze else random.Random(args.seed).sample(range(len(data)),args.limit)
    if not 1<=len(ids)<=50 or len(set(ids))!=len(ids):p.error('invalid cohort')
    inputs={i:allowed_input(data[i]) for i in ids}
    if args.env_file:
        if not args.allow_network or not args.env_file.is_file():p.error('explicit existing env file needs network flag')
        from dotenv import dotenv_values
        key=dotenv_values(args.env_file).get('EXA_API_KEY')
        if key:os.environ['EXA_API_KEY']=key
    args.output.mkdir(parents=True)
    manifest={'dataset':'AVeriTeC','role':'development','selected_ids':ids,'seed':args.seed,
        'dataset_sha256':sha256(args.data),'allowed_inputs':inputs,'excluded_domains':EXCLUDED,'blocked_paths':BLOCKED_PATHS,
        'conditions':args.conditions,'policy':{'local_pages':5,'wider_local_pages':10,'exa_results':5,'passage_chars':600,'context_chars':4000,
            'query':'original claim only','date_cutoff':'claim-date end of day UTC','unknown_dates':'keep and report',
            'max_new_requests':0 if args.analyze else args.max_api_requests,'retries':0},
        'source_sha256':{s:sha256(s) for s in ('retrieval/eval_averitec.py','retrieval/exa.py','retrieval/metrics.py')},
        'temporal_limit':'Publication filtering does not establish historical text validity. Wikipedia baseline is the FEVER-2017 snapshot.',
        'annotation_limit':'Gold QAs/URLs/labels/justifications never enter collection. Wayback wrappers and tracking parameters removed; content query parameters preserved.',
        'source_prediction_sha256':sha256(args.analyze/'predictions.jsonl') if args.analyze else None,
        'source_manifest_sha256':sha256(args.analyze/'manifest.json') if args.analyze else None}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if args.analyze:
        rows=[json.loads(l) for l in (args.analyze/'predictions.jsonl').read_text().splitlines()]
        if [r['id'] for r in rows]!=ids or any(r['input']!=inputs[r['id']] for r in rows):raise ValueError('source cohort/input mismatch')
    else:
        from retrieval.fever_retrieval import FeverRetriever
        r=FeverRetriever(method='bm25',candidate_pages=5)
        client=ExaClient(args.cache_dir,allow_network=args.allow_network,max_requests=args.max_api_requests)
        rows=[]
        try:
            with (args.output/'predictions.jsonl').open('x') as f:
                for i in ids:
                    row=dict(collect(inputs[i],r,client),id=i)
                    rows.append(row);f.write(json.dumps(row)+'\n');f.flush()
                    print(f'{len(rows)}/{len(ids)}',flush=True)
        finally:r.db.close()
    rows = [dict(row, conditions={arm: row['conditions'][arm] for arm in args.conditions}) for row in rows]
    metrics=evaluate_averitec(rows,data,semantic=args.semantic_similarity)
    metrics['new_network_requests_this_analysis']=0 if args.analyze else metrics['operations']['network_attempts']
    (args.output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps({k:v for k,v in metrics.items() if k not in ('per_claim','qa_similarity')},indent=2))


if __name__=='__main__':main()
