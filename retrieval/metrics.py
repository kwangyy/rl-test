"""Pure retrieval scoring; no tools, network or verifier imports."""
from collections import Counter
import math
import random
import re


def normalized(text):
    return ' '.join(re.findall(r'\w+', text.casefold()))


def contains_complete_set(groups, texts, chunks):
    """Conservative verbatim diagnostic; never concatenate unrelated passages."""
    haystacks = [' '+normalized(chunk)+' ' for chunk in chunks]
    return any(all(texts.get(key) and any(' '+normalized(texts[key])+' ' in value
                                         for value in haystacks) for key in group)
               for group in groups)


def lexical_cosine(query, text):
    """Bag-of-words cosine: topic-overlap diagnostic, NOT relevance labels/recall."""
    a, b = (Counter(re.findall(r'\w+', value.casefold())) for value in (query, text))
    denom = math.sqrt(sum(v*v for v in a.values()) * sum(v*v for v in b.values()))
    return sum(v*b.get(term, 0) for term,v in a.items())/denom if denom else 0.0


def annotations(claim):
    groups = []
    for group in claim.get('evidence', []):
        items = {(str(e[2]), int(e[3])) for e in group
                 if len(e) >= 4 and e[2] is not None and e[3] is not None}
        if items:
            groups.append(items)
    return groups


def evidence_metrics(claim, pages, pool, final):
    groups = annotations(claim)
    gold_union = set().union(*groups) if groups else set()
    keys = lambda hits: {(h['page'], h['line']) for h in hits}
    selected = keys(final)
    page_sets = [{page for page,line in group} for group in groups]
    return {
        'coverage_eligible': bool(groups),
        'complete_gold_page_coverage': any(group <= set(pages) for group in page_sets) if groups else None,
        'pool_complete_set_coverage': any(group <= keys(pool) for group in groups) if groups else None,
        'complete_set_coverage': any(group <= selected for group in groups) if groups else None,
        'any_gold_sentence': bool(selected & gold_union) if groups else None,
        'annotated_sentence_recall': len(selected & gold_union)/len(gold_union) if groups else None,
    }



def recall_at_k(claims, results, ks=(1, 3, 5, 10), level='pages'):
    """Shared complete-set semantics for historical page/sentence commands."""
    if not claims or len(claims) != len(results) or level not in ('pages', 'sentences'):
        raise ValueError('need nonempty aligned claims/results and pages/sentences level')
    out = {}
    for k in ks:
        scored = []
        for claim, retrieved in zip(claims, results):
            top = retrieved[:k]
            pages = top if level == 'pages' else [h['page'] for h in top]
            metrics = evidence_metrics(claim, pages, [], top if level == 'sentences' else [])
            value = metrics['complete_gold_page_coverage' if level == 'pages' else 'complete_set_coverage']
            if value is not None:
                scored.append(value)
        out[k] = sum(scored)/len(scored) if scored else None
    return out


def paired_outcomes(pairs, seed=0):
    """One implementation of gained/lost, net change and paired uncertainty."""
    differences = [int(b)-int(a) for a,b in pairs]
    if not differences:
        return {'gained': 0, 'lost': 0, 'recall_delta': None, 'paired_bootstrap_95': None}
    rng = random.Random(seed)
    draws = sorted(sum(rng.choices(differences, k=len(differences)))/len(differences) for _ in range(2000))
    return {'gained': sum(not a and b for a,b in pairs),
            'lost': sum(a and not b for a,b in pairs),
            'recall_delta': sum(differences)/len(differences),
            'paired_bootstrap_95': [draws[49], draws[1949]]}

def summarize(rows, arms, seed=0):
    eligible = [r for r in rows if next(iter(r['conditions'].values()))['coverage_eligible']]
    summary = {'claims': len(rows), 'coverage_claims': len(eligible),
               'excluded_no_gold': len(rows)-len(eligible), 'conditions': {}, 'paired': {}, 'slices': {}}
    for arm in arms:
        values = [r['conditions'][arm] for r in rows]
        mean = lambda key: sum(v[key] for v in values if v[key] is not None)/sum(v[key] is not None for v in values) if any(v[key] is not None for v in values) else None
        metrics = {key: mean(key) for key in (
            'complete_gold_page_coverage','pool_complete_set_coverage','complete_set_coverage',
            'any_gold_sentence','annotated_sentence_recall','source_jaccard_vs_initial',
            'new_sources','new_candidate_sentences','new_retained_sentences','retained_chars',
            'mean_query_lexical_cosine','mean_query_semantic_cosine','seconds','diagnostic_seconds')}
        metrics['failed_actions'] = sum(v['failed_actions'] for v in values)
        metrics['failed_action_statuses'] = dict(Counter(t['result']['status'] for v in values for t in v['trace']
                                                        if t['result']['status'] not in ('ok','already_visited')))
        metrics['empty_snapshot_pages'] = sum(v['empty_snapshot_pages'] for v in values)
        metrics['mean_retained_sentences'] = sum(len(v['retained']) for v in values)/len(values)
        sources = [s for v in values for t in v['trace'] for s in t['result'].get('sources', [])]
        metrics.update(
            exa_returned_sources=len(sources),
            web_text_extraction_rate=sum(s['text_extracted'] for s in sources)/len(sources) if sources else None,
            duplicate_url_rate=sum(s['duplicate_url'] for s in sources)/len(sources) if sources else None,
            duplicate_page_rate=sum(s['duplicate_page'] for s in sources)/len(sources) if sources else None,
            snapshot_mapping_rate=sum(s['mapping_status'] in ('resolved','duplicate') for s in sources)/len(sources) if sources else None,
            unresolved_snapshot_urls=sum(s['mapping_status'] in ('not_in_snapshot','invalid_url') for s in sources))
        events = [e for v in values for e in v['api_calls']]
        metrics['network_attempts'] = sum(e['network_attempted'] for e in events)
        metrics['cache_hits'] = sum(e['cache_hit'] for e in events)
        metrics['provider_failed_urls'] = sum(status.get('status') != 'success'
                                              for e in events for status in e.get('provider_statuses', []))
        known_costs = [e['recorded_cost_dollars'] for e in events if e.get('recorded_cost_dollars') is not None]
        metrics['recorded_response_cost_sum'] = sum(known_costs) if known_costs else None
        metrics['cost_note'] = 'Includes replayed response estimates; not actual new billing. Per-request details in predictions.'
        summary['conditions'][arm] = metrics
    for reference in ('initial','local'):
        if reference not in arms or not eligible:
            continue
        for arm in arms:
            if arm == reference:
                continue
            pairs = [(r['conditions'][reference]['complete_set_coverage'],r['conditions'][arm]['complete_set_coverage']) for r in eligible]
            summary['paired'][arm+'-vs-'+reference] = {
                **paired_outcomes(pairs, seed), 'seed': seed,
                'note': 'Exploratory development comparison; not independent validation after tuning.'}
            overlaps = []
            for row in rows:
                a = set(row['conditions'][reference]['expansion_pages'])
                b = set(row['conditions'][arm]['expansion_pages'])
                overlaps.append(len(a&b)/len(a|b) if a|b else 1.0)
            summary['paired'][arm+'-vs-'+reference]['mean_expansion_source_jaccard'] = sum(overlaps)/len(overlaps)
    for multi_page in (False,True):
        subset = [r for r in eligible if r['requires_multiple_pages'] == multi_page]
        summary['slices']['requires_multiple_pages' if multi_page else 'single_page_sufficient'] = {
            'claims':len(subset), 'complete_set_recall':{
                arm:sum(r['conditions'][arm]['complete_set_coverage'] for r in subset)/len(subset)
                if subset else None for arm in arms}}
    return summary


def analyze(rows, manifest):
    if not rows or [r['id'] for r in rows] != manifest['selected_ids']:
        raise ValueError('predictions must exactly match the completed manifest cohort/order')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('duplicate claim IDs')
    arms = [a for a in manifest['config']['arms'] if a != 'exa-live']
    ks = sorted(set(int(k) for k in rows[0]['conditions'][arms[0]]['by_k']))
    eligible = [r for r in rows if r['conditions'][arms[0]]['coverage_eligible']]
    result = {'claims':len(rows),'coverage_claims':len(eligible),'ks':ks,'conditions':{},'paired':{}}
    for arm in arms:
        scored = [r['conditions'][arm] for r in eligible]
        required = {str(k) for k in ks}
        if any(set(v['by_k']) != required for v in scored):
            raise ValueError('inconsistent k budgets')
        examples = {name:[] for name in ('missing_source','missing_snapshot_sentence','ranking_loss','fixed_discovery','coverage_regression')}
        for r in eligible:
            value = r['conditions'][arm]
            baseline = r['conditions'].get('initial',value)
            categories = []
            if not value['complete_gold_page_coverage']:categories.append('missing_source')
            elif not value['pool_complete_set_coverage']:categories.append('missing_snapshot_sentence')
            elif not value['complete_set_coverage']:categories.append('ranking_loss')
            if not baseline['pool_complete_set_coverage'] and value['pool_complete_set_coverage']:categories.append('fixed_discovery')
            if baseline['complete_set_coverage'] and not value['complete_set_coverage']:categories.append('coverage_regression')
            for name in categories:
                examples[name].append({'id':r['id'],'claim':r['claim'],'expansion_pages':value['expansion_pages']})
        all_values = [r['conditions'][arm] for r in rows]
        first_k = {str(k):0 for k in ks}
        first_k['not_covered_at_largest_k'] = 0
        for value in scored:
            first = next((str(k) for k in ks if value['by_k'][str(k)]['complete_set_coverage']),
                         'not_covered_at_largest_k')
            first_k[first] += 1
        result['conditions'][arm] = {
            'first_successful_tested_k_counts':first_k,
            'complete_set_recall_by_k':{str(k):sum(v['by_k'][str(k)]['complete_set_coverage'] for v in scored)/len(scored)
                                         if scored else None for k in ks},
            'complete_page_coverage':sum(v['complete_gold_page_coverage'] for v in scored)/len(scored) if scored else None,
            'mean_new_sources':sum(v['new_sources'] for v in all_values)/len(rows),
            'mean_seconds':sum(v['seconds'] for v in all_values)/len(rows),
            'mean_retained_chars':sum(v['retained_chars'] for v in all_values)/len(rows),
            'recorded_response_cost_sum': (sum(e['recorded_cost_dollars'] for v in all_values
                for e in v['api_calls'] if e.get('recorded_cost_dollars') is not None)
                if any(e.get('recorded_cost_dollars') is not None for v in all_values for e in v['api_calls']) else None),
            'network_attempts': sum(e['network_attempted'] for v in all_values for e in v['api_calls']),
            'failures':{name:{'count':len(items),'examples':items[:10]} for name,items in examples.items()}}
    comparisons = [(reference, arm) for reference in ('initial', 'local')
                   if reference in arms for arm in arms if arm != reference]
    # Fixed-pool hybrid ablations need the within-ranker follow comparison too.
    comparisons.extend((reference, target) for reference, target in (
        ('hybrid_local', 'hybrid_follow'), ('follow', 'hybrid_follow'))
        if reference in arms and target in arms)
    for reference, arm in comparisons:
        result['paired'][arm+'-vs-'+reference] = {str(k): paired_outcomes([
            (r['conditions'][reference]['by_k'][str(k)]['complete_set_coverage'],
             r['conditions'][arm]['by_k'][str(k)]['complete_set_coverage'])
            for r in eligible]) for k in ks}
    if 'new_network_requests' in manifest:
        result['new_network_requests'] = manifest['new_network_requests']
    return result


def report(result):
    def pct(v):return f'{100*v:.2f}%' if v is not None else 'n/a'
    lines = ['# Retrieval expansion analysis','',
             f"Claims: {result['claims']}; gold-coverage denominator: {result['coverage_claims']}.",'',
             'The table scores frozen FEVER retrieval; live text, when enabled, is reported separately.','',
             '| Condition | Gold pages | '+ ' | '.join('@'+str(k) for k in result['ks'])+' |',
             '|---|---|'+ '|'.join('---' for _ in result['ks'])+'|']
    for name,value in result['conditions'].items():
        lines.append('| '+name+' | '+pct(value['complete_page_coverage'])+' | '+
                     ' | '.join(pct(value['complete_set_recall_by_k'][str(k)]) for k in result['ks'])+' |')
    if result['paired']:
        lines += ['', '## Does expansion help?', '',
                  '| Comparison | k | Recovered / lost claims | Net recall change | Paired 95% interval |',
                  '|---|---|---|---|---|']
        for comparison, curves in result['paired'].items():
            if comparison.startswith('initial-vs-'):
                continue
            for k, value in curves.items():
                if '10' in curves and k != '10':
                    continue
                ci = value.get('paired_bootstrap_95')
                interval = ('['+ ', '.join(f'{100*x:+.2f} pp' for x in ci)+']') if ci else 'n/a'
                delta = value.get('recall_delta')
                lines.append(f"| {comparison} | {k} | {value['gained']} / {value['lost']} | "
                             + (f'{100*delta:+.2f} pp' if delta is not None else 'n/a') + f' | {interval} |')
        lines += ['', 'A discovery gain with unchanged/lower retained recall is not an end-to-end retrieval gain.',
                  'An all-zero bootstrap interval when no sampled claim changes is not proof of population equivalence.',
                  'Intervals describe this development cohort; they are not independent validation after tuning.']
    lines += ['','## Diagnostics','']
    for name,value in result['conditions'].items():
        lines.append(f"- {name}: mean new pages {value['mean_new_sources']:.2f}; "
                     f"mean retrieval seconds {value['mean_seconds']:.3f}; "
                     f"recorded response cost USD {value.get('recorded_response_cost_sum')}; "
                     f"network attempts {value.get('network_attempts', 'n/a')}; "
                     +'; '.join(f"{category}={record['count']}" for category,record in value['failures'].items()))
    lines += ['', 'Recorded response costs can include replayed estimates; they are not new billing.',
              'Replay timing does not measure live provider latency. Use the original live-request records for that.']
    if 'live' in result:
        import json
        lines += ['', '## Current-text diagnostics', '',
                  'Not FEVER sentence recall. Similarity and source novelty do not establish evidence sufficiency.',
                  'Use automatic reference diagnostics separately; compare text volumes before claiming relevance gains.',
                  '', '```json', json.dumps(result['live'], indent=2), '```']
    lines += ['','Examples and paired gained/lost counts by k are in analysis.json.',
              'Local is wider search unless a recorded retry-query file was supplied.',
              'These are exploratory development results; inspect paired changes and uncertainty before adopting defaults.']
    return '\n'.join(lines)+'\n'



def analyze_saved(run, output):
    import hashlib
    import json
    from pathlib import Path
    run, output = Path(run), Path(output)
    if output.exists():
        raise ValueError('analysis directory must be new')
    manifest = json.loads((run/'manifest.json').read_text())
    prediction_path = run / 'predictions.jsonl'
    if prediction_path.exists():
        text = prediction_path.read_text()
    else:
        import gzip
        prediction_path = run / 'predictions.jsonl.gz'
        with gzip.open(prediction_path, 'rt', encoding='utf-8') as stream:
            text = stream.read()
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    result = analyze(rows, manifest)
    result['prediction_sha256'] = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    result['manifest_sha256'] = hashlib.sha256((run/'manifest.json').read_bytes()).hexdigest()
    if any('live' in row for row in rows):
        result['live'] = summarize_live(rows)
    output.mkdir(parents=True, exist_ok=False)
    (output/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    (output/'report.md').write_text(report(result))
    return result


def summarize_live(rows):
    values = [row['live'] for row in rows if 'live' in row]
    count = len(values)
    total = sum(v['returned_sources'] for v in values)
    known_cost = [e['recorded_cost_dollars'] for v in values for e in v['api_calls']
                  if e.get('recorded_cost_dollars') is not None]
    return {'claims': count, 'failed_searches': sum(v['status'] != 'ok' for v in values),
            'empty_context_claims': sum(not v['retained'] for v in values),
            'returned_sources': total,
            'text_extraction_rate_on_returned_sources': sum(v['extracted_sources'] for v in values)/total if total else None,
            'duplicate_urls': sum(v['duplicate_urls'] for v in values),
            'duplicate_contents': sum(v['duplicate_contents'] for v in values),
            'mean_query_lexical_cosine': (sum(v['query_lexical_cosine'] for v in values if v['query_lexical_cosine'] is not None)
                /sum(v['query_lexical_cosine'] is not None for v in values)
                if any(v['query_lexical_cosine'] is not None for v in values) else None),
            'mean_unique_urls': sum(v['unique_urls'] for v in values)/count if count else None,
            'mean_novel_mapped_pages': sum(v['novel_mapped_pages'] for v in values)/count if count else None,
            'mean_overlap_initial_pages': sum(v['overlap_initial_snapshot_pages'] for v in values)/count if count else None,
            'mean_retained_chars': sum(v['retained_chars'] for v in values)/count if count else None,
            'network_attempts': sum(e['network_attempted'] for v in values for e in v['api_calls']),
            'cache_hits': sum(e['cache_hit'] for v in values for e in v['api_calls']),
            'recorded_response_cost_sum': sum(known_cost) if known_cost else None,
            'cost_note': 'Replay estimates are not new billing; shared requests must not be summed across conditions.',
            'gold_sentence_recall': None,
            'note': 'All claims retained in failure/empty rates. Extraction denominator is returned sources. Extraction and overlap do not establish semantic evidence coverage.'}


# Optional offline similarity diagnostics; torch/transformers load only on demand.
import numpy as np

MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
REVISION = 'c9745ed1d9f207416be6d2e6f8de32d1f16199bf'
THRESHOLDS = (0.6, 0.7, 0.8)


def sentence_views(hit, max_characters=600):
    """Same evaluation-only sentence/window segmentation for every condition.

    Page titles supply entity context. Windows overlap 100 characters for long
    unpunctuated spans; embeddings have a separate 256-token truncation limit.
    """
    if max_characters <= 100:
        raise ValueError('sentence window must exceed overlap of 100 characters')
    title = hit.get('title', '').replace('_', ' ')
    out = []
    for sentence in re.split(r'(?<=[.!?])\s+|\n+', hit['text']):
        sentence = sentence.strip()
        if not sentence:
            continue
        start = 0
        while True:
            chunk = sentence[start:start+max_characters]
            out.append(title+' | '+chunk)
            if start+max_characters >= len(sentence):
                break
            start += max_characters-100
    return list(dict.fromkeys(out))


class LocalEncoder:
    """No key, API request or model download; fail explicitly if not cached."""
    def __init__(self):
        import torch
        from transformers import AutoModel, AutoTokenizer
        torch.set_num_threads(4)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
        self.model = AutoModel.from_pretrained(MODEL, revision=REVISION, local_files_only=True).eval()

    def encode(self, texts):
        import torch
        import torch.nn.functional as F
        out = []
        with torch.inference_mode():
            for start in range(0, len(texts), 64):
                tokens = self.tokenizer(texts[start:start+64], padding=True, truncation=True,
                                        max_length=256, return_tensors='pt')
                hidden = self.model(**tokens).last_hidden_state
                mask = tokens['attention_mask'].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden*mask).sum(1)/mask.sum(1).clamp(min=1e-9)
                out.append(F.normalize(pooled, p=2, dim=1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.model.config.hidden_size))


def reference_similarity(groups, references, candidate_vectors, reference_vectors):
    """Best alternative gold set, weakest required sentence, best retained match."""
    if not groups:
        raise ValueError('nonempty reference groups required')
    lookup = {key: i for i, key in enumerate(references)}
    if any(key not in lookup for group in groups for key in group):
        raise ValueError('every gold reference must have an embedding')
    if not len(candidate_vectors):
        return 0.0
    similarities = np.clip(reference_vectors @ candidate_vectors.T, -1, 1)
    nearest = similarities.max(axis=1)
    return float(max(min(nearest[lookup[key]] for key in group) for group in groups))


def evaluate_similarity(rows, references, *, encoder=None):
    """Score fixed retained contexts against gold and a mismatched-claim control.

    references is one {groups, hits: {(page,line): {title,text}}} record per row.
    Thresholds were fixed before scoring; no threshold is labelled semantic recall.
    """
    if len(rows) < 2 or len(rows) != len(references):
        raise ValueError('at least two aligned rows/references required for shuffled control')
    arms = tuple(rows[0]['conditions'])
    texts = set()
    views = []
    gold_views, negative_views = [], []
    for row, reference in zip(rows, references):
        if tuple(row['conditions']) != arms:
            raise ValueError('all conditions must occur in every row, in the same order')
        item = {arm: list(dict.fromkeys(v for h in row['conditions'][arm]['retained']
                                      for v in sentence_views(h))) for arm in arms}
        views.append(item)
        gv = {key: h['title'].replace('_', ' ')+' | '+h['text'] for key,h in reference['hits'].items()}
        nv = {key: h['title'].replace('_', ' ')+' | It is not true that '+h['text'] for key,h in reference['hits'].items()}
        gold_views.append(gv); negative_views.append(nv)
        texts.update(gv.values()); texts.update(nv.values())
        for value in item.values(): texts.update(value)
        texts.add(row['claim'])
    ordered = sorted(texts)
    encoder = encoder or LocalEncoder()
    matrix = np.asarray(encoder.encode(ordered))
    if matrix.ndim != 2 or matrix.shape[0] != len(ordered) or not np.isfinite(matrix).all():
        raise ValueError('encoder must return one finite vector per text')
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0): raise ValueError('encoder returned a zero vector')
    matrix = matrix/norms
    lookup = dict(zip(ordered, matrix))
    vectors = lambda values: np.asarray([lookup[v] for v in values]).reshape(-1, matrix.shape[1])
    per_claim = []
    for i,(row,reference) in enumerate(zip(rows,references)):
        ref_keys = list(reference['hits'])
        gv = vectors(gold_views[i].values())
        nv = vectors(negative_views[i].values())
        wrong = (i+1) % len(rows)
        wrong_keys = list(references[wrong]['hits'])
        wrong_vectors = vectors(gold_views[wrong].values())
        record = {'id': row['id'], 'conditions': {}}
        for arm in arms:
            cv = vectors(views[i][arm])
            score = reference_similarity(reference['groups'], ref_keys, cv, gv)
            shuffled = reference_similarity(references[wrong]['groups'], wrong_keys, cv, wrong_vectors)
            negated = reference_similarity(reference['groups'], ref_keys, cv, nv)
            query_score = float(np.clip(cv @ lookup[row['claim']], -1, 1).max()) if len(cv) else 0.0
            record['conditions'][arm] = {'gold_similarity': score, 'shuffled_gold_similarity': shuffled,
                'negated_gold_similarity': negated, 'query_similarity': query_score, 'sentence_views': len(cv)}
        per_claim.append(record)
    result = {'model': MODEL, 'revision': REVISION, 'max_tokens': 256, 'sentence_window_characters': 600,
        'claims': len(rows), 'thresholds': THRESHOLDS, 'conditions': {}, 'paired_vs_local': {}, 'per_claim': per_claim,
        'note': 'Automatic cosine diagnostics, not entailment, semantic precision/recall or verdict correctness. Shuffled and negated-reference controls expose limitations.'}
    for arm in arms:
        values = [r['conditions'][arm] for r in per_claim]
        result['conditions'][arm] = {
            'mean_gold_similarity': float(np.mean([v['gold_similarity'] for v in values])),
            'mean_shuffled_gold_similarity': float(np.mean([v['shuffled_gold_similarity'] for v in values])),
            'mean_query_similarity': float(np.mean([v['query_similarity'] for v in values])),
            'thresholds': {str(t): {
                'gold_match_rate': sum(v['gold_similarity'] >= t for v in values)/len(values),
                'shuffled_match_rate': sum(v['shuffled_gold_similarity'] >= t for v in values)/len(values),
                'negated_match_rate': sum(v['negated_gold_similarity'] >= t for v in values)/len(values)} for t in THRESHOLDS}}
        if arm != 'local':
            differences = np.asarray([r['conditions'][arm]['gold_similarity']-r['conditions']['local']['gold_similarity'] for r in per_claim])
            rng = np.random.default_rng(20261024)
            draws = np.mean(rng.choice(differences,size=(2000,len(differences)),replace=True),axis=1)
            result['paired_vs_local'][arm] = {
                'mean_cosine_delta': float(differences.mean()),
                'paired_bootstrap_95': np.quantile(draws,[0.025,0.975]).tolist(),
                'thresholds': {str(t): paired_outcomes([
                    (r['conditions']['local']['gold_similarity']>=t,r['conditions'][arm]['gold_similarity']>=t)
                    for r in per_claim]) for t in THRESHOLDS}}
    return result
