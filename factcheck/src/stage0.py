"""Stage 0: prompted baseline on FEVER dev, no training.

Retrieve top-k pages with BM25, show their sentences, ask the model for
{verdict, evidence ids, confidence} as JSON. Runs the local small model or
any OpenRouter model on the same inputs, so the two are directly comparable.

Two confidences are recorded per claim: the one the model states in the
JSON, and the probability it puts on its verdict label at the first token
of the verdict (renormalised over the three labels).

    python stage0.py --model local --n 50
    python stage0.py --model qwen/qwen3.8-flash --n 50
    python stage0.py --summarize ../data/stage0/local_n50.jsonl

Results are appended per claim to factcheck/data/stage0/<model>_n<N>.jsonl,
and a rerun skips claims already in the file.
"""

import argparse
import json
import math
import os
import random
import re
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

from build_bm25_index import DATA
from reward import evidence_f1

LOCAL_MODEL = "Qwen/Qwen3.5-2B"
BIG_MODEL = "qwen/qwen3.8-flash"
SEED = 0
K_PAGES = 5
MAX_SENTS_PER_PAGE = 20
MAX_CITED = 5  # FEVER scoring only looks at the first 5 predicted sentences

LABELS = ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]
FEVER_LABEL = {"SUPPORTS": "SUPPORTED", "REFUTES": "REFUTED", "NOT ENOUGH INFO": "NOT_ENOUGH_INFO"}

SYSTEM = """You are a fact-checker. You get a claim and numbered evidence sentences from Wikipedia.
Decide whether the evidence SUPPORTS the claim, REFUTES it, or gives NOT_ENOUGH_INFO to decide. Use only the evidence given.

Reply with only a JSON object, nothing else:
{"verdict": "SUPPORTED" | "REFUTED" | "NOT_ENOUGH_INFO", "evidence": ["E3", ...], "confidence": <number 0 to 1>}

- evidence: ids of the sentences that justify the verdict, at most 5. Use [] for NOT_ENOUGH_INFO.
- confidence: the probability that your verdict is correct."""

OUT_DIR = os.path.join(DATA, "..", "stage0")
VERDICT_PREFIX = re.compile(r'"verdict"\s*:\s*"')


# ---------------------------------------------------------------- data

def sample_claims(n, seed=SEED):
    """n claims from FEVER dev, balanced across the three labels."""
    with open(os.path.join(DATA, "shared_task_dev.jsonl"), encoding="utf-8") as f:
        dev = [json.loads(line) for line in f]
    rng = random.Random(seed)
    out = []
    for i, label in enumerate(FEVER_LABEL):
        pool = [c for c in dev if c["label"] == label]
        out += rng.sample(pool, n // 3 + (1 if i < n % 3 else 0))
    rng.shuffle(out)
    return out


def gold_sets(claim):
    """Gold evidence as a list of complete sets of (page, line)."""
    if claim["label"] == "NOT ENOUGH INFO":
        return []
    return [{(e[2], e[3]) for e in s} for s in claim["evidence"]]


def build_context(retriever, claims):
    """Per claim: the shown sentences as [(eid, page, line, text)] plus
    retrieval diagnostics."""
    pages = retriever.search([c["claim"] for c in claims], k=K_PAGES)
    out = []
    for c, ps in zip(claims, pages):
        shown = []
        for p in ps:
            for line, text in retriever.sentences(p)[:MAX_SENTS_PER_PAGE]:
                shown.append((f"E{len(shown) + 1}", p, line, text))
        shown_keys = {(p, line) for _, p, line, _ in shown}
        gs = gold_sets(c)
        out.append({
            "shown": shown,
            "gold_page_retrieved": any(all(p in ps for p, _ in s) for s in gs) if gs else None,
            "gold_in_context": any(s <= shown_keys for s in gs) if gs else None,
        })
    return out


def render(claim, shown):
    lines = [f"{eid} [{page.replace('_', ' ')}] {text}" for eid, page, _, text in shown]
    return f"Claim: {claim}\n\nEvidence:\n" + "\n".join(lines)


# ---------------------------------------------------------------- parsing

def parse(text):
    """Return (parsed dict or None, per-field validity)."""
    m = re.search(r"\{.*\}", text, re.S)
    obj = None
    if m:
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    ok = {"json": isinstance(obj, dict)}
    obj = obj if ok["json"] else {}
    ok["verdict"] = obj.get("verdict") in LABELS
    ev = obj.get("evidence")
    ok["evidence"] = isinstance(ev, list) and all(isinstance(e, str) and re.fullmatch(r"E\d+", e) for e in ev)
    conf = obj.get("confidence")
    ok["confidence"] = isinstance(conf, (int, float)) and 0 <= conf <= 1
    return obj, ok


def label_probs(tokens):
    """Verdict distribution from the first token of the verdict value.

    tokens: generated tokens in order, each (text, [(alt_text, logprob), ...]).
    Finds the token that starts the verdict value, then assigns each
    alternative's probability to the label it is a prefix of. Returns
    {label: p} renormalised over the labels, or None if not found.
    """
    def value_start(s):
        m = VERDICT_PREFIX.search(s)
        return s[m.end():].upper() if m else ""

    text = ""
    for tok, alts in tokens:
        if value_start(text + tok):
            probs = dict.fromkeys(LABELS, 0.0)
            for alt, lp in alts:
                cont = value_start(text + alt)
                for L in LABELS:
                    if cont and (L.startswith(cont) or cont.startswith(L)):
                        probs[L] += math.exp(lp)
                        break
            z = sum(probs.values())
            return {L: p / z for L, p in probs.items()} if z > 0 else None
        text += tok
    return None


# ---------------------------------------------------------------- models

class LocalModel:
    def __init__(self, name=LOCAL_MODEL, batch_size=4):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda")
        self.batch_size = batch_size
        self.name = name

    def run(self, prompts):
        out = []
        for i in range(0, len(prompts), self.batch_size):
            out += self._batch(prompts[i:i + self.batch_size])
        return out

    def _batch(self, prompts):
        texts = [self.tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False) for p in prompts]
        inp = self.tok(texts, return_tensors="pt", padding=True).to("cuda")
        self.torch.manual_seed(SEED)
        gen = self.model.generate(**inp, max_new_tokens=160, do_sample=False,
                                  output_logits=True, return_dict_in_generate=True)
        n_in = inp["input_ids"].shape[1]
        res = []
        for b in range(len(prompts)):
            ids = gen.sequences[b, n_in:].tolist()
            tokens = []
            for step, tid in enumerate(ids):
                if tid in (self.tok.eos_token_id, self.tok.pad_token_id):
                    break
                lp = self.torch.log_softmax(gen.logits[step][b].float(), -1)
                top = lp.topk(20)
                alts = [(self.tok.decode([j]), v) for j, v in zip(top.indices.tolist(), top.values.tolist())]
                tokens.append((self.tok.decode([tid]), alts))
            res.append({"text": self.tok.decode(ids, skip_special_tokens=True),
                        "tokens": tokens, "provider": "local",
                        "in_tokens": int(inp["attention_mask"][b].sum()), "out_tokens": len(tokens)})
        return res


class OpenRouterModel:
    def __init__(self, name=BIG_MODEL):
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1",
                             api_key=os.environ["OPENROUTER_API_KEY"], max_retries=3)
        self.name = name

    def run(self, prompts):
        return [self._one(p) for p in prompts]

    def _one(self, prompt):
        r = self.client.chat.completions.create(
            model=self.name, temperature=0, seed=SEED, max_tokens=160,
            logprobs=True, top_logprobs=5,  # Alibaba, the only provider, caps this at 5
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            extra_body={"reasoning": {"enabled": False}, "provider": {"require_parameters": True}},
        )
        ch = r.choices[0]
        tokens = []
        if ch.logprobs and ch.logprobs.content:
            tokens = [(t.token, [(a.token, a.logprob) for a in t.top_logprobs]) for t in ch.logprobs.content]
        return {"text": ch.message.content or "", "tokens": tokens,
                "provider": getattr(r, "provider", None),
                "in_tokens": r.usage.prompt_tokens, "out_tokens": r.usage.completion_tokens}


# ---------------------------------------------------------------- run

def score(claim, ctx, gen):
    obj, ok = parse(gen["text"])
    gold = FEVER_LABEL[claim["label"]]
    pred = obj.get("verdict") if ok["verdict"] else None
    by_eid = {eid: (p, line) for eid, p, line, _ in ctx["shown"]}
    cited = [by_eid[e] for e in (obj.get("evidence") or []) if ok["evidence"] and e in by_eid][:MAX_CITED]
    gs = gold_sets(claim)
    has_evidence = any(s <= set(cited) for s in gs) if gs else True
    gold_ids = sorted({f"{p}:{l}" for s in gs for p, l in s})
    probs = label_probs(gen["tokens"])
    return {
        "id": claim["id"], "claim": claim["claim"], "gold": gold, "pred": pred,
        "correct": pred == gold, "fever_score": pred == gold and has_evidence,
        "evidence_f1": evidence_f1(gold_ids, [f"{p}:{l}" for p, l in cited]) if gs else None,
        "stated_conf": obj.get("confidence") if ok["confidence"] else None,
        "token_conf": probs[pred] if probs and pred else None, "label_probs": probs,
        "format_ok": all(ok.values()), "format": ok,
        "gold_page_retrieved": ctx["gold_page_retrieved"], "gold_in_context": ctx["gold_in_context"],
        "n_shown": len(ctx["shown"]), "cited": cited,
        "provider": gen["provider"], "in_tokens": gen["in_tokens"], "out_tokens": gen["out_tokens"],
        "raw": gen["text"],
    }


def run(model_name, n, chunk=20):
    from fever_retrieval import FeverRetriever
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{model_name.replace('/', '_')}_n{n}.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f}
    claims = [c for c in sample_claims(n) if c["id"] not in done]
    print(f"{len(done)} done, {len(claims)} to go -> {path}")
    if not claims:
        return path
    model = LocalModel() if model_name == "local" else OpenRouterModel(model_name)
    retriever = FeverRetriever()
    t0 = time.time()
    for i in range(0, len(claims), chunk):
        batch = claims[i:i + chunk]
        ctxs = build_context(retriever, batch)
        gens = model.run([render(c["claim"], x["shown"]) for c, x in zip(batch, ctxs)])
        with open(path, "a", encoding="utf-8") as f:
            for c, x, g in zip(batch, ctxs, gens):
                f.write(json.dumps({"model": model.name, **score(c, x, g)}, ensure_ascii=False) + "\n")
        print(f"  {i + len(batch)}/{len(claims)}  {time.time() - t0:.0f}s", flush=True)
    return path


# ---------------------------------------------------------------- metrics

def ece(rows, key, bins=10):
    rows = [r for r in rows if r[key] is not None]
    if not rows:
        return None, []
    table = []
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        in_bin = [r for r in rows if lo <= r[key] < hi or (b == bins - 1 and r[key] == 1.0)]
        if in_bin:
            conf = sum(r[key] for r in in_bin) / len(in_bin)
            acc = sum(r["correct"] for r in in_bin) / len(in_bin)
            total += len(in_bin) / len(rows) * abs(acc - conf)
            table.append((lo, hi, len(in_bin), conf, acc))
    return total, table


def summarize(path, cost_in=None, cost_out=None):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    n = len(rows)
    print(f"\n== {rows[0]['model']}  ({n} claims, {os.path.basename(path)}) ==")
    print(f"  format adherence      {mean([r['format_ok'] for r in rows]):.3f}   "
          + "  ".join(f"{k}={mean([r['format'][k] for r in rows]):.2f}" for k in rows[0]["format"]))
    print(f"  label accuracy        {mean([r['correct'] for r in rows]):.3f}")
    print(f"  FEVER score           {mean([r['fever_score'] for r in rows]):.3f}")
    f1 = [r["evidence_f1"] for r in rows if r["evidence_f1"] is not None]
    print(f"  evidence F1 (non-NEI) {mean(f1):.3f}")
    for L in LABELS:
        sub = [r for r in rows if r["gold"] == L]
        print(f"  recall {L:16s} {mean([r['pred'] == L for r in sub]):.3f}  (n={len(sub)})")
    for key in ("stated_conf", "token_conf"):
        e, table = ece(rows, key)
        cov = mean([r[key] is not None for r in rows])
        print(f"  ECE {key:11s}       {e if e is None else round(e, 3)}   (available {cov:.2f})")
        for lo, hi, cnt, conf, acc in table:
            print(f"      [{lo:.1f},{hi:.1f})  n={cnt:<4d} conf={conf:.2f}  acc={acc:.2f}")
    ne = [r for r in rows if r["gold"] != "NOT_ENOUGH_INFO"]
    for flag in (True, False):
        sub = [r for r in ne if r["gold_in_context"] is flag]
        print(f"  non-NEI, gold {'in ' if flag else 'NOT in'} context: n={len(sub):<4d} "
              f"acc={mean([r['correct'] for r in sub]):.3f}  FEVER={mean([r['fever_score'] for r in sub]):.3f}")
    tin, tout = sum(r["in_tokens"] for r in rows), sum(r["out_tokens"] for r in rows)
    print(f"  tokens in/out         {tin / n:.0f} / {tout / n:.0f} per claim")
    if cost_in is not None:
        print(f"  cost                  ${(tin * cost_in + tout * cost_out) / 1e6:.4f}")
    provs = {}
    for r in rows:
        provs[r["provider"]] = provs.get(r["provider"], 0) + 1
    print(f"  providers             {provs}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="local", help='"local" or an OpenRouter model id')
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--summarize", help="only print metrics for an existing results file")
    a = ap.parse_args()
    PRICES = {BIG_MODEL: (0.15, 0.47)}  # $/M tokens, OpenRouter 2026-09-19
    if a.summarize:
        summarize(a.summarize)
    else:
        p = run(a.model, a.n)
        summarize(p, *PRICES.get(a.model, (None, None)))
