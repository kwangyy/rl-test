"""Evaluate a search-calling (multi-turn) verifier on frozen FEVER claim IDs.

The model gets only the claim, calls `search` itself, then answers with the
verifier JSON. Everything the model sees matches grpo_multiturn.py training:
the same system prompt, FeverSearchEnv tool and budget, chat template
(thinking off) and TRL's parse_response for tool calls. Decoding is greedy, as
in the shared single-turn evaluator. The final answer is scored with
verifier.model.score and the metrics with score_predictions.summarize, so the
numbers line up with single-turn results on the same IDs.

    python -m verifier.evaluation.eval_multiturn \
        --adapter <sft4500 checkpoints>,<grpo checkpoints> \
        --method bm25 --search-k 5 --max-calls 2 \
        --output verifier/evaluation/results/<condition>/<run>

Each predictions.jsonl row adds n_search, n_refused, ended_on_tool_call,
queries and messages to the usual fields; "cited" and "gold_in_context" refer
to every sentence the model's own searches returned.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.fever_eval import LEGACY_IDS, load_manifest_claims
from verifier.evaluation.score_predictions import summarize
from verifier.experiments._shared.grpo_multiturn import SYSTEM, FeverSearchEnv, retriever
from verifier.model import DATA, LOCAL_MODEL, SEED, SPLIT_FILE, adapter_label, gold_sets, score


def load_model(name, adapter):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl.chat_template_utils import add_response_schema

    tok = add_response_schema(AutoTokenizer.from_pretrained(name))  # as GRPOTrainer does for tool parsing
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda")
    for ad in (adapter or "").split(","):  # "sft,grpo": GRPO LoRA trained on the merged SFT model
        if ad:
            model = PeftModel.from_pretrained(model, ad).merge_and_unload()
    model.eval()
    return tok, model


def call_tool(env, call):
    """Run one parsed tool call the way GRPOTrainer._tool_call_loop does."""
    fn = call.get("function") or {}
    try:
        if call.get("type") != "function" or fn.get("name") != "search":
            raise ValueError(f"Tool {fn.get('name')} not found.")
        return str(env.search(**fn["arguments"]))
    except Exception as e:  # TRL returns the error to the model as the tool result
        return str({"error": str(e)})


def run_batch(tok, model, claims, a):
    import torch
    from trl.chat_template_utils import parse_response

    system = SYSTEM.format(max_calls=a.max_calls)
    envs = [FeverSearchEnv() for _ in claims]
    msgs = [[{"role": "system", "content": system}, {"role": "user", "content": f"Claim: {c['claim']}"}]
            for c in claims]
    final = [None] * len(claims)
    in_tokens, out_tokens = [0] * len(claims), [0] * len(claims)
    stop = {tok.eos_token_id, tok.pad_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    rounds = a.max_calls + a.budget_grace + 1  # TRL: max_tool_calling_iterations tool rounds, then one more turn
    for rnd in range(rounds):
        active = [i for i, f in enumerate(final) if f is None]
        if not active:
            break
        texts = [tok.apply_chat_template(msgs[i], tools=[envs[i].search], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False) for i in active]
        inp = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        torch.manual_seed(SEED)
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=a.max_new_tokens, do_sample=False,
                                 output_logits=True, return_dict_in_generate=True)
        n_in = inp["input_ids"].shape[1]
        for b, i in enumerate(active):
            ids = gen.sequences[b, n_in:].tolist()
            end = next((k for k, t in enumerate(ids) if t in stop), len(ids))
            prompt_ids = inp["input_ids"][b][inp["attention_mask"][b].bool()].tolist()
            msg = parse_response(tok, ids[:end + 1], prefix=prompt_ids, tools=[envs[i].search])
            in_tokens[i] += len(prompt_ids)
            out_tokens[i] += end
            msgs[i].append(msg)
            calls = msg.get("tool_calls")
            if calls and rnd < rounds - 1:
                for call in calls:
                    msgs[i].append({"role": "tool", "name": (call.get("function") or {}).get("name", "unknown"),
                                    "content": call_tool(envs[i], call)})
                continue
            tokens = []  # (text, top-20 alternatives) per answer token, for verdict-token probabilities
            if not calls:
                for step in range(end):
                    lp = torch.log_softmax(gen.logits[step][b].float(), -1)
                    top = lp.topk(20)
                    tokens.append((tok.decode([ids[step]]),
                                   [(tok.decode([j]), v) for j, v in zip(top.indices.tolist(), top.values.tolist())]))
            final[i] = {"text": "" if calls else (msg.get("content") or ""), "tokens": tokens}
    rows = []
    for c, env, m, f, n_in_c, n_out_c in zip(claims, envs, msgs, final, in_tokens, out_tokens):
        shown = []
        for eid, key in env.shown.items():
            page, line = key.rsplit(":", 1)
            shown.append((eid, page, int(line), ""))
        gs = gold_sets(c)
        keys = {(p, line) for _, p, line, _ in shown}
        ctx = {"shown": shown, "gold_page_retrieved": None,
               "gold_in_context": any(s <= keys for s in gs) if gs else None}
        gen = {"text": f["text"], "tokens": f["tokens"], "provider": "local",
               "in_tokens": n_in_c, "out_tokens": n_out_c}
        row = score(c, ctx, gen)
        row.update({"model": a.model_label, "retrieval": a.retrieval_label, "prompt_style": "multiturn",
                    "n_search": env.n_search, "n_refused": env.n_refused,
                    "ended_on_tool_call": bool(m[-1].get("tool_calls")),
                    "queries": [((call.get("function") or {}).get("arguments") or {}).get("query")
                                for msg in m if msg.get("role") == "assistant"
                                for call in msg.get("tool_calls") or []],
                    "messages": m})
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=LOCAL_MODEL)
    ap.add_argument("--adapter", default=None, help="LoRA adapter(s) merged in order, e.g. 'sft,grpo'")
    ap.add_argument("--ids", default=str(LEGACY_IDS), help="frozen dev ID manifest (default: the 900 dev IDs)")
    ap.add_argument("--n", type=int, default=0, help="evaluate only the first n IDs (smoke test)")
    ap.add_argument("--method", choices=("bm25", "hybrid"), default="bm25")
    ap.add_argument("--search-k", type=int, default=5)
    ap.add_argument("--candidate-pages", type=int, default=3)
    ap.add_argument("--max-calls", type=int, default=2)
    ap.add_argument("--budget-grace", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=256, help="per assistant turn")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--output", required=True, help="new results/<condition>/<run> directory")
    a = ap.parse_args()
    out = Path(a.output)
    if out.exists():
        raise SystemExit(f"{out} exists; choose a new run directory")
    FeverSearchEnv.method = a.method
    FeverSearchEnv.k_pages, FeverSearchEnv.search_k = a.candidate_pages, a.search_k
    FeverSearchEnv.max_calls, FeverSearchEnv.budget_grace = a.max_calls, a.budget_grace
    a.model_label = a.model if not a.adapter else f"{a.model}+{adapter_label(a.adapter)}"
    a.retrieval_label = f"agent-{a.method}-p{a.candidate_pages}-k{a.search_k}-calls{a.max_calls}"

    claims = load_manifest_claims(os.path.join(DATA, SPLIT_FILE["dev"]), a.ids)
    if a.n:
        claims = claims[:a.n]
    retriever()
    tok, model = load_model(a.model, a.adapter)
    out.mkdir(parents=True)
    t0, rows = time.time(), []
    with open(out / "predictions.jsonl", "w", encoding="utf-8") as f:
        for s in range(0, len(claims), a.batch_size):
            batch = run_batch(tok, model, claims[s:s + a.batch_size], a)
            for r in batch:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            f.flush()
            rows += batch
            print(f"{len(rows)}/{len(claims)} claims, acc {sum(r['correct'] for r in rows) / len(rows):.3f}",
                  flush=True)
    n = len(rows)
    non_nei = [r for r in rows if r["gold"] != "NOT_ENOUGH_INFO"]
    metrics = summarize(rows)
    metrics["multiturn"] = {
        "searches_per_claim": sum(r["n_search"] for r in rows) / n,
        "search_counts": {k: sum(r["n_search"] == k for r in rows) for k in range(a.max_calls + 1)},
        "refused_rate": sum(r["n_refused"] > 0 for r in rows) / n,
        "ended_on_tool_call_rate": sum(r["ended_on_tool_call"] for r in rows) / n,
        "gold_in_context_non_nei": sum(bool(r["gold_in_context"]) for r in non_nei) / max(1, len(non_nei)),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    manifest = {**vars(a), "n_claims": n, "seconds": round(time.time() - t0), "seed": SEED, "decoding": "greedy"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
