"""Prepare and validate evidence-grounded CoT SFT data.

Supports SoCLaaS chat completions and Gemini generateContent. Requests are
sequential and resumable; 429 stops the run for a later retry.
"""

import argparse
import collections
import getpass
import hashlib
import json
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "verifier/data/train/sft/fever-oracle-full/labeled_prompts.jsonl"
DEFAULT_OUT = ROOT / "verifier/data/train/sft/fever-oracle-cot-gemini"
SOCLAAS_OUT = ROOT / "verifier/data/train/sft/fever-oracle-cot-soclaas"
SOCLAAS_URL = "https://soclaas-api.comp.nus.edu.sg/v1"
MODEL = "gemini-3.8-flash"
SYSTEM = (
    "Write 1-3 short English sentences explaining a fact-verification label using only "
    "the numbered evidence in the user input. Cite E IDs for evidence used. The gold "
    "verdict and gold evidence IDs are given as targets, but if the shown sentences do "
    "not actually justify them, set gold_justified to false. For NEI, explain the "
    "missing connection without asserting a refutation. Do not invent evidence or "
    "use outside knowledge. Return only a JSON object with keys reasoning "
    "(string), relation (supports/refutes/insufficient), cited_evidence "
    "(array of E IDs), and gold_justified (boolean)."
)
SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "relation": {"type": "string", "enum": ["supports", "refutes", "insufficient"]},
        "cited_evidence": {"type": "array", "items": {"type": "string"}},
        "gold_justified": {"type": "boolean"},
    },
    "required": ["reasoning", "relation", "cited_evidence", "gold_justified"],
}
RELATION = {"SUPPORTED": "supports", "REFUTED": "refutes", "NOT_ENOUGH_INFO": "insufficient"}
EID = re.compile(r"\bE\d+\b")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Avoid forwarding an API credential to a redirected host."""

    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


OPENER = urllib.request.build_opener(NoRedirect)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def select_rows(rows, per_class, seed):
    if per_class == 0:
        return rows
    rng = random.Random(seed)
    selected = []
    for label in RELATION:
        group = [row for row in rows if row["gold"] == label]
        if per_class > len(group):
            raise ValueError(f"{label}: requested {per_class}, only {len(group)} available")
        selected.extend(rng.sample(group, per_class))
    rng.shuffle(selected)
    return selected


def request_for(row, provider="gemini", model=None):
    target = json.loads(row["completion"])
    user = (
        f"{row['prompt']}\n\nGold verdict: {row['gold']}\n"
        f"Gold evidence IDs: {json.dumps(target['evidence'])}\n"
        "Explain the decision using only the shown evidence."
    )
    if provider == "soclaas":
        if not model:
            raise ValueError("SoCLaaS requires a model ID from GET /v1/models")
        return {
            "model": model,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            "reasoning_effort": "none",
            "max_tokens": 1024,
        }
    return {
        "contents": [{"parts": [{"text": user}]}],
        "systemInstruction": {"parts": [{"text": SYSTEM}]},
        "generationConfig": {
            "responseFormat": {"text": {"mimeType": "application/json", "schema": SCHEMA}},
            "maxOutputTokens": 1024,
        },
    }


def prepare(args):
    from utils.fever_eval import assert_not_reserved
    if args.per_class < 0:
        raise ValueError("per_class must be nonnegative")
    if args.provider == "soclaas" and not args.model:
        raise ValueError("Use `models` to list allowed SoCLaaS IDs, then pass --model")
    out = Path(args.out or ((SOCLAAS_OUT if args.provider == "soclaas" else DEFAULT_OUT) / "full1"))
    source_bytes = Path(args.source).read_bytes()
    source = read_jsonl(args.source)
    chosen = select_rows(source, args.per_class, args.seed)
    assert_not_reserved(chosen, "teacher-target generation")
    ids = [str(row["id"]) for row in chosen]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate claim IDs in selected rows")
    frozen_dev = ROOT / "verifier/data/eval/fever-dev-900-ids.json"
    dev_ids = {str(cid) for cid in json.loads(frozen_dev.read_text(encoding="utf-8"))["ids"]}
    if set(ids) & dev_ids:
        raise ValueError("selected train rows overlap frozen FEVER dev evaluation IDs")
    source_path = Path(args.source).resolve()
    try:
        source_label = str(source_path.relative_to(ROOT))
    except ValueError:
        source_label = str(source_path)
    out.mkdir(parents=True, exist_ok=False)
    write_jsonl(out / "requests.jsonl", [{"id": row["id"], "body": request_for(row, args.provider, args.model)} for row in chosen])
    write_jsonl(out / "source_rows.jsonl", chosen)
    meta = {
        "source": source_label,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "model": args.model,
        "provider": args.provider,
        "base_url": args.base_url if args.provider == "soclaas" else None,
        "seed": args.seed,
        "per_class": args.per_class,
        "count": len(chosen),
        "label_counts": dict(collections.Counter(row["gold"] for row in chosen)),
        "schema_version": 2,
    }
    (out / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(chosen)} requests in {out}. No API request was sent.")


def soclaas_url(base_url, endpoint):
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme != "https" or parsed.hostname != "soclaas-api.comp.nus.edu.sg" or parsed.path.rstrip("/") != "/v1":
        raise ValueError("SoCLaaS URL must be https://soclaas-api.comp.nus.edu.sg/v1")
    return base_url.rstrip("/") + "/" + endpoint


def send_one(provider, model, body, key, base_url=None):
    if provider == "soclaas":
        url = soclaas_url(base_url, "chat/completions")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    else:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        headers = {"Content-Type": "application/json", "x-goog-api-key": key}
    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), method="POST",
        headers=headers,
    )
    with OPENER.open(request, timeout=120) as response:
        return json.load(response)


def secret_key(provider):
    if not sys.stdin.isatty():
        raise SystemExit("Run in an interactive terminal so the API key can be entered privately")
    key = getpass.getpass(f"{'SoCLaaS' if provider == 'soclaas' else 'Gemini'} API key (hidden): ").strip()
    if not key:
        raise SystemExit("No API key entered")
    return key


def models(args):
    url = soclaas_url(args.base_url, "models")
    key = secret_key("soclaas")
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    with OPENER.open(req, timeout=30) as response:
        catalogue = json.load(response)
    for item in catalogue.get("data", []):
        print(item.get("id"))


def run(args):
    if args.max_requests < 0 or (args.delay is not None and args.delay < 0):
        raise ValueError("max_requests and delay must be nonnegative")
    out = Path(args.out)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    model = manifest["model"]
    provider = manifest.get("provider", "gemini")
    delay = args.delay if args.delay is not None else (1.0 if provider == "soclaas" else 10.0)
    base_url = manifest.get("base_url")
    if provider == "soclaas":
        soclaas_url(base_url, "chat/completions")
    requests = read_jsonl(out / "requests.jsonl")
    responses_path = out / "responses.jsonl"
    done = {str(row["id"]) for row in read_jsonl(responses_path)} if responses_path.exists() else set()
    remaining = [row for row in requests if str(row["id"]) not in done]
    limit = min(args.max_requests, len(remaining)) if args.max_requests else len(remaining)
    print(f"Model {model}: {len(done)} done, sending at most {limit} of {len(remaining)} remaining")
    if not limit:
        return
    key = secret_key(provider)
    sent = 0
    with open(responses_path, "a", encoding="utf-8") as f:
        for row in remaining[:limit]:
            try:
                result = send_one(provider, model, row["body"], key, base_url)
            except urllib.error.HTTPError as exc:
                print(f"Stopped on HTTP {exc.code} at claim {row['id']}")
                print("Completed responses are saved; rerun the same command to resume.")
                return
            except (urllib.error.URLError, TimeoutError) as exc:
                print(f"Stopped on network error at claim {row['id']}: {exc}")
                print("Completed responses are saved; rerun the same command to resume.")
                return
            f.write(json.dumps({"id": row["id"], "response": result}, ensure_ascii=False) + "\n")
            f.flush()
            sent += 1
            if sent % 10 == 0 or sent == limit:
                print(f"Sent {sent}/{limit} this session", flush=True)
            if sent < limit:
                time.sleep(delay)


def response_text(body, provider="gemini"):
    if provider == "soclaas":
        choices = body.get("choices") or []
        if not choices or choices[0].get("finish_reason") != "stop":
            return ""
        content = choices[0].get("message", {}).get("content", "")
        return content if isinstance(content, str) else ""
    candidates = body.get("candidates") or []
    if not candidates or candidates[0].get("finishReason") != "STOP":
        return ""
    return "".join(part.get("text", "") for part in candidates[0].get("content", {}).get("parts", []))


def parse_teacher_answer(text):
    """Accept JSON alone or inside one complete Markdown JSON code fence."""
    text = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", text, re.IGNORECASE | re.DOTALL)
    return json.loads(fenced.group(1) if fenced else text)


def validate(row, answer):
    if not isinstance(answer, dict):
        return "invalid_json"
    reasoning = answer.get("reasoning")
    if not isinstance(reasoning, str) or not 20 <= len(reasoning.strip()) <= 900:
        return "reasoning_length"
    if answer.get("relation") != RELATION[row["gold"]]:
        return "wrong_relation"
    if answer.get("gold_justified") is not True:
        return "gold_not_justified_by_context"
    cited = answer.get("cited_evidence")
    if not isinstance(cited, list) or any(not isinstance(e, str) or not EID.fullmatch(e) for e in cited):
        return "invalid_citation_list"
    if len(cited) != len(set(cited)) or len(cited) > 5:
        return "duplicate_or_too_many_citations"
    shown = set(re.findall(r"(?m)^E\d+\b", row["prompt"]))
    mentioned = set(EID.findall(reasoning))
    # NEI may describe visible context that fails to establish the claim while
    # citing no supporting evidence. Its final FEVER evidence target stays [].
    if (not set(cited) <= shown or not mentioned <= shown or not set(cited) <= mentioned
            or (mentioned and not cited and row["gold"] != "NOT_ENOUGH_INFO")):
        return "citation_mismatch"
    gold_eids = set(json.loads(row["completion"])["evidence"])
    if row["gold"] != "NOT_ENOUGH_INFO" and not gold_eids <= mentioned:
        return "gold_evidence_not_explained"
    return None


def collect(args):
    out = Path(args.out)
    result_dir = Path(args.result_dir) if args.result_dir else out
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    provider = manifest.get("provider", "gemini")
    source = read_jsonl(out / "source_rows.jsonl")
    by_id = {str(row["id"]): row for row in source}
    received, accepted, rejected = set(), [], []
    usage = collections.Counter()
    for result in read_jsonl(out / "responses.jsonl"):
        cid = str(result.get("id"))
        if cid not in by_id or cid in received:
            raise ValueError(f"unexpected or duplicate response ID: {cid}")
        received.add(cid)
        row = by_id[cid]
        body = result.get("response") or {}
        if provider == "soclaas":
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                usage[key] += (body.get("usage") or {}).get(key, 0)
        else:
            for key in ("promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount", "totalTokenCount"):
                usage[key] += (body.get("usageMetadata") or {}).get(key, 0)
        try:
            answer = parse_teacher_answer(response_text(body, provider))
            reason = validate(row, answer)
        except (ValueError, TypeError, KeyError):
            answer, reason = None, "invalid_json_or_incomplete"
        if reason:
            rejected.append({"id": row["id"], "reason": reason, "response": result})
            continue
        gold = json.loads(row["completion"])
        completion = {"reasoning": answer["reasoning"].strip(), "verdict": gold["verdict"], "evidence": gold["evidence"]}
        accepted.append({"id": row["id"], "gold": row["gold"], "prompt": row["prompt"],
                         "completion": json.dumps(completion, ensure_ascii=False)})
    if len(received) != len(source):
        raise ValueError(f"only {len(received)}/{len(source)} responses saved; finish `run` before `collect`")
    cot_sft_path = result_dir / "cot_sft_labeled_prompts.jsonl"
    if any((result_dir / name).exists() for name in
           ("cot_sft_labeled_prompts.jsonl", "rejected.jsonl", "summary.json")):
        raise FileExistsError("Collected output already exists; use a fresh directory")
    result_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(cot_sft_path, accepted)
    write_jsonl(result_dir / "rejected.jsonl", rejected)
    summary = {
        "source_run": str(out), "validation_rule": "nei_visible_context_refs_v2",
        "source_rows": len(source), "received": len(received), "accepted": len(accepted),
        "rejected": len(rejected), "rejection_reasons": dict(collections.Counter(x["reason"] for x in rejected)),
        "usage": dict(usage),
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Accepted {len(accepted)}/{len(source)}; rejected {len(rejected)}")
    print(f"CoT SFT data: {cot_sft_path}")
    print(f"Token usage: {dict(usage)}; details: {result_dir / 'summary.json'}")
    print("Automated checks cannot prove that every explanation is semantically faithful.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source", default=str(SOURCE))
    prep.add_argument("--out")
    prep.add_argument("--provider", choices=("soclaas", "gemini"), default="soclaas")
    prep.add_argument("--model", help="SoCLaaS: choose from `models`; Gemini default is gemini-3.8-flash")
    prep.add_argument("--base-url", default=SOCLAAS_URL)
    prep.add_argument("--per-class", type=int, default=100, help="0 uses all source rows")
    prep.add_argument("--seed", type=int, default=0)
    listing = sub.add_parser("models", help="list models available to your SoCLaaS key")
    listing.add_argument("--base-url", default=SOCLAAS_URL)
    sending = sub.add_parser("run")
    sending.add_argument("--out", required=True)
    sending.add_argument("--max-requests", type=int, default=0, help="0 sends all remaining rows")
    sending.add_argument("--delay", type=float, help="seconds between requests (default: SoCLaaS 1, Gemini 10)")
    collection = sub.add_parser("collect")
    collection.add_argument("--out", required=True)
    collection.add_argument("--result-dir", help="fresh directory for validated outputs; preserves an earlier collection")
    args = ap.parse_args()
    if args.command == "prepare" and args.provider == "gemini" and not args.model:
        args.model = MODEL
    {"prepare": prepare, "models": models, "run": run, "collect": collect}[args.command](args)


if __name__ == "__main__":
    main()
