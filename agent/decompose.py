"""Decomposer (AG-2): the big model splits a claim into sub-claims, interfaces.Decompose.

Temperature 0, so results are cached once per claim text and reused by every
policy and by RL-3's offline controller env:

    python -m agent.decompose --n 900      # writes verifier/data/decompose/fever_dev_<model>_n900.jsonl

Each cache row: id, claim, subclaims, model, tokens_in, tokens_out. A claim that
is already simple, or whose reply cannot be parsed, comes back as [claim].
"""

import argparse
import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from retrieval.build_bm25_index import DATA
from verifier.model import BIG_MODEL, SEED, sample_claims
from utils.fever_eval import manifest_tag, validate_resume_ids

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CACHE_DIR = os.path.join(DATA, "..", "decompose")

DECOMPOSE_V1 = """Split a factual claim into the smallest set of simpler sub-claims that are
all true exactly when the claim is true. Each sub-claim must stand on its own: repeat
the subject instead of using pronouns. A claim that states one fact stays as one.
Reply with a JSON list of strings and nothing else.

Claim: Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.
["Nikolaj Coster-Waldau worked with the Fox Broadcasting Company."]

Claim: Brad Wilk co-founded Rage Against the Machine with Tom Morello in 1991.
["Brad Wilk co-founded Rage Against the Machine.", "Tom Morello co-founded Rage Against the Machine.", "Rage Against the Machine was founded in 1991."]"""

# v2 (2026-10-17): v1 split claims whose parts depend on each other ("stars an actor
# that died in 1961" -> "That actor died in 1961"), and dropped negations and "only".
# See docs/agent-loop-notes.md section 4. The examples are not FEVER dev claims.
DECOMPOSE_V2 = """Split a factual claim into the smallest set of simpler sub-claims that are
all true exactly when the claim is true. Each sub-claim must be checkable on its own:
name every person, place and work explicitly, and never use a pronoun or a phrase such
as "that actor" or "the band" that points to another sub-claim. Keep the claim's exact
meaning, including negations, "only", numbers and dates.

Keep the claim as a single item when:
- it states one fact;
- one part identifies something that another part is about ("X stars an actor who was
  born in Sweden": no part can be checked without knowing which actor);
- splitting would change a negation, "only", "exclusively" or a count.

Reply with a JSON list of strings and nothing else.

Claim: Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.
["Nikolaj Coster-Waldau worked with the Fox Broadcasting Company."]

Claim: Brad Wilk co-founded Rage Against the Machine with Tom Morello in 1991.
["Brad Wilk co-founded Rage Against the Machine.", "Tom Morello co-founded Rage Against the Machine.", "Rage Against the Machine was founded in 1991."]

Claim: Casablanca stars an actress who was born in Sweden.
["Casablanca stars an actress who was born in Sweden."]

Claim: Titanic was not directed by Steven Spielberg.
["Titanic was not directed by Steven Spielberg."]"""

PROMPTS = {"v1": DECOMPOSE_V1, "v2": DECOMPOSE_V2}
DEFAULT_PROMPT = "v2"
DECOMPOSE_SYSTEM = PROMPTS[DEFAULT_PROMPT]


def load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(ROOT, ".env"))
    except ImportError:
        pass


class BigModel:
    """Text-in, text-out calls to the big model on OpenRouter."""

    def __init__(self, name=BIG_MODEL, max_tokens=200):
        from openai import OpenAI
        load_env()
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1",
                             api_key=os.environ["OPENROUTER_API_KEY"], max_retries=3)
        self.name = name
        # Claude sometimes reasons in text before its JSON; 200 tokens cut that off (smoke run)
        self.max_tokens = max(max_tokens, 1000) if name.startswith("anthropic/") else max_tokens

    def __call__(self, system, user):
        # Claude 5.x on OpenRouter cannot disable reasoning and takes no temperature;
        # "minimal" effort is the closest to off (0 reasoning tokens on a test call).
        claude = self.name.startswith("anthropic/")
        r = self.client.chat.completions.create(
            model=self.name, max_tokens=self.max_tokens,
            **({} if claude else {"temperature": 0, "seed": SEED}),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            extra_body={"reasoning": {"effort": "minimal"} if claude else {"enabled": False}},
        )
        return {"text": r.choices[0].message.content or "",
                "in_tokens": r.usage.prompt_tokens, "out_tokens": r.usage.completion_tokens}


class ClaudeModel:
    """BigModel's interface on the Anthropic API (ANTHROPIC_API_KEY in .env), for claude-* names.

    Thinking is off ("between_tools"), like BigModel's reasoning: enabled False. Claude 5.x
    models take no temperature or seed, so repeated calls are not guaranteed identical."""

    def __init__(self, name, max_tokens=200):
        import anthropic
        load_env()
        self.client = anthropic.Anthropic(max_retries=8)
        self.name = name
        self.max_tokens = max_tokens

    def __call__(self, system, user):
        r = self.client.messages.create(
            model=self.name, max_tokens=self.max_tokens, system=system,
            thinking={"type": "between_tools"},
            messages=[{"role": "user", "content": user}],
        )
        return {"text": "".join(b.text for b in r.content if b.type == "text"),
                "in_tokens": r.usage.input_tokens, "out_tokens": r.usage.output_tokens}


def big_model(name=BIG_MODEL, max_tokens=200):
    """ClaudeModel for claude-* names, otherwise BigModel on OpenRouter."""
    return (ClaudeModel if name.startswith("claude-") else BigModel)(name, max_tokens)


def first_json(text, kind):
    m = re.search(r"\{.*\}" if kind is dict else r"\[.*\]", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, kind) else None


def cache_path(model, n, dataset="fever", split="dev", claim_ids=None, prompt=DEFAULT_PROMPT):
    """v1 keeps its historical filename; later prompts add _prompt<v> so caches never mix."""
    suffix = "" if prompt == "v1" else f"_prompt{prompt}"
    suffix += f"_{manifest_tag(claim_ids)}" if claim_ids else ""
    return os.path.join(CACHE_DIR, f"{dataset}_{split}_{model.replace('/', '_')}_n{n}{suffix}.jsonl")


class Decomposer:
    """interfaces.Decompose. run() also returns the token cost of the original call, also when
    served from the cache, so cost accounting does not depend on whether a cache was used."""

    def __init__(self, big=None, cache=None, prompt=DEFAULT_PROMPT):
        self.big = big
        self.name = big.name if big else BIG_MODEL
        self.prompt = prompt
        self.system = PROMPTS[prompt]
        self.cache = {}
        if cache and os.path.exists(cache):
            with open(cache, encoding="utf-8") as f:
                self.cache = {r["claim"]: {k: r[k] for k in ("subclaims", "tokens_in", "tokens_out")}
                              for r in map(json.loads, f)}

    def __call__(self, claim):
        return self.run(claim)["subclaims"]

    def run(self, claim):
        if claim in self.cache:
            return {**self.cache[claim], "cached": True}
        if self.big is None:
            self.big = big_model(self.name)
        g = self.big(self.system, f"Claim: {claim}")
        subs = [s.strip() for s in (first_json(g["text"], list) or []) if isinstance(s, str) and s.strip()]
        subs = list(dict.fromkeys(subs))
        if len(subs) < 2:
            subs = [claim]
        self.cache[claim] = {"subclaims": subs, "tokens_in": g["in_tokens"], "tokens_out": g["out_tokens"]}
        return {**self.cache[claim], "cached": False}

    def batch(self, claims, workers=8):
        with ThreadPoolExecutor(workers) as pool:
            return list(pool.map(self.run, claims))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BIG_MODEL)
    ap.add_argument("--n", type=int, help="default 900; ignored when --claim-ids is supplied")
    ap.add_argument("--claim-ids", help="frozen FEVER dev ID manifest")
    ap.add_argument("--prompt", choices=sorted(PROMPTS), default=DEFAULT_PROMPT, help="decomposer prompt version")
    a = ap.parse_args()

    claims = sample_claims(a.n if a.n is not None or a.claim_ids else 900, SEED,
                           ids_file=a.claim_ids)
    a.n = len(claims)
    path = cache_path(a.model, a.n, claim_ids=a.claim_ids, prompt=a.prompt)
    os.makedirs(CACHE_DIR, exist_ok=True)
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            done_ids = [json.loads(line)["id"] for line in f]
            validate_resume_ids(done_ids, [c["id"] for c in claims])
            done = set(done_ids)
    todo = [c for c in claims if c["id"] not in done]
    print(f"{len(done)} done, {len(todo)} to go -> {path}")
    d = Decomposer(BigModel(a.model), prompt=a.prompt)
    for i in range(0, len(todo), 50):
        chunk = todo[i:i + 50]
        with open(path, "a", encoding="utf-8") as f:
            for c, r in zip(chunk, d.batch([c["claim"] for c in chunk])):
                f.write(json.dumps({"id": c["id"], "claim": c["claim"], "subclaims": r["subclaims"],
                                    "model": d.name, "prompt": a.prompt, "tokens_in": r["tokens_in"],
                                    "tokens_out": r["tokens_out"]}, ensure_ascii=False) + "\n")
        print(f"  {i + len(chunk)}/{len(todo)}", flush=True)

    with open(path, encoding="utf-8") as f:
        rows = list(map(json.loads, f))
    label = {c["id"]: c["label"] for c in claims}
    sizes = Counter(len(r["subclaims"]) for r in rows)
    print(f"\n{len(rows)} claims; sub-claim counts {dict(sorted(sizes.items()))}")
    for L in ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO"):
        sub = [r for r in rows if label.get(r["id"]) == L]
        print(f"  split {L:16s} {sum(len(r['subclaims']) > 1 for r in sub) / max(len(sub), 1):.3f}  (n={len(sub)})")
    print(f"  tokens per claim {sum(r['tokens_in'] for r in rows) / len(rows):.0f} in, "
          f"{sum(r['tokens_out'] for r in rows) / len(rows):.0f} out")
