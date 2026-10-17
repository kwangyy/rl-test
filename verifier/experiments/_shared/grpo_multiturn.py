"""Multi-turn GRPO spike: the verifier calls search() itself, then answers.

Week-1 test for AG-2. Same model, reward and output JSON as grpo_verifier.py;
only the loop differs. The claim comes in with no evidence, the model calls the
`search` tool (TRL's environment_factory: one FeverSearchEnv per rollout), reads
the numbered sentences it gets back, and ends with the untrained baseline JSON. Reward is
reward.total on the final answer, with n_ops = number of searches.

Tool results are masked out of the loss by TRL (tool_mask, multiplied into the
loss mask in GRPOTrainer), so the model is trained only on what it wrote: its
queries and its verdict. Tool tokens still count toward max_completion_length.

    python grpo_multiturn.py                              # smoke: 16 claims, 3 steps
    python grpo_multiturn.py --num-generations 4          # if 8 does not fit in 16 GB
    python grpo_multiturn.py --init sft                   # start from the SFT verifier
    python grpo_multiturn.py --max-calls 3 --max-completion-length 3072   # more searches, more memory

Pass --method bm25 or hybrid for complete ranked sentence search, with
--candidate-pages and --search-k. Ranked runs require a new --config-name and
--run-name. Omitting --method preserves the historical three-page, first-ten-
sentences policy used by the existing experiment wrapper.

What the spike has to answer (write the numbers into docs/grpo-results.md):
does it run, do the rollouts call search and end with valid JSON
(rollouts.jsonl), peak GPU GB and seconds per step (meta.json).
"""

import argparse
import json
import os
import random
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core import reward as R
from utils.experiment_layout import artifacts_dir
from utils.grpo_common import DEFAULT_SFT, FORMAT_FAIL, VERDICT
from verifier.model import FEVER_LABEL, LOCAL_MODEL, MAX_CITED, SEED, gold_sets, parse, sample_claims


SYSTEM = """You are a fact-checker. You get a claim. Use the search tool to find Wikipedia evidence, then decide whether the evidence SUPPORTS the claim, REFUTES it, or gives NOT_ENOUGH_INFO to decide. Use only evidence the tool returned.

search returns numbered sentences (E1, E2, ...); the numbering continues across searches. Search at most {max_calls} times.

When you are done searching, reply with only a JSON object, nothing else:
{{"verdict": "SUPPORTED" | "REFUTED" | "NOT_ENOUGH_INFO", "evidence": ["E3", ...], "confidence": <number 0 to 1>}}

- evidence: ids of the sentences that justify the verdict, at most 5. Use [] for NOT_ENOUGH_INFO.
- confidence: the probability that your verdict is correct."""

_RETRIEVER = None


def retriever():
    """One FeverRetriever per process; every env instance shares it (the index is ~2.3 GB)."""
    global _RETRIEVER
    if _RETRIEVER is None:
        from retrieval.fever_retrieval import FeverRetriever
        _RETRIEVER = FeverRetriever(method=FeverSearchEnv.method or "bm25",
                                    candidate_pages=FeverSearchEnv.k_pages)
    return _RETRIEVER


class FeverSearchEnv:
    """One rollout's state: the sentences shown so far and how many searches ran.

    Public methods other than reset are the model's tools; TRL builds the tool
    schema from the signature and docstring.
    """

    method = None  # Omitted --method preserves historical page-prefix search.
    k_pages = 3
    search_k = 30
    sents_per_page = 10

    def __init__(self):
        self.reset()

    def reset(self, **row):
        self.shown = {}  # "E3" -> "page:line"
        self.n_search = 0
        return None

    def search(self, query: str) -> str:
        """Search Wikipedia and return evidence under the configured retrieval policy.

        Args:
            query: What to look for, for example the names and facts in the claim.

        Returns:
            Numbered evidence sentences, one per line, as "E<n> [Page title] sentence".
        """
        self.n_search += 1
        r = retriever()
        seen = set(self.shown.values())
        lines = []
        if self.method is None:
            sentences = ((page, line, text)
                         for page in r.search_with_titles([query], k=self.k_pages)[0]
                         for line, text in r.sentences(page)[:self.sents_per_page])
        else:
            sentences = ((h["page"], h["line"], h["text"])
                         for h in r.search([query], k=self.search_k)[0])
        for page, line, text in sentences:
            key = f"{page}:{line}"
            if key in seen:
                continue
            seen.add(key)
            eid = f"E{len(self.shown) + 1}"
            self.shown[eid] = key
            lines.append(f"{eid} [{page.replace('_', ' ')}] {text}")
        return "\n".join(lines) or "No new results."


def load_rows(n, max_calls):
    """n balanced FEVER train claims, no evidence in the prompt."""
    system = SYSTEM.format(max_calls=max_calls)
    rows = []
    for c in sample_claims(n, SEED, split="train"):
        rows.append({
            "prompt": [{"role": "system", "content": system}, {"role": "user", "content": f"Claim: {c['claim']}"}],
            "claim": c["claim"],
            "gold": VERDICT[FEVER_LABEL[c["label"]]],
            "gold_ids": sorted({f"{p}:{l}" for s in gold_sets(c) for p, l in s}),
        })
    return rows


def score_one(messages, env, gold, gold_ids, cfg, rng):
    """(reward, parts) for one rollout; parts is None if it never gave a valid final answer."""
    last = messages[-1] if messages else {}
    text = last.get("content") or ""
    obj, ok = parse(text) if not last.get("tool_calls") else ({}, {"json": False})
    if not all(ok.values()):
        return FORMAT_FAIL, None
    cited = [env.shown[e] for e in obj["evidence"][:MAX_CITED] if e in env.shown]
    return R.total(gold, VERDICT[obj["verdict"]], float(obj["confidence"]), gold_ids, cited,
                   n_ops=env.n_search, format_ok=True, cfg=cfg, rng=rng)


def make_reward_funcs(cfg, seed, log_path):
    """The training reward, plus zero-weight copies of its parts so TRL logs them.
    Every rollout is appended to log_path for inspection."""
    rng = random.Random(seed)
    cache = {}

    def scored(completions, environments, gold, gold_ids, claim):
        key = id(completions)
        if key not in cache:
            cache.clear()
            res = [score_one(m, e, g, ids, cfg, rng)
                   for m, e, g, ids in zip(completions, environments, gold, gold_ids)]
            cache[key] = res
            with open(log_path, "a", encoding="utf-8") as f:
                for m, e, g, c, (r, p) in zip(completions, environments, gold, claim, res):
                    f.write(json.dumps({"claim": c, "gold": g, "reward": r, "parts": p, "n_search": e.n_search,
                                        "shown": e.shown, "messages": m}, ensure_ascii=False, default=str) + "\n")
        return cache[key]

    def verifier_reward(completions, environments, gold, gold_ids, claim, **_):
        return [r for r, _ in scored(completions, environments, gold, gold_ids, claim)]

    def part(name):
        def f(completions, environments, gold, gold_ids, claim, **_):
            return [p.get(name, 0.0) if p else 0.0 for _, p in scored(completions, environments, gold, gold_ids, claim)]
        f.__name__ = name
        return f

    def format_ok(completions, environments, gold, gold_ids, claim, **_):
        return [0.0 if p is None else 1.0 for _, p in scored(completions, environments, gold, gold_ids, claim)]

    def searches(completions, environments, **_):
        return [float(e.n_search) for e in environments]

    funcs = [verifier_reward, part("correctness"), part("calibration"), part("evidence"), format_ok, searches]
    return funcs, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def train(a, out):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    ds = Dataset.from_list(load_rows(a.n_train, a.max_calls))
    print(f"{len(ds)} claims, init={a.init}, beta={a.beta}")
    retriever()  # load the index before the first rollout, not inside it

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda")
    if a.init == "sft":
        model = PeftModel.from_pretrained(model, a.sft).merge_and_unload()
    with open(os.path.join(a.sft, "adapter_config.json"), encoding="utf-8") as f:
        sc = json.load(f)  # same LoRA shape as SFT and single-turn GRPO
    peft_cfg = LoraConfig(r=sc["r"], lora_alpha=sc["lora_alpha"], lora_dropout=sc["lora_dropout"],
                          target_modules=sc["target_modules"], task_type="CAUSAL_LM")

    funcs, weights = make_reward_funcs(R.ABLATIONS[a.reward], a.seed,
                                      os.path.join(os.path.dirname(out), "rollouts.jsonl"))
    cfg = GRPOConfig(
        output_dir=out, seed=a.seed,
        num_generations=a.num_generations,
        per_device_train_batch_size=a.num_generations,  # one prompt's group per micro-batch
        gradient_accumulation_steps=a.prompts_per_step,
        max_completion_length=a.max_completion_length,  # includes the (masked) tool results
        max_tool_calling_iterations=a.max_calls,
        temperature=1.0, beta=a.beta,
        chat_template_kwargs={"enable_thinking": False},
        reward_weights=weights,
        learning_rate=a.lr, lr_scheduler_type="constant_with_warmup", warmup_steps=min(10, a.max_steps) if a.max_steps else 10,
        num_train_epochs=1, max_steps=a.max_steps if a.max_steps else -1,
        bf16=True, gradient_checkpointing=True,
        logging_steps=1, log_completions=False,
        save_strategy="no", report_to="none",
    )
    trainer = GRPOTrainer(model=model, args=cfg, train_dataset=ds, reward_funcs=funcs,
                          processing_class=tok, peft_config=peft_cfg, environment_factory=FeverSearchEnv)
    t0 = time.time()
    trainer.train()
    secs = time.time() - t0
    trainer.save_model(out)
    log_dir = os.path.join(os.path.dirname(out), "logs")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "train_log.json"), "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=1)
    return secs, torch.cuda.max_memory_allocated() / 2**30, trainer.state.global_step


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=LOCAL_MODEL)
    ap.add_argument("--sft", default=DEFAULT_SFT, help="SFT adapter: LoRA shape always, weights with --init sft")
    ap.add_argument("--init", default="base", choices=["base", "sft"],
                    help="base: the instruct model, which already calls tools; sft: merged SFT verifier "
                         "(trained single-turn, may have lost tool calling)")
    ap.add_argument("--reward", default="full", choices=sorted(R.ABLATIONS))
    ap.add_argument("--n-train", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=3)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--prompts-per-step", type=int, default=2)
    ap.add_argument("--max-calls", type=int, default=2,
                    help="search budget per claim; each result is ~860 tokens, so 3 calls need --max-completion-length 3072")
    ap.add_argument("--max-completion-length", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tag", default="")
    ap.add_argument("--run-name", default=None, help="repeat label under verifier/experiments/<configuration>/runs/")
    ap.add_argument("--output-root", default=None, help="experiments directory for run artifacts (default: verifier/experiments)")
    ap.add_argument("--config-name", default=None, help="directory name for this training configuration")
    ap.add_argument("--method", choices=("bm25", "hybrid"),
                    help="enable ranked sentence search; omit for the historical page-prefix policy")
    ap.add_argument("--search-k", type=int, default=30, help="returned sentence budget for ranked search")
    ap.add_argument("--candidate-pages", type=int, default=3, help="candidate-page budget for ranked search")
    a = ap.parse_args()
    if a.search_k < 1 or a.candidate_pages < 1:
        ap.error("search-k and candidate-pages must be positive")
    if a.method is None and (a.search_k != 30 or a.candidate_pages != 3):
        ap.error("search-k/candidate-pages overrides require --method bm25 or hybrid")
    if a.method and (not a.config_name or not a.run_name):
        ap.error("ranked search requires explicit --config-name and --run-name")
    FeverSearchEnv.method = a.method
    FeverSearchEnv.k_pages, FeverSearchEnv.search_k = a.candidate_pages, a.search_k

    name = f"{a.model.replace('/', '_')}_mt_{a.init}_{a.reward}_n{a.n_train}" + (f"_{a.tag}" if a.tag else "")
    artifacts = artifacts_dir("grpo_mt", name, a.seed, a.run_name, a.output_root, a.config_name)
    if artifacts.exists():
        raise SystemExit(f"run already exists: {artifacts}; use --run-name for a new run")
    out = artifacts / "checkpoints"
    out.mkdir(parents=True)
    secs, peak_gb, steps = train(a, str(out))
    meta = {**vars(a), "steps": steps, "train_seconds": round(secs),
            "seconds_per_step": round(secs / max(steps, 1), 1), "peak_gpu_gb": round(peak_gb, 2),
            "k_pages": FeverSearchEnv.k_pages,
            "search_policy": "ranked" if a.method else "page_prefix",
            **({"sents_per_page": FeverSearchEnv.sents_per_page} if a.method is None else {})}
    with open(artifacts / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))
    print(f"\nrollouts: {artifacts / 'rollouts.jsonl'}")
