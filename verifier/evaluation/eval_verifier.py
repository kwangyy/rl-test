"""Run the FEVER verifier and save claim-level evaluation results."""

import argparse

from verifier.model import BIG_MODEL, run, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="local", help='"local", "local:<hf id>", or an OpenRouter model id')
    parser.add_argument("--n", type=int, help="default 50; ignored when --claim-ids is supplied")
    parser.add_argument("--claim-ids", help="frozen FEVER dev ID manifest; evaluates every ID in order")
    parser.add_argument("--retrieval", choices=["bm25", "titles", "oracle"], default="bm25")
    parser.add_argument("--prompt-style", choices=["direct", "reasoning", "cot_chain", "cot_brief", "cot_sft", "cot_native"], default="direct")
    parser.add_argument("--max-new-tokens", type=int, help="override the prompt style's output-token cap")
    parser.add_argument("--adapter", help="LoRA checkpoint dir from an SFT run (local models only)")
    parser.add_argument("--output", help="write to this JSONL file instead of the default (resume by claim id if it exists)")
    parser.add_argument("--summarize", help="only print metrics for an existing results file")
    args = parser.parse_args()
    prices = {BIG_MODEL: (0.15, 0.47)}  # $/M tokens, OpenRouter 2026-09-19
    if args.summarize:
        summarize(args.summarize)
    else:
        n = args.n if args.n is not None or args.claim_ids else 50
        path = run(args.model, n, args.retrieval, adapter=args.adapter,
                   output=args.output, prompt_style=args.prompt_style,
                   max_new_tokens=args.max_new_tokens, claim_ids=args.claim_ids)
        summarize(path, *prices.get(args.model, (None, None)))


if __name__ == "__main__":
    main()
