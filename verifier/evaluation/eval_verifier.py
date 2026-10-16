"""Run the FEVER verifier and save claim-level evaluation results."""

import argparse

from verifier.model import (BIG_MODEL, K_PAGES,
                            MAX_SENTS_PER_PAGE, resolve_context_mode, run, summarize)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="local", help='"local", "local:<hf id>", or an OpenRouter model id')
    parser.add_argument("--n", type=int, help="default 50; ignored when --claim-ids is supplied")
    parser.add_argument("--claim-ids", help="frozen FEVER dev ID manifest; evaluates every ID in order")
    parser.add_argument("--context-mode", "--retrieval", dest="context_mode", metavar="MODE",
                        default="bm25",
                        help="page_bm25 (default), page_titles, oracle, sentence_bm25 or sentence_hybrid; "
                             "old bm25/titles/hybrid aliases still work")
    parser.add_argument("--search-k", type=int, default=K_PAGES * MAX_SENTS_PER_PAGE,
                        help="returned sentence budget for sentence_bm25/sentence_hybrid")
    parser.add_argument("--candidate-pages", type=int, default=K_PAGES,
                        help="candidate-page budget for sentence_bm25/sentence_hybrid")
    parser.add_argument("--prompt-style", choices=["direct", "reasoning", "cot_chain", "cot_brief", "cot_sft", "cot_native"], default="direct")
    parser.add_argument("--max-new-tokens", type=int, help="override the prompt style's output-token cap")
    parser.add_argument("--adapter", help="LoRA checkpoint dir from an SFT run (local models only)")
    parser.add_argument("--output", help="write to this JSONL file instead of the default (resume by claim id if it exists)")
    parser.add_argument("--summarize", help="only print metrics for an existing results file")
    args = parser.parse_args()
    try:
        mode = resolve_context_mode(args.context_mode)
    except ValueError as exc:
        parser.error(str(exc))
    if args.search_k < 1 or args.candidate_pages < 1:
        parser.error("search-k and candidate-pages must be positive")
    if mode not in ("sentence_bm25", "sentence_hybrid") and (args.search_k != K_PAGES * MAX_SENTS_PER_PAGE or args.candidate_pages != K_PAGES):
        parser.error("search-k/candidate-pages overrides require sentence_bm25 or sentence_hybrid")
    prices = {BIG_MODEL: (0.15, 0.47)}  # $/M tokens, OpenRouter 2026-09-19
    if args.summarize:
        summarize(args.summarize)
    else:
        n = args.n if args.n is not None or args.claim_ids else 50
        path = run(args.model, n, args.context_mode, adapter=args.adapter,
                   output=args.output, prompt_style=args.prompt_style,
                   max_new_tokens=args.max_new_tokens, claim_ids=args.claim_ids,
                   search_k=args.search_k, candidate_pages=args.candidate_pages)
        summarize(path, *prices.get(args.model, (None, None)))


if __name__ == "__main__":
    main()
