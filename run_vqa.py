"""CLI entry point. Drop-in replacement for the old monolithic vqa.py — same flags, same outputs.

Example (local vLLM on GPU 0):
    python run_vqa.py --dataset meva \
        --data_dir /nas/neurosymbolic/multi-cam-dataset-organized/meva \
        --model OpenGVLab/InternVL3-8B --gpu 0 \
        --strategy uniform --num_frames 8
"""
import argparse
import os

import runner
from dataloaders.qa_json import _DEFAULT_CATEGORIES, load_datasets, shard_datasets
from models.clients import OPENAI_MODELS, TextLLM, VLLMClient
from models.cost import METER, estimate_cost, load_prices, resolve_openai_key

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(_DEFAULT_CATEGORIES), default="meva")
    parser.add_argument("--data_dir", required=True, help="directory holding qa_<category>.json")
    parser.add_argument("--strategy", choices=["uniform", "stitched", "decentralized"], default="uniform")
    parser.add_argument("--gpu", type=int, default=0, help="sets api_base to localhost:800{gpu} for local vLLM")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--num_frames", type=int, default=8)
    # decentralized-harness options (ignored by uniform/stitched)
    parser.add_argument("--aggregator_model", type=str, default="same",
                        help="text-only LLM for the aggregation pass. 'same' = reuse the served VLM "
                             "(local, no cloud cost); an OpenAI id (e.g. gpt-5.2) uses the OpenAI API; "
                             "any other id is assumed served on the local vLLM endpoint.")
    parser.add_argument("--frame_budget", choices=["per_camera", "total"], default="per_camera",
                        help="per_camera: each camera gets --num_frames; total: split --num_frames across cameras")
    parser.add_argument("--summary_max_tokens", type=int, default=512,
                        help="max tokens for each per-camera summary")
    parser.add_argument("--max_usd", type=float, default=10.0,
                        help="hard cap: halt the run before exceeding this USD (OpenAI models only; local=free)")
    parser.add_argument("--prices", type=str, default=os.path.join(_REPO_ROOT, "configs", "model_prices.yaml"),
                        help="YAML price table (USD per 1M tokens) for cost accounting")
    parser.add_argument("--passes", type=int, default=1, help="independent passes for error bars")
    parser.add_argument("--seeds", type=str, default="1", help="comma-separated seeds, one per pass")
    parser.add_argument("--temperature", type=float, default=None,
                        help="sampling temperature applied to all passes (None = greedy per-task defaults)")
    parser.add_argument("--log_reasoning", action="store_true",
                        help="decentralized: aggregator justifies its choice (stored as answer_reasoning) "
                             "and GT reasoning is logged, for greybox debugging. Off by default (clean parsing).")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="base output dir (default: <repo>/results). Set a distinct dir per run/shard "
                             "so parallel processes don't overwrite each other's per-category result files.")
    parser.add_argument("--shard", type=str, default=None,
                        help="i:N — keep only questions whose index mod N == i within each category (sorted "
                             "keys). Enables running N parallel processes over disjoint question slices.")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()] or [1]

    if args.out_dir:
        runner.set_out_base(args.out_dir)
        print(f"[OUT] writing results under {os.path.abspath(args.out_dir)}")

    METER.configure(prices=load_prices(args.prices), max_usd=args.max_usd)

    datasets_by_category = load_datasets(args.dataset, args.data_dir)
    if args.shard:
        datasets_by_category = shard_datasets(datasets_by_category, args.shard)
    total = sum(len(d) for d in datasets_by_category.values())
    print(f"\nLoaded {total} questions across {len(datasets_by_category)} categories ({args.dataset})")

    est, n = estimate_cost(datasets_by_category, args.strategy, args.num_frames, args.model, METER.prices)
    if est is not None:
        print(f"[COST] rough pre-run estimate for {args.model}: ~${est:.2f} over {n} questions "
              f"(hard cap ${args.max_usd:.2f}). This is approximate; the live cap is the real guard.")

    vllm_client = VLLMClient(
        api_base=f"http://localhost:800{args.gpu}/v1",
        model=args.model, dataset=args.dataset, strategy=args.strategy, num_frames=args.num_frames,
    )
    aggregator = None
    if args.strategy == "decentralized":
        api_base = f"http://localhost:800{args.gpu}/v1"
        agg_model = args.model if args.aggregator_model == "same" else args.aggregator_model
        if agg_model in OPENAI_MODELS:
            aggregator = TextLLM(agg_model, api_key=resolve_openai_key())
        else:
            aggregator = TextLLM(agg_model, api_key="EMPTY", base_url=api_base)
        print(f"Decentralized: per-camera summaries via {args.model}, "
              f"text aggregation via {agg_model} "
              f"({'OpenAI' if agg_model in OPENAI_MODELS else 'local vLLM'}, frame_budget={args.frame_budget})")
    runner.run_experiment(datasets_by_category, vllm_client, vllm_client.model, args.num_frames, args.dataset,
                          args.strategy, frame_budget=args.frame_budget, aggregator=aggregator,
                          summary_max_tokens=args.summary_max_tokens,
                          passes=args.passes, seeds=seeds, temperature=args.temperature,
                          log_reasoning=args.log_reasoning)

    print(f"\n[USAGE] calls={METER.calls} prompt_tokens={METER.prompt_tokens} "
          f"completion_tokens={METER.completion_tokens} cost=${METER.usd:.4f} (cap ${args.max_usd:.2f})")


if __name__ == "__main__":
    main()
