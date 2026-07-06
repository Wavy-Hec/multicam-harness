# multicam-harness

Shared eval code for the multi-camera harness experiments: how does the *packaging* of
multi-camera video for a frozen VLM (the "harness") change QA accuracy, at equal frame budget?

Three harnesses, everything else held fixed (questions, model call, scoring):

| Strategy | What the model sees |
|---|---|
| `uniform` | per-camera uniformly-sampled frames, shown sequentially (CVBench-native) |
| `stitched` | time-synchronized frames stitched into labeled grid montages (centralized) |
| `decentralized` | per-camera query-conditioned **text** summaries → text-only LLM aggregation |

## Layout

```
run_vqa.py        CLI entry point (same flags as the old monolithic vqa.py)
runner.py         experiment loop: iterate questions → harness → score → write results
dataloaders/      qa_json.py (CrossView-style qa_<category>.json), video.py (ffmpeg frames)
harnesses/        uniform.py, stitched.py, decentralized.py — the independent variable
models/           clients.py (VLLMClient / TextLLM), cost.py (token+USD metering, budget cap)
evaluation/       scoring.py (MC / counting partial credit / ROUGE), llm_judge.py (LLM-as-judge)
plotting/         plot_results.py — grouped per-category bars across harnesses
configs/          datasets.yaml (categories per dataset), model_prices.yaml (USD per 1M tokens)
scripts/          serve_vllm.sh
```

Run everything **from the repo root** (imports are top-level packages).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# ffmpeg/ffprobe must be on PATH.
# OpenAI models/judge: put OPENAI_API_KEY=... (or OPENAI_API_KEY1=) in .env at repo root.
```

## Run

```bash
# 1. serve a VLM (skip for OpenAI models)
bash scripts/serve_vllm.sh Qwen/Qwen2.5-VL-7B-Instruct 0 8000   # GPU 0 → port 8000

# 2. evaluate (port is derived from --gpu: localhost:800<gpu>)
python run_vqa.py --dataset meva \
    --data_dir /nas/neurosymbolic/multi-cam-dataset-organized/meva \
    --model Qwen/Qwen2.5-VL-7B-Instruct --gpu 0 \
    --strategy uniform --num_frames 8

# swap --strategy stitched | decentralized for the other harnesses
# decentralized extras: --aggregator_model gpt-5.2 --frame_budget total --log_reasoning
# error bars: --passes 4 --seeds 1,2,3,4 --temperature 0.7
# parallel shards: --shard i:N --out_dir results/run_name_shard_i
```

Results land in `results/<model>/<dataset>/<strategy>/` (per-category JSONs + `results.txt`
tables); raw request logs in `logs/`. Both are gitignored — outputs never go in git.

## Summarization scoring

ROUGE is computed inline for continuity, but the headline metric is the LLM judge
(G-Eval-style, see `evaluation/llm_judge.py` docstring):

```bash
python -m evaluation.llm_judge results/<...>/<model>_<ds>_<strat>_summarization.json
# add --claims for the claim-level (interpretable) variant
```

## Plotting

```bash
python -m plotting.plot_results results/ --out figures/
```

## Ground rules

- **Inference-only.** No training; the harness is the variable.
- **Equal budget.** Compare harnesses at the same frame budget (watch token parity too).
- **One change at a time.** Only visual packaging differs between harnesses.
- **Inspect by hand.** Watch the clips before trusting a number.
- **Never commit** data, video, weights, or run outputs (see `.gitignore`).
- Single greedy passes are preliminary — no error bars until `--passes 4`.

## Data

QA JSONs (per category: best_camera, counting, event_ordering, spatial, summarization,
temporal): `/nas/neurosymbolic/multi-cam-dataset-organized/meva/` on the swarmcluster.
MEVA raw video: `/nas/mars/dataset/MEVA`. If you're on a different machine, copy the QA
JSONs + referenced clips locally and pass your local `--data_dir`.
