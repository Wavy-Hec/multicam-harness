# multicam-harness

Shared eval code for the multi-camera harness experiments: how does the **packaging** of
multi-camera video for a frozen VLM (the *harness*) change QA accuracy, at an equal frame
budget? This is the in-process HF-transformers sibling of the vLLM-serving variant of the
harness — models load in-process via `run_vqa.py` (`scripts/serve_vllm.sh` is a standalone
helper for serving a VLM externally; nothing in this pipeline uses it).

## Strategies

| Harness | What the model sees | Calls/question |
|---|---|---|
| `uniform` | All clips' frames in one sequential prompt; the frame budget is split across clips by duration (`temporal_weighted`) or evenly (`--weighting even` — rows are recorded as `temporal_even`); `cvbench_native` = the original CVBench packaging | 1 |
| `stitched` | Time-synchronized frames stitched into labeled grid montages (centralized) | 1 |
| `decentralized` | Per-stream query-conditioned text descriptions → a text-only aggregation call; `--stream-kind camera\|video` | k+1 |
| `clip_select` | The budget is spent on question-relevant clips: cached summaries → an LLM router (`summary_select_*`), or CLIP/SigLIP thumbnail scoring (`clip_select[_<scorer>][_opt]_top<m>`) | 1–2 |
| `frame_select` | One shared budget of the most question-relevant frames chosen **globally** across every clip, shown grouped by source clip in temporal order (`frame_select[_<scorer>][_opt]`) | 1 |
| option-union (`_optu`) | A frame (`frame_select[_<scorer>]_optu`) or clip (`clip_select[_<scorer>\|_viclip]_optu`) is kept when **any answer option** would retrieve it; kept sets are unioned, never backfilled — showing fewer, more relevant items is the lever under test | 1 |
| `query_search` | The backend first writes short visual search phrases from Question+Options (one text-only call), then CLIP/SigLIP retrieves the top-budget frames matching any phrase (`query_search[_<scorer>]`) | 2 |
| `segment_select` | Each clip is split into equal-time segments; the most relevant segments per clip are kept, their frames pooled, near-duplicates removed question-wide, and the unique set thinned evenly in time to the budget (`segment_select[_<scorer>][_opt]`) | 1 |
| `blind` | The identical prompt scaffold with **zero** visual items — the text-prior floor every sighted arm must clear | 1 |
| `single_view<i>` | Only view *i*, with the scaffold and the view's true marker unchanged; records with fewer than *i* views are skipped | 1 |

Selection-arm knobs worth knowing before a run:

- `--budget` is the TOTAL frames per question. Omitted: the legacy selection arms
  (`clip_select*`, `frame_select`, `temporal_weighted`, `summary_select_*`) keep their
  historic 64, while `_optu`/`query_search`/`segment_select` default to **matched**
  (`nframes × K`, the sequential arm's budget). Explicit `--budget 0` also means matched
  and is legal only for those newer arms — the legacy arms refuse it at submit time.
- Scorer tags: `siglip`, `siglip2` (default scorer: `--clip-model
  openai/clip-vit-base-patch32`); `viclip` only for `clip_select_*_optu` (video-native,
  loaded from `VICLIP_DIR`).
- Spelling trap: the option-guided variant is `_opt` (`segment_select_opt`,
  `frame_select_siglip_opt`); `_optu` is the option-union family. `segment_select`
  rejects an `_optu` tag at submit time with a hint.
- `segment_select --dedup-tau` lives in (0, 1]: **1.0 disables dedup, 0 is invalid** —
  deliberately NOT the `--sel-tau` "0 = off" convention. Static-camera footage collapses
  at the 0.95 default; calibrate the tau or run 1.0 there, and read `n_unique` /
  `dedup_dropped` in `frame_alloc` before trusting a leg.

All arms accept `--total-frames`, which holds the **total** frame count per question
fixed (split evenly across its clips) instead of a flat `--nframes` per clip — use it
whenever two harnesses are being compared, since equal frames per clip is not an equal
budget when clip counts differ.

Multi-view **still-image** records (`image_1..image_N`) are supported alongside video
records by the uniform/stitched/decentralized/blind/single_view arms; their prompts and
montage labels say "View i" to match the question text. The selection arms
(`clip_select*`, `frame_select*`, `segment_select`, `query_search`) are video-only and
fail loudly on image records.

## Layout

```
run_vqa.py        # CLI entry: subset × methods × backends × passes → results JSONL
runner.py         # run loop: sharding, resume keys, per-pass seeding
dataloaders/      # qa_json.py (record → messages, vendored), video.py (frame sampling)
harnesses/        # base.py + uniform.py / stitched.py / decentralized.py / clip_select.py
                  #   / option_union.py / segment_select.py (+ viclip_scorer.py)
                  #   + blind.py, single_view.py (controls)
models/           # clients.py — Qwen3-VL and InternVL3 backends (InternVL import stays lazy);
                  #   cost.py — token pricing (used only by the unwired llm_judge)
evaluation/       # scoring.py (parse_choice / gt_choice), chance.py, report.py + run
                  #   metrics & summaries (llm_judge.py is vestigial — see below)
plotting/         # plot_results.py — Table 1 + Plots 1–4; frame-sweep + Task-1 figure scripts
configs/          # datasets.yaml — video roots, subset paths, summary-cache path
scripts/          # run_bench.sbatch, gen_clip_summaries.sbatch (SLURM)
scripts/data/     # download_videos.py, fetch_meva_videos.py, fetch_egoexo_videos.py, fetch_agibot_videos.py
data/subsets/     # committed question-subset JSONs (they define the benchmark)
docs/             # ported spec + analysis writeups (provenance: docs/PORTING.md)
envs/             # cvbench.yml, internvl.yml conda environments
tests/            # compare_prompts_vs_fork.py — prompt-equivalence gate vs the reference;
                  #   validate_scoring.py + fixtures/ — CPU scoring gate, runs with no GPU/data
```

Run everything from the repo root — modules are top-level packages
(`from harnesses.uniform import ...`).

## Setup

```bash
conda env create -f envs/cvbench.yml
conda env create -f envs/internvl.yml
```

> ⚠️ **The two-env split is mandatory.** Qwen legs run under `cvbench`; InternVL3 runs
> under `internvl` — a transformers version conflict breaks InternVL3's remote code
> under `cvbench`.

Models load from the local HF cache; pre-download them on a login node
(`HF_HUB_OFFLINE=1` works after that). Backend aliases → HF ids: `qwen3vl` =
`Qwen/Qwen3-VL-8B-Thinking`, `qwen3vl-instruct` = `Qwen/Qwen3-VL-8B-Instruct`,
`internvl3` = `OpenGVLab/InternVL3-8B`.

## Run

Sanity first. This runs green on a fresh clone with no GPU, videos, or weights:

```bash
python -m tests.validate_scoring   # CPU scoring gate against the committed fixture
```

Then a 5-record GPU smoke before committing to a full run:

```bash
conda activate internvl
python run_vqa.py --subset data/subsets/cvbench_full_runnable_subset.json \
    --methods temporal_weighted --backends internvl3 \
    --video-root <your CVBench video dir> --limit 5 \
    --passes 4 --seeds 1,2,3,4 --temperature 0.7
```

Drop `--limit 5` for the real run:

```bash
python run_vqa.py --subset data/subsets/cvbench_full_runnable_subset.json \
    --methods temporal_weighted --backends internvl3 \
    --video-root <your CVBench video dir> \
    --passes 4 --seeds 1,2,3,4 --temperature 0.7
```

The same run under SLURM, sharded 8 ways:

```bash
ENV=internvl SUBSET=data/subsets/cvbench_full_runnable_subset.json \
    METHODS=temporal_weighted BACKENDS=internvl3 \
    CHUNK=8 sbatch --array=0-7 scripts/run_bench.sbatch
```

One leg of the stitch frame-budget sweep — the `centralized` 2×2 arm at `NFRAMES` frames
per clip with nothing else changed (design + findings in `docs/stitch_frame_sweep.md`):

```bash
ENV=internvl SUBSET=data/subsets/cvbench_full_runnable_subset.json BACKENDS=internvl3 \
    METHODS=centralized MONTAGE_KIND=video NFRAMES=32 TAG=_fullstitch32 CHUNK=8 \
    VIDEO_ROOT=<your CVBench video dir> \
    sbatch --array=0-7 scripts/run_bench.sbatch
```

The sbatch partition defaults to `gpul40q`; override with `sbatch -p <partition>`.
Error-bar convention for anything reported: `--passes 4 --seeds 1,2,3,4 --temperature 0.7`.

## Results

Runs append to `results/<subset>_<...>.jsonl` — one row per question × method × backend ×
pass — and write a `*_summary.json` next to it. `results/` and `logs/` are gitignored —
outputs never go in git. `scripts/finalize_cvbench_full.sh` pools the full-1000 3-way
shards into one report; `scripts/pool_stitch_sweep.sh` does the same for the stitch
frame-budget sweep legs (renaming each leg's rows to `stitch<NN>_f<N>` by its TAG so the
budgets stay distinct arms). Caution: finalize's glob matches *every* full-1000 shard,
sweep legs included — since sweep rows record `method='centralized'`, running it with
sweep shards present folds all budgets into its centralized arm, so move sweep shards
aside (or use only `pool_stitch_sweep.sh`) when both run families coexist.

## Summarization / clip-selection scoring

Answers are scored by deterministic choice parsing (`evaluation/scoring.py`
`parse_choice`; an unparseable answer = abstain = wrong). `clip_select` needs the
per-clip summary cache: generate it with `scripts/gen_clip_summaries.sbatch` →
`results/clip_summaries_internvl3.jsonl`. MCQ answers are never judge-scored:
`evaluation/llm_judge.py` is a summarization G-Eval judge kept from the original split —
nothing in this pipeline produces its input, and it is not wired in.

## Plotting

```bash
python plotting/plot_results.py --jsonl results/<run>.jsonl --out-dir results/figs
```

`--jsonl` accepts one or more results files (shards are pooled); `--out-dir` defaults to
a `figs/` directory next to the first input. Writes Table 1 (`table1.md/.csv`) and
Plots 1–4.

Frame-budget sweep figures, from a pooled sweep JSONL (see `docs/stitch_frame_sweep.md`):

```bash
python -m plotting.frame_sweep_by_category      --jsonl results/<pooled>.jsonl
python -m plotting.frame_budget_smallmultiples  --jsonl results/<pooled>.jsonl
```

Both draw each series against its own random-guessing floor, computed from the subset's
answer-option lists by `evaluation/chance.py` (`python -m evaluation.chance` prints the
per-category table). Figures land in `figures/frame_sweep/` as png/pdf/svg.

## Ground rules

- **Inference-only** — no training; the harness is the variable.
- **Equal budget** — compare harnesses at the same frame budget (watch token parity too).
- **One change at a time** — only visual packaging differs between harnesses.
- **Inspect by hand** — watch the clips before trusting a number.
- **Never commit data, video, weights, or run outputs** (see `.gitignore`).
- **Single greedy passes are preliminary** — no error bars until `--passes 4`.
- **Ours:** the two-conda-env split is mandatory; harness equivalence claims are enforced
  by `tests/compare_prompts_vs_fork.py` (prompts must stay byte-identical to the
  reference implementation). The gate needs `CVBENCH_FORK` set **and** the reference
  checkout's CVBench videos on disk — otherwise it SKIPs with exit 0, which is **not** a
  pass — and it covers the originally ported arms only.

## Data

Subset JSONs are committed under `data/subsets/` (they define the benchmark); videos are
never committed. CVBench videos: `scripts/data/download_videos.py` (HF). CrossView: the
release annotations plus `scripts/data/fetch_meva_videos.py` (public S3),
`scripts/data/fetch_egoexo_videos.py` (license-gated), and
`scripts/data/fetch_agibot_videos.py` (gated HF). Set your video roots in
`configs/datasets.yaml` or pass `--video-root`.
