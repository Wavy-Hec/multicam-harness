# Ported from Wavy-Hec/CVBench bench/run_bench.py @ 619af02e8e65762f725bccff302b2fd3de379d36
"""Run the multi-camera benchmark: methods x backends x passes over a subset.

Usage (from repo root):
  python run_vqa.py --subset data/subsets/crossview_combined_subset.json \
      --methods centralized,per_stream --backends qwen3vl \
      --passes 4 --seeds 1,2,3,4 --temperature 0.7 --limit 5

A "pass" = one sampled generation (temperature>0) at a fixed seed, with the
(deterministic) frames held fixed; Table 1's std is taken over the passes.
Writes one Result per line to a JSONL (resumable on (id,method,backend,pass)),
then prints + saves a per-(method/backend) summary. Backends load once and are
reused across methods/passes. ``--chunk N --offset i`` shards the subset
(``data[i::N]``) for Slurm-array sweeps.

Cluster-specific argparse defaults (video root, summary cache) are read from
``configs/datasets.yaml`` when present; the relative-path fallbacks below apply
otherwise. The experiment loop itself lives in ``runner.run``.
"""
import argparse
import re

import yaml

from harnesses.stitched import CentralizedMethod
from harnesses.decentralized import PerStreamMethod
from harnesses.uniform import CVBenchNativeMethod, TemporalWeightedMethod
from harnesses.blind import BlindMethod
from harnesses.single_view import SingleViewMethod
from harnesses.clip_select import (SummarySelectMethod, ClipScoreSelectMethod,
                                   FrameSelectMethod)
from models.clients import make_backend
import runner


def load_datasets_config(path="configs/datasets.yaml"):
    """configs/datasets.yaml as a dict ({} if the file is missing). Used only to
    fill cluster-specific argparse defaults; harness/model code keeps taking
    these values as ordinary parameters."""
    try:
        with open(path) as fh:
            return yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return {}


_CFG = load_datasets_config()
DEFAULT_VIDEO_ROOT = (_CFG.get("video_roots") or {}).get(
    "crossview", "crossview-release-annotations/crossview-release")
DEFAULT_SUMMARIES = _CFG.get("summaries_cache") or "results/clip_summaries_internvl3.jsonl"

METHODS = {"centralized": CentralizedMethod, "per_stream": PerStreamMethod,
           "cvbench_native": CVBenchNativeMethod,
           "temporal_weighted": TemporalWeightedMethod,
           "blind": BlindMethod,
           # (adaptive_content / adaptive_query — the within-clip frame-selection
           # ablation — were retired 2026-07-02 after losing/tying uniform; the
           # archived result lives in the fork's git history.)
           # D3 clip selection: spend the budget on the clips a question needs.
           # summary_select_* = cached per-clip summaries -> one text-only
           # selector call (route may answer ALL; top1 forces one clip);
           # clip_select_top1 = CLIP question-vs-thumbnail scoring, no LLM call.
           "summary_select_route": SummarySelectMethod,
           "summary_select_top1": SummarySelectMethod,
           "clip_select_top1": ClipScoreSelectMethod}

# clip_select method names are generated, not enumerated: an optional scorer
# tag (must be in SCORER_ALIASES), an optional _opt marker, then the top-m
# count, e.g. clip_select_top2, clip_select_siglip_top1,
# clip_select_siglip_opt_top1. The matched string becomes the method's recorded
# name, so scorer and query variants never collide in rows/resume keys.
# _opt scores frames against each ANSWER OPTION instead of the question — the
# answer-choice-guided selection arm. The tag cannot itself be "opt".
CLIP_SELECT_RE = re.compile(
    r"^clip_select(?:_(?P<tag>(?!opt(?:_|$))[a-z0-9]+))?(?P<opt>_opt)?_top(?P<m>\d+)$")
# single_view<i>: feed only view/clip i (harnesses/single_view.py); records with
# fewer than i views are skipped, so single_view1..13 sweeps a mixed-K pool.
SINGLE_VIEW_RE = re.compile(r"^single_view(?P<i>\d+)$")
# frame_select: global top-budget frame selection across ALL clips (optional
# scorer tag, e.g. frame_select_siglip; optional _opt for option-guided
# scoring); the budget comes from --budget.
FRAME_SELECT_RE = re.compile(
    r"^frame_select(?:_(?P<tag>(?!opt(?:_|$))[a-z0-9]+))?(?P<opt>_opt)?$")
SCORER_ALIASES = {"siglip": "google/siglip-so400m-patch14-384",
                  "siglip2": "google/siglip2-so400m-patch14-384"}


def make_method(mname, backend, args):
    if mname == "centralized":
        return CentralizedMethod(backend, nframes=args.nframes,
                                 max_new_tokens=args.max_new_tokens,
                                 temperature=args.temperature,
                                 montage_frames=args.montage_frames, cell_px=args.cell_px,
                                 montage_kind=args.montage_kind,
                                 total_frames=args.total_frames,
                                 reasoning=not args.no_reasoning)
    if mname == "per_stream":
        return PerStreamMethod(backend, nframes=args.nframes,
                               max_new_tokens=args.max_new_tokens,
                               temperature=args.temperature,
                               perception_max_new_tokens=args.perception_max_new_tokens,
                               stream_kind=args.stream_kind,
                               total_frames=args.total_frames,
                               reasoning=not args.no_reasoning)
    if mname == "cvbench_native":
        return CVBenchNativeMethod(backend, nframes=args.nframes,
                                   max_new_tokens=args.max_new_tokens,
                                   temperature=args.temperature,
                                   total_frames=args.total_frames,
                                   reasoning=not args.no_reasoning)
    sv = SINGLE_VIEW_RE.match(mname)
    if sv:
        return SingleViewMethod(backend, view_idx=int(sv.group("i")),
                                nframes=args.nframes,
                                max_new_tokens=args.max_new_tokens,
                                temperature=args.temperature, name=mname,
                                reasoning=not args.no_reasoning)
    if mname == "temporal_weighted":
        return TemporalWeightedMethod(backend, budget=args.budget, floor=args.floor,
                                      weighting=args.weighting, nframes=args.nframes,
                                      max_new_tokens=args.max_new_tokens,
                                      temperature=args.temperature,
                                      reasoning=not args.no_reasoning)
    if mname.startswith("summary_select_"):
        return SummarySelectMethod(
            backend, summaries_path=args.summaries,
            mode=mname.rsplit("_", 1)[1], budget=args.budget, floor=args.floor,
            sel_max_new_tokens=args.sel_max_new_tokens, nframes=args.nframes,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            reasoning=not args.no_reasoning)
    fm = FRAME_SELECT_RE.match(mname)
    if fm:
        tag = fm.group("tag")
        if tag and tag not in SCORER_ALIASES:
            raise SystemExit(f"unknown frame_select scorer tag '{tag}'. "
                             f"Known: {list(SCORER_ALIASES)}")
        return FrameSelectMethod(
            backend, budget=args.budget,
            candidates_per_video=args.frame_candidates,
            clip_model=SCORER_ALIASES[tag] if tag else args.clip_model,
            cell_px=args.cell_px, name=mname,
            nframes=args.nframes, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, reasoning=not args.no_reasoning,
            query="options" if fm.group("opt") else "question")
    mm = CLIP_SELECT_RE.match(mname)
    if mm:
        tag = mm.group("tag")
        if tag and tag not in SCORER_ALIASES:
            raise SystemExit(f"unknown clip_select scorer tag '{tag}'. "
                             f"Known: {list(SCORER_ALIASES)}")
        return ClipScoreSelectMethod(
            backend, top_m=int(mm.group("m")), thumbs=args.sel_thumbs,
            clip_model=SCORER_ALIASES[tag] if tag else args.clip_model,
            stat=args.sel_stat, name=mname,
            budget=args.budget, floor=args.floor,
            nframes=args.nframes, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, reasoning=not args.no_reasoning,
            query="options" if mm.group("opt") else "question")
    return METHODS[mname](backend, nframes=args.nframes,
                          max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                          reasoning=not args.no_reasoning)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--methods", default="centralized")
    ap.add_argument("--backends", default="qwen3vl")
    ap.add_argument("--limit", type=int, default=0, help="only first N records (smoke test)")
    ap.add_argument("--video-root", default=DEFAULT_VIDEO_ROOT)
    ap.add_argument("--nframes", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--perception-max-new-tokens", type=int, default=1024,
                    help="per_stream: token cap for each per-view perception call "
                         "(1024 truncates thinking backends mid-<think>; raise for those)")
    ap.add_argument("--passes", type=int, default=4, help="independent sampled passes for std")
    ap.add_argument("--seeds", default="1,2,3,4", help="comma seeds; len must cover --passes")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--no-reasoning", action="store_true",
                    help="direct-answer prompt: no <think> trace requested. There is no "
                         "model-side switch — the visible reasoning is produced BY the "
                         "prompt, so turning it off means asking for the answer directly. "
                         "The <answer> tags stay, so one parser serves both modes.")
    ap.add_argument("--budget", type=int, default=64,
                    help="temporal_weighted: TOTAL frames per question, split across clips")
    ap.add_argument("--total-frames", type=int, default=0,
                    help="centralized/cvbench_native/per_stream: hold the TOTAL frame "
                         "count per question fixed (split evenly across its clips) "
                         "instead of a flat --nframes per clip; 0 = off")
    ap.add_argument("--floor", type=int, default=2,
                    help="temporal_weighted: per-clip minimum frames")
    ap.add_argument("--weighting", default="duration", choices=["duration", "even"],
                    help="temporal_weighted: split the budget by clip duration "
                         "('duration') or evenly ('even', the budget-matched control)")
    ap.add_argument("--clip-model", default="openai/clip-vit-base-patch32",
                    help="clip_select_*: HF CLIP/SigLIP model id for image-text "
                         "scoring; a scorer tag in the method name (e.g. "
                         "clip_select_siglip_top1) takes precedence")
    ap.add_argument("--summaries",
                    default=DEFAULT_SUMMARIES,
                    help="summary_select_*: per-clip summary cache JSONL (or glob); "
                         "generate with scripts/gen_clip_summaries.py")
    ap.add_argument("--sel-thumbs", type=int, default=8,
                    help="clip_select_*: uniform thumbnails scored per clip")
    ap.add_argument("--sel-stat", default="max", choices=["max", "mean"],
                    help="clip_select_*: rank clips by max- or mean-over-thumbnails "
                         "similarity (both are recorded in frame_alloc regardless)")
    ap.add_argument("--frame-candidates", type=int, default=32,
                    help="frame_select: uniform candidate frames decoded PER clip; "
                         "the global top-(--budget) across all clips' candidates is kept")
    ap.add_argument("--sel-max-new-tokens", type=int, default=512,
                    help="summary_select_*: token cap for the selector call")
    ap.add_argument("--montage-frames", type=int, default=0,
                    help="centralized montages per question (0 -> = nframes)")
    ap.add_argument("--cell-px", type=int, default=448)
    ap.add_argument("--stream-kind", default="camera", choices=["camera", "video", "view"],
                    help="per_stream: label/phrase clips as synced 'camera' views "
                         "(MEVA, byte-identical to the original prompt) or independent "
                         "'video' clips (CVBench — matches the questions' 'Video k' "
                         "wording, mirroring --montage-kind)")
    ap.add_argument("--montage-kind", default="camera", choices=["camera", "video", "view"],
                    help="centralized montage framing: 'camera' (synced views, default) "
                         "or 'video' (independent clips — corrected CVBench preamble + 'Video i' labels)")
    ap.add_argument("--internvl-max-tiles", type=int, default=1,
                    help="InternVL tiles per montage image (4 lets a 2x2 montage keep per-camera 448 res)")
    ap.add_argument("--chunk", type=int, default=0, help="number of shards (Slurm array)")
    ap.add_argument("--offset", type=int, default=0, help="this shard index in [0,chunk)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    runner.run(args, METHODS, (CLIP_SELECT_RE, FRAME_SELECT_RE, SINGLE_VIEW_RE),
               make_backend, make_method)


if __name__ == "__main__":
    main()
