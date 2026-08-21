#!/bin/bash
# Ported from Wavy-Hec/CVBench bench/run_bench.sbatch @ 619af02e8e65762f725bccff302b2fd3de379d36
### Multi-camera benchmark sweep (2 models x 2 harnesses x 4 passes).
### Supports Slurm-array sharding (OFFSET=$SLURM_ARRAY_TASK_ID, CHUNK=array size)
### and an env split: Qwen legs run under `cvbench`, InternVL3 under `internvl`
### (cvbench's transformers breaks the InternVL3 remote code).
###
### Usage:
###   # tiny smoke (Qwen, 5 records):
###   LIMIT=5 BACKENDS=qwen3vl sbatch scripts/run_bench.sbatch
###   # full meva1033 Qwen leg, sharded 8 ways:
###   ENV=cvbench SUBSET=data/subsets/crossview_meva1033_subset.json BACKENDS=qwen3vl \
###     CHUNK=8 sbatch --array=0-7 scripts/run_bench.sbatch
###   # full meva1033 InternVL3 leg, sharded 8 ways:
###   ENV=internvl SUBSET=data/subsets/crossview_meva1033_subset.json BACKENDS=internvl3 \
###     CHUNK=8 sbatch --array=0-7 scripts/run_bench.sbatch
#SBATCH --job-name=bench-multicam
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH -N 1
#SBATCH -p gpul40q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=08:00:00

set -euo pipefail
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$REPO"
CONDA_SH=${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}
source "$CONDA_SH"
ENV=${ENV:-cvbench}
conda activate "$ENV"

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export DECORD_EOF_RETRY_MAX=${DECORD_EOF_RETRY_MAX:-20480}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}

SUBSET=${SUBSET:-data/subsets/crossview_combined_subset.json}
METHODS=${METHODS:-centralized,per_stream}
BACKENDS=${BACKENDS:-qwen3vl}
LIMIT=${LIMIT:-0}
NFRAMES=${NFRAMES:-8}
PASSES=${PASSES:-4}
SEEDS=${SEEDS:-1,2,3,4}
TEMPERATURE=${TEMPERATURE:-0.7}
MONTAGE_FRAMES=${MONTAGE_FRAMES:-0}
MONTAGE_KIND=${MONTAGE_KIND:-camera}   # 'camera' (synced views) | 'video' (independent clips)
STREAM_KIND=${STREAM_KIND:-camera}     # per_stream labels: 'camera' (MEVA) | 'video' (CVBench)
BUDGET=${BUDGET:-}                     # selection arms: TOTAL frames per question. Empty = per-arm default
                                       # (legacy arms 64; _optu/query_search MATCHED = nframes x K). Explicit 0
                                       # = matched, legal only for _optu/query_search.
SEL_TAU=${SEL_TAU:-0}                  # _optu arms: absolute per-option similarity cutoff; 0 = quantile mode
SEL_TAU_Q=${SEL_TAU_Q:-0.85}           # _optu arms: per-option quantile when SEL_TAU=0
N_QUERIES=${N_QUERIES:-4}              # query_search: generated search phrases per question
SEGMENTS_PER_VIDEO=${SEGMENTS_PER_VIDEO:-8}   # segment_select: segments each clip is split into
SEGMENTS_KEEP=${SEGMENTS_KEEP:-4}             # segment_select: most-relevant segments kept per clip
FRAMES_PER_SEGMENT=${FRAMES_PER_SEGMENT:-8}   # segment_select: frames sampled per segment
DEDUP_TAU=${DEDUP_TAU:-0.95}                  # segment_select: near-duplicate cosine cutoff, in (0,1].
                                              # 1 = dedup OFF; 0 is INVALID (fails fast) — NOT the SEL_TAU
                                              # '0 = off' convention. Static cams (MEVA) collapse at 0.95.
QUERY_MAX_NEW_TOKENS=${QUERY_MAX_NEW_TOKENS:-256}  # query_search: phrase-generation token cap (raise for thinking backends)
export VICLIP_DIR=${VICLIP_DIR:-$HOME/models/ViCLIP}   # clip_select_viclip_optu: local OpenGVLab/ViCLIP download
FLOOR=${FLOOR:-2}                      # temporal_weighted: per-clip minimum frames
WEIGHTING=${WEIGHTING:-duration}       # temporal_weighted: 'duration' (weighted) | 'even' (control)
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8192}
PERCEPTION_MAX_NEW_TOKENS=${PERCEPTION_MAX_NEW_TOKENS:-1024}  # per_stream per-view cap (raise for thinking backends)
TOTAL_FRAMES=${TOTAL_FRAMES:-0}        # >0: fixed TOTAL frames/question for native/centralized/per_stream
CELL_PX=${CELL_PX:-448}                # montage cell size (raise so centralized isn't token-starved vs native)
INTERNVL_MAX_TILES=${INTERNVL_MAX_TILES:-1}
REASONING=${REASONING:-1}              # 0 = direct-answer prompt, no <think> trace requested
# v2 answer-hygiene prompt (enumerate all letters + no-refusal sentence). It
# changes generation, so never flip it mid-series; rows record it either way.
export STRICT_ANSWER_PROMPT=${STRICT_ANSWER_PROMPT:-0}
# reasoning is imposed BY the prompt, not a model switch, so this swaps templates
NOREASON_ARG=""; [ "$REASONING" = "0" ] && NOREASON_ARG="--no-reasoning"
VIDEO_ROOT=${VIDEO_ROOT:-}   # override for non-MEVA sets (e.g. CVBench videos live elsewhere)
CHUNK=${CHUNK:-0}
# shard index: Slurm array task id if present, else OFFSET (default 0)
OFFSET=${SLURM_ARRAY_TASK_ID:-${OFFSET:-0}}

TAG=${TAG:-}   # optional suffix to separate runs that share subset+env (e.g. _dur/_even/_stitch)
BASE="results/bench_$(basename "${SUBSET%.json}")_${ENV}${TAG}"
if [ "$CHUNK" != "0" ]; then
  OUT=${OUT:-${BASE}_shard${OFFSET}.jsonl}
  SHARD_ARG="--chunk $CHUNK --offset $OFFSET"
else
  OUT=${OUT:-${BASE}.jsonl}
  SHARD_ARG=""
fi
LIMIT_ARG=""
if [ "$LIMIT" != "0" ]; then LIMIT_ARG="--limit $LIMIT"; fi
VR_ARG=""
if [ -n "$VIDEO_ROOT" ]; then VR_ARG="--video-root $VIDEO_ROOT"; fi
SUMMARIES=${SUMMARIES:-}   # summary_select_*: per-clip summary cache JSONL (or glob)
SUM_ARG=()                 # array + quoted expansion so a glob reaches python intact
if [ -n "$SUMMARIES" ]; then SUM_ARG=(--summaries "$SUMMARIES"); fi

echo "RUN @ $(date '+%F %T') env=$ENV subset=$SUBSET methods=$METHODS backends=$BACKENDS"
echo "     passes=$PASSES seeds=$SEEDS temp=$TEMPERATURE budget=${BUDGET:-auto} weighting=$WEIGHTING chunk=$CHUNK offset=$OFFSET out=$OUT video_root=${VIDEO_ROOT:-<default>}"
echo "     sel_tau=$SEL_TAU sel_tau_q=$SEL_TAU_Q n_queries=$N_QUERIES query_max_new_tokens=$QUERY_MAX_NEW_TOKENS viclip_dir=$VICLIP_DIR"
echo "     segments_per_video=$SEGMENTS_PER_VIDEO segments_keep=$SEGMENTS_KEEP frames_per_segment=$FRAMES_PER_SEGMENT dedup_tau=$DEDUP_TAU"
echo "     nframes=$NFRAMES total_frames=$TOTAL_FRAMES montage_frames=$MONTAGE_FRAMES cell_px=$CELL_PX internvl_max_tiles=$INTERNVL_MAX_TILES montage_kind=$MONTAGE_KIND stream_kind=$STREAM_KIND reasoning=$REASONING strict_prompt=$STRICT_ANSWER_PROMPT"
python run_vqa.py --subset "$SUBSET" --methods "$METHODS" --backends "$BACKENDS" \
    --nframes "$NFRAMES" --passes "$PASSES" --seeds "$SEEDS" --temperature "$TEMPERATURE" \
    --montage-frames "$MONTAGE_FRAMES" --montage-kind "$MONTAGE_KIND" --stream-kind "$STREAM_KIND" \
    --total-frames "$TOTAL_FRAMES" \
    ${BUDGET:+--budget "$BUDGET"} --floor "$FLOOR" --weighting "$WEIGHTING" \
    --sel-tau "$SEL_TAU" --sel-tau-q "$SEL_TAU_Q" --n-queries "$N_QUERIES" \
    --query-max-new-tokens "$QUERY_MAX_NEW_TOKENS" \
    --segments-per-video "$SEGMENTS_PER_VIDEO" --segments-keep "$SEGMENTS_KEEP" \
    --frames-per-segment "$FRAMES_PER_SEGMENT" --dedup-tau "$DEDUP_TAU" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --perception-max-new-tokens "$PERCEPTION_MAX_NEW_TOKENS" --cell-px "$CELL_PX" \
    --internvl-max-tiles "$INTERNVL_MAX_TILES" --out "$OUT" $NOREASON_ARG $SHARD_ARG $LIMIT_ARG $VR_ARG "${SUM_ARG[@]}"
echo "DONE @ $(date '+%F %T')"
