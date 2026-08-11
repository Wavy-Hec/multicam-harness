# `inprocess/` — in-process harness arms

A self-contained implementation of the **centralized**, **decentralized**, and
**clip/frame-selection** arms, kept in its own package so it can be read, run and
compared without disturbing anything already in the repo.

**This directory adds files only.** Nothing outside `inprocess/` is modified: the
existing `harnesses/`, `dataloaders/`, `evaluation/` and `models/` packages are
untouched, so the two implementations can sit side by side and be diffed against
each other on the same subset. Take whatever is useful and leave the rest — the
package has no hooks into the rest of the tree.

## Arms

| Class | Module | What the model sees | Calls/question |
|---|---|---|---|
| `CentralizedMethod` | `harnesses.stitched` | Time-synchronized frames from all views tiled into labeled grid montages, fed as one visual input | 1 |
| `PerStreamMethod` | `harnesses.decentralized` | One query-conditioned perception pass per view, then a text-only aggregation pass over the descriptions | k+1 |
| `SummarySelectMethod` | `harnesses.clip_select` | Cached per-clip text summaries routed by the same model, which picks the clips the question needs | 1–2 |
| `ClipScoreSelectMethod` | `harnesses.clip_select` | Clips scored by CLIP/SigLIP text-image similarity over thumbnails; keep the top-m | 1 |
| `FrameSelectMethod` | `harnesses.clip_select` | One shared budget of the most question-relevant frames chosen globally across every clip, grouped by source clip | 1 |

Montage geometry is `cols = ceil(sqrt(K))`, `rows = ceil(K / cols)` — 2x2 at four
views, up to 4x4 at the thirteen-slot cap.

## Usage

Every arm shares one interface, so swapping the harness is the only variable:

```python
from inprocess.harnesses.stitched import CentralizedMethod
from inprocess.harnesses.decentralized import PerStreamMethod
from inprocess.harnesses.clip_select import FrameSelectMethod

m = CentralizedMethod(backend, nframes=8)      # any backend exposing .generate()
res = m.answer(record, video_root, seed=1)     # -> Result dataclass
```

`Result` carries the prediction, the gold answer, the reasoning trace, latency,
token counts and a `frame_alloc` dict recording exactly how the frame budget was
spent — so an accuracy difference can always be checked against what each arm was
actually given.

Records are the `video_1..video_N` / `image_1..image_N` schema already used by
`data/subsets/`. Multi-view still-image records are supported by the centralized
and decentralized arms; the selection arms sample frames out of clips and raise a
clear error on a still-image record rather than silently running with no visual
input.

## Reasoning mode and option-guided selection

Two knobs were added since the initial drop, both default-off so existing calls
behave exactly as before:

- **`reasoning=False`** on any arm swaps the prompt for a direct-answer
  template. There is no model-side thinking switch — the visible trace is
  produced by the prompt — so turning it off means asking for the answer
  directly. The `<answer>` tags stay, so one parser serves both modes, and the
  parser also accepts the tagless letter-plus-option-text shape that
  direct-answer models produce.
- **`query="options"`** on `ClipScoreSelectMethod` / `FrameSelectMethod` scores
  frames against EACH answer option separately (reduced by max) instead of the
  question. Options are embedded one by one, so nothing is lost to the text
  encoder's 64/77-token cap. Rows record `query_mode` and the query count in
  `frame_alloc`, so an option-guided row is distinguishable from a
  question-guided one without the launch environment. A record with no usable
  option text falls back to the question and says so in the row.

Setting `STRICT_ANSWER_PROMPT=1` in the environment enables a v2 answer-hygiene
prompt (every legal letter enumerated, options declared exhaustive so N/A-style
refusals are ruled out). It changes generation, so it is off by default — and
because the flag lives in the environment, a runner that adopts it should record
its value alongside each row (ours stamps a `strict_prompt` field at write
time), so runs on different prompt versions can never silently pool.

The selection arms now fail loudly rather than degrade: an unreadable clip, or a
similarity scorer that fails to load or score, raises instead of silently
falling back to unguided selection — a row that looks option-guided while
carrying no guidance is worse than a crashed leg.

## How this respects the ground rules

- **Inference-only.** No training anywhere in this package; the harness is the
  only variable.
- **Equal budget.** `PerStreamMethod` takes `total_frames`, which fixes the
  *total* frames per question and splits them across its clips, so two harnesses
  can be compared at one budget rather than at equal frames-per-clip (which is not
  an equal budget when clip counts differ). `Result.frame_alloc` records the split
  and the token counts, so token parity is checkable and not assumed.
- **One change at a time.** The arms share a single prompt scaffold and answer
  parser; only the visual packaging differs between them. The selection arms'
  all-clips branch emits a prompt byte-identical to the sequential baseline, so any
  delta is attributable to the clips it pruned.
- **Never commit data, video, weights or run outputs.** This package is code only.
- **Passes.** Selection is deterministic and cached across passes of the same
  question, so a multi-pass standard deviation isolates the answer stage rather
  than re-rolling the selection.

## Provenance

Ported from the CVBench evaluation fork; each module carries a header naming the
source file and commit it was taken from, plus any deliberate delta. The
selection arms additionally guard against two failure modes that do not raise on
their own: running with zero visual items on a still-image record, and a global
top-k ranking starving whole cameras of frames, which biases exactly the
camera-count axis these arms are compared on.
