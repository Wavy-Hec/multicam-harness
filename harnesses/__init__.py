"""The harness is the independent variable: same model, same questions, same scoring —
only how the multi-camera video is packaged for the model changes.

  uniform       — CVBench-native: per-camera uniform frames, shown sequentially
  stitched      — centralized: synchronized frames stitched into grid montages
  decentralized — per-camera text summaries, then text-only aggregation
"""
from harnesses.decentralized.decentralized import decentralized_frames
from harnesses.centralized.stitched import stitched_frames_sampling_strategy
from harnesses.uniform import uniform_sampling_strategy

STRATEGIES = ("uniform", "stitched", "decentralized")


def get_frames(strategy, entry, num_frames, frame_budget="per_camera"):
    """Package a question's videos into model-ready frames for the chosen harness."""
    if strategy == "uniform":
        return uniform_sampling_strategy(entry["video_paths"], num_frames)
    if strategy == "stitched":
        return stitched_frames_sampling_strategy(entry["video_paths"], num_frames,
                                                 camera_names=entry.get("camera_names", {}))
    if strategy == "decentralized":
        return decentralized_frames(entry["video_paths"], num_frames, frame_budget)
    raise ValueError(f"unknown strategy: {strategy}")
