"""CVBench-native harness: uniform per-camera frame sampling, clips shown sequentially."""
import numpy as np
import os

from dataloaders.video import get_video_frame_count, load_frames_ffmpeg


def uniform_sampling_strategy(video_paths, num_samples):
    frames_by_cam = {}
    for video_path in video_paths:
        if not os.path.exists(video_path):
            continue
        cam_name = os.path.splitext(os.path.basename(video_path))[0]
        frame_count = get_video_frame_count(video_path)
        if frame_count < num_samples:
            frame_indices = np.arange(frame_count)
        else:
            frame_indices = np.linspace(0, frame_count - 1, num_samples, dtype=int)
        frames = load_frames_ffmpeg(video_path, frame_indices)
        if frames:
            frames_by_cam[cam_name] = frames
    return frames_by_cam
