"""Centralized harness: time-synchronized frames from all cameras stitched into grid montages."""
import cv2
import math
import numpy as np
import os

from dataloaders.video import get_video_frame_count, load_frames_ffmpeg


def create_camera_grid(frames_dict, labels_dict):
    num_cameras = len(frames_dict)
    if num_cameras == 0:
        return None
    grid_cols = math.ceil(math.sqrt(num_cameras))
    grid_rows = math.ceil(num_cameras / grid_cols)
    camera_names = sorted(frames_dict.keys())
    frames = [frames_dict[cam] for cam in camera_names]
    if not frames:
        return None
    cell_height = max(f.shape[0] for f in frames)
    cell_width = max(f.shape[1] for f in frames)
    labeled_frames = []
    for cam_name, frame in zip(camera_names, frames):
        labeled_frame = frame.copy()
        label = labels_dict.get(cam_name, cam_name)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.8, 2)
        cv2.rectangle(labeled_frame, (5, 5), (15 + tw, 15 + th), (0, 0, 0), -1)
        cv2.putText(labeled_frame, label, (10, 10 + th), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        labeled_frames.append(labeled_frame)
    grid_canvas = np.zeros((grid_rows * cell_height, grid_cols * cell_width, 3), dtype=np.uint8)
    for idx, frame in enumerate(labeled_frames):
        row, col = idx // grid_cols, idx % grid_cols
        h, w = frame.shape[:2]
        y0 = row * cell_height + (cell_height - h) // 2
        x0 = col * cell_width + (cell_width - w) // 2
        grid_canvas[y0:y0 + h, x0:x0 + w] = frame
    return grid_canvas


def stitched_frames_sampling_strategy(video_paths, num_samples, camera_names=None):
    if not video_paths:
        return []
    video_frame_counts, valid_paths = {}, []
    for video_path in video_paths:
        if not os.path.exists(video_path):
            continue
        fc = get_video_frame_count(video_path)
        if fc > 0:
            video_frame_counts[video_path] = fc
            valid_paths.append(video_path)
    if not valid_paths:
        return []
    min_frame_count = min(video_frame_counts.values())
    if min_frame_count < num_samples:
        frame_indices = np.arange(min_frame_count)
    else:
        frame_indices = np.linspace(0, min_frame_count - 1, num_samples, dtype=int)
    frames_by_video = {}
    for video_path in valid_paths:
        cam_name = os.path.splitext(os.path.basename(video_path))[0]
        frames = load_frames_ffmpeg(video_path, frame_indices)
        if frames:
            frames_by_video[cam_name] = frames
    if not frames_by_video:
        return []
    labels_dict = {c: (camera_names or {}).get(c, c) for c in frames_by_video}
    stitched_grids = []
    for time_idx in range(len(next(iter(frames_by_video.values())))):
        frames_at_time = {c: f[time_idx] for c, f in frames_by_video.items() if time_idx < len(f)}
        grid = create_camera_grid(frames_at_time, labels_dict)
        if grid is not None:
            stitched_grids.append(grid)
    return stitched_grids
