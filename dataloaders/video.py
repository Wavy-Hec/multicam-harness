"""ffmpeg/ffprobe frame extraction shared by all harnesses."""
import numpy as np
import subprocess


def get_video_frame_count(video_path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=nb_frames",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path], capture_output=True, text=True)
    try:
        return int(result.stdout.strip())
    except ValueError:
        return 0


def load_frames_ffmpeg(video_path, frame_indices):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "csv=p=0", video_path], capture_output=True, text=True)
    try:
        w, h = map(int, result.stdout.strip().split(','))
    except ValueError:
        return []
    select_expr = '+'.join(f'eq(n\\,{int(idx)})' for idx in frame_indices)
    result = subprocess.run(
        ["ffmpeg", "-i", video_path, "-vf", f"select={select_expr}", "-vsync", "0",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], capture_output=True)
    data = result.stdout
    frame_size = w * h * 3
    frames = []
    for i in range(len(frame_indices)):
        chunk = data[i * frame_size:(i + 1) * frame_size]
        if len(chunk) == frame_size:
            frames.append(np.frombuffer(chunk, dtype=np.uint8).reshape(h, w, 3).copy())
    return frames
