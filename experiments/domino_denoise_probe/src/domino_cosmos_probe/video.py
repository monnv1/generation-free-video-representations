from __future__ import annotations

from pathlib import Path

import av
import numpy as np
from PIL import Image


def read_video_frames(path: str | Path) -> list[Image.Image]:
    """Decode an mp4 into RGB PIL frames.

    DOMINO episodes are short enough that decoding one episode at a time keeps
    the extraction code simple and avoids repeated random seeks.
    """
    frames: list[Image.Image] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            arr = frame.to_ndarray(format="rgb24")
            frames.append(Image.fromarray(arr))
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    return frames


def select_history_frames(frames: list[Image.Image], t: int, history_frames: int) -> list[Image.Image]:
    start = t - history_frames + 1
    if start < 0:
        raise ValueError(f"Cannot select {history_frames} history frames at t={t}")
    return frames[start : t + 1]


def normalize_uv(uv: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = np.asarray([max(width - 1, 1), max(height - 1, 1)], dtype=np.float32)
    return uv.astype(np.float32) / scale
