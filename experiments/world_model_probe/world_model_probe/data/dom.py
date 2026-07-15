from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

from world_model_probe.data.base import BackboneInput, ProbeSample


def _episode_path(dom_root: Path, chunk_id: int, episode_index: int) -> Path:
    return dom_root / "data" / f"chunk-{chunk_id:03d}" / f"episode_{episode_index:06d}.parquet"


def _video_path(dom_root: Path, chunk_id: int, video_key: str, episode_index: int) -> Path:
    return dom_root / "videos" / f"chunk-{chunk_id:03d}" / video_key / f"episode_{episode_index:06d}.mp4"


def load_info(dom_root: str | Path) -> dict[str, Any]:
    with open(Path(dom_root) / "meta" / "info.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_tasks(dom_root: str | Path) -> dict[int, str]:
    tasks: dict[int, str] = {}
    path = Path(dom_root) / "meta" / "tasks.jsonl"
    if not path.exists():
        return tasks
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "task_index" in row and "task" in row:
                tasks[int(row["task_index"])] = str(row["task"])
    return tasks


def load_episode_table(path: str | Path) -> pd.DataFrame:
    return pq.read_table(path).to_pandas()


def dynamic_vla_instruction(task: str) -> str:
    """Convert DOM structured task metadata to the natural-language prompt used by dynamic-vla."""
    try:
        instruction_metadata = json.loads(task)
    except json.JSONDecodeError:
        return task

    if not isinstance(instruction_metadata, dict) or "task" not in instruction_metadata:
        return task

    # Older local DOM exports use an underscore; dynamic-vla's native generator expects a hyphen.
    if instruction_metadata.get("task") == "long_horizon":
        instruction_metadata = copy.deepcopy(instruction_metadata)
        instruction_metadata["task"] = "long-horizon"

    dynamic_vla_root = Path("/data/repos/dynamic-vla")
    if dynamic_vla_root.exists() and str(dynamic_vla_root) not in sys.path:
        sys.path.insert(0, str(dynamic_vla_root))

    try:
        from utils.instruction_generator import InstructionGenerator
    except ImportError as exc:
        raise ImportError(
            "Could not import dynamic-vla InstructionGenerator. "
            "Ensure /data/repos/dynamic-vla exists or is on PYTHONPATH."
        ) from exc

    return InstructionGenerator.generate_instruction(instruction_metadata)


def _stack_future(df: pd.DataFrame, col: str, frame_index: int, horizons: list[int], indices: list[int]) -> np.ndarray:
    rows = []
    for horizon in horizons:
        value = np.asarray(df.iloc[frame_index + horizon][col], dtype=np.float32)
        rows.append(value[indices])
    return np.stack(rows, axis=0).astype(np.float32)


def build_targets(
    df: pd.DataFrame,
    frame_index: int,
    horizons: list[int],
    obj_pos_indices: list[int],
    obj_vel_indices: list[int],
    arm_pos_indices: list[int],
) -> dict[str, np.ndarray]:
    return {
        "obj_pos": _stack_future(df, "observation.environment_state", frame_index, horizons, obj_pos_indices),
        "obj_vel": _stack_future(df, "observation.environment_state", frame_index, horizons, obj_vel_indices),
        "arm_pos": _stack_future(df, "observation.state", frame_index, horizons, arm_pos_indices),
    }


def iter_probe_samples(cfg: dict[str, Any], split: str) -> Iterable[ProbeSample]:
    data_cfg = cfg["data"]
    target_cfg = cfg["targets"]
    dom_root = Path(data_cfg["dom_root"])
    chunk_id = int(data_cfg.get("chunk_id", 0))
    video_key = str(data_cfg.get("video_key", "observation.images.opst_cam"))
    split_range = data_cfg["splits"][split]
    start_ep, end_ep = int(split_range[0]), int(split_range[1])
    horizons = [int(h) for h in target_cfg["horizons"]]
    max_horizon = max(horizons)
    sample_stride = int(data_cfg.get("sample_stride", 5))
    input_frames = int(data_cfg.get("input_frames", 8))
    frame_stride = int(data_cfg.get("frame_stride", 1))
    image_size = data_cfg.get("image_size")
    max_samples_per_episode = data_cfg.get("max_samples_per_episode")
    max_samples_per_episode = None if max_samples_per_episode is None else int(max_samples_per_episode)
    obj_pos_indices = [int(i) for i in target_cfg.get("obj_pos_indices", [0, 1, 2])]
    obj_vel_indices = [int(i) for i in target_cfg.get("obj_vel_indices", [6, 7, 8])]
    arm_pos_indices = [int(i) for i in target_cfg.get("arm_pos_indices", [0, 1, 2])]
    tasks = load_tasks(dom_root)

    for episode_index in range(start_ep, end_ep):
        parquet_path = _episode_path(dom_root, chunk_id, episode_index)
        video_path = _video_path(dom_root, chunk_id, video_key, episode_index)
        if not parquet_path.exists() or not video_path.exists():
            continue
        df = load_episode_table(parquet_path)
        needed = {"observation.environment_state", "observation.state"}
        missing = needed - set(df.columns)
        if missing:
            raise KeyError(f"{parquet_path} missing columns: {sorted(missing)}")
        limit = max(0, len(df) - max_horizon)
        emitted = 0
        for frame_index in range(0, limit, sample_stride):
            frame_indices = clip_frame_indices(frame_index, input_frames, frame_stride)
            frames = read_video_frames(video_path, frame_indices, image_size)
            targets = build_targets(
                df,
                frame_index,
                horizons,
                obj_pos_indices,
                obj_vel_indices,
                arm_pos_indices,
            )
            task_index = int(df.iloc[frame_index].get("task_index", -1))
            raw_task = tasks.get(task_index, "")
            instruction = dynamic_vla_instruction(raw_task)
            metadata = {
                "episode_index": episode_index,
                "frame_index": frame_index,
                "parquet_path": str(parquet_path),
                "video_path": str(video_path),
                "task_index": task_index,
                "task": raw_task,
                "raw_task": raw_task,
                "instruction": instruction,
            }
            yield ProbeSample(
                sample_id=f"episode_{episode_index:06d}_frame_{frame_index:06d}",
                backbone_input=BackboneInput(
                    frames=frames,
                    semantic=instruction,
                    metadata=metadata,
                ),
                targets=targets,
                valid=np.ones(len(horizons), dtype=np.float32),
                metadata=metadata,
            )
            emitted += 1
            if max_samples_per_episode is not None and emitted >= max_samples_per_episode:
                break


def clip_frame_indices(frame_index: int, input_frames: int, frame_stride: int) -> list[int]:
    indices = [frame_index - frame_stride * i for i in reversed(range(input_frames))]
    first = max(0, indices[0])
    return [max(first, idx) for idx in indices]


def read_video_frames(video_path: str | Path, indices: list[int], image_size: list[int] | tuple[int, int] | None) -> list[Image.Image]:
    try:
        import decord

        vr = decord.VideoReader(str(video_path), num_threads=1)
        max_idx = len(vr) - 1
        safe_indices = [min(max(0, int(i)), max_idx) for i in indices]
        frames = vr.get_batch(safe_indices).asnumpy()
    except Exception:
        frames = _read_video_frames_pyav(video_path, indices)
    images: list[Image.Image] = []
    for arr in frames:
        img = Image.fromarray(arr)
        if image_size is not None:
            width, height = int(image_size[0]), int(image_size[1])
            img = img.resize((width, height), Image.BICUBIC)
        images.append(img)
    return images


def _read_video_frames_pyav(video_path: str | Path, indices: list[int]) -> np.ndarray:
    import av

    wanted = [max(0, int(i)) for i in indices]
    wanted_set = set(wanted)
    found: dict[int, np.ndarray] = {}
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame_idx, frame in enumerate(container.decode(stream)):
            if frame_idx in wanted_set:
                found[frame_idx] = frame.to_ndarray(format="rgb24")
            if frame_idx > max(wanted_set) and len(found) == len(wanted_set):
                break
    if not found:
        raise RuntimeError(f"Could not decode any requested frames from {video_path}")
    ordered = []
    last = found[min(found)]
    for idx in wanted:
        if idx in found:
            last = found[idx]
        ordered.append(last)
    return np.stack(ordered, axis=0)
