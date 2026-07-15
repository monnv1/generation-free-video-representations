from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .video import normalize_uv


@dataclass(frozen=True)
class EpisodeData:
    task: str
    episode_index: int
    prompt: str
    length: int
    parquet_path: Path
    video_path: Path
    frame_index: np.ndarray
    object_xyz: np.ndarray
    object_uv: np.ndarray
    object_depth_m: np.ndarray
    in_frame: np.ndarray
    visible: np.ndarray
    contact: np.ndarray
    success: np.ndarray
    out_of_bounds: np.ndarray


def _load_episode_meta(task_dir: Path) -> dict[int, dict]:
    path = task_dir / "meta" / "episodes.jsonl"
    out: dict[int, dict] = {}
    with open(path, "r") as f:
        for line in f:
            item = json.loads(line)
            out[int(item["episode_index"])] = item
    return out


def _episode_video_path(task_dir: Path, camera_key: str, episode_index: int) -> Path:
    return (
        task_dir
        / "videos"
        / "chunk-000"
        / camera_key
        / f"episode_{episode_index:06d}.mp4"
    )


def _read_parquet_episode(task: str, task_dir: Path, episode_index: int, camera_key: str, image_width: int, image_height: int) -> EpisodeData:
    parquet_path = task_dir / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    meta = _load_episode_meta(task_dir)[episode_index]
    prompt = str((meta.get("tasks") or [""])[0])

    object_xyz = np.asarray(df["env.object_keypoint_xyz"].tolist(), dtype=np.float32)
    object_uv_raw = np.asarray(df["env.object_keypoint_uv.cam_high"].tolist(), dtype=np.float32)
    object_uv = normalize_uv(object_uv_raw, image_width, image_height)
    depth = np.asarray(df["env.object_keypoint_depth.cam_high"], dtype=np.float32).reshape(-1, 1) / 1000.0

    return EpisodeData(
        task=task,
        episode_index=episode_index,
        prompt=prompt,
        length=len(df),
        parquet_path=parquet_path,
        video_path=_episode_video_path(task_dir, camera_key, episode_index),
        frame_index=np.asarray(df["frame_index"], dtype=np.int64),
        object_xyz=object_xyz,
        object_uv=object_uv,
        object_depth_m=depth,
        in_frame=np.asarray(df["env.object_keypoint_in_frame.cam_high"], dtype=bool),
        visible=np.asarray(df["env.object_keypoint_visible.cam_high"], dtype=bool),
        contact=np.asarray(df["env.gripper_contact"], dtype=bool),
        success=np.asarray(df["env.task_success"], dtype=bool),
        out_of_bounds=np.asarray(df["env.object_out_of_bounds"], dtype=bool),
    )


def _split_episodes(episode_ids: list[int], train_fraction: float, val_fraction: float, rng: np.random.Generator) -> dict[int, str]:
    ids = list(episode_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(n * train_fraction))
    n_val = int(round(n * val_fraction))
    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train : n_train + n_val])
    split = {}
    for episode_id in ids:
        if episode_id in train_ids:
            split[episode_id] = "train"
        elif episode_id in val_ids:
            split[episode_id] = "val"
        else:
            split[episode_id] = "test"
    return split


def _time_to_contact_bucket(contact_future: np.ndarray) -> int:
    hits = np.flatnonzero(contact_future.astype(bool))
    if len(hits) == 0:
        return int(contact_future.shape[0])
    return int(hits[0])


def build_slice_cache(cfg: dict, run_dir: Path) -> tuple[Path, Path]:
    data_cfg = cfg["data"]
    root = Path(data_cfg["root"])
    tasks = list(data_cfg["tasks"])
    camera = str(data_cfg.get("camera", "observation.images.cam_high"))
    history = int(data_cfg.get("history_frames", 5))
    horizons = [int(x) for x in data_cfg.get("horizons", [1, 2, 4, 8, 15])]
    max_horizon = max(horizons)
    max_slices_per_task = int(data_cfg.get("max_slices_per_task", 1200))
    seed = int(cfg.get("run", {}).get("seed", 42))
    image_width = int(data_cfg.get("image_width", 320))
    image_height = int(data_cfg.get("image_height", 240))
    train_fraction = float(data_cfg.get("train_fraction", 0.70))
    val_fraction = float(data_cfg.get("val_fraction", 0.15))

    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    label_chunks: dict[str, list[np.ndarray]] = {
        "current_xyz": [],
        "current_uv": [],
        "current_depth_m": [],
        "current_uv_valid": [],
        "current_contact": [],
        "current_success": [],
        "current_out_of_bounds": [],
        "future_xyz": [],
        "future_uv": [],
        "future_depth_m": [],
        "future_uv_valid": [],
        "future_velocity_xyz": [],
        "future_contact": [],
        "future_success": [],
        "future_out_of_bounds": [],
        "time_to_contact": [],
    }

    row_id = 0
    for task in tasks:
        task_dir = root / task
        if not task_dir.exists():
            raise FileNotFoundError(f"Task directory does not exist: {task_dir}")
        episode_paths = sorted((task_dir / "data" / "chunk-000").glob("episode_*.parquet"))
        episode_ids = [int(path.stem.split("_")[1]) for path in episode_paths]
        episode_split = _split_episodes(episode_ids, train_fraction, val_fraction, rng)

        task_candidates: list[tuple[EpisodeData, int]] = []
        for episode_id in episode_ids:
            ep = _read_parquet_episode(task, task_dir, episode_id, camera, image_width, image_height)
            if not ep.video_path.exists():
                raise FileNotFoundError(f"Video file does not exist: {ep.video_path}")
            first_t = history - 1
            last_t = ep.length - max_horizon - 1
            if last_t < first_t:
                continue
            for t in range(first_t, last_t + 1):
                task_candidates.append((ep, t))

        if len(task_candidates) > max_slices_per_task:
            chosen = rng.choice(len(task_candidates), size=max_slices_per_task, replace=False)
            task_candidates = [task_candidates[int(i)] for i in np.sort(chosen)]

        for ep, t in task_candidates:
            fidx = np.asarray([t + h for h in horizons], dtype=np.int64)
            current_uv_valid = bool(ep.visible[t] and ep.in_frame[t])
            future_uv_valid = np.logical_and(ep.visible[fidx], ep.in_frame[fidx]).astype(np.float32)
            future_xyz = ep.object_xyz[fidx]
            future_velocity_xyz = (future_xyz - ep.object_xyz[t][None, :]) / np.asarray(horizons, dtype=np.float32)[:, None]
            future_contact = ep.contact[fidx].astype(np.float32)

            rows.append(
                {
                    "row_id": row_id,
                    "task": task,
                    "episode_index": ep.episode_index,
                    "split": episode_split[ep.episode_index],
                    "t": t,
                    "history_start": t - history + 1,
                    "history_end": t,
                    "prompt": ep.prompt,
                    "video_path": str(ep.video_path),
                    "parquet_path": str(ep.parquet_path),
                }
            )
            label_chunks["current_xyz"].append(ep.object_xyz[t].astype(np.float32))
            label_chunks["current_uv"].append(ep.object_uv[t].astype(np.float32))
            label_chunks["current_depth_m"].append(ep.object_depth_m[t].astype(np.float32))
            label_chunks["current_uv_valid"].append(np.asarray(current_uv_valid, dtype=np.float32))
            label_chunks["current_contact"].append(np.asarray(ep.contact[t], dtype=np.float32))
            label_chunks["current_success"].append(np.asarray(ep.success[t], dtype=np.float32))
            label_chunks["current_out_of_bounds"].append(np.asarray(ep.out_of_bounds[t], dtype=np.float32))
            label_chunks["future_xyz"].append(future_xyz.astype(np.float32))
            label_chunks["future_uv"].append(ep.object_uv[fidx].astype(np.float32))
            label_chunks["future_depth_m"].append(ep.object_depth_m[fidx].astype(np.float32))
            label_chunks["future_uv_valid"].append(future_uv_valid.astype(np.float32))
            label_chunks["future_velocity_xyz"].append(future_velocity_xyz.astype(np.float32))
            label_chunks["future_contact"].append(future_contact)
            label_chunks["future_success"].append(ep.success[fidx].astype(np.float32))
            label_chunks["future_out_of_bounds"].append(ep.out_of_bounds[fidx].astype(np.float32))
            label_chunks["time_to_contact"].append(np.asarray(_time_to_contact_bucket(future_contact), dtype=np.int64))
            row_id += 1

    if not rows:
        raise RuntimeError("No slices were built. Check task names, frame counts, and horizons.")

    index_path = run_dir / "slice_index.parquet"
    labels_path = run_dir / "labels.npz"
    pd.DataFrame(rows).to_parquet(index_path, index=False)
    labels = {key: np.stack(values, axis=0) for key, values in label_chunks.items()}
    labels["horizons"] = np.asarray(horizons, dtype=np.int64)
    np.savez_compressed(labels_path, **labels)
    return index_path, labels_path
