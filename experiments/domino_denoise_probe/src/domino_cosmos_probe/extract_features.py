from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import dump_json, ensure_run_dir, load_config, load_json, set_seed
from .cosmos_features import CosmosDenoiseFeatureExtractor
from .video import read_video_frames, select_history_frames


def _build_cosmos_cfg(cfg: dict) -> tuple[dict, dict]:
    cosmos_cfg = dict(cfg["cosmos"])
    data_cfg = cfg["data"]
    cosmos_cfg["image_width"] = int(data_cfg.get("image_width", 320))
    cosmos_cfg["image_height"] = int(data_cfg.get("image_height", 240))
    cosmos_cfg["history_frames"] = int(data_cfg.get("history_frames", 5))
    cosmos_cfg["future_frames"] = int(data_cfg.get("future_frames", 16))
    return cosmos_cfg, data_cfg


def _feature_cache_spec(cfg: dict, extractor: CosmosDenoiseFeatureExtractor, n: int) -> tuple[tuple[int, ...], np.dtype, dict]:
    data_cfg = cfg["data"]
    cosmos_cfg = dict(cfg["cosmos"])
    sources = extractor.source_names
    layers = extractor.layers
    cond_latent_frames = extractor._latent_frame_count(int(data_cfg.get("history_frames", 5)))
    total_latent_frames = extractor._latent_frame_count(
        int(data_cfg.get("history_frames", 5)) + int(data_cfg.get("future_frames", 16))
    )
    future_latent_frames = total_latent_frames - cond_latent_frames
    if future_latent_frames <= 0:
        raise ValueError("future_frames did not create any future latent slots.")
    if extractor.feature_pool == "future_steps":
        feature_shape = (n, len(sources), len(layers), future_latent_frames, extractor.hidden_size)
    else:
        feature_shape = (n, len(sources), len(layers), extractor.hidden_size)
    dtype = np.float16 if str(cosmos_cfg.get("save_dtype", "float16")).lower() in {"fp16", "float16"} else np.float32
    meta = {
        "shape": list(feature_shape),
        "dtype": str(dtype),
        "sources": sources,
        "layers": layers,
        "hidden_size": extractor.hidden_size,
        "feature_pool": extractor.feature_pool,
        "num_inference_steps": extractor.num_inference_steps,
        "capture_steps": extractor.capture_steps,
        "temporal_factor": extractor.temporal_factor,
        "cond_latent_frames": cond_latent_frames,
        "total_latent_frames": total_latent_frames,
        "future_latent_frames": future_latent_frames,
        "notes": "raw_no_denoise uses clean history latents only; denoise_tau=* uses clean history condition latents plus random future latent slots. No future pixels are fed to Cosmos.",
    }
    return feature_shape, dtype, meta


def _prepare_feature_cache(cfg: dict, run_dir: Path) -> tuple[Path, Path, tuple[int, ...], np.dtype]:
    index_path = run_dir / "slice_index.parquet"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing slice index: {index_path}")
    n = len(pd.read_parquet(index_path, columns=["row_id"]))
    cosmos_cfg, _data_cfg = _build_cosmos_cfg(cfg)
    extractor = CosmosDenoiseFeatureExtractor(cosmos_cfg)
    try:
        feature_shape, dtype, meta = _feature_cache_spec(cfg, extractor, n)
    finally:
        extractor.close()

    feature_path = run_dir / "features.npy"
    features = np.lib.format.open_memmap(feature_path, mode="w+", dtype=dtype, shape=feature_shape)
    features.flush()
    meta["feature_path"] = str(feature_path)
    meta_path = run_dir / "feature_meta.json"
    dump_json(meta, meta_path)
    return feature_path, meta_path, feature_shape, dtype


def extract_feature_cache(
    cfg: dict,
    run_dir: Path,
    *,
    num_shards: int = 1,
    shard_id: int = 0,
    init_only: bool = False,
) -> tuple[Path, Path]:
    num_shards = int(num_shards)
    shard_id = int(shard_id)
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if not 0 <= shard_id < num_shards:
        raise ValueError(f"shard_id must be in [0, {num_shards}), got {shard_id}")

    index_path = run_dir / "slice_index.parquet"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing slice index: {index_path}")
    index = pd.read_parquet(index_path).sort_values(["task", "episode_index", "t"]).reset_index(drop=True)

    feature_path = run_dir / "features.npy"
    meta_path = run_dir / "feature_meta.json"
    should_initialize = init_only or num_shards == 1 or not feature_path.exists() or not meta_path.exists()
    if should_initialize:
        _feature_path, _meta_path, feature_shape, dtype = _prepare_feature_cache(cfg, run_dir)
        if init_only:
            return _feature_path, _meta_path
    else:
        meta = load_json(meta_path)
        feature_shape = tuple(int(x) for x in meta["shape"])
        dtype = np.dtype("float16" if "float16" in str(meta.get("dtype", "float16")) else "float32")

    cosmos_cfg, data_cfg = _build_cosmos_cfg(cfg)
    features = np.load(feature_path, mmap_mode="r+")
    if tuple(features.shape) != tuple(feature_shape):
        raise ValueError(f"Feature shape mismatch: {features.shape} vs {feature_shape}")

    extractor = CosmosDenoiseFeatureExtractor(cosmos_cfg)
    seed_base = int(cfg.get("run", {}).get("seed", 42))
    current_video_path: str | None = None
    current_frames = None
    shard_rows = [(out_i, row) for out_i, row in index.iterrows() if out_i % num_shards == shard_id]

    try:
        desc = "Extracting Cosmos features" if num_shards == 1 else f"Extracting Cosmos features shard {shard_id + 1}/{num_shards}"
        for out_i, row in tqdm(shard_rows, total=len(shard_rows), desc=desc):
            video_path = str(row["video_path"])
            if video_path != current_video_path:
                current_frames = read_video_frames(video_path)
                current_video_path = video_path
            assert current_frames is not None
            history = select_history_frames(
                current_frames,
                int(row["t"]),
                int(data_cfg.get("history_frames", 5)),
            )
            row_seed = seed_base * 1_000_003 + int(row["row_id"])
            rep = extractor.extract_one(history, str(row["prompt"]), seed=row_seed)
            features[out_i] = rep.astype(dtype, copy=False)
    finally:
        extractor.close()

    features.flush()
    return feature_path, meta_path

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frozen Cosmos raw/denoise hidden features.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--init-only", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("run", {}).get("seed", 42)) + int(args.shard_id))
    run_dir = ensure_run_dir(cfg, run_id=args.run_id)
    feature_path, meta_path = extract_feature_cache(
        cfg,
        run_dir,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        init_only=args.init_only,
    )
    print(f"features={feature_path}")
    print(f"feature_meta={meta_path}")


if __name__ == "__main__":
    main()
