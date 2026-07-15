from __future__ import annotations

import argparse
from typing import Any

import torch

from world_model_probe.backbones import build_backbone
from world_model_probe.config import apply_overrides, load_config
from world_model_probe.data.base import ProbeSample, resolve_data_adapter
from world_model_probe.utils import append_jsonl, ensure_dir, seed_everything, torch_dtype, write_json


def _fmt(path: str, cfg: dict[str, Any]) -> str:
    return path.format(backbone=cfg["backbone"]["name"], run_name=cfg["project"]["run_name"])


def _select_tokens(tokens: torch.Tensor, max_tokens: int | None) -> torch.Tensor:
    if max_tokens is None or tokens.shape[0] <= max_tokens:
        return tokens
    idx = torch.linspace(0, tokens.shape[0] - 1, steps=max_tokens).round().long()
    return tokens.index_select(0, idx)


def _format_prompt(template: str, semantic: str, metadata: dict[str, Any]) -> str:
    fields = dict(metadata)
    fields.update(
        {
            "task": semantic,
            "instruction": semantic,
            "semantic": semantic,
        }
    )
    return template.format(**fields)


def _sample_metadata(sample: ProbeSample, split: str, semantic: str, prompt: str) -> dict[str, Any]:
    metadata = dict(sample.backbone_input.metadata)
    metadata.update(sample.metadata)
    metadata.setdefault("task", semantic)
    metadata.setdefault("instruction", semantic)
    metadata.update(
        {
            "sample_id": sample.sample_id,
            "semantic": semantic,
            "prompt": prompt,
            "split": split,
        }
    )
    return metadata


def cache_split(cfg: dict[str, Any], split: str, limit: int | None, overwrite: bool) -> int:
    data_cfg = cfg["data"]
    backbone_cfg = cfg["backbone"]
    latent_root = ensure_dir(_fmt(cfg["paths"]["latent_root"], cfg))
    split_dir = ensure_dir(latent_root / split)
    index_path = latent_root / f"index_{split}.jsonl"
    if index_path.exists():
        index_path.unlink()

    backbone = build_backbone(cfg)
    iter_samples = resolve_data_adapter(cfg)
    max_tokens = backbone_cfg.get("max_tokens")
    max_tokens = None if max_tokens is None else int(max_tokens)
    cache_dtype = torch_dtype(backbone_cfg.get("cache_dtype", "float16"))
    prompt_template = str(backbone_cfg.get("prompt_template", "{task}"))

    count = 0
    for sample in iter_samples(cfg, split):
        latent_path = split_dir / f"{sample.sample_id}.pt"
        semantic = sample.backbone_input.semantic
        merged_metadata = dict(sample.backbone_input.metadata)
        merged_metadata.update(sample.metadata)
        prompt = _format_prompt(prompt_template, semantic, merged_metadata)
        metadata = _sample_metadata(sample, split, semantic, prompt)
        if not latent_path.exists() or overwrite:
            tokens = backbone.extract_tokens(sample.backbone_input.frames, prompt=prompt)
            tokens = _select_tokens(tokens, max_tokens).to(dtype=cache_dtype)
            payload = {
                "tokens": tokens.cpu(),
                "targets": {k: torch.from_numpy(v).float() for k, v in sample.targets.items()},
                "valid": torch.from_numpy(sample.valid).float(),
                "metadata": metadata,
            }
            torch.save(payload, latent_path)

        row = {
            "latent_path": str(latent_path),
            "sample_id": sample.sample_id,
            "split": split,
            "valid": sample.valid.tolist(),
        }
        for key in ("episode_index", "frame_index", "task_index", "video_path", "parquet_path"):
            if key in metadata:
                row[key] = metadata[key]
        append_jsonl(index_path, row)
        count += 1
        if count % 100 == 0:
            print(f"[cache] split={split} cached/indexed {count} samples", flush=True)
        if limit is not None and count >= limit:
            break

    write_json(
        latent_root / f"meta_{split}.json",
        {
            "split": split,
            "num_samples": count,
            "index_path": str(index_path),
            "backbone": cfg["backbone"],
            "targets": cfg["targets"],
            "data": data_cfg,
        },
    )
    print(f"[cache] split={split} done: {count} samples -> {index_path}", flush=True)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache frozen world-model latents for DOM probing.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="all", choices=["all", "train", "eval"])
    parser.add_argument("--limit", type=int, default=None, help="Optional debug limit per selected split.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    seed_everything(int(cfg["project"].get("seed", 0)))
    splits = ["train", "eval"] if args.split == "all" else [args.split]
    for split in splits:
        cache_split(cfg, split, args.limit, args.overwrite)


if __name__ == "__main__":
    main()
