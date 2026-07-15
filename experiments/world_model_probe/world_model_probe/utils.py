from __future__ import annotations

import importlib
import json
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def torch_dtype(name: str | None) -> torch.dtype:
    if name is None:
        return torch.float32
    value = str(name).lower()
    if value in {"fp32", "float32"}:
        return torch.float32
    if value in {"fp16", "float16"}:
        return torch.float16
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported torch dtype: {name}")


def import_object(path: str) -> Any:
    if ":" in path:
        module_name, attr = path.split(":", 1)
    else:
        module_name, attr = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    obj: Any = module
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_custom_callable(path: str | None) -> Callable[..., Any] | None:
    if not path:
        return None
    return import_object(path)

