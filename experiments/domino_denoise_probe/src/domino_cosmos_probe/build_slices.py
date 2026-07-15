from __future__ import annotations

import argparse

from .config import ensure_run_dir, load_config, set_seed
from .data import build_slice_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DOMINO slice index and label cache.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("run", {}).get("seed", 42)))
    run_dir = ensure_run_dir(cfg, run_id=args.run_id)
    index_path, labels_path = build_slice_cache(cfg, run_dir)
    print(f"slice_index={index_path}")
    print(f"labels={labels_path}")


if __name__ == "__main__":
    main()
