#!/usr/bin/env python3
"""Extract rollout and async/RTC timing evidence from a DOMINO evaluation log."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
FIELD_RE = re.compile(r"([a-zA-Z_]+)=([^ ]+)")
SUCCESS_RE = re.compile(r"Success rate:\s*(\d+)/(\d+)\s*=>\s*([0-9.]+)%")


def numeric(rows: list[dict[str, str]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        value = row.get(key)
        if value not in (None, "NA", "None", "none"):
            try:
                values.append(float(value))
            except ValueError:
                pass
    return np.asarray(values, dtype=np.float64)


def stats(values: np.ndarray) -> dict[str, float | int] | None:
    if values.size == 0:
        return None
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    text = ANSI_RE.sub("", args.log.read_text(encoding="utf-8", errors="replace"))
    timing_rows = []
    success = None
    for line in text.splitlines():
        if "[TIMING_EXEC]" in line:
            timing_rows.append(dict(FIELD_RE.findall(line)))
        match = SUCCESS_RE.search(line)
        if match:
            success = {
                "successes": int(match.group(1)),
                "episodes": int(match.group(2)),
                "success_rate_percent": float(match.group(3)),
            }

    activated = sum(row.get("async_result_status") == "activated" for row in timing_rows)
    rtc_applied = sum(row.get("rtc_applied") == "1" for row in timing_rows)
    sync_block = sum(row.get("sync_block") == "1" for row in timing_rows)
    result = {
        "source_log": args.log.name,
        "rollout": success,
        "timing_rows": len(timing_rows),
        "async_enabled_rows": sum(row.get("async_inference") == "1" for row in timing_rows),
        "async_activations": activated,
        "rtc_applied_rows": rtc_applied,
        "sync_block_rows": sync_block,
        "timing_ms": {
            key: stats(numeric(timing_rows, key))
            for key in (
                "eval_total_ms",
                "websocket_roundtrip_ms",
                "server_total_ms",
                "server_model_inference_ms",
                "server_action_head_ms",
                "chunk_request_interval_ms",
            )
        },
        "rtc_inference_delay_steps": stats(numeric(timing_rows, "rtc_inference_delay_steps")),
        "interpretation_limit": (
            "Characterizes one asynchronous run. A matched synchronous run is required to "
            "attribute a latency or success-rate change to asynchronous execution."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
