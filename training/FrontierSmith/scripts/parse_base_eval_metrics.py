#!/usr/bin/env python3
"""Extract Qwen3.5 base eval metrics from a VERL validation log."""

from __future__ import annotations

import re
import sys
from pathlib import Path

NUMBER = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"

METRICS = {
    "frontiercs_avg5": r"val-core/frontiercs/reward/mean@5['\"]?\s*:\s*" + NUMBER,
    "frontiercs_best5": r"val-core/frontiercs/reward/best@5/mean['\"]?\s*:\s*" + NUMBER,
    "alebench_perf_avg5": r"val-core/alebench/performance/mean@5['\"]?\s*:\s*" + NUMBER,
    "alebench_perf_best5": r"val-core/alebench/performance/best@5/mean['\"]?\s*:\s*" + NUMBER,
    "alebench_reward_avg5": r"val-core/alebench/reward/mean@5['\"]?\s*:\s*" + NUMBER,
    "alebench_reward_best5": r"val-core/alebench/reward/best@5/mean['\"]?\s*:\s*" + NUMBER,
}


def read_text(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    return Path(path).read_text(errors="replace")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: scripts/parse_base_eval_metrics.py <log-file-or->", file=sys.stderr)
        raise SystemExit(2)

    text = read_text(sys.argv[1])
    values = {}
    for name, pattern in METRICS.items():
        matches = re.findall(pattern, text)
        values[name] = matches[-1] if matches else ""

    print(",".join(values.keys()))
    print(",".join(values.values()))


if __name__ == "__main__":
    main()
