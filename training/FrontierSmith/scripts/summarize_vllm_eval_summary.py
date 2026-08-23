#!/usr/bin/env python3
"""Print compact metrics from a vLLM eval summary.json."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


def _get(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def summarize(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cfg = data.get("config", {})
    return {
        "summary_json": str(path),
        "source": cfg.get("source", ""),
        "model": cfg.get("model", ""),
        "n_samples": cfg.get("n_samples", ""),
        "complete_problem_count": data.get("complete_problem_count", ""),
        "scored_sample_count": data.get("scored_sample_count", ""),
        "frontiercs_reward_mean@5": _get(data, ("metrics", "frontiercs", "reward", "mean@5")),
        "frontiercs_score_mean@5": _get(data, ("metrics", "frontiercs", "score", "mean@5")),
        "frontiercs_reward_best@5_mean": _get(data, ("metrics", "frontiercs", "reward", "best@5/mean")),
        "frontiercs_reward_oracle_best@5": _get(data, ("oracle_best", "frontiercs", "reward/oracle_best@5")),
        "alebench_performance_mean@5": _get(data, ("metrics", "alebench", "performance", "mean@5")),
        "alebench_performance_best@5_mean": _get(data, ("metrics", "alebench", "performance", "best@5/mean")),
        "alebench_performance_oracle_best@5": _get(data, ("oracle_best", "alebench", "performance/oracle_best@5")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_json", nargs="+", type=Path)
    parser.add_argument("--format", choices=["table", "csv"], default="table")
    args = parser.parse_args()

    rows = [summarize(path) for path in args.summary_json]
    fields = list(rows[0])
    if args.format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        return

    for row in rows:
        print(row["summary_json"])
        for field in fields[1:]:
            print(f"  {field}: {_fmt(row[field])}")


if __name__ == "__main__":
    main()
