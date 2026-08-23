#!/usr/bin/env python3
"""Summarize sharded HF base-eval JSONL files."""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "eval_base_model_qwen35_hf.py"


def _load_eval_module():
    spec = importlib.util.spec_from_file_location("eval_base_model_qwen35_hf", EVAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {EVAL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _expand_patterns(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(m) for m in matches)
        else:
            paths.append(Path(pattern))
    return sorted(set(paths))


def _load_records(paths: list[Path]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            print(f"Missing: {path}", file=sys.stderr)
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = (str(rec["data_source"]), str(rec["ground_truth"]), int(rec["sample_idx"]))
                by_key[key] = rec
    return list(by_key.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="*",
        default=[str(PROJECT_ROOT / "outputs" / "base_eval_qwen35_9b_hf" / "shards" / "shard_*" / "samples.jsonl")],
    )
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "base_eval_qwen35_9b_hf" / "summary.json",
    )
    args = parser.parse_args()

    module = _load_eval_module()
    paths = _expand_patterns(args.paths)
    print(f"Reading {len(paths)} JSONL file(s)")
    records = _load_records(paths)
    print(f"Loaded {len(records)} unique sample record(s)")

    summary = module._summarize(records, args.n_samples, args.seed)
    summary["samples_jsonl_files"] = [str(path) for path in paths]
    sources = sorted({str(rec.get("data_source", "")) for rec in records if rec.get("data_source")})
    summary["config"] = {
        "n_samples": args.n_samples,
        "seed": args.seed,
        "source": "both" if len(sources) > 1 else (sources[0] if sources else ""),
        "sources": sources,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=module._json_default) + "\n")
    print(f"Saved summary to {args.output_json}")
    module._print_metrics(summary)


if __name__ == "__main__":
    main()
