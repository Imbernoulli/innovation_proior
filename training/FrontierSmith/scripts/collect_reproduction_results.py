#!/usr/bin/env python3
"""Collect FrontierSmith reproduction eval summaries into one CSV.

The script is intentionally tolerant of missing summaries so it can be run
while the Slurm pipeline is still pending/running.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ExpectedResult:
    label: str
    stage: str
    model_path: str
    summary_json: Path


def expected_results() -> list[ExpectedResult]:
    rows = [
        ExpectedResult(
            "qwen35_9b_base",
            "base_eval_reference",
            "models/Qwen3.5-9B",
            PROJECT_ROOT / "outputs/base_eval_qwen35_9b_vllm/summary.json",
        ),
        ExpectedResult(
            "qwen35_9b_base_model",
            "base_eval_reference",
            "models/Qwen3.5-9B-Base",
            PROJECT_ROOT / "outputs/base_eval_qwen35_9b_base_vllm/summary.json",
        ),
        ExpectedResult(
            "qwen35_9b_mixed_no_thinking",
            "trained_eval_reference",
            "models/cc_qwen35_9b_mixed_hf_fixed",
            PROJECT_ROOT / "outputs/cc_eval_qwen35_9b_mixed_fixed_vllm/summary.json",
        ),
        ExpectedResult(
            "qwen35_9b_mixed_thinking",
            "trained_eval_reference",
            "models/cc_qwen35_9b_mixed_hf_fixed",
            PROJECT_ROOT / "outputs/cc_eval_qwen35_9b_mixed_thinking_general_32k_both_vllm/summary.json",
        ),
        ExpectedResult(
            "qwen35_9b_mixed_step20_thinking",
            "trained_eval",
            "models/cc_qwen35_9b_mixed_hf_step20",
            PROJECT_ROOT / "outputs/cc_eval_qwen35_9b_mixed_step20_thinking_general_32k_both_vllm/summary.json",
        ),
        ExpectedResult(
            "qwen3_8b_mixed_public",
            "trained_eval",
            "models/qwen3_8b_mixed_public_hf",
            PROJECT_ROOT / "outputs/eval_qwen3_8b_mixed_public_thinking_general_32k_both_vllm/summary.json",
        ),
        ExpectedResult(
            "qwen3_8b_mixed_public_step25",
            "trained_eval_fallback",
            "models/qwen3_8b_mixed_public_step25_hf",
            PROJECT_ROOT
            / "outputs/eval_qwen3_8b_mixed_public_step25_thinking_general_32k_both_vllm/summary.json",
        ),
    ]

    for tag, model_path in [
        ("qwen3_8b", "models/Qwen3-8B"),
        ("qwen3_8b_base", "models/Qwen3-8B-Base"),
        ("qwen3_8b_alpha0p25", "models/Qwen3-8B-linear-alpha-0p25"),
        ("qwen3_8b_alpha0p50", "models/Qwen3-8B-linear-alpha-0p50"),
        ("qwen3_8b_alpha0p75", "models/Qwen3-8B-linear-alpha-0p75"),
    ]:
        rows.append(
            ExpectedResult(
                tag,
                "base_eval",
                model_path,
                PROJECT_ROOT / f"outputs/eval_{tag}_thinking_general_both_vllm/summary.json",
            )
        )

    for tag, stage in [
        ("qwen3_8b_base", "trained_eval"),
        ("qwen3_8b_alpha0p25", "trained_eval_invalid_port_collision"),
        ("qwen3_8b_alpha0p50", "trained_eval"),
        ("qwen3_8b_alpha0p75", "trained_eval"),
    ]:
        rows.append(
            ExpectedResult(
                f"{tag}_mixed_public",
                stage,
                f"models/{tag}_mixed_public_hf",
                PROJECT_ROOT / f"outputs/eval_{tag}_mixed_public_thinking_general_both_vllm/summary.json",
            )
        )

    rows.append(
        ExpectedResult(
            "qwen3_8b_alpha0p25_mixed_public_step25",
            "trained_eval_fallback",
            "models/qwen3_8b_alpha0p25_mixed_public_step25_hf",
            PROJECT_ROOT
            / "outputs/eval_qwen3_8b_alpha0p25_mixed_public_step25_thinking_general_32k_both_vllm/summary.json",
        )
    )

    rows.append(
        ExpectedResult(
            "qwen3_8b_alpha0p25_mixed_public_step30",
            "trained_eval_fallback",
            "models/qwen3_8b_alpha0p25_mixed_public_step30_hf",
            PROJECT_ROOT
            / "outputs/eval_qwen3_8b_alpha0p25_mixed_public_step30_thinking_general_32k_both_vllm/summary.json",
        )
    )

    rows.append(
        ExpectedResult(
            "qwen3_8b_alpha0p50_mixed_public_step25",
            "trained_eval_fallback",
            "models/qwen3_8b_alpha0p50_mixed_public_step25_hf",
            PROJECT_ROOT
            / "outputs/eval_qwen3_8b_alpha0p50_mixed_public_step25_thinking_general_32k_both_vllm/summary.json",
        )
    )

    rows.append(
        ExpectedResult(
            "qwen3_8b_alpha0p50_mixed_public_step30",
            "trained_eval_fallback",
            "models/qwen3_8b_alpha0p50_mixed_public_step30_hf",
            PROJECT_ROOT
            / "outputs/eval_qwen3_8b_alpha0p50_mixed_public_step30_thinking_general_32k_both_vllm/summary.json",
        )
    )

    rows.append(
        ExpectedResult(
            "qwen3_8b_alpha0p75_mixed_public_step25",
            "trained_eval_fallback",
            "models/qwen3_8b_alpha0p75_mixed_public_step25_hf",
            PROJECT_ROOT
            / "outputs/eval_qwen3_8b_alpha0p75_mixed_public_step25_thinking_general_32k_both_vllm/summary.json",
        )
    )

    rows.append(
        ExpectedResult(
            "qwen3_8b_alpha0p75_mixed_public_step30",
            "trained_eval_fallback",
            "models/qwen3_8b_alpha0p75_mixed_public_step30_hf",
            PROJECT_ROOT
            / "outputs/eval_qwen3_8b_alpha0p75_mixed_public_step30_thinking_general_32k_both_vllm/summary.json",
        )
    )

    return rows


def get_nested(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def infer_n_samples(data: dict[str, Any]) -> str:
    metric_keys: set[str] = set()
    for source_metrics in data.get("metrics", {}).values():
        if not isinstance(source_metrics, dict):
            continue
        for metric_values in source_metrics.values():
            if isinstance(metric_values, dict):
                metric_keys.update(str(key) for key in metric_values)
    candidates = []
    for key in metric_keys:
        match = re.fullmatch(r"(?:mean|best|worst)@(\d+)(?:/.*)?", key)
        if match:
            candidates.append(int(match.group(1)))
    return str(max(candidates)) if candidates else ""


def infer_source(data: dict[str, Any]) -> str:
    sources = sorted(data.get("metrics", {}))
    if not sources:
        return ""
    return "both" if len(sources) > 1 else sources[0]


def samples_field(data: dict[str, Any]) -> str:
    if data.get("samples_jsonl"):
        return str(data["samples_jsonl"])
    files = data.get("samples_jsonl_files")
    if isinstance(files, list):
        return ";".join(str(path) for path in files)
    return ""


def summarize(item: ExpectedResult) -> dict[str, Any]:
    row: dict[str, Any] = {
        "label": item.label,
        "stage": item.stage,
        "model_path": item.model_path,
        "summary_json": str(item.summary_json.relative_to(PROJECT_ROOT)),
        "exists": item.summary_json.is_file(),
    }
    if not item.summary_json.is_file():
        partial_count, partial_errors = count_partial_samples(item.summary_json.parent)
        row["partial_sample_count"] = partial_count
        row["partial_error_count"] = partial_errors
        return row

    data = json.loads(item.summary_json.read_text(encoding="utf-8"))
    cfg = data.get("config", {})
    row.update(
        {
            "source": cfg.get("source", "") or infer_source(data),
            "served_model": cfg.get("model", ""),
            "n_samples": cfg.get("n_samples", "") or infer_n_samples(data),
            "complete_problem_count": data.get("complete_problem_count", ""),
            "scored_sample_count": data.get("scored_sample_count", ""),
            "samples_jsonl": samples_field(data),
            "frontiercs_reward_mean@5": get_nested(data, ("metrics", "frontiercs", "reward", "mean@5")),
            "frontiercs_reward_best@5_mean": get_nested(
                data, ("metrics", "frontiercs", "reward", "best@5/mean")
            ),
            "frontiercs_reward_oracle_best@5": get_nested(
                data, ("oracle_best", "frontiercs", "reward/oracle_best@5")
            ),
            "alebench_performance_mean@5": get_nested(data, ("metrics", "alebench", "performance", "mean@5")),
            "alebench_performance_best@5_mean": get_nested(
                data, ("metrics", "alebench", "performance", "best@5/mean")
            ),
            "alebench_performance_oracle_best@5": get_nested(
                data, ("oracle_best", "alebench", "performance/oracle_best@5")
            ),
        }
    )
    return row


def count_partial_samples(output_dir: Path) -> tuple[int, int]:
    paths = sorted(output_dir.glob("samples.jsonl"))
    paths.extend(sorted(output_dir.glob("shards/shard_*/samples.jsonl")))
    seen: set[tuple[str, str, int]] = set()
    errors = 0
    for path in paths:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                key = (str(rec["data_source"]), str(rec["ground_truth"]), int(rec["sample_idx"]))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                errors += 1
                continue
            seen.add(key)
            if rec.get("error"):
                errors += 1
    return len(seen), errors


def write_table(rows: list[dict[str, Any]]) -> None:
    fields = [
        "label",
        "stage",
        "exists",
        "complete_problem_count",
        "partial_sample_count",
        "partial_error_count",
        "frontiercs_reward_best@5_mean",
        "alebench_performance_best@5_mean",
    ]
    widths = {field: max(len(field), *(len(fmt(row.get(field))) for row in rows)) for field in fields}
    print("  ".join(field.ljust(widths[field]) for field in fields))
    print("  ".join("-" * widths[field] for field in fields))
    for row in rows:
        print("  ".join(fmt(row.get(field)).ljust(widths[field]) for field in fields))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-csv", type=Path, default=PROJECT_ROOT / "outputs/reproduction_results.csv")
    parser.add_argument("--no-table", action="store_true")
    args = parser.parse_args()

    rows = [summarize(item) for item in expected_results()]
    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "label",
        "stage",
        "exists",
        "model_path",
        "summary_json",
        "source",
        "served_model",
        "n_samples",
        "complete_problem_count",
        "scored_sample_count",
        "partial_sample_count",
        "partial_error_count",
        "frontiercs_reward_mean@5",
        "frontiercs_reward_best@5_mean",
        "frontiercs_reward_oracle_best@5",
        "alebench_performance_mean@5",
        "alebench_performance_best@5_mean",
        "alebench_performance_oracle_best@5",
        "samples_jsonl",
    ]
    fieldnames = preferred + [field for field in fieldnames if field not in preferred]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row.get(key)) for key in fieldnames})
    print(f"Wrote {args.output_csv}")

    if not args.no_table:
        write_table(rows)


if __name__ == "__main__":
    main()
