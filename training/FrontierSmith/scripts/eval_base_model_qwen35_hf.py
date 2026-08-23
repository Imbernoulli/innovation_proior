#!/usr/bin/env python3
"""Evaluate Qwen3.5 base models with Transformers generation.

This is the fallback evaluator for Qwen3.5 models when the installed vLLM
does not implement Qwen3_5ForConditionalGeneration. It keeps the VERL
validation metric convention by calling process_validation_metrics on the
scored samples.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import sys
import time
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "verl"))
sys.path.insert(0, str(PROJECT_ROOT / "ALE-Bench" / "src"))

from verl.trainer.ppo.metric_utils import process_validation_metrics
from verl.utils.reward_score import alebench, frontiercs


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _as_messages(prompt: Any) -> list[dict[str, str]]:
    if isinstance(prompt, np.ndarray):
        prompt = prompt.tolist()
    if isinstance(prompt, str):
        prompt = ast.literal_eval(prompt) if prompt.startswith("[") else [{"role": "user", "content": prompt}]
    messages = []
    for msg in prompt:
        if isinstance(msg, dict):
            messages.append({"role": str(msg["role"]), "content": str(msg["content"])})
        else:
            messages.append({"role": str(msg["role"]), "content": str(msg["content"])})
    return messages


def _ground_truth(row: pd.Series) -> str:
    reward_model = row.get("reward_model", {})
    if isinstance(reward_model, dict):
        return str(reward_model.get("ground_truth", ""))
    return str(row.get("ground_truth", ""))


def _load_problems(path: Path, source: str, limit: int | None) -> list[dict[str, Any]]:
    df = pd.read_parquet(path)
    problems: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        data_source = str(row.get("data_source", source))
        if data_source != source:
            data_source = source
        problems.append(
            {
                "data_source": data_source,
                "ground_truth": _ground_truth(row),
                "messages": _as_messages(row["prompt"]),
            }
        )
    if limit is not None:
        problems = problems[:limit]
    return problems


def _load_existing(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    records: dict[tuple[str, str, int], dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (str(rec["data_source"]), str(rec["ground_truth"]), int(rec["sample_idx"]))
            records[key] = rec
    return records


def _first_device(model: torch.nn.Module) -> torch.device:
    for param in model.parameters():
        return param.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _dtype(name: str) -> torch.dtype | str:
    match name:
        case "bfloat16":
            return torch.bfloat16
        case "float16":
            return torch.float16
        case "float32":
            return torch.float32
        case "auto":
            return "auto"
        case _:
            raise ValueError(f"Unsupported dtype: {name}")


def _tokenize_prompt(tokenizer: Any, messages: list[dict[str, str]], enable_thinking: bool | None) -> torch.Tensor:
    kwargs: dict[str, Any] = {"add_generation_prompt": True, "tokenize": True, "return_tensors": "pt"}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    input_ids = tokenizer.apply_chat_template(messages, **kwargs)
    if isinstance(input_ids, Mapping):
        input_ids = input_ids["input_ids"]
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor([input_ids], dtype=torch.long)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    return input_ids


def _generate_one(
    model: torch.nn.Module,
    tokenizer: Any,
    messages: list[dict[str, str]],
    args: argparse.Namespace,
) -> tuple[str, int, int, float]:
    prompt_ids = _tokenize_prompt(tokenizer, messages, args.enable_thinking)
    prompt_len = int(prompt_ids.shape[-1])
    if prompt_len > args.max_prompt_length:
        raise ValueError(f"Prompt has {prompt_len} tokens, above --max-prompt-length={args.max_prompt_length}")

    device = _first_device(model)
    prompt_ids = prompt_ids.to(device)
    attention_mask = torch.ones_like(prompt_ids, device=device)

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_token_id = tokenizer.eos_token_id
    gen_kwargs: dict[str, Any] = {
        "input_ids": prompt_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_token_id,
        "use_cache": True,
    }
    if args.do_sample:
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_p"] = args.top_p
        gen_kwargs["top_k"] = args.top_k

    start = time.time()
    with torch.inference_mode():
        output_ids = model.generate(**gen_kwargs)
    elapsed = time.time() - start

    completion_ids = output_ids[0, prompt_len:]
    text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    return text, prompt_len, int(completion_ids.shape[-1]), elapsed


def _score(data_source: str, text: str, ground_truth: str, judge_url: str) -> dict[str, float | None]:
    if data_source == "frontiercs":
        score = float(frontiercs.compute_score("frontiercs", text, ground_truth, judge_url=judge_url))
        return {"reward": score, "score": score}
    if data_source == "alebench":
        res = alebench.compute_score("alebench", text, ground_truth)
        performance = float(res.get("performance") or 0.0)
        return {
            "reward": float(res.get("score") or performance),
            "score": float(res.get("score") or performance),
            "performance": performance,
            "rank": None if res.get("rank") is None else float(res["rank"]),
            "overall_absolute_score": float(res.get("overall_absolute_score") or 0.0),
            "overall_relative_score": None
            if res.get("overall_relative_score") is None
            else float(res["overall_relative_score"]),
        }
    raise ValueError(f"Unsupported data_source: {data_source}")


def _summarize(records: list[dict[str, Any]], n_samples: int, seed: int) -> dict[str, Any]:
    complete_records = []
    grouped: dict[tuple[str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for rec in records:
        grouped[(rec["data_source"], rec["ground_truth"])][int(rec["sample_idx"])] = rec

    complete_problem_count = 0
    for (_source, _uid), sample_map in sorted(grouped.items()):
        if all(i in sample_map for i in range(n_samples)):
            complete_problem_count += 1
            complete_records.extend(sample_map[i] for i in range(n_samples))

    if not complete_records:
        return {"complete_problem_count": 0, "metrics": {}, "oracle_best": {}}

    metric_names = sorted({k for rec in complete_records for k in rec.get("metrics", {}).keys()})
    infos: dict[str, list[Any]] = {name: [] for name in metric_names}
    data_sources: list[str] = []
    sample_uids: list[str] = []
    for rec in complete_records:
        data_sources.append(str(rec["data_source"]))
        sample_uids.append(str(rec["ground_truth"]))
        for name in metric_names:
            infos[name].append(rec.get("metrics", {}).get(name))

    metrics = process_validation_metrics(data_sources, sample_uids, infos, seed=seed)

    oracle_best: dict[str, dict[str, float]] = defaultdict(dict)
    by_source_metric: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (source, _uid), sample_map in grouped.items():
        if not all(i in sample_map for i in range(n_samples)):
            continue
        per_metric: dict[str, list[float]] = defaultdict(list)
        for i in range(n_samples):
            for name, val in sample_map[i].get("metrics", {}).items():
                if val is not None and isinstance(val, (int, float)):
                    per_metric[name].append(float(val))
        for name, vals in per_metric.items():
            if len(vals) == n_samples:
                best_val = min(vals) if name == "rank" else max(vals)
                by_source_metric[source][name].append(best_val)
    for source, metric2vals in by_source_metric.items():
        for name, vals in metric2vals.items():
            oracle_best[source][f"{name}/oracle_best@{n_samples}"] = float(np.mean(vals))

    return {
        "complete_problem_count": complete_problem_count,
        "scored_sample_count": len(complete_records),
        "metrics": metrics,
        "oracle_best": oracle_best,
    }


def _print_metrics(summary: dict[str, Any]) -> None:
    metrics = summary.get("metrics", {})
    for source in sorted(metrics):
        for var_name in sorted(metrics[source]):
            for metric_name in sorted(metrics[source][var_name]):
                value = metrics[source][var_name][metric_name]
                print(f"METRIC val-core/{source}/{var_name}/{metric_name}: {value:.6f}", flush=True)
    for source in sorted(summary.get("oracle_best", {})):
        for metric_name in sorted(summary["oracle_best"][source]):
            value = summary["oracle_best"][source][metric_name]
            print(f"METRIC val-core/{source}/{metric_name}: {value:.6f}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "Qwen3.5-9B")
    parser.add_argument("--frontiercs-data", type=Path, default=PROJECT_ROOT / "data" / "frontiercs" / "full.parquet")
    parser.add_argument("--alebench-data", type=Path, default=PROJECT_ROOT / "data" / "alebench" / "val.parquet")
    parser.add_argument("--source", choices=["both", "frontiercs", "alebench"], default="both")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "base_eval_qwen35_9b_hf")
    parser.add_argument("--samples-jsonl", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-frontiercs", type=int, default=None)
    parser.add_argument("--limit-alebench", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-idx", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=16000)
    parser.add_argument("--max-prompt-length", type=int, default=10240)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32", "auto"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--model-loader",
        choices=["image-text-to-text", "causal-lm", "auto"],
        default="image-text-to-text",
        help="Qwen3.5 config maps to Qwen3_5ForConditionalGeneration through AutoModelForImageTextToText.",
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--judge-url", default=os.environ.get("FRONTIERCS_JUDGE_URL", "http://127.0.0.1:8082"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = args.samples_jsonl or (args.output_dir / "samples.jsonl")
    summary_path = args.summary_json or (args.output_dir / "summary.json")

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_idx < args.num_shards:
        raise ValueError("--shard-idx must satisfy 0 <= shard_idx < num_shards")

    problems: list[dict[str, Any]] = []
    if args.source in ("both", "frontiercs"):
        problems.extend(_load_problems(args.frontiercs_data, "frontiercs", args.limit_frontiercs))
    if args.source in ("both", "alebench"):
        problems.extend(_load_problems(args.alebench_data, "alebench", args.limit_alebench))

    total_problem_count = len(problems)
    if args.num_shards > 1:
        problems = [problem for idx, problem in enumerate(problems) if idx % args.num_shards == args.shard_idx]

    print(f"Loaded {len(problems)} problems", flush=True)
    if args.num_shards > 1:
        print(
            f"Shard {args.shard_idx}/{args.num_shards}: {len(problems)} of {total_problem_count} total problems",
            flush=True,
        )
    print(
        f"Sources: frontiercs={sum(p['data_source'] == 'frontiercs' for p in problems)}, "
        f"alebench={sum(p['data_source'] == 'alebench' for p in problems)}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt_lengths = [
        int(_tokenize_prompt(tokenizer, problem["messages"], args.enable_thinking).shape[-1]) for problem in problems
    ]
    if prompt_lengths:
        print(
            "Prompt tokens: "
            f"max={max(prompt_lengths)} p95={np.percentile(prompt_lengths, 95):.1f} "
            f"mean={np.mean(prompt_lengths):.1f}",
            flush=True,
        )
    if args.dry_run:
        print("Dry run complete; model was not loaded.", flush=True)
        return

    print(f"Loading model from {args.model}", flush=True)
    model_kwargs: dict[str, Any] = {
        "dtype": _dtype(args.dtype),
        "device_map": args.device_map,
        "trust_remote_code": True,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    loader_cls = {
        "image-text-to-text": AutoModelForImageTextToText,
        "causal-lm": AutoModelForCausalLM,
        "auto": AutoModel,
    }[args.model_loader]
    model = loader_cls.from_pretrained(args.model, **model_kwargs)
    model.eval()
    print(f"Model loaded. First parameter device: {_first_device(model)}", flush=True)

    planned_keys = {
        (problem["data_source"], problem["ground_truth"], sample_idx)
        for problem in problems
        for sample_idx in range(args.n_samples)
    }
    existing = _load_existing(samples_path) if args.resume else {}
    existing = {key: rec for key, rec in existing.items() if key in planned_keys}
    if existing:
        print(f"Resuming from {samples_path}: {len(existing)} completed samples", flush=True)

    records = dict(existing)
    total = len(problems) * args.n_samples
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    with samples_path.open(mode, encoding="utf-8") as out:
        pbar = tqdm(total=total, initial=len(existing), desc="Eval", unit="sample")
        for problem_idx, problem in enumerate(problems):
            for sample_idx in range(args.n_samples):
                key = (problem["data_source"], problem["ground_truth"], sample_idx)
                if key in records:
                    continue
                started = time.time()
                try:
                    text, prompt_len, completion_len, gen_seconds = _generate_one(model, tokenizer, problem["messages"], args)
                    metrics = _score(problem["data_source"], text, problem["ground_truth"], args.judge_url)
                    error = None
                except Exception as exc:
                    text = ""
                    prompt_len = 0
                    completion_len = 0
                    gen_seconds = 0.0
                    metrics = {"reward": 0.0, "score": 0.0}
                    error = repr(exc)
                    print(
                        f"ERROR {problem['data_source']} {problem['ground_truth']} sample {sample_idx}: {error}",
                        flush=True,
                    )

                rec = {
                    "data_source": problem["data_source"],
                    "ground_truth": problem["ground_truth"],
                    "problem_idx": problem_idx,
                    "sample_idx": sample_idx,
                    "prompt_tokens": prompt_len,
                    "completion_tokens": completion_len,
                    "generation_seconds": gen_seconds,
                    "total_seconds": time.time() - started,
                    "metrics": metrics,
                    "error": error,
                    "text": text,
                }
                out.write(json.dumps(rec, ensure_ascii=False, default=_json_default) + "\n")
                out.flush()
                records[key] = rec
                reward = metrics.get("reward")
                reward_str = "NA" if reward is None else f"{float(reward):.4f}"
                print(
                    f"SAMPLE {len(records)}/{total} {problem['data_source']} "
                    f"{problem['ground_truth']} sample={sample_idx} reward={reward_str} "
                    f"gen_tokens={completion_len} gen_sec={gen_seconds:.1f}",
                    flush=True,
                )
                pbar.update(1)
        pbar.close()

    summary = _summarize(list(records.values()), args.n_samples, args.seed)
    summary["config"] = vars(args)
    summary["samples_jsonl"] = str(samples_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default) + "\n")
    print(f"Saved samples to {samples_path}", flush=True)
    print(f"Saved summary to {summary_path}", flush=True)
    _print_metrics(summary)


if __name__ == "__main__":
    main()
