#!/usr/bin/env python3
"""Evaluate Qwen3.5 through a vLLM OpenAI-compatible server."""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OFFICIAL_FRONTIERCS_ROOT = PROJECT_ROOT / ".cache" / "Frontier-CS-official"
LOCAL_FRONTIERCS_ROOT = PROJECT_ROOT / "Frontier-CS"
if OFFICIAL_FRONTIERCS_ROOT.exists():
    sys.path.insert(0, str(OFFICIAL_FRONTIERCS_ROOT))
    sys.path.insert(0, str(OFFICIAL_FRONTIERCS_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "verl"))
sys.path.insert(0, str(PROJECT_ROOT / "ALE-Bench" / "src"))

from verl.trainer.ppo.metric_utils import process_validation_metrics
from verl.utils.reward_score import alebench, frontiercs

try:
    from algorithmic.scripts.generate_solutions import CPP_SYSTEM_PROMPT, extract_cpp_code as official_extract_cpp_code
    from frontier_cs.runner.algorithmic_local import AlgorithmicLocalRunner
except Exception as exc:  # pragma: no cover - surfaced at runtime with actionable error.
    CPP_SYSTEM_PROMPT = ""
    official_extract_cpp_code = None
    AlgorithmicLocalRunner = None
    _OFFICIAL_FRONTIERCS_IMPORT_ERROR = exc
else:
    _OFFICIAL_FRONTIERCS_IMPORT_ERROR = None


_THREAD_LOCAL = threading.local()


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
    messages: list[dict[str, str]] = []
    for msg in prompt:
        if isinstance(msg, Mapping):
            messages.append({"role": str(msg["role"]), "content": str(msg["content"])})
        else:
            messages.append({"role": str(msg["role"]), "content": str(msg["content"])})
    return messages


def _ground_truth(row: pd.Series) -> str:
    reward_model = row.get("reward_model", {})
    if isinstance(reward_model, Mapping):
        return str(reward_model.get("ground_truth", ""))
    return str(row.get("ground_truth", ""))


def _frontiercs_statement(problem_id: str) -> str:
    for root in (LOCAL_FRONTIERCS_ROOT, OFFICIAL_FRONTIERCS_ROOT):
        stmt = root / "algorithmic" / "problems" / str(problem_id) / "statement.txt"
        if stmt.is_file():
            return stmt.read_text(encoding="utf-8")
    raise FileNotFoundError(f"FrontierCS statement not found for problem {problem_id}")


INNOV_SYS_TMPL = "It is now year {year}. You are a good researcher."


def _researcher_system_message() -> dict[str, str] | None:
    """Innovation meta-conditioning system prompt with the year set to the PRESENT.
    Matches the time-conditioned system prompt the SFT data was trained on. Enabled
    by env EVAL_RESEARCHER_YEAR (no effect on pre-SFT/base models per design)."""
    year = os.environ.get("EVAL_RESEARCHER_YEAR", "").strip()
    return {"role": "system", "content": INNOV_SYS_TMPL.format(year=year)} if year else None


def _frontiercs_official_messages(problem_id: str) -> list[dict[str, str]]:
    statement = _frontiercs_statement(problem_id)
    prompt = f"{CPP_SYSTEM_PROMPT}\n\nProblem:\n\n{statement}\n\nGenerate solution code:"
    return [{"role": "user", "content": prompt}]


def _load_problems(
    path: Path,
    source: str,
    limit: int | None,
    *,
    frontiercs_prompt_source: str = "official",
) -> list[dict[str, Any]]:
    df = pd.read_parquet(path)
    problems: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        ground_truth = _ground_truth(row)
        if source == "frontiercs" and frontiercs_prompt_source == "official":
            messages = _frontiercs_official_messages(ground_truth)
            prompt_variant = "frontiercs:official-generate_solutions"
        else:
            messages = _as_messages(row["prompt"])
            prompt_variant = f"{source}:parquet"
        _sys = _researcher_system_message()
        if _sys is not None:
            if messages and messages[0].get("role") == "system":
                # already has a system prompt -> prepend the innovation conditioning to it
                messages = [{"role": "system", "content": _sys["content"] + "\n\n" + messages[0]["content"]}, *messages[1:]]
            else:
                messages = [_sys, *messages]
        problems.append(
            {
                "data_source": source,
                "ground_truth": ground_truth,
                "messages": messages,
                "prompt_variant": prompt_variant,
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
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = (str(rec["data_source"]), str(rec["ground_truth"]), int(rec["sample_idx"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                print(
                    f"WARNING: skipping invalid JSONL line {lineno} in {path}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            records[key] = rec
    return records


def _record_compatible(rec: dict[str, Any], args: argparse.Namespace) -> bool:
    data_source = str(rec.get("data_source", ""))
    if data_source == "frontiercs":
        expected_prompt = (
            "frontiercs:official-generate_solutions"
            if args.frontiercs_prompt_source == "official"
            else "frontiercs:parquet"
        )
        return rec.get("prompt_variant") == expected_prompt and rec.get("score_backend") == args.frontiercs_score_backend
    if data_source == "alebench":
        return rec.get("score_backend") in (None, "alebench:official-private-eval")
    return True


def _client(base_url: str, timeout: float):
    from openai import OpenAI

    cache_key = (base_url, timeout)
    client_cache = getattr(_THREAD_LOCAL, "client_cache", {})
    if cache_key not in client_cache:
        client_cache[cache_key] = OpenAI(base_url=base_url, api_key="dummy", timeout=timeout)
        _THREAD_LOCAL.client_cache = client_cache
    return client_cache[cache_key]


def _is_judge_infra_failure(res: Any) -> bool:
    """True if an EvaluationResult reflects a judge/infrastructure failure
    (judge unreachable, submission failed, or evaluation timed out) rather than a
    legitimate model-side 0 (code that failed to compile or failed test cases).

    Infra failures must be surfaced as errors, not silently scored 0, otherwise a
    down/misconfigured judge is indistinguishable from genuine model weakness.
    """
    status = getattr(res, "status", None)
    status_val = getattr(status, "value", status)
    if status_val == "timeout":
        return True
    if status_val != "error":
        return False
    message = (getattr(res, "message", None) or "").lower()
    # Legitimate model-side failures from the judge engine: the submitted code did
    # not compile, crashed, or was rejected. These are real 0s, keep them as 0.
    code_side_markers = (
        "compile failed",
        "compilation failed",
        "wrong answer",
        "runtime error",
        "time limit",
        "memory limit",
        "signalled",
        "non-zero exit",
        "empty code submission",
    )
    if any(marker in message for marker in code_side_markers):
        return False
    # Explicit infrastructure markers emitted by AlgorithmicLocalRunner.
    infra_markers = (
        "not available",
        "submission failed",
        "judge server",
        "connection",
        "unavailable",
    )
    if any(marker in message for marker in infra_markers):
        return True
    # Unknown 'error' with no recognizable code-side cause: treat as infra failure
    # so it is loud rather than silently zeroed.
    return True


def _frontiercs_runner(judge_url: str):
    if AlgorithmicLocalRunner is None:
        raise RuntimeError(
            "Could not import official FrontierCS AlgorithmicLocalRunner. "
            f"Original error: {_OFFICIAL_FRONTIERCS_IMPORT_ERROR!r}"
        )
    cache_key = judge_url.rstrip("/")
    runner_cache = getattr(_THREAD_LOCAL, "frontiercs_runner_cache", {})
    if cache_key not in runner_cache:
        runner_cache[cache_key] = AlgorithmicLocalRunner(
            judge_url=cache_key,
            base_dir=LOCAL_FRONTIERCS_ROOT,
            auto_start=False,
        )
        _THREAD_LOCAL.frontiercs_runner_cache = runner_cache
    return runner_cache[cache_key]


def _generate_one(
    args: argparse.Namespace,
    messages: list[dict[str, str]],
    request_seed: int | None,
) -> tuple[str, int | None, float]:
    kwargs: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    if args.presence_penalty is not None:
        kwargs["presence_penalty"] = args.presence_penalty
    if args.frequency_penalty is not None:
        kwargs["frequency_penalty"] = args.frequency_penalty
    if request_seed is not None:
        kwargs["seed"] = request_seed
    extra_body: dict[str, Any] = {}
    if args.enable_thinking is not None:
        extra_body["chat_template_kwargs"] = {"enable_thinking": bool(args.enable_thinking)}
    if args.top_k is not None:
        extra_body["top_k"] = args.top_k
    if args.min_p is not None:
        extra_body["min_p"] = args.min_p
    if args.repetition_penalty is not None:
        extra_body["repetition_penalty"] = args.repetition_penalty
    if extra_body:
        kwargs["extra_body"] = extra_body

    start = time.time()
    resp = _client(args.base_url, args.timeout).chat.completions.create(**kwargs)
    elapsed = time.time() - start
    text = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    completion_tokens = None if usage is None else getattr(usage, "completion_tokens", None)
    return text, completion_tokens, elapsed


def _score(
    data_source: str,
    text: str,
    ground_truth: str,
    judge_url: str,
    *,
    frontiercs_score_backend: str,
) -> dict[str, float | None]:
    if data_source == "frontiercs":
        if frontiercs_score_backend == "official":
            if official_extract_cpp_code is None:
                raise RuntimeError(
                    "Could not import official FrontierCS extract_cpp_code. "
                    f"Original error: {_OFFICIAL_FRONTIERCS_IMPORT_ERROR!r}"
                )
            # Strip the <think> reasoning trace BEFORE the official extractor runs.
            # The official extractor picks the LONGEST fenced block; with thinking
            # enabled a model often writes scratch ```cpp blocks inside <think> that
            # can be longer than the final answer, so the unstripped extractor would
            # score reasoning scratch instead of the real solution (under-reporting
            # capability and diverging from the RL reward, which already strips think).
            code = official_extract_cpp_code(frontiercs.strip_think(text))
            if not code:
                return {"reward": 0.0, "score": 0.0, "score_unbounded": 0.0}
            res = _frontiercs_runner(judge_url).evaluate(str(ground_truth), code)
            # CRITICAL: distinguish a legitimate model-side 0 (code that compiled
            # but failed tests, or a genuine compile error in the model's output)
            # from an INFRASTRUCTURE failure (judge unreachable / submission failed
            # / evaluation timed out). Previously every non-SUCCESS result was
            # silently mapped to 0.0, so a down/misconfigured judge was
            # indistinguishable from "the model scored 0" -- which is exactly how a
            # whole eval can come back all-zeros and look like model weakness.
            # Raise on infra failures so they are recorded in the `error` field
            # instead of being scored 0 and polluting the ablation.
            if not res.success and _is_judge_infra_failure(res):
                raise RuntimeError(
                    f"FrontierCS judge infrastructure failure for problem "
                    f"{ground_truth} (status={getattr(res.status, 'value', res.status)}): "
                    f"{res.message}"
                )
            score = float(res.score or 0.0) if res.success else 0.0
            score_unbounded = None if res.score_unbounded is None else float(res.score_unbounded)
            return {
                "reward": score,
                "score": score,
                "score_unbounded": score if score_unbounded is None else score_unbounded,
            }
        score = float(frontiercs.compute_score("frontiercs", text, ground_truth, judge_url=judge_url))
        return {"reward": score, "score": score, "score_unbounded": score}
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


def _run_one(args: argparse.Namespace, problem: dict[str, Any], problem_idx: int, sample_idx: int) -> dict[str, Any]:
    started = time.time()
    request_seed = None if args.seed is None else int(args.seed) + problem_idx * args.n_samples + sample_idx
    try:
        text, completion_tokens, gen_seconds = _generate_one(args, problem["messages"], request_seed)
        metrics = _score(
            problem["data_source"],
            text,
            problem["ground_truth"],
            args.judge_url,
            frontiercs_score_backend=args.frontiercs_score_backend,
        )
        error = None
    except Exception as exc:
        text = ""
        completion_tokens = None
        gen_seconds = 0.0
        metrics = {"reward": 0.0, "score": 0.0}
        error = repr(exc)

    rec = {
        "data_source": problem["data_source"],
        "ground_truth": problem["ground_truth"],
        "problem_idx": problem_idx,
        "sample_idx": sample_idx,
        "completion_tokens": completion_tokens,
        "generation_seconds": gen_seconds,
        "total_seconds": time.time() - started,
        "metrics": metrics,
        "error": error,
        "prompt_variant": problem.get("prompt_variant"),
        "score_backend": args.frontiercs_score_backend
        if problem["data_source"] == "frontiercs"
        else "alebench:official-private-eval",
        "request_seed": request_seed,
    }
    if args.save_text:
        rec["text"] = text
    elif args.text_preview_chars > 0:
        rec["text_preview"] = text[: args.text_preview_chars]
        rec["text_chars"] = len(text)
    return rec


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
    parser.add_argument("--frontiercs-data", type=Path, default=PROJECT_ROOT / "data" / "frontiercs" / "full.parquet")
    parser.add_argument("--alebench-data", type=Path, default=PROJECT_ROOT / "data" / "alebench" / "val.parquet")
    parser.add_argument("--source", choices=["both", "frontiercs", "alebench"], default="both")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "base_eval_qwen35_9b_vllm")
    parser.add_argument("--samples-jsonl", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-frontiercs", type=int, default=None)
    parser.add_argument("--limit-alebench", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-idx", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen35-9b")
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-p", type=float, default=None)
    parser.add_argument("--presence-penalty", type=float, default=None)
    parser.add_argument("--frequency-penalty", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--max-errors", type=int, default=0)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--judge-url", default=os.environ.get("FRONTIERCS_JUDGE_URL", "http://127.0.0.1:8082"))
    parser.add_argument("--frontiercs-prompt-source", choices=["official", "parquet"], default="official")
    parser.add_argument("--frontiercs-score-backend", choices=["official", "legacy"], default="official")
    parser.add_argument("--save-text", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--text-preview-chars", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    problems: list[dict[str, Any]] = []
    if args.source in ("both", "frontiercs"):
        problems.extend(
            _load_problems(
                args.frontiercs_data,
                "frontiercs",
                args.limit_frontiercs,
                frontiercs_prompt_source=args.frontiercs_prompt_source,
            )
        )
    if args.source in ("both", "alebench"):
        problems.extend(_load_problems(args.alebench_data, "alebench", args.limit_alebench))

    total_problem_count = len(problems)
    if args.num_shards > 1:
        problems = [problem for idx, problem in enumerate(problems) if idx % args.num_shards == args.shard_idx]

    print(f"Loaded {len(problems)} problems", flush=True)
    if args.num_shards > 1:
        print(f"Shard {args.shard_idx}/{args.num_shards}: {len(problems)} of {total_problem_count}", flush=True)
    print(
        f"Sources: frontiercs={sum(p['data_source'] == 'frontiercs' for p in problems)}, "
        f"alebench={sum(p['data_source'] == 'alebench' for p in problems)}",
        flush=True,
    )
    print(f"vLLM endpoint: {args.base_url} model={args.model} concurrency={args.concurrency}", flush=True)
    if args.dry_run:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = args.samples_jsonl or (args.output_dir / "samples.jsonl")
    summary_path = args.summary_json or (args.output_dir / "summary.json")

    planned_keys = {
        (problem["data_source"], problem["ground_truth"], sample_idx)
        for problem in problems
        for sample_idx in range(args.n_samples)
    }
    existing = _load_existing(samples_path) if args.resume else {}
    records = {key: rec for key, rec in existing.items() if key in planned_keys and _record_compatible(rec, args)}
    if records:
        print(f"Resuming from {samples_path}: {len(records)} completed samples", flush=True)

    tasks = []
    for problem_idx, problem in enumerate(problems):
        for sample_idx in range(args.n_samples):
            key = (problem["data_source"], problem["ground_truth"], sample_idx)
            if key not in records:
                tasks.append((problem_idx, problem, sample_idx))

    # COMPILE STARTUP SELF-TEST for ALE-Bench. If this node's compile substrate
    # is systematically broken, every ALE submission would be scored 0 (a fake
    # uniform-zero indistinguishable from "all models are weak"). Probe the real
    # compile-container path ONCE, up front, on the main thread -- so a broken
    # substrate aborts the eval LOUDLY instead of being smeared across per-sample
    # errors inside the thread pool. Only runs when there are ALE tasks to score.
    if any(problem["data_source"] == "alebench" for problem_idx, problem, sample_idx in tasks):
        from verl.utils.reward_score.ale_selftest import ale_compile_selftest

        ale_compile_selftest(alebench.AleInfraError)

    samples_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    with samples_path.open(mode, encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = {ex.submit(_run_one, args, problem, problem_idx, sample_idx): (problem, sample_idx) for problem_idx, problem, sample_idx in tasks}
            pbar = tqdm(as_completed(futures), total=len(futures), initial=0, desc="Eval", unit="sample")
            for fut in pbar:
                rec = fut.result()
                key = (rec["data_source"], rec["ground_truth"], int(rec["sample_idx"]))
                out.write(json.dumps(rec, ensure_ascii=False, default=_json_default) + "\n")
                out.flush()
                records[key] = rec
                reward = rec.get("metrics", {}).get("reward")
                reward_str = "NA" if reward is None else f"{float(reward):.4f}"
                error = rec.get("error")
                if error:
                    print(
                        f"ERROR {rec['data_source']} {rec['ground_truth']} sample={rec['sample_idx']}: {error}",
                        flush=True,
                    )
                print(
                    f"SAMPLE {len(records)}/{len(planned_keys)} {rec['data_source']} "
                    f"{rec['ground_truth']} sample={rec['sample_idx']} reward={reward_str} "
                    f"gen_tokens={rec['completion_tokens']} gen_sec={rec['generation_seconds']:.1f}",
                    flush=True,
                )
                pbar.update(0)

    summary = _summarize(list(records.values()), args.n_samples, args.seed)
    summary["config"] = vars(args)
    summary["samples_jsonl"] = str(samples_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default) + "\n")
    print(f"Saved samples to {samples_path}", flush=True)
    print(f"Saved summary to {summary_path}", flush=True)
    _print_metrics(summary)

    planned_records = [records[key] for key in planned_keys if key in records]
    missing_count = len(planned_keys) - len(planned_records)
    error_records = [rec for rec in planned_records if rec.get("error")]
    if missing_count:
        print(f"ERROR: {missing_count} planned samples were not completed", file=sys.stderr, flush=True)
        sys.exit(2)
    if len(error_records) > args.max_errors:
        examples = "; ".join(str(rec.get("error"))[:240] for rec in error_records[:3])
        print(
            f"ERROR: {len(error_records)} sample(s) failed, above --max-errors={args.max_errors}. "
            f"Examples: {examples}",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
