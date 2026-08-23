#!/usr/bin/env python3
"""Evaluate Qwen3.5 through a vLLM OpenAI-compatible server."""

from __future__ import annotations

import argparse
import ast
import json
import os
import queue
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

# Research-track (no-Docker) evaluator: enumeration, canonical prompt, code
# extraction, and direct evaluator.py scoring. Import-safe (no google.generativeai).
try:
    import frontiercs_research_eval as fcs_research
except Exception as exc:  # pragma: no cover
    fcs_research = None
    _FCS_RESEARCH_IMPORT_ERROR = exc
else:
    _FCS_RESEARCH_IMPORT_ERROR = None

# CPU-subset research adapter: families that use a non-Triton CLI / output shape
# (cant_be_late*, cloudcast, grammar_fuzzing, imagenet_pareto, llm_router,
# llm_sql, nbody_simulation [C++], symbolic_regression, vdb_pareto). These run
# their evaluator.py in-place (cwd = the leaf) with the official per-family args.
try:
    import frontiercs_research_cpu_eval as fcs_research_cpu
except Exception as exc:  # pragma: no cover
    fcs_research_cpu = None
    _FCS_RESEARCH_CPU_IMPORT_ERROR = exc
else:
    _FCS_RESEARCH_CPU_IMPORT_ERROR = None

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

# The FULL system prompt the innovation SFT data was actually trained with
# (86.95% of corpus records carry this exact clause; the rest use an expert-CP
# prompt). INNOV_SYS_TMPL above drops the delivery clause. The corpus teaches
# "fall back to the simplest correct approach and ship that" almost entirely as
# a prompt-conditional instruction: audited demonstration rate in trace text is
# ~2% of turns (an earlier 23.5% figure here was wrong -- independent audit,
# 2026-08-17). MEASURED A/B RESULT (n=850/arm, full data): restoring this clause
# does NOT significantly change FCS for either soup arm (soupWD03_20 7.397 ->
# 6.751 paired p=0.35; soupNEW10 7.310 -> 6.975 p=0.60); mid-run numbers that
# suggested a big verbosity reduction did not survive the full run. Kept because
# eval-time conditioning should match training (user's time-conditioning design),
# not because it rescues these arms. Opt in with EVAL_SYS_PROMPT_MODE=full.
INNOV_SYS_FULL_TMPL = (
    "It is now year {year}. You are a good researcher. When you write code, "
    "deliver a single, self-contained, runnable implementation that respects any "
    "stated input/output contract; if an idea is not converging within the budget, "
    "fall back to the simplest correct approach and ship that."
)


def _researcher_system_message() -> dict[str, str] | None:
    """Innovation meta-conditioning system prompt with the year set to the PRESENT.
    Matches the time-conditioned system prompt the SFT data was trained on. Enabled
    by env EVAL_RESEARCHER_YEAR (no effect on pre-SFT/base models per design).
    EVAL_SYS_PROMPT_MODE=full selects the complete training string including the
    delivery clause; anything else (default) keeps the historical short form."""
    year = os.environ.get("EVAL_RESEARCHER_YEAR", "").strip()
    if not year:
        return None
    mode = os.environ.get("EVAL_SYS_PROMPT_MODE", "short").strip().lower()
    # "bare" = time only ("It is now year 2026.") -- the going-forward convention
    # (user decision 2026-08-18): a normal system prompt just states the time; the
    # persona and delivery-clause sentences were legacy additions from the SFT
    # remediation and should not propagate to RL or evaluation. rlv13 trains with
    # this bare prompt, so its evaluation must use the same.
    if mode == "bare":
        return {"role": "system", "content": f"It is now year {year}."}
    tmpl = INNOV_SYS_FULL_TMPL if mode == "full" else INNOV_SYS_TMPL
    return {"role": "system", "content": tmpl.format(year=year)}


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
        elif source == "frontiercs_research":
            # The research parquet already carries the canonical (system + readme)
            # generation prompt built by frontiercs_research_eval.build_research_messages.
            messages = _as_messages(row["prompt"])
            prompt_variant = "frontiercs_research:official-readme"
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
    # An errored record (judge infra failure, generation failure, ...) was scored
    # 0 as a placeholder, NOT as a real result. Treat it as incomplete so --resume
    # regenerates/rescores it instead of baking the fake 0 into the summary.
    # (Job 10972217 had 236/910 such zeros -> FCS 1.57 was mostly infra noise.)
    if rec.get("error"):
        return False
    data_source = str(rec.get("data_source", ""))
    if data_source == "frontiercs":
        expected_prompt = (
            "frontiercs:official-generate_solutions"
            if args.frontiercs_prompt_source == "official"
            else "frontiercs:parquet"
        )
        return rec.get("prompt_variant") == expected_prompt and rec.get("score_backend") == args.frontiercs_score_backend
    if data_source == "frontiercs_research":
        return rec.get("score_backend") in (None, "frontiercs_research:official-evaluator")
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
            # CORRECT extraction for OPEN thinking models: strip <think> first,
            # THEN run the official extractor. The official FrontierCS pipeline only
            # ever calls CLOSED APIs (OpenAI/Gemini/Claude/DeepSeek/Grok); those
            # return `choices[0].message.content` = the ANSWER ONLY (reasoning lives
            # in a separate field the client discards), so the official
            # extract_cpp_code (`max(matches, key=len)`) never sees <think> and
            # "longest block" is safe. A local Qwen with thinking emits <think>
            # INLINE in content; feeding that raw makes "longest" grab reasoning
            # scratch -> artificially low (this is why the paper's Qwen base reads
            # 1.80 instead of the fair ~7.05). Stripping <think> reconstructs the
            # same answer-only input the closed models get, and matches the RL
            # training reward (frontiercs.py extract_cpp, which also strips).
            # Set FRONTIERCS_STRIP_THINK_EXTRACT=0 for the raw/paper-artifact view.
            _strip_think = os.environ.get("FRONTIERCS_STRIP_THINK_EXTRACT", "1") == "1"
            code = official_extract_cpp_code(
                frontiercs.strip_think(text) if _strip_think else text
            )
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
    if data_source == "frontiercs_research":
        # Research track: extract the Python `Solution` from the (think-stripped)
        # response, then run the official evaluator.py directly (no Docker).
        # Score is 0..100, same scale as the leaderboard's per-problem score.
        if fcs_research is None:
            raise RuntimeError(
                "Could not import frontiercs_research_eval. "
                f"Original error: {_FCS_RESEARCH_IMPORT_ERROR!r}"
            )
        pid = str(ground_truth)
        # CPU families have non-Triton CLIs / output shapes (and nbody is C++);
        # route them through the dedicated adapter.
        use_cpu = (
            fcs_research_cpu is not None
            and fcs_research_cpu.is_cpu_family(pid)
        )
        stripped = frontiercs.strip_think(text)
        if use_cpu and fcs_research_cpu.solution_language(pid) == "cpp":
            # nbody_simulation: extract C++ (fall back to python-style fence strip).
            if official_extract_cpp_code is not None:
                code = official_extract_cpp_code(stripped)
            else:
                code = fcs_research.extract_python_code(stripped)
        else:
            code = fcs_research.extract_python_code(stripped)
        if not code:
            return {"reward": 0.0, "score": 0.0, "score_unbounded": 0.0}
        # ResearchInfraError propagates -> recorded as `error` (never silent 0),
        # exactly like the algorithmic judge-infra path above.
        if use_cpu:
            res = fcs_research_cpu.evaluate_cpu_research_solution(pid, code)
        else:
            res = fcs_research.evaluate_research_solution(pid, code)
        score = float(res.get("score") or 0.0)
        su = res.get("score_unbounded")
        return {
            "reward": score,
            "score": score,
            "score_unbounded": score if su is None else float(su),
        }
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


def _judge_feedback_message(round_idx: int, metrics: dict[str, Any]) -> dict[str, str]:
    """Build the follow-up user turn that feeds the judge's result back to the
    model between iterative-refinement rounds.

    Mirrors the FEEDBACK an agent gets from `submit.sh` in the official Harbor
    adapter (a sanitized normalized score + status), NOT any hidden test data --
    the adapter only ever exposes the aggregate score, so this stays faithful to
    the "controlled feedback" contract (parity-mode agent gets no test cases).
    """
    score = metrics.get("score")
    score_str = "0" if score is None else f"{float(score):.2f}"
    return {
        "role": "user",
        "content": (
            f"Your previous C++ solution was graded by the judge and scored "
            f"{score_str}/100 (partial credit = fraction of test cases passed). "
            f"This is attempt {round_idx + 1}. Analyze why it did not score 100, "
            f"fix or improve the solution, and output the FULL corrected C++ code "
            f"wrapped in ```cpp and ```. Output ONLY the code."
        ),
    }


def _generate_and_score_iterative(
    args: argparse.Namespace,
    problem: dict[str, Any],
    request_seed: int | None,
) -> tuple[str, int | None, float, dict[str, Any], int]:
    """Iterative-refinement generation for the ALGORITHMIC frontiercs track.

    Runs up to args.frontiercs_iterative_rounds rounds. Each round: generate,
    score against the judge, then (if not the last round and not already 100)
    append a sanitized judge-feedback turn and let the model revise. Returns the
    text / tokens / seconds / metrics of the BEST-scoring round and the number of
    rounds actually run -- mirroring the official agent adapter's "records the
    higher of the final solution score and the best iterative submission"
    (README.md + tests/evaluate.py best_submission()).

    A judge INFRA failure in any round propagates (recorded as `error`), exactly
    like the single-shot path -- a down judge must never be masked as a 0.
    """
    messages = list(problem["messages"])
    best_text = ""
    best_tokens: int | None = None
    total_gen_seconds = 0.0
    best_metrics: dict[str, Any] = {"reward": 0.0, "score": 0.0}
    best_score = -1.0
    rounds = max(1, int(args.frontiercs_iterative_rounds))
    for round_idx in range(rounds):
        seed = None if request_seed is None else request_seed + round_idx * 100003
        text, tokens, gen_seconds = _generate_one(args, messages, seed)
        total_gen_seconds += gen_seconds
        metrics = _score(
            problem["data_source"],
            text,
            problem["ground_truth"],
            args.judge_url,
            frontiercs_score_backend=args.frontiercs_score_backend,
        )
        score = float(metrics.get("score") or 0.0)
        if score > best_score:
            best_score, best_metrics, best_text, best_tokens = score, metrics, text, tokens
        # Stop early on a perfect score, and don't waste a round after the last.
        if score >= 100.0 or round_idx == rounds - 1:
            return best_text, best_tokens, total_gen_seconds, best_metrics, round_idx + 1
        # Feed the assistant's answer + judge feedback back for the next round.
        messages = [*messages, {"role": "assistant", "content": text}, _judge_feedback_message(round_idx, metrics)]
    return best_text, best_tokens, total_gen_seconds, best_metrics, rounds


def _run_one(args: argparse.Namespace, problem: dict[str, Any], problem_idx: int, sample_idx: int) -> dict[str, Any]:
    started = time.time()
    request_seed = None if args.seed is None else int(args.seed) + problem_idx * args.n_samples + sample_idx
    # Iterative-refinement (agentic) mode is OPT-IN and applies ONLY to the
    # algorithmic frontiercs track (the only track the official Harbor agent
    # adapter covers). rounds<=1 (default) is byte-for-byte the old single-shot
    # path, so existing runs are unchanged.
    iterative = (
        int(getattr(args, "frontiercs_iterative_rounds", 1)) > 1
        and problem["data_source"] == "frontiercs"
    )
    rounds_run = 1
    try:
        if iterative:
            text, completion_tokens, gen_seconds, metrics, rounds_run = _generate_and_score_iterative(
                args, problem, request_seed
            )
        else:
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
        "score_backend": (
            args.frontiercs_score_backend
            if problem["data_source"] == "frontiercs"
            else "frontiercs_research:official-evaluator"
            if problem["data_source"] == "frontiercs_research"
            else "alebench:official-private-eval"
        ),
        "request_seed": request_seed,
    }
    if iterative:
        rec["iterative_rounds_run"] = rounds_run
    if args.save_text:
        rec["text"] = text
    elif args.text_preview_chars > 0:
        rec["text_preview"] = text[: args.text_preview_chars]
        rec["text_chars"] = len(text)
    return rec


def _official_research_metrics(
    records: list[dict[str, Any]], n_samples: int
) -> dict[str, dict[str, float | int]]:
    """Compute the OFFICIAL Frontier-CS leaderboard metrics (Score@1, Avg@5,
    Score@5) exactly as the official batch aggregator does, for BOTH the
    algorithmic (data_source="frontiercs") and research
    (data_source="frontiercs_research") tracks.

    Reference: .cache/Frontier-CS-official/src/frontier_cs/batch/state.py:493-591
    (export_aggregated_csv / aggregate_by_model). Per problem, with per-variant
    scores clamped to [0,100]:
      - Score@1 = the variant-0 (sample_idx 0) score          (state.py:566)
      - Avg@5   = sum(score[i] for i in range(5)) / 5          (state.py:570;
                  MISSING variants count as 0)
      - Score@5 = max(score[i] for i in range(5))              (state.py:571;
                  MISSING variants count as 0)
    Each is then averaged across all problems that HAVE a variant-0 result
    (mirroring the official "only count problems where 0 in variant_scores",
    state.py:561-562).

    This is the leaderboard-faithful counterpart to process_validation_metrics'
    mean@5 / best@5, which diverge from the official numbers in three ways:
      (a) Score@1 -- official = the base variant (idx 0) ONLY; mean@1 is not even
          emitted when n_samples>1, and mean@5 is Avg@5 not Score@1.
      (b) Score@5 -- official = the TRUE per-problem max over 5 (== our
          oracle_best@5); process_validation_metrics' best@5/mean is a BOOTSTRAP
          estimate (resampled 1000x with replacement) that is biased low vs the
          true max, so it must NOT be reported as Score@5.
      (c) the @5 denominator is ALWAYS 5 -- a problem that only produced 3 samples
          still divides by 5, and problems missing some of the 5 are NOT dropped;
          they contribute 0 for the missing variants.
    For a fully-complete run (all n_samples present on every problem) with
    n_samples==5, Avg@5 coincides with mean@5 and Score@5 with oracle_best@5; on a
    PARTIAL run they diverge, which is exactly when faithfulness matters.

    Returned per data_source (both frontiercs and frontiercs_research):
      {score_at_1, avg_at_k, score_at_k, pass_at_1, pass_at_k, num_problems, k}.
    k is min(n_samples, 5) -- the official metric fixes k=5; if a run used fewer
    samples we report the variant window actually used so the number is not
    silently misread as @5.
    """
    K = min(int(n_samples), 5)
    # Sources whose scores live on the [0,100] Frontier-CS scale and whose
    # leaderboard metric is the official Score@1/Avg@5/Score@5. ALE-Bench uses a
    # different (relative/rank) leaderboard metric and is intentionally excluded.
    OFFICIAL_SOURCES = ("frontiercs", "frontiercs_research")
    # {source: {problem_id: {variant_idx: clamped_score}}}
    by_src_prob: dict[str, dict[str, dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for rec in records:
        source = str(rec["data_source"])
        if source not in OFFICIAL_SOURCES:
            continue
        # An errored sample (infra failure) has no legitimate score -> official
        # "successful results with scores" excludes it (variant simply absent ->
        # counts as 0 in the @5 window, exactly like a missing variant).
        if rec.get("error"):
            continue
        metrics = rec.get("metrics") or {}
        raw = metrics.get("score")
        if raw is None:
            continue
        try:
            score = max(0.0, min(100.0, float(raw)))
        except (TypeError, ValueError):
            continue
        by_src_prob[source][str(rec["ground_truth"])][int(rec["sample_idx"])] = score

    out: dict[str, dict[str, float | int]] = {}
    for source, prob_scores in by_src_prob.items():
        s1, a5, m5 = [], [], []
        pass1 = pass5 = 0
        n_problems = 0
        for _pid, variant_scores in prob_scores.items():
            # Official: only count a problem if it has the base variant (idx 0).
            if 0 not in variant_scores:
                continue
            n_problems += 1
            score1 = variant_scores.get(0, 0.0)
            window = [variant_scores.get(i, 0.0) for i in range(K)]
            s1.append(score1)
            a5.append(sum(window) / K)
            m5.append(max(window))
            if score1 > 0:
                pass1 += 1
            if max(window) > 0:
                pass5 += 1
        if n_problems == 0:
            continue
        out[source] = {
            "score_at_1": sum(s1) / len(s1),
            f"avg_at_{K}": sum(a5) / len(a5),
            f"score_at_{K}": sum(m5) / len(m5),
            "pass_at_1": pass1 / n_problems,
            f"pass_at_{K}": pass5 / n_problems,
            "num_problems": n_problems,
            "k": K,
        }
    return out


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

    # Official Frontier-CS RESEARCH leaderboard metrics (Score@1/Avg@5/Score@5),
    # computed over ALL records with the official "missing variant = 0" semantics
    # (not just the complete problems). Only the frontiercs_research source(s) are
    # included; harmless (empty) for other sources.
    official_leaderboard = _official_research_metrics(records, n_samples)

    return {
        "complete_problem_count": complete_problem_count,
        "scored_sample_count": len(complete_records),
        "metrics": metrics,
        "oracle_best": oracle_best,
        # Canonical key. "official_research_metrics" retained as an alias so any
        # existing downstream reader (parse scripts, dashboards) keeps working.
        "official_leaderboard_metrics": official_leaderboard,
        "official_research_metrics": official_leaderboard,
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
    # Official Frontier-CS leaderboard metrics (Score@1/Avg@k/Score@k) for both
    # the algorithmic (frontiercs) and research (frontiercs_research) tracks.
    for source in sorted(summary.get("official_leaderboard_metrics", {})):
        block = summary["official_leaderboard_metrics"][source]
        for metric_name in sorted(block):
            value = block[metric_name]
            if isinstance(value, (int, float)):
                print(f"METRIC official/{source}/{metric_name}: {value:.6f}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontiercs-data", type=Path, default=PROJECT_ROOT / "data" / "frontiercs" / "full.parquet")
    parser.add_argument("--alebench-data", type=Path, default=PROJECT_ROOT / "data" / "alebench" / "val.parquet")
    parser.add_argument("--research-data", type=Path, default=PROJECT_ROOT / "data" / "frontiercs" / "research.parquet")
    parser.add_argument("--source", choices=["both", "frontiercs", "alebench", "research", "all"], default="both")
    parser.add_argument("--limit-research", type=int, default=None)
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
    parser.add_argument(
        "--frontiercs-iterative-rounds",
        type=int,
        default=int(os.environ.get("FRONTIERCS_ITERATIVE_ROUNDS", "1")),
        help=(
            "Rounds of iterative refinement per sample for the ALGORITHMIC frontiercs "
            "track (agentic protocol). 1 (default) = the leaderboard single-shot "
            "generate->extract->judge path (unchanged). >1 feeds the judge's score back "
            "each round and records the BEST-scoring round, mirroring the official Harbor "
            "agent adapter's max(final, best-submission)."
        ),
    )
    parser.add_argument("--save-text", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--text-preview-chars", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--decouple-scoring",
        action="store_true",
        help=(
            "Decouple generation from scoring: `concurrency` generation workers feed a "
            "queue drained by `score-concurrency` scoring workers, so a slow judge never "
            "starves the vLLM engine of requests. Outputs are identical to the coupled "
            "loop (same seeds/prompts/scorer); only the scheduling changes. Single-shot "
            "sources only (frontiercs/frontiercs_research/alebench); refuses iterative mode."
        ),
    )
    parser.add_argument(
        "--score-concurrency",
        type=int,
        default=int(os.environ.get("EVAL_SCORE_CONCURRENCY", "8")),
        help="Number of scoring workers in --decouple-scoring mode (the judge is CPU-bound; ~8 on an 8-CPU job).",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _decoupled_preflight(args: argparse.Namespace, problems: list[dict[str, Any]]) -> None:
    """Refuse decoupled mode for anything that is NOT plain single-shot generate-then-score.

    The decoupled path generates once and scores the same text (the request_seed is
    identical to _run_one's, so outputs are unchanged). Anything that changes the
    message stream based on an intermediate score -- the iterative-refinement loop
    (_generate_and_score_iterative), which re-prompts the model with the judge's
    feedback -- cannot be decoupled. Rather than silently doing the wrong thing,
    fail loudly.
    """
    if int(getattr(args, "frontiercs_iterative_rounds", 1)) > 1:
        raise SystemExit(
            "FATAL: --decouple-scoring cannot be combined with --frontiercs-iterative-rounds>1. "
            "Iterative refinement re-prompts the model with the judge's score, so generation "
            "depends on scoring and cannot be decoupled. Run without --decouple-scoring."
        )
    for problem in problems:
        if problem.get("data_source") not in ("frontiercs", "frontiercs_research", "alebench"):
            raise SystemExit(
                f"FATAL: --decouple-scoring only supports single-shot sources "
                f"(frontiercs / frontiercs_research / alebench); got {problem.get('data_source')!r}."
            )


def _decoupled_gen_worker(
    args: argparse.Namespace,
    problem: dict[str, Any],
    problem_idx: int,
    sample_idx: int,
    request_seed: int | None,
    gen_q: "queue.Queue",
    gen_active: list,
    gen_lock: threading.Lock,
) -> None:
    """Generation-only worker. Produces the completion, then hands everything the
    scorer needs to gen_q. NEVER blocks on a judge. On exit it decrements
    gen_active; when the last generator finishes, it posts one sentinel per scorer
    so the scorers drain the queue and terminate.
    """
    started = time.time()
    try:
        text, completion_tokens, gen_seconds = _generate_one(args, problem["messages"], request_seed)
        gen_error = None
    except Exception as exc:
        text, completion_tokens, gen_seconds = "", None, 0.0
        gen_error = repr(exc)
    gen_q.put((problem, problem_idx, sample_idx, request_seed, text, completion_tokens, gen_seconds, started, gen_error))
    with gen_lock:
        gen_active[0] -= 1


def _decoupled_score_worker(
    args: argparse.Namespace,
    gen_q: "queue.Queue",
    result_q: "queue.Queue",
) -> None:
    """Scoring-only worker. Pulls completed generations off gen_q, scores them, and
    pushes finished records to result_q. Builds a record IDENTICAL in shape to
    _run_one's single-shot branch. Generation errors skip the judge and yield an
    error record (same as _run_one's catch)."""
    while True:
        item = gen_q.get()
        if item is None:
            gen_q.task_done()
            break
        problem, problem_idx, sample_idx, request_seed, text, completion_tokens, gen_seconds, started, gen_error = item
        try:
            if gen_error is not None:
                raise RuntimeError(gen_error)
            metrics = _score(
                problem["data_source"],
                text,
                problem["ground_truth"],
                args.judge_url,
                frontiercs_score_backend=args.frontiercs_score_backend,
            )
            error = None
        except Exception as exc:
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
            "score_backend": (
                args.frontiercs_score_backend
                if problem["data_source"] == "frontiercs"
                else "frontiercs_research:official-evaluator"
                if problem["data_source"] == "frontiercs_research"
                else "alebench:official-private-eval"
            ),
            "request_seed": request_seed,
        }
        if args.save_text:
            rec["text"] = text
        elif args.text_preview_chars > 0:
            rec["text_preview"] = text[: args.text_preview_chars]
            rec["text_chars"] = len(text)
        result_q.put(rec)
        gen_q.task_done()


def _run_decoupled(
    args: argparse.Namespace,
    tasks: list,
    planned_keys: set,
    records: dict,
    out,
) -> None:
    """Decoupled eval loop: `args.concurrency` generation workers feed a queue that
    `args.score_concurrency` scoring workers drain. Generation never blocks on the
    judge, so the vLLM engine is never starved of requests. Output records are
    identical in shape and content to the coupled loop's (same seeds, same prompts,
    same scorer); only the SCHEDULING changed. The single writer keeps
    samples.jsonl appends serialized, exactly as before.
    """
    gen_q: "queue.Queue" = queue.Queue()
    result_q: "queue.Queue" = queue.Queue()
    gen_active = [len(tasks)]
    gen_lock = threading.Lock()

    scorers = [
        threading.Thread(target=_decoupled_score_worker, args=(args, gen_q, result_q), daemon=True, name=f"scorer-{i}")
        for i in range(int(getattr(args, "score_concurrency", 8)))
    ]
    for t in scorers:
        t.start()

    def _writer() -> None:
        pbar = tqdm(total=len(tasks), desc="Eval", unit="sample")
        while True:
            rec = result_q.get()
            if rec is None:
                result_q.task_done()
                break
            key = (rec["data_source"], rec["ground_truth"], int(rec["sample_idx"]))
            out.write(json.dumps(rec, ensure_ascii=False, default=_json_default) + "\n")
            out.flush()
            records[key] = rec
            reward = rec.get("metrics", {}).get("reward")
            reward_str = "NA" if reward is None else f"{float(reward):.4f}"
            if rec.get("error"):
                print(
                    f"ERROR {rec['data_source']} {rec['ground_truth']} sample={rec['sample_idx']}: {rec['error']}",
                    flush=True,
                )
            print(
                f"SAMPLE {len(records)}/{len(planned_keys)} {rec['data_source']} "
                f"{rec['ground_truth']} sample={rec['sample_idx']} reward={reward_str} "
                f"gen_tokens={rec['completion_tokens']} gen_sec={rec['generation_seconds']:.1f}",
                flush=True,
            )
            pbar.update(1)
            result_q.task_done()
        pbar.close()

    writer = threading.Thread(target=_writer, daemon=True, name="writer")
    writer.start()

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        gen_futs = [
            ex.submit(
                _decoupled_gen_worker, args, problem, problem_idx, sample_idx,
                (None if args.seed is None else int(args.seed) + problem_idx * args.n_samples + sample_idx),
                gen_q, gen_active, gen_lock,
            )
            for problem_idx, problem, sample_idx in tasks
        ]
        for f in gen_futs:
            f.result()  # surface a generator exception instead of swallowing it
    # Every generation has now been PUT on gen_q. Only NOW is it safe to post the
    # sentinels: posting them earlier (when the last generator FINISHED its put)
    # raced the scorers, which had already dequeued earlier samples and were busy
    # in _score -- the sentinels landed behind in-flight items and let scorers
    # exit before the queue was drained. Post after the generator pool exits.
    for _ in range(int(getattr(args, "score_concurrency", 8))):
        gen_q.put(None)
    gen_q.join()        # all generations scored (sentinels consumed last)
    result_q.put(None)  # tell the writer to stop after the last record
    result_q.join()
    for t in scorers:
        t.join(timeout=5)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    problems: list[dict[str, Any]] = []
    if args.source in ("both", "frontiercs", "all"):
        problems.extend(
            _load_problems(
                args.frontiercs_data,
                "frontiercs",
                args.limit_frontiercs,
                frontiercs_prompt_source=args.frontiercs_prompt_source,
            )
        )
    if args.source in ("both", "alebench", "all"):
        problems.extend(_load_problems(args.alebench_data, "alebench", args.limit_alebench))
    if args.source in ("research", "all"):
        problems.extend(_load_problems(args.research_data, "frontiercs_research", args.limit_research))

    total_problem_count = len(problems)
    if args.num_shards > 1:
        problems = [problem for idx, problem in enumerate(problems) if idx % args.num_shards == args.shard_idx]

    print(f"Loaded {len(problems)} problems", flush=True)
    if args.num_shards > 1:
        print(f"Shard {args.shard_idx}/{args.num_shards}: {len(problems)} of {total_problem_count}", flush=True)
    print(
        f"Sources: frontiercs={sum(p['data_source'] == 'frontiercs' for p in problems)}, "
        f"frontiercs_research={sum(p['data_source'] == 'frontiercs_research' for p in problems)}, "
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
        if getattr(args, "decouple_scoring", False):
            _decoupled_preflight(args, problems)
            print(
                f"[decouple] generation workers={args.concurrency} scoring workers={args.score_concurrency} "
                "(generation never blocks on the judge)",
                flush=True,
            )
            _run_decoupled(args, tasks, planned_keys, records, out)
        else:
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
