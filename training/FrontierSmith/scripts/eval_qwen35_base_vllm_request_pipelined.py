#!/usr/bin/env python3
"""Single-shot vLLM evaluation with generation and scoring decoupled.

This is a NEW entry point.  It deliberately reuses the established evaluator's
prompt construction, request construction, seeds, extraction, scoring, resume,
and summary code; it changes only scheduling.  ``--concurrency`` workers issue
generation requests, completed texts enter an unbounded in-process scoring queue,
and a smaller scoring pool drains that queue.  Therefore a scorer never occupies
a generation worker.

Generated text and sampling inputs are unchanged.  Record order and
``total_seconds`` can change because results are written when scoring completes.
Iterative FrontierCS evaluation is rejected: later generation rounds depend on
the prior judge result and cannot be safely decoupled.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import random
import sys
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from tqdm import tqdm

# Executing this file as ``python scripts/...`` places scripts/ on sys.path.
# Importing helpers rather than copying them is intentional: the exact prompt,
# sampling request, extractor, judge backend, resume checks, and summary remain
# the established evaluator's implementation.
import eval_qwen35_base_vllm_request as base


T = TypeVar("T")
U = TypeVar("U")
_STOP = object()


def run_generation_scoring_pipeline(
    items: Iterable[T],
    *,
    generation_workers: int,
    scoring_workers: int,
    generate: Callable[[T], U],
    score: Callable[[U], Any],
    consume: Callable[[Any], None],
) -> None:
    """Run an unbounded producer/consumer pipeline.

    The queue has no capacity limit, so ``put`` never waits for a slow score.
    ``generate`` functions execute only in ``generation_workers``; ``score``
    functions execute only in the separate scoring threads.  This small generic
    core is also exercised by ``--pipeline-smoke`` without a model or judge.
    """
    item_list = list(items)
    generated_q: queue.SimpleQueue[U | object] = queue.SimpleQueue()
    result_q: queue.Queue[tuple[bool, Any]] = queue.Queue()

    def scoring_loop() -> None:
        while True:
            generated = generated_q.get()
            if generated is _STOP:
                return
            try:
                result_q.put((True, score(generated)))
            except BaseException as exc:  # propagate programmer failures to main
                result_q.put((False, exc))

    scorers = [
        threading.Thread(target=scoring_loop, name=f"eval-score-{idx}", daemon=False)
        for idx in range(scoring_workers)
    ]
    for scorer in scorers:
        scorer.start()

    try:
        # Futures are consumed as soon as generation returns; enqueueing a text is
        # independent of (and never waits for) every judge/simulator invocation.
        with ThreadPoolExecutor(max_workers=generation_workers, thread_name_prefix="eval-generate") as pool:
            futures = [pool.submit(generate, item) for item in item_list]
            for future in as_completed(futures):
                generated_q.put(future.result())
    finally:
        for _ in scorers:
            generated_q.put(_STOP)

    try:
        for _ in item_list:
            ok, result = result_q.get()
            if not ok:
                raise result
            consume(result)
    finally:
        for scorer in scorers:
            scorer.join()


def _is_iterative(args: argparse.Namespace, problem: dict[str, Any]) -> bool:
    return int(getattr(args, "frontiercs_iterative_rounds", 1)) > 1 and problem["data_source"] == "frontiercs"


def _preflight(args: argparse.Namespace, problems: list[dict[str, Any]]) -> None:
    if int(getattr(args, "frontiercs_iterative_rounds", 1)) > 1:
        raise SystemExit(
            "FATAL: the pipelined evaluator supports only single-shot scoring. "
            "Iterative FrontierCS re-prompts with judge feedback, so decoupling "
            "would change the request sequence."
        )
    unsupported = sorted({str(p.get("data_source")) for p in problems} - {"frontiercs", "frontiercs_research", "alebench"})
    if unsupported:
        raise SystemExit(f"FATAL: unsupported source(s) for pipelining: {unsupported}")
    if args.concurrency < 1 or args.score_concurrency < 1:
        raise SystemExit("FATAL: --concurrency and --score-concurrency must both be positive")


def _generation_item(
    args: argparse.Namespace,
    task: tuple[int, dict[str, Any], int],
) -> dict[str, Any]:
    """Generate exactly the old single-shot request, but do not call the scorer."""
    problem_idx, problem, sample_idx = task
    started = time.time()
    request_seed = None if args.seed is None else int(args.seed) + problem_idx * args.n_samples + sample_idx
    try:
        text, completion_tokens, generation_seconds = base._generate_one(args, problem["messages"], request_seed)
        generation_error = None
    except Exception as exc:
        # Matches _run_one's generation-failure values.  The scoring stage preserves
        # this exact error and does not try to judge an empty failed generation.
        text, completion_tokens, generation_seconds = "", None, 0.0
        generation_error = repr(exc)
    return {
        "problem": problem,
        "problem_idx": problem_idx,
        "sample_idx": sample_idx,
        "request_seed": request_seed,
        "started": started,
        "text": text,
        "completion_tokens": completion_tokens,
        "generation_seconds": generation_seconds,
        "generation_error": generation_error,
    }


def _score_item(args: argparse.Namespace, item: dict[str, Any]) -> dict[str, Any]:
    """Score an already-generated text and build the established record schema."""
    problem = item["problem"]
    text = item["text"]
    completion_tokens = item["completion_tokens"]
    generation_seconds = item["generation_seconds"]
    if item["generation_error"] is None:
        try:
            metrics = base._score(
                problem["data_source"],
                item["text"],
                problem["ground_truth"],
                args.judge_url,
                frontiercs_score_backend=args.frontiercs_score_backend,
            )
            error = None
        except Exception as exc:
            # _run_one's one large try/except resets these same fields when a
            # scorer fails. Keep failure records byte-for-byte compatible too,
            # not only successful generated texts and sampling requests.
            text, completion_tokens, generation_seconds = "", None, 0.0
            metrics = {"reward": 0.0, "score": 0.0}
            error = repr(exc)
    else:
        text, completion_tokens, generation_seconds = "", None, 0.0
        metrics = {"reward": 0.0, "score": 0.0}
        error = item["generation_error"]

    record: dict[str, Any] = {
        "data_source": problem["data_source"],
        "ground_truth": problem["ground_truth"],
        "problem_idx": item["problem_idx"],
        "sample_idx": item["sample_idx"],
        "completion_tokens": completion_tokens,
        "generation_seconds": generation_seconds,
        "total_seconds": time.time() - item["started"],
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
        "request_seed": item["request_seed"],
    }
    if args.save_text:
        record["text"] = text
    elif args.text_preview_chars > 0:
        record["text_preview"] = text[: args.text_preview_chars]
        record["text_chars"] = len(text)
    return record


def _write_record(
    record: dict[str, Any],
    *,
    out: Any,
    records: dict[tuple[str, str, int], dict[str, Any]],
    planned_keys: set[tuple[str, str, int]],
    pbar: tqdm,
) -> None:
    key = (record["data_source"], record["ground_truth"], int(record["sample_idx"]))
    out.write(json.dumps(record, ensure_ascii=False, default=base._json_default) + "\n")
    out.flush()
    records[key] = record
    reward = record.get("metrics", {}).get("reward")
    reward_str = "NA" if reward is None else f"{float(reward):.4f}"
    if record.get("error"):
        print(
            f"ERROR {record['data_source']} {record['ground_truth']} sample={record['sample_idx']}: {record['error']}",
            flush=True,
        )
    print(
        f"SAMPLE {len(records)}/{len(planned_keys)} {record['data_source']} "
        f"{record['ground_truth']} sample={record['sample_idx']} reward={reward_str} "
        f"gen_tokens={record['completion_tokens']} gen_sec={record['generation_seconds']:.1f}",
        flush=True,
    )
    pbar.update(1)


def _load_tasks(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[tuple[int, dict[str, Any], int]], set[tuple[str, str, int]], dict, Path, Path]:
    problems: list[dict[str, Any]] = []
    if args.source in ("both", "frontiercs", "all"):
        problems.extend(base._load_problems(args.frontiercs_data, "frontiercs", args.limit_frontiercs, frontiercs_prompt_source=args.frontiercs_prompt_source))
    if args.source in ("both", "alebench", "all"):
        problems.extend(base._load_problems(args.alebench_data, "alebench", args.limit_alebench))
    if args.source in ("research", "all"):
        problems.extend(base._load_problems(args.research_data, "frontiercs_research", args.limit_research))

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
    print(
        f"vLLM endpoint: {args.base_url} model={args.model} generation_concurrency={args.concurrency} "
        f"score_concurrency={args.score_concurrency}",
        flush=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = args.samples_jsonl or (args.output_dir / "samples.jsonl")
    summary_path = args.summary_json or (args.output_dir / "summary.json")
    planned_keys = {(p["data_source"], p["ground_truth"], sample_idx) for p in problems for sample_idx in range(args.n_samples)}
    existing = base._load_existing(samples_path) if args.resume else {}
    records = {key: rec for key, rec in existing.items() if key in planned_keys and base._record_compatible(rec, args)}
    if records:
        print(f"Resuming from {samples_path}: {len(records)} completed samples", flush=True)
    tasks = [
        (problem_idx, problem, sample_idx)
        for problem_idx, problem in enumerate(problems)
        for sample_idx in range(args.n_samples)
        if (problem["data_source"], problem["ground_truth"], sample_idx) not in records
    ]
    return problems, tasks, planned_keys, records, samples_path, summary_path


def _pipeline_smoke() -> None:
    """CPU-only proof that slow scoring does not delay any generation worker."""
    generation_finished: list[float] = []
    score_finished: list[float] = []
    lock = threading.Lock()

    def fake_generate(value: int) -> int:
        time.sleep(0.01)
        with lock:
            generation_finished.append(time.monotonic())
        return value

    def fake_score(value: int) -> int:
        time.sleep(0.06)  # deliberately 6x slower than generation
        with lock:
            score_finished.append(time.monotonic())
        return value

    got: list[int] = []
    run_generation_scoring_pipeline(
        range(12), generation_workers=3, scoring_workers=1,
        generate=fake_generate, score=fake_score, consume=got.append,
    )
    assert sorted(got) == list(range(12))
    assert len(generation_finished) == len(score_finished) == 12
    # With a coupled worker the first 0.06s score would hold one of three
    # generators.  Here all four 0.01s generation waves finish before any score.
    assert max(generation_finished) < min(score_finished)
    print("PIPELINE_SMOKE_PASS: all 12 generations completed before the first slow score finished")


def _parse_args() -> argparse.Namespace | None:
    outer = argparse.ArgumentParser(add_help=False)
    outer.add_argument("--score-concurrency", type=int, default=int(os.environ.get("EVAL_SCORE_CONCURRENCY", "6")))
    outer.add_argument("--pipeline-smoke", action="store_true")
    outer_args, remaining = outer.parse_known_args()
    if outer_args.pipeline_smoke:
        _pipeline_smoke()
        return None
    # Delegate every established evaluator option verbatim to the original parser.
    sys.argv = [sys.argv[0], *remaining]
    args = base.parse_args()
    args.score_concurrency = outer_args.score_concurrency
    return args


def main() -> None:
    args = _parse_args()
    if args is None:
        return
    random.seed(args.seed)
    np.random.seed(args.seed)
    problems, tasks, planned_keys, records, samples_path, summary_path = _load_tasks(args)
    _preflight(args, problems)
    if args.dry_run:
        return

    if any(problem["data_source"] == "alebench" for _, problem, _ in tasks):
        from verl.utils.reward_score.ale_selftest import ale_compile_selftest

        ale_compile_selftest(base.alebench.AleInfraError)

    samples_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    print(
        f"[pipelined] generation workers={args.concurrency}; scoring workers={args.score_concurrency}; "
        "generation queue is unbounded",
        flush=True,
    )
    with samples_path.open(mode, encoding="utf-8") as out:
        pbar = tqdm(total=len(tasks), desc="Eval", unit="sample")
        try:
            run_generation_scoring_pipeline(
                tasks,
                generation_workers=args.concurrency,
                scoring_workers=args.score_concurrency,
                generate=lambda task: _generation_item(args, task),
                score=lambda item: _score_item(args, item),
                consume=lambda record: _write_record(record, out=out, records=records, planned_keys=planned_keys, pbar=pbar),
            )
        finally:
            pbar.close()

    summary = base._summarize(list(records.values()), args.n_samples, args.seed)
    summary["config"] = vars(args)
    summary["samples_jsonl"] = str(samples_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=base._json_default) + "\n")
    print(f"Saved samples to {samples_path}", flush=True)
    print(f"Saved summary to {summary_path}", flush=True)
    base._print_metrics(summary)
    planned_records = [records[key] for key in planned_keys if key in records]
    missing_count = len(planned_keys) - len(planned_records)
    error_records = [record for record in planned_records if record.get("error")]
    if missing_count:
        print(f"ERROR: {missing_count} planned samples were not completed", file=sys.stderr, flush=True)
        raise SystemExit(2)
    if len(error_records) > args.max_errors:
        examples = "; ".join(str(record.get("error"))[:240] for record in error_records[:3])
        print(f"ERROR: {len(error_records)} sample(s) failed, above --max-errors={args.max_errors}. Examples: {examples}", file=sys.stderr, flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
