#!/usr/bin/env python3
"""Faithful ALE-Bench agentic runner (repeated-sampling + self-refinement).

WHY THIS EXISTS
---------------
`generate_alebench_lite_vllm_oneshot.sh` + `evaluate_alebench_lite_outputs.py`
implement a ONE-SHOT simplification: a single generation per sample, best-of-k
scored against the private judge. That is NOT the official ALE-Bench protocol.

The official protocol (ALE-Bench/src/ale_bench_eval/{__main__,scaffolds,
selection}.py) is a 4-phase agentic loop, PER PROBLEM:

  Phase 1 — Repeated sampling: draw ``n_repeated_sampling`` INDEPENDENT
            single-turn solutions from the same initial prompt; score each on
            the PUBLIC cases (session.public_eval).
  Phase 2 — Selection: pick ONE repeated-sampling solution by
            ``selection_method`` (default "median" of public abs score;
            "best" also supported). (selection.py:34-84)
  Phase 3 — Self-refinement: seed a conversation from the selected solution and
            iterate ``n_self_refine`` turns; each turn feeds the PUBLIC-eval
            feedback back into the SAME growing message history
            (scaffolds.py:143-291).
  Phase 4 — Private evaluation: run the true hidden judge (session.private_eval)
            on the selected repeated-sampling solution AND on the best
            self-refine solution at power-of-two refine-count checkpoints
            (1,2,4,...,n) (__main__.py:35-40,137-162). Report rank + performance.

This runner reproduces that loop faithfully against an OpenAI-compatible
(vLLM) chat endpoint, so it needs NO pydantic_ai / provider stack. It reuses
the official ALE-Bench `ale_bench` session library for evaluation and scoring,
so raw-score -> overall_absolute_score -> standings-rank -> performance is the
OFFICIAL mapping (ale_bench/data.py + session.py).

Faithful defaults (mirroring the official harness):
  * session_duration              = 240h  (safe_ale_session.py:9,24; makes the
                                    contest wall-clock non-binding, so the
                                    iteration count is the budget)
  * code_language / judge_version = cpp20 / 202301  (__main__.py:261-262)
  * selection_method              = median  (__main__.py:266; selection.py:37)
  * time-limit / memory-limit     = the problem's own constraints (the judge
                                    uses them automatically inside public/private
                                    eval; per-problem, per ale_bench)
  * prompts                       = byte-for-byte the official English system /
                                    consideration / implementation / feedback
                                    texts (prompts/texts.py)

Two knobs match the two dominant published settings; set them on the CLI:
  --n-repeated-sampling  (paper uses e.g. 1 or 8)
  --n-self-refine        (paper "iterative refinement" uses e.g. 4/8)

DIVERGENCE we do NOT reproduce (documented): the official `safe_generation`
retries when no code block is found and re-prompts with `no_code_block_message`;
we retry the generation instead and fall back to a compile-error stub for empty
code (same net scoring effect: an unparseable/empty solution gets the worst
score). We also do not send statement images (text statements only), matching
`use_statement_image=False` (__main__.py:267).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from string import Template
from typing import Any

import statistics

# ---------------------------------------------------------------------------
# Official prompt texts (English), copied verbatim from
# ALE-Bench/src/ale_bench_eval/prompts/texts.py so this runner has no import
# dependency on the heavy ale_bench_eval package (which pulls in pydantic_ai).
# Keep these in sync if the upstream prompts change.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_EN = (
    "You are a world-class algorithm engineer, and you are very good at programming. "
    "Now, you are participating in a programming contest. "
    "You are asked to solve a heuristic problem, known as an NP-hard problem."
)
CONSIDERATION_PROMPT_EN = (
    "There is a problem statement at the end of this message. "
    "First, please analyze the problem statement. "
    "Please think about the essential points of the problem and possible algorithms "
    "to get higher rank in the contest. "
)
IMPLEMENTATION_SPECIFIC_PROMPT_EN = Template(
    "Next, please implement your solution in ${language}. "
    "Your solution code should be written in the ${code_block} code block. "
    "You can use external libraries as follows:\n${libraries}\n\n"
)
PROBLEM_HEADER_PROMPT_EN = Template(
    "[Problem statement]\n"
    "Execution time limit: ${time_limit} sec / Memory limit: ${memory_limit} MiB\n"
)
FEEDBACK_PROMPT_EN = Template(
    "${feedback}\n\n"
    "Based on the above feedback, please consider the ways to improve your solution. "
    "Firstly, please analyze this given feedback and list what insights can be gained from it. "
    "Then, based on the insights, please refine your code to achieve better performance. "
    "It can be a simple bug fix, the introduction of a new algorithm, or any degree of "
    "change from minor to major. "
)
REFINE_SPECIFIC_PROMPT_EN = Template("Your solution code should be written in the ${code_block} code block.")

# Language display strings + libraries (judge 202301), from texts.py.
CODE_LANGUAGE_STRING_202301 = {
    "cpp17": "C++17 (gcc 12.2.0)",
    "cpp20": "C++20 (gcc 12.2.0)",
    "cpp23": "C++23 (gcc 12.2.0)",
}
CODE_LANGUAGE_LIBRARIES_202301 = {
    "cpp17": "- AC Library@1.5.1\n- Boost@1.82.0",
    "cpp20": "- AC Library@1.5.1\n- Boost@1.82.0\n- GMP@6.2.1\n- Eigen@3.4.0-2ubuntu2",
    "cpp23": "- AC Library@1.5.1\n- Boost@1.82.0\n- GMP@6.2.1\n- Eigen@3.4.0-2ubuntu2",
}
CODE_BLOCK_STRING = {"cpp17": "```cpp ```", "cpp20": "```cpp ```", "cpp23": "```cpp ```"}

# Worst score by score direction (selection.py:30).
WORST_MAXIMIZE = -1
WORST_MINIMIZE = 1_000_000_000_000_000_000

# Official session wall-clock override (safe_ale_session.py:9).
SESSION_DURATION_HOURS_DEFAULT = 240

_CPP_BLOCK_RE = re.compile(r"```cpp\s*\n(.+?)\n```", re.DOTALL)


def utc_now() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).isoformat()


def add_ale_bench_to_path(src: Path) -> None:
    src = src.resolve()
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def build_system_prompt() -> str:
    return SYSTEM_PROMPT_EN


def build_initial_user_message(problem: Any, code_language: str) -> str:
    """Reproduce create_initial_message (English, text-only) from builder.py."""
    parts = [CONSIDERATION_PROMPT_EN]
    parts.append(
        IMPLEMENTATION_SPECIFIC_PROMPT_EN.substitute(
            language=CODE_LANGUAGE_STRING_202301.get(code_language, code_language),
            code_block=CODE_BLOCK_STRING.get(code_language, "```cpp ```"),
            libraries=CODE_LANGUAGE_LIBRARIES_202301.get(code_language, "- (none)"),
        )
    )
    parts.append(
        PROBLEM_HEADER_PROMPT_EN.substitute(
            time_limit=problem.constraints.time_limit,
            memory_limit=problem.constraints.memory_limit // 1024 // 1024,
        )
    )
    parts.append(problem.statement)
    return "".join(parts)


def result_feedback(result: Any) -> str:
    """Reproduce result_feedback() from builder.py for a public Result."""
    from ale_bench.result import JudgeResult

    if result is None:
        return "No public result is available. Mainly because no valid code block was found."
    feedback = f"[Public test result]\nOverall judge result: {result.overall_judge_result.value}\n"
    if result.overall_judge_result == JudgeResult.ACCEPTED:
        feedback += f"Overall absolute score: {result.overall_absolute_score}\n"
        feedback += "\n".join(
            f"- Case {i}: {c.absolute_score}" for i, c in enumerate(result.case_results, 1)
        )
    else:
        sel = 0
        for idx, c in enumerate(result.case_results):
            if c.judge_result == result.overall_judge_result:
                sel = idx
                break
        c = result.case_results[sel]
        feedback += (
            f"- Case {sel + 1}:\n"
            f"    Absolute score: {c.absolute_score}\n"
            f"    Execution time: {c.execution_time:.3f} sec\n"
            f"    Memory usage: {c.memory_usage // 1024 // 1024} MB\n"
            f"    Standard error: \"{c.error_str}\"\n"
            f"    Message: \"{c.message}\""
        )
    return feedback


def build_feedback_message(result: Any, code_language: str) -> str:
    fb = result_feedback(result)
    return FEEDBACK_PROMPT_EN.substitute(feedback=fb) + REFINE_SPECIFIC_PROMPT_EN.substitute(
        code_block=CODE_BLOCK_STRING.get(code_language, "```cpp ```")
    )


def extract_code(response_text: str) -> str:
    """Official get_code_from_response: last ```cpp block; else empty."""
    matches = _CPP_BLOCK_RE.findall(response_text or "")
    if matches:
        return matches[-1]
    return ""


def worst_score(score_type_value: str) -> int:
    return WORST_MAXIMIZE if score_type_value == "maximize" else WORST_MINIMIZE


def power_of_two_indices(n: int) -> list[int]:
    """__main__.py:35-40 — 1,2,4,...,<=n, always including n."""
    if n <= 0:
        return []
    out, i = [], 1
    while i < n:
        out.append(i)
        i *= 2
    out.append(n)
    return sorted(set(out))


class ChatClient:
    def __init__(self, base_url: str, model: str, api_key: str, timeout: float, max_retries: int):
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> tuple[str, dict[str, Any]]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                req = urllib.request.Request(self.endpoint, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
                parsed = json.loads(raw)
                choice = parsed["choices"][0]
                msg = choice.get("message") or {}
                text = msg.get("content")
                if not isinstance(text, str):
                    text = choice.get("text", "") or ""
                usage = parsed.get("usage") or {}
                return text, {"finish_reason": choice.get("finish_reason"), "usage": usage}
            except (urllib.error.URLError, socket.timeout, KeyError, json.JSONDecodeError) as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(min(60.0, 5.0 * (2 ** attempt)))
        raise RuntimeError(f"generation failed after {self.max_retries + 1} attempts: {last_exc!r}")


def public_eval_score(session: Any, code: str, code_language: str, judge_version: str) -> tuple[Any, int]:
    """public_eval + official score extraction (scaffolds.py:97-106)."""
    from ale_bench.result import JudgeResult

    st = session.problem.metadata.score_type.value
    if not code.strip():
        return None, worst_score(st)
    try:
        result = session.public_eval(code, code_language, judge_version=judge_version)
    except Exception:
        return None, worst_score(st)
    score = (
        result.overall_absolute_score
        if result.overall_judge_result == JudgeResult.ACCEPTED
        else worst_score(st)
    )
    return result, score


def select_repeated_sampling(scores: list[int], score_type_value: str, method: str) -> int:
    """selection.py:34-84. Returns index into `scores`."""
    import numpy as np

    arr = np.array(scores, dtype=float)
    if method == "median":
        med = float(np.median(arr))
        return int(np.argmin(np.abs(arr - med)))
    # "best"
    if score_type_value == "maximize":
        return int(np.argmax(arr))
    return int(np.argmin(arr))


def select_best(scores: list[int], score_type_value: str) -> int:
    import numpy as np

    arr = np.array(scores, dtype=float)
    if score_type_value == "maximize":
        return int(np.argmax(arr))
    return int(np.argmin(arr))


def reset_private_eval_quota(session: Any) -> None:
    """Allow multiple private evals on one session (evaluate.py:152-158)."""
    from ale_bench.result import ResourceUsage

    u = session.current_resource_usage
    session._current_resource_usage = ResourceUsage(  # noqa: SLF001
        num_case_gen=u.num_case_gen,
        num_case_eval=u.num_case_eval,
        num_call_public_eval=u.num_call_public_eval,
        num_call_private_eval=0,
        execution_time_case_eval=u.execution_time_case_eval,
    )


def private_eval(session: Any, code: str, code_language: str, judge_version: str) -> dict[str, Any]:
    if not code.strip():
        # Empty code -> official CE fallback (evaluate.py:107-110). A CE gets the
        # worst rank/performance from the judge, same as an empty submission.
        code = "this code intentionally fails\n"
    reset_private_eval_quota(session)
    result, rank, performance = session.private_eval(code, code_language, judge_version=judge_version)
    return {
        "overall_judge_result": result.overall_judge_result.value,
        "overall_absolute_score": result.overall_absolute_score,
        "overall_relative_score": result.overall_relative_score,
        "rank": int(rank),
        "performance": int(performance),
    }


def run_problem(
    problem_id: str,
    client: ChatClient,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any]:
    import ale_bench

    session = ale_bench.start(
        problem_id,
        lite_version=args.lite,
        session_duration=dt.timedelta(hours=args.session_hours),
        num_workers=args.num_workers,
        run_visualization_server=False,
    )
    problem = session.problem
    score_type = problem.metadata.score_type.value
    system_prompt = build_system_prompt()
    initial_user = build_initial_user_message(problem, args.code_language)

    per_problem: dict[str, Any] = {
        "problem_id": problem_id,
        "score_type": score_type,
        "lite": args.lite,
        "code_language": args.code_language,
        "judge_version": args.judge_version,
        "n_repeated_sampling": args.n_repeated_sampling,
        "n_self_refine": args.n_self_refine,
        "selection_method": args.selection_method,
        "started_at": utc_now(),
    }
    try:
        # -- Phase 1: repeated sampling (independent single-turn generations) --
        rs: list[dict[str, Any]] = []
        for i in range(args.n_repeated_sampling):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": initial_user},
            ]
            text, meta = client.generate(messages, args.temperature, args.top_p, args.max_tokens)
            code = extract_code(text)
            pub_result, score = public_eval_score(session, code, args.code_language, args.judge_version)
            rs.append(
                {
                    "index": i,
                    "code": code,
                    "public_score": score,
                    "public_judge": (pub_result.overall_judge_result.value if pub_result else None),
                    "response_text": text,
                    "usage": meta.get("usage"),
                }
            )
            print(f"[{problem_id}] repeated_sampling {i + 1}/{args.n_repeated_sampling} "
                  f"public_score={score}", flush=True)

        rs_scores = [r["public_score"] for r in rs]
        sel_idx = select_repeated_sampling(rs_scores, score_type, args.selection_method)
        selected = rs[sel_idx]
        per_problem["repeated_sampling"] = {
            "public_scores": rs_scores,
            "selected_index": sel_idx,
            "selected_public_score": selected["public_score"],
        }

        # Seed the self-refine conversation from the selected solution.
        # The assistant turn is the selected solution's raw response, then the
        # feedback message from ITS public result (public re-eval, cheap).
        selected_pub_result, _ = public_eval_score(
            session, selected["code"], args.code_language, args.judge_version
        )
        message_history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_user},
            {"role": "assistant", "content": selected["response_text"]},
        ]

        # -- Phase 3: self-refinement (shared growing conversation) --
        sr: list[dict[str, Any]] = []
        cur_pub_result = selected_pub_result
        for i in range(args.n_self_refine):
            fb = build_feedback_message(cur_pub_result, args.code_language)
            turn_messages = message_history + [{"role": "user", "content": fb}]
            text, meta = client.generate(turn_messages, args.temperature, args.top_p, args.max_tokens)
            code = extract_code(text)
            cur_pub_result, score = public_eval_score(
                session, code, args.code_language, args.judge_version
            )
            sr.append({"index": i, "code": code, "public_score": score, "response_text": text})
            message_history = turn_messages + [{"role": "assistant", "content": text}]
            print(f"[{problem_id}] self_refine {i + 1}/{args.n_self_refine} "
                  f"public_score={score}", flush=True)

        # -- Phase 4: private evaluation of candidates --
        # (a) selected repeated-sampling solution -> method "repeated_sampling"
        methods: dict[str, dict[str, Any]] = {}
        methods["repeated_sampling"] = private_eval(
            session, selected["code"], args.code_language, args.judge_version
        )
        print(f"[{problem_id}] PRIVATE repeated_sampling -> "
              f"rank={methods['repeated_sampling']['rank']} "
              f"perf={methods['repeated_sampling']['performance']}", flush=True)

        # (b) best self-refine solution at power-of-two checkpoints
        if sr:
            sr_scores = [s["public_score"] for s in sr]
            for k in power_of_two_indices(args.n_self_refine):
                # best among the first k refinements (indices 0..k-1)
                subset = sr_scores[:k]
                best_i = select_best(subset, score_type)
                best_code = sr[best_i]["code"]
                methods[f"self_refine_{k}"] = private_eval(
                    session, best_code, args.code_language, args.judge_version
                )
                print(f"[{problem_id}] PRIVATE self_refine_{k} -> "
                      f"rank={methods[f'self_refine_{k}']['rank']} "
                      f"perf={methods[f'self_refine_{k}']['performance']}", flush=True)

        per_problem["methods"] = methods
        # Best-so-far headline for this problem = max performance across methods.
        best_method = max(methods.items(), key=lambda kv: kv[1]["performance"])
        per_problem["best_method"] = best_method[0]
        per_problem["best_performance"] = best_method[1]["performance"]
        per_problem["best_rank"] = best_method[1]["rank"]
    finally:
        session.close()

    per_problem["finished_at"] = utc_now()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{problem_id}.json").write_text(
        json.dumps(per_problem, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return per_problem


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Reproduce analyze_results.aggregate_results: per-method mean/median rank
    and performance across the problem set."""
    by_method: dict[str, dict[str, list[float]]] = {}
    for r in results:
        for name, m in r.get("methods", {}).items():
            slot = by_method.setdefault(name, {"ranks": [], "performances": []})
            slot["ranks"].append(float(m["rank"]))
            slot["performances"].append(float(m["performance"]))
    summary: dict[str, Any] = {}
    for name, slot in by_method.items():
        perfs = slot["performances"]
        ranks = slot["ranks"]
        summary[name] = {
            "count": len(perfs),
            "mean_performance": statistics.mean(perfs) if perfs else None,
            "median_performance": statistics.median(perfs) if perfs else None,
            "mean_rank": statistics.mean(ranks) if ranks else None,
            "median_rank": statistics.median(ranks) if ranks else None,
        }
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default=os.environ.get("ALE_BASE_URL", "http://localhost:8000/v1"),
                   help="OpenAI-compatible chat endpoint base (…/v1).")
    p.add_argument("--model", default=os.environ.get("ALE_MODEL", "Qwen3.5-9B"))
    p.add_argument("--api-key", default=os.environ.get("API_KEY", "dummy"))
    p.add_argument("--problems", nargs="*", default=None,
                   help="Problem ids. Default: the whole lite/full set.")
    p.add_argument("--lite", dest="lite", action="store_true", default=True,
                   help="Lite problem+seed set (default; 10 problems, 5 public + 10%% private seeds).")
    p.add_argument("--no-lite", dest="lite", action="store_false",
                   help="Full problem+seed set (leaderboard-valid rank/performance).")
    p.add_argument("--n-repeated-sampling", type=int, default=1)
    p.add_argument("--n-self-refine", type=int, default=0,
                   help="Self-refine turns after selection (0 = repeated-sampling only).")
    p.add_argument("--selection-method", choices=["median", "best"], default="median")
    p.add_argument("--code-language", default="cpp20")
    p.add_argument("--judge-version", default="202301")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--session-hours", type=float, default=SESSION_DURATION_HOURS_DEFAULT)
    # Sampling: harness itself sets none; per-model config decides. Faithful
    # Claude configs use temperature 1.0. Expose all three.
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=32000)
    p.add_argument("--request-timeout", type=float, default=3600.0)
    p.add_argument("--max-retries", type=int, default=30)
    p.add_argument("--ale-bench-src", type=Path, default=Path("ALE-Bench/src"))
    p.add_argument("--ale-bench-data", type=Path, default=Path("data/alebench/local_data"))
    p.add_argument("--ale-bench-cache", type=Path, default=Path("eval_work/alebench_cache"))
    p.add_argument("--output-root", type=Path,
                   default=Path("outputs/alebench_faithful"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    add_ale_bench_to_path(args.ale_bench_src)
    os.environ["ALE_BENCH_DATA"] = str(args.ale_bench_data.resolve())
    os.environ["ALE_BENCH_CACHE"] = str(args.ale_bench_cache.resolve())
    args.ale_bench_cache.mkdir(parents=True, exist_ok=True)

    from ale_bench.data import list_problem_ids

    problems = args.problems or list_problem_ids(lite_version=args.lite)
    print(f"Faithful ALE-Bench run: {len(problems)} problems "
          f"(lite={args.lite}); n_rs={args.n_repeated_sampling} "
          f"n_sr={args.n_self_refine} selection={args.selection_method}", flush=True)

    client = ChatClient(
        args.base_url, args.model, args.api_key, args.request_timeout, args.max_retries
    )

    run_dir = args.output_root / f"{args.model.replace('/', '_')}"
    per_problem_dir = run_dir / "problems"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "created_at": utc_now(),
                "host": socket.gethostname(),
                "base_url": args.base_url,
                "model": args.model,
                "lite": args.lite,
                "problems": problems,
                "n_repeated_sampling": args.n_repeated_sampling,
                "n_self_refine": args.n_self_refine,
                "selection_method": args.selection_method,
                "code_language": args.code_language,
                "judge_version": args.judge_version,
                "session_hours": args.session_hours,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    results: list[dict[str, Any]] = []
    for pid in problems:
        cached = per_problem_dir / f"{pid}.json"
        if cached.exists():
            print(f"[{pid}] cached", flush=True)
            results.append(json.loads(cached.read_text()))
            continue
        try:
            results.append(run_problem(pid, client, args, per_problem_dir))
        except Exception as exc:  # noqa: BLE001
            print(f"[{pid}] FAILED: {exc!r}", flush=True)
            results.append({"problem_id": pid, "error": repr(exc), "methods": {}})

    summary = aggregate(results)
    (run_dir / "summary.json").write_text(
        json.dumps({"per_method": summary, "problems": results}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("\n=== per-method summary (mean/median performance & rank) ===", flush=True)
    for name, s in summary.items():
        print(
            f"{name}: n={s['count']} "
            f"mean_perf={s['mean_performance']} median_perf={s['median_performance']} "
            f"mean_rank={s['mean_rank']} median_rank={s['median_rank']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
