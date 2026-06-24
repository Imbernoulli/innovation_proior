#!/usr/bin/env python3
"""Run a set of MLS-Bench CPU tasks against a local vLLM OpenAI server, IN ONE JOB.

This is the in-job worker pool for the FrontierSmith MLS-Bench CPU eval. It does
NOT submit any Slurm jobs: each task is run with `mlsbench agent` using a config
that has container_runtime=apptainer and NO `slurm:` block, so MLS-Bench runs the
task's Apptainer container locally (direct subprocess) on the current node
(see MLS-Bench src/mlsbench/agent/tools.py _run_all_cmds_direct). After each agent
run we call `mlsbench score <task>` to read back the normalized [0,1] task score.

Robustness:
  - Each task runs in its own subprocess with a hard wall-clock timeout. A slow or
    hanging task is killed and recorded as status="timeout" without blocking others.
  - A failing task is recorded as status="error" (with the tail of its log) and
    never aborts the whole run.
  - A bounded local worker pool (CONCURRENCY) runs tasks in parallel within the job.
  - summary.json is written incrementally after every task completes, so partial
    results survive even if the job is killed.

Env / args (all optional, sensible defaults):
  MODEL          model string passed to `mlsbench agent --model` (default vllm/qwen3-8b)
  MLSBENCH_ROOT  MLS-Bench repo root (default /scratch/gpfs/CHIJ/bohan/MLS-Bench)
  MLSBENCH_CONFIG path to the generated MLS-Bench config yaml (REQUIRED via --config)
  CONCURRENCY    number of tasks to run in parallel (default 4)
  TASK_TIMEOUT   per-task wall-clock timeout in seconds (default 5400 = 90 min)
  EVAL_RESEARCHER_YEAR  if set, MLS-Bench prepends the researcher system prompt
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# The 20 non-agent CPU target tasks (frozen public-140 set). The 2 agent CPU
# tasks (agent-tool-reasoning, mas-topology) are intentionally excluded: they are
# agent-loop tasks AND lack local SIF images.
DEFAULT_CPU_TASKS = [
    "causal-discovery-discrete",
    "causal-observational-linear-gaussian",
    "causal-observational-linear-non-gaussian",
    "causal-observational-nonlinear",
    "causal-treatment-effect",
    "ml-active-learning",
    "ml-anomaly-detection",
    "ml-calibration",
    "ml-clustering-algorithm",
    "ml-dimensionality-reduction",
    "ml-ensemble-boosting",
    "ml-missing-data-imputation",
    "ml-selective-deferral",
    "ml-subgroup-calibration-shift",
    "ml-symbolic-regression",
    "mlsys-moe-load-balance",
    "optimization-evolution-strategy",
    "optimization-hyperparameter-search",
    "optimization-multi-objective",
    "optimization-nas",
]


def log(msg: str) -> None:
    print(f"[mlsbench-cpu] {msg}", flush=True)


def run_one(
    task: str,
    *,
    root: Path,
    config: str,
    model: str,
    log_dir: Path,
    timeout: int,
    python_exe: str,
    env_base: dict,
) -> dict:
    """Run `mlsbench agent` for one task, then `mlsbench score`. Return a result dict."""
    t0 = time.time()
    task_log = log_dir / f"{task}.log"
    rec: dict = {"task": task, "model": model, "status": "unknown", "score": None}

    env = dict(env_base)
    # Make concurrent runs of the same model log to distinct dirs (MLS-Bench hook).
    env["MLSBENCH_LOG_LABEL"] = f"cc-{os.environ.get('SLURM_JOB_ID', 'manual')}-{task}"

    agent_cmd = [
        python_exe, "-m", "mlsbench", "agent", task,
        "--model", model,
        "--config", config,
    ]
    with open(task_log, "w") as lf:
        lf.write(f"### AGENT  {task}\n# {' '.join(agent_cmd)}\n\n")
        lf.flush()
        try:
            proc = subprocess.run(
                agent_cmd, cwd=str(root), env=env,
                stdout=lf, stderr=subprocess.STDOUT, timeout=timeout,
            )
            rec["agent_returncode"] = proc.returncode
            if proc.returncode != 0:
                rec["status"] = "agent_failed"
        except subprocess.TimeoutExpired:
            rec["status"] = "timeout"
            rec["agent_returncode"] = None
            lf.write(f"\n### TIMEOUT after {timeout}s\n")

    # Always attempt to score: the agent auto-submits its last test even on a
    # non-zero exit, so a leaderboard row may exist regardless.
    score_cmd = [
        python_exe, "-m", "mlsbench", "score", task,
        "--model", model, "--format", "json",
    ]
    try:
        # IMPORTANT: keep stderr SEPARATE from stdout. `mlsbench score` prints
        # its JSON to stdout but also emits Python UserWarnings (scoring
        # calibration: solve_gamma degenerate / pathological bounded_power) to
        # stderr. Merging them (stderr=STDOUT) prepends warning text to the
        # captured stream, so json.loads() fails with "Expecting value: line 1
        # column 1 (char 0)" and a perfectly good score is discarded. Capture
        # stdout for parsing; log stderr separately.
        out = subprocess.run(
            score_cmd, cwd=str(root), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=600, text=True,
        )
        with open(task_log, "a") as lf:
            lf.write("\n### SCORE\n# " + " ".join(score_cmd) + "\n")
            lf.write(out.stdout or "")
            if out.stderr:
                lf.write("\n### SCORE (stderr)\n")
                lf.write(out.stderr)
        # Primary parse: stdout should be pure JSON (warnings go to stderr, which
        # we keep separate above). Defensive fallback: if any non-JSON preamble
        # ever leaks onto stdout (e.g. a banner, or warnings misrouted to stdout by
        # a future env/config change), recover by slicing from the first top-level
        # '{' to the matching last '}', so a valid computed score is never silently
        # discarded as it was before the stderr/stdout split.
        try:
            data = json.loads(out.stdout)
        except json.JSONDecodeError:
            s = out.stdout or ""
            i, j = s.find("{"), s.rfind("}")
            if i == -1 or j == -1 or j < i:
                raise
            data = json.loads(s[i : j + 1])
            rec["score_parse_recovered"] = True
        entries = data.get(task, [])
        # Pick the entry whose model matches ours (InteractiveAgent uses the raw
        # --model string as the leaderboard model column).
        match = next((e for e in entries if e.get("model") == model), None)
        if match is None and entries:
            match = entries[-1]
        if match is not None:
            rec["score"] = match.get("task_score")
            rec["settings"] = [
                {"name": s.get("name"), "score": s.get("score")}
                for s in match.get("settings", [])
            ]
            if rec["status"] in ("unknown", "agent_failed", "timeout") and rec["score"] is not None:
                # We got a score even though the agent exited non-zero/timed out.
                rec["status"] = "scored" if rec["status"] == "unknown" else rec["status"] + "+scored"
            elif rec["status"] == "unknown":
                rec["status"] = "scored"
        else:
            if rec["status"] == "unknown":
                rec["status"] = "no_leaderboard_row"
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:  # noqa: BLE001
        rec.setdefault("score_error", str(e))
        if rec["status"] == "unknown":
            rec["status"] = "score_failed"

    rec["elapsed_s"] = round(time.time() - t0, 1)
    rec["log"] = str(task_log)
    log(f"DONE {task:42s} status={rec['status']:18s} score={rec['score']} ({rec['elapsed_s']}s)")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="MLS-Bench config yaml (no slurm block)")
    ap.add_argument("--model", default=os.environ.get("MODEL", "vllm/qwen3-8b"))
    ap.add_argument("--root", default=os.environ.get("MLSBENCH_ROOT", "/scratch/gpfs/CHIJ/bohan/MLS-Bench"))
    ap.add_argument("--out", required=True, help="summary.json output path")
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--concurrency", type=int, default=int(os.environ.get("CONCURRENCY", "4")))
    ap.add_argument("--timeout", type=int, default=int(os.environ.get("TASK_TIMEOUT", "5400")))
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="explicit task list (default: the 20 CPU targets)")
    ap.add_argument("--limit", type=int, default=None, help="run only the first N tasks")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir).resolve() if args.log_dir else out_path.parent / "task_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    tasks = list(args.tasks) if args.tasks else list(DEFAULT_CPU_TASKS)
    if args.limit:
        tasks = tasks[: args.limit]

    # Base env for every subprocess: ensure the in-tree src is importable, offline,
    # and inline-apptainer (never the local scheduler daemon).
    env_base = dict(os.environ)
    env_base["PYTHONPATH"] = f"{root}/src:" + env_base.get("PYTHONPATH", "")
    env_base["MLSBENCH_SCHEDULER_MANAGED"] = "1"   # defensive: force inline apptainer
    env_base["MLSBENCH_NO_PREBUILT"] = env_base.get("MLSBENCH_NO_PREBUILT", "1")  # offline: never pull
    env_base.setdefault("HF_HUB_OFFLINE", "1")
    env_base.setdefault("TRANSFORMERS_OFFLINE", "1")

    log(f"root={root}")
    log(f"config={args.config} model={args.model}")
    log(f"tasks={len(tasks)} concurrency={args.concurrency} timeout={args.timeout}s")
    log(f"out={out_path}")
    log(f"researcher_year={env_base.get('EVAL_RESEARCHER_YEAR', '<unset>')}")

    results: dict[str, dict] = {}
    lock = threading.Lock()

    def write_summary() -> None:
        scored = [r for r in results.values() if isinstance(r.get("score"), (int, float))]
        scores = [r["score"] for r in scored]
        summary = {
            "model": args.model,
            "config": args.config,
            "researcher_year": env_base.get("EVAL_RESEARCHER_YEAR"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "n_tasks": len(tasks),
            "n_completed": len(results),
            "n_scored": len(scored),
            "mean_score": (sum(scores) / len(scores)) if scores else None,
            "tasks": [results[t] for t in tasks if t in results],
        }
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(summary, indent=2))
        tmp.replace(out_path)

    write_summary()  # write skeleton up front
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        futs = {
            ex.submit(
                run_one, t, root=root, config=args.config, model=args.model,
                log_dir=log_dir, timeout=args.timeout, python_exe=args.python,
                env_base=env_base,
            ): t
            for t in tasks
        }
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                rec = fut.result()
            except Exception as e:  # noqa: BLE001
                rec = {"task": t, "status": "harness_error", "score": None, "error": str(e)}
            with lock:
                results[t] = rec
                write_summary()

    write_summary()
    scored = [r for r in results.values() if isinstance(r.get("score"), (int, float))]
    log(f"ALL DONE: {len(scored)}/{len(tasks)} scored. summary -> {out_path}")
    if scored:
        mean = sum(r["score"] for r in scored) / len(scored)
        log(f"mean task_score over {len(scored)} scored tasks = {mean:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
