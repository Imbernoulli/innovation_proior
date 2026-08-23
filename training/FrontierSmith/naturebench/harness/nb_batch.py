"""nb_batch.py — run a NatureBench task subset for ONE model on ONE GPU job.

Efficiency: vLLM is loaded ONCE and the official eval service is started ONCE
per job; all tasks in the shard reuse both. (Per-task vLLM loading cost ~130 s
would otherwise dominate a 4-6 min agent run.)

Sharding mirrors the repo's cc_eval_*_autopart.sh pattern:
  * a shard is a deterministic slice of the task list (SHARD/NSHARDS)
  * per-task output dir under results/<batch>/<task>/
  * --resume skips tasks that already have a completed result.json
  * aggregation is a separate step (nb_aggregate.py) that refuses to
    aggregate an incomplete subset unless explicitly forced

Usage:
  python nb_batch.py --tasks-file task-sets/subset.txt --model-path /path/to/hf \
      --served-name qwen35-9b --batch mybatch --shard 0 --nshards 4
  python nb_batch.py ... --openai-base-url http://127.0.0.1:8000/v1   # reuse server
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import List, Optional

from nb_run import (NB_ROOT, DEFAULT_DATA, DEFAULT_SIF, DEFAULT_EVAL_PY,
                    run_task, start_eval_service, stop_eval_service, log)

FS_ROOT = NB_ROOT.parent          # FrontierSmith repo root (vLLM launcher lives there)


def load_tasks(args) -> List[str]:
    tasks: List[str] = list(args.task)
    if args.tasks_file:
        tasks += [l.strip() for l in open(args.tasks_file)
                  if l.strip() and not l.startswith("#")]
    tasks = list(dict.fromkeys(tasks))
    if args.nshards > 1:
        tasks = [t for i, t in enumerate(tasks) if i % args.nshards == args.shard]
    return tasks


def start_vllm(model_path: str, served: str, port: int, log_path: Path,
               max_model_len: int, gpu_mem: str) -> subprocess.Popen:
    env = dict(os.environ)
    env.update({"PORT": str(port), "SERVED_MODEL_NAME": served,
                "MODEL_PATH": model_path, "MAX_MODEL_LEN": str(max_model_len),
                "MAX_NUM_SEQS": "8", "GPU_MEMORY_UTILIZATION": gpu_mem})
    log(f"starting vLLM: {model_path} as '{served}' on :{port}")
    proc = subprocess.Popen(["bash", str(FS_ROOT / "scripts" / "start_vllm_server.sh")],
                            env=env, stdout=open(log_path, "w"),
                            stderr=subprocess.STDOUT, cwd=str(FS_ROOT))
    for i in range(240):
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited early (rc={proc.returncode}); see {log_path}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3) as r:
                if served in r.read().decode():
                    log(f"vLLM ready after ~{i*5}s")
                    return proc
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError(f"vLLM did not become ready; see {log_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", action="append", default=[])
    ap.add_argument("--tasks-file", default=None)
    ap.add_argument("--batch", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--results-dir", default=str(NB_ROOT / "results"))
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA))
    ap.add_argument("--sif", default=str(DEFAULT_SIF))
    ap.add_argument("--eval-python", default=str(DEFAULT_EVAL_PY))
    ap.add_argument("--eval-port", type=int, default=0, help="0 = derive from shard")
    # model / serving
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--served-name", default="nb-model")
    ap.add_argument("--vllm-port", type=int, default=0, help="0 = derive from shard")
    ap.add_argument("--openai-base-url", default=None, help="reuse a running server")
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--gpu-mem-util", default="0.85")
    # agent / task budgets
    ap.add_argument("--mode", choices=["agent", "reference", "probe"], default="agent")
    ap.add_argument("--ref-solver-dir", default=str(NB_ROOT / "harness" / "ref_solvers"))
    ap.add_argument("--timeout", type=int, default=1500)
    ap.add_argument("--max-rounds", type=int, default=6)
    ap.add_argument("--run-timeout", type=int, default=600)
    ap.add_argument("--task-gpu", action="store_true", help="give task containers --nv")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--setup-extra-file", default=str(NB_ROOT / "task-sets" / "setup_extra.json"))
    args = ap.parse_args()

    tasks = load_tasks(args)
    if not tasks:
        print("no tasks selected", file=sys.stderr)
        return 2
    results_dir = Path(args.results_dir)
    batch_dir = results_dir / args.batch
    batch_dir.mkdir(parents=True, exist_ok=True)
    eval_port = args.eval_port or (8400 + args.shard)
    vllm_port = args.vllm_port or (8500 + args.shard)

    setup_extra = {}
    if Path(args.setup_extra_file).exists():
        setup_extra = json.loads(Path(args.setup_extra_file).read_text())

    log(f"batch={args.batch} shard={args.shard}/{args.nshards} tasks={len(tasks)}: {tasks}")

    # ---- resume: drop tasks already completed -----------------------------
    todo = []
    for t in tasks:
        rj = batch_dir / t / "result.json"
        if not args.no_resume and rj.exists():
            try:
                if json.loads(rj.read_text()).get("status") == "done":
                    log(f"[{t}] already done, skipping (resume)")
                    continue
            except Exception:
                pass
        todo.append(t)
    if not todo:
        log("nothing to do (all tasks already complete)")
        return 0

    vllm_proc: Optional[subprocess.Popen] = None
    base_url = args.openai_base_url
    if args.mode == "agent" and not base_url:
        if not args.model_path:
            print("agent mode needs --model-path or --openai-base-url", file=sys.stderr)
            return 2
        vllm_proc = start_vllm(args.model_path, args.served_name, vllm_port,
                               batch_dir / f"vllm_shard{args.shard}.log",
                               args.max_model_len, args.gpu_mem_util)
        base_url = f"http://127.0.0.1:{vllm_port}/v1"
        os.environ.setdefault("OPENAI_API_KEY", "EMPTY")

    eval_proc = start_eval_service(eval_port, batch_dir / f"eval_service_shard{args.shard}.log",
                                   args.eval_python, timeout=args.timeout)
    eval_url = f"http://127.0.0.1:{eval_port}"
    summary = {"batch": args.batch, "shard": args.shard, "nshards": args.nshards,
               "model": args.served_name if args.mode == "agent" else "reference",
               "model_path": args.model_path, "mode": args.mode,
               "started": time.strftime("%Y-%m-%d %H:%M:%S"), "tasks": {}}
    try:
        for t in todo:
            out_dir = batch_dir / t
            out_dir.mkdir(parents=True, exist_ok=True)
            # Atomic per-task lock so two shards / a twin job on another
            # partition never run the same task into the same directory.
            lock = out_dir / ".lock"
            try:
                lock.mkdir()
            except FileExistsError:
                age = time.time() - lock.stat().st_mtime
                if age < 7200:
                    log(f"[{t}] locked by another job ({age:.0f}s old), skipping")
                    continue
                log(f"[{t}] stale lock ({age:.0f}s), taking over")
                lock.touch()
            ref = None
            if args.mode == "reference":
                ref = Path(args.ref_solver_dir) / f"{t}.py"
                if not ref.exists():
                    log(f"[{t}] no reference solver, skipping")
                    summary["tasks"][t] = {"status": "no_ref_solver"}
                    continue
                ref = str(ref)
            try:
                res = run_task(
                    t, eval_url=eval_url, batch=args.batch, out_dir=out_dir,
                    data_dir=Path(args.data_dir), sif=Path(args.sif),
                    mode=args.mode, model=args.served_name, openai_base_url=base_url or "",
                    ref_solver=ref, gpu=args.task_gpu, timeout=args.timeout,
                    max_rounds=args.max_rounds, run_timeout=args.run_timeout,
                    setup_extra=setup_extra.get(t), skip_setup=True,
                    overlay_root=results_dir / "_overlays",
                    pip_cache=results_dir / "_pipcache",
                )
                summary["tasks"][t] = {"g": res.get("g"), "status": res.get("status"),
                                       "seconds": res.get("task_seconds"),
                                       "attempts": res.get("total_attempts")}
            except Exception as e:  # one broken task must not kill the shard
                log(f"[{t}] FAILED: {type(e).__name__}: {e}")
                summary["tasks"][t] = {"status": "error", "error": f"{type(e).__name__}: {e}"}
                (out_dir / "result.json").write_text(json.dumps(
                    {"task": t, "batch": args.batch, "status": "error",
                     "error": f"{type(e).__name__}: {e}", "g": None}, indent=2))
            (batch_dir / f"shard{args.shard}_summary.json").write_text(
                json.dumps(summary, indent=2, default=str))
            try:
                lock.rmdir()
            except OSError:
                pass
    finally:
        stop_eval_service(eval_proc)
        if vllm_proc is not None:
            vllm_proc.terminate()
            try:
                vllm_proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                vllm_proc.kill()
    summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (batch_dir / f"shard{args.shard}_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    log(f"shard {args.shard} done: " +
        ", ".join(f"{k}={v.get('g')}" for k, v in summary["tasks"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
