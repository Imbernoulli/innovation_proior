#!/usr/bin/env python3
"""Self-check MLS-Bench training-variant tasks (login-node, CPU, tiny budgets).

Per task it runs the SAME code path the RL agent loop uses
(scripts/mlsbench_rl_episode_worker.py, one process per task):

  1. loader     : `init` -> mlsbench builds the workspace + initial prompt
  2. scaffold   : `dispatch test` -> runs the task's test_cmds in Apptainer with
                  the untouched (spliced) scaffold
  3. scoring    : `finalize` -> force-submit + mlsbench.scoring.evaluate_task,
                  must return a numeric reward in [0, 1]
  4. overlap    : max shared-sentence similarity between the variant's
                  "## Research Question" and the eval task's, plus the fraction
                  of variant RQ sentences that appear verbatim in the eval RQ.

Usage:
  python scripts/mlsbench_check_train_tasks.py --tasks ml-clustering-algorithm_tv1
  python scripts/mlsbench_check_train_tasks.py --all --timeout 420 --out /tmp/check.json
"""
from __future__ import annotations

import argparse
import difflib
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKER_SCRIPT = str(PROJECT_ROOT / "scripts" / "mlsbench_rl_episode_worker.py")
TRAIN_ROOT = Path("/scratch/gpfs/CHIJ/bohan/MLS-Bench-train")
DEV_ROOT = Path("/scratch/gpfs/CHIJ/bohan/MLS-Bench-dev")


def _worker_client_cls():
    spec = importlib.util.spec_from_file_location("mlsbench_rl_episode_worker", WORKER_SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.WorkerClient


def call_t(client, timeout: float, **req) -> dict:
    """WorkerClient.call with a hard wall-clock cap (kills the worker's process group)."""
    import threading

    timer = threading.Timer(timeout, client.kill)
    timer.start()
    try:
        return client.call(**req)
    finally:
        timer.cancel()


# ---------------------------------------------------------------------------
# research-question overlap
# ---------------------------------------------------------------------------

def _rq_section(md_path: Path) -> str:
    if not md_path.exists():
        return ""
    lines = md_path.read_text().split("\n")
    out, on = [], False
    for ln in lines:
        if ln.strip().lower().startswith("## research question"):
            on = True
            continue
        if on and ln.startswith("## "):
            break
        if on:
            out.append(ln)
    return "\n".join(out).strip()


def _sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[.!?;])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) > 25]


def rq_overlap(variant_md: Path, base_md: Path) -> dict:
    a, b = _rq_section(variant_md), _rq_section(base_md)
    sa, sb = _sentences(a), _sentences(b)
    best, best_pair = 0.0, ("", "")
    verbatim = 0
    for x in sa:
        for y in sb:
            r = difflib.SequenceMatcher(None, x.lower(), y.lower()).ratio()
            if r > best:
                best, best_pair = r, (x[:110], y[:110])
        if x.lower() in b.lower():
            verbatim += 1
    wa = set(re.findall(r"[a-z]{4,}", a.lower()))
    wb = set(re.findall(r"[a-z]{4,}", b.lower()))
    jac = len(wa & wb) / max(1, len(wa | wb))
    return {
        "max_shared_sentence": round(best, 3),
        "verbatim_sentences": verbatim,
        "n_sentences_variant": len(sa),
        "word_jaccard": round(jac, 3),
        "closest_pair": best_pair,
    }


# ---------------------------------------------------------------------------
# episode
# ---------------------------------------------------------------------------

def check_task(task: str, args) -> dict:
    base = re.sub(r"_(?:tv\d+|train)$", "", task)
    rec = {"task": task, "base": base}
    rec["overlap"] = rq_overlap(
        Path(args.train_root) / "tasks" / task / "task_description.md",
        Path(args.dev_root) / "tasks" / base / "task_description.md",
    )

    WorkerClient = _worker_client_cls()
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ws = Path(args.workspace_root) / task
    client = WorkerClient(
        worker_python=args.worker_python,
        worker_script=WORKER_SCRIPT,
        env={"MLS_RL_MLSBENCH_ROOT": args.train_root, "MLS_RL_PRUNE_LEADERBOARD": "1"},
        stderr_path=str(log_dir / f"{task}.worker.log"),
    )
    t0 = time.time()
    try:
        init = call_t(
            client, args.init_timeout,
            cmd="init", task=task, model_name=f"vllm/selfcheck-{task}",
            max_steps=2, max_tests=1,
            save_path=str(Path(args.out_base) / "saves"),
            workspace_root=str(ws), data_root=args.data_root, use_replace=True,
        )
        rec["loader_ok"] = True
        rec["prompt_chars"] = len(init.get("initial_prompt", ""))
        rec["init_s"] = round(time.time() - t0, 1)

        t1 = time.time()
        try:
            res = call_t(client, args.timeout, cmd="dispatch", name="test", args={})
            fb = str(res.get("result", ""))
            rec["test_s"] = round(time.time() - t1, 1)
            rec["test_ok"] = True
            rec["test_feedback_head"] = fb[:400]
            rec["test_has_metrics"] = bool(re.search(r"[A-Za-z_]+:\s*-?\d", fb))
        except Exception as e:  # noqa: BLE001
            rec["test_ok"] = False
            rec["test_error"] = repr(e)[:300]
            rec["test_s"] = round(time.time() - t1, 1)

        t2 = time.time()
        try:
            fin = call_t(client, args.finalize_timeout, cmd="finalize")
            rec["reward"] = fin.get("reward")
            rec["score_ok"] = isinstance(fin.get("reward"), (int, float))
            rec["score_detail"] = fin.get("detail", {})
            rec["finalize_s"] = round(time.time() - t2, 1)
        except Exception as e:  # noqa: BLE001
            rec["score_ok"] = False
            rec["score_error"] = repr(e)[:300]
        try:
            call_t(client, 120, cmd="cleanup", keep_workspace=False)
        except Exception:
            pass
    except Exception as e:  # noqa: BLE001
        rec["loader_ok"] = False
        rec["loader_error"] = repr(e)[:400]
    finally:
        try:
            client.quit()
        except Exception:
            pass
    rec["total_s"] = round(time.time() - t0, 1)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--train-root", default=str(TRAIN_ROOT))
    ap.add_argument("--dev-root", default=str(DEV_ROOT))
    ap.add_argument("--data-root",
                    default="/scratch/gpfs/CHIJ/st3812/projects/MLS-Bench/vendor/data")
    ap.add_argument("--worker-python", default="/home/bl3615/miniconda3/bin/python")
    ap.add_argument("--out-base", default="/tmp/mls_tv1_check")
    ap.add_argument("--workspace-root", default="/tmp/mls_tv1_check/ws")
    ap.add_argument("--log-dir", default="/tmp/mls_tv1_check/logs")
    ap.add_argument("--out", default="/tmp/mls_tv1_check/report.json")
    ap.add_argument("--init-timeout", type=int, default=600)
    ap.add_argument("--timeout", type=int, default=420)
    ap.add_argument("--finalize-timeout", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()

    tasks = args.tasks or []
    if args.all or not tasks:
        tasks = sorted(d.name for d in (Path(args.train_root) / "tasks").iterdir() if d.is_dir())
    os.makedirs(args.out_base, exist_ok=True)

    results = []
    if args.jobs > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(check_task, t, args): t for t in tasks}
            for f in as_completed(futs):
                rec = f.result()
                results.append(rec)
                print(json.dumps(rec)[:600], flush=True)
                Path(args.out).write_text(json.dumps(results, indent=2))
    else:
        for t in tasks:
            rec = check_task(t, args)
            results.append(rec)
            print(json.dumps(rec)[:600], flush=True)
            Path(args.out).write_text(json.dumps(results, indent=2))

    Path(args.out).write_text(json.dumps(results, indent=2))
    n_ok = sum(1 for r in results if r.get("loader_ok") and r.get("score_ok"))
    print(f"\n[check] {n_ok}/{len(results)} tasks passed loader+scoring -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
