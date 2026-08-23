#!/usr/bin/env python3
"""
Frontier-CS Research **GPU-21** score-only job: rescore existing generations.

WHY THIS EXISTS (2026-08-19 audit):
  The research track's 64-problem scope has ALWAYS contained the 21 Triton GPU
  problems, and the June-era 1-GPU launcher (slurm/cc_eval_research_autopart.sh,
  GPU_MEMORY_UTILIZATION=0.55, envs/research_overlay evaluator python) scored
  them correctly (anchor_base 2026-08-15: vector_addition ~50, qknorm 100/61.5).
  The NEWER self-hosted pool path (slurm/cc_eval_pool_selfhosted.sh ->
  scripts/eval_base_model_qwen35_9b_vllm_request.sh) does NOT export
  FRONTIERCS_RESEARCH_PYTHON / JULIA_* and serves vLLM at
  GPU_MEMORY_UTILIZATION=0.90 x TP=2 on the same GPUs, so under it the GPU
  evaluators run in a squeezed / wrong environment: every GPU problem comes back
  0.0 (or qknorm "No module named 'flashinfer'") for EVERY arm since ~08-16.

  GENERATION is unaffected (same prompts, same protocol) -- only SCORING of the
  21 GPU problems is unreliable under the pool. So: rescore those samples from
  the stored samples.jsonl text on a node with a free GPU + the overlay env,
  and report them as a SEPARATE "Research-GPU (21)" column. Never fold these
  into the historical Research-64 numbers (口径 must not change retroactively).

USAGE
  python scripts/frontiercs_research_gpu_rescore.py \
      --samples <samples.jsonl | eval-output-dir> [more ...] \
      --tag base_anchor \
      [--out-dir outputs/research_gpu/<tag>] \
      [--problems vector_addition/2_20,flash_attn] [--n-samples 5] \
      [--timeout 2400] [--skip-control] [--resume]

  Inputs are the EXISTING eval products (any research eval's samples.jsonl with
  --save-text); only records with data_source=frontiercs_research and a GPU
  problem id are rescored. EVAL-ONLY: reads outputs/, writes outputs/ -- this
  never touches training data.

Faithfulness: identical to the online scoring path in
scripts/eval_qwen35_base_vllm_request.py `_score` for frontiercs_research:
strip <think> (rpartition "</think>"), extract_python_code, then
frontiercs_research_eval.evaluate_research_solution (official evaluator.py,
result.json / last-line score parse). ResearchInfraError -> recorded in `error`
(loud), never a silent 0.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# The evaluator interpreter MUST come from the research overlay (torch 2.10 +
# triton 3.6 from the sft_lf base, PLUS flashinfer for qknorm). Default it
# BEFORE importing frontiercs_research_eval, which snapshots the env at import.
_OVERLAY = "/scratch/gpfs/CHIJ/bohan/fs/envs/research_overlay/bin/python"
if "FRONTIERCS_RESEARCH_PYTHON" not in os.environ and Path(_OVERLAY).exists():
    os.environ["FRONTIERCS_RESEARCH_PYTHON"] = _OVERLAY
# The RL-era host-OOM guard (RLIMIT_AS, default 24 GiB) KILLS large-VA GPU
# evaluators: vector_addition/2_28 dies rc=127 under 24 GiB but scores ~50 at
# >=48 GiB (CUDA reserves big virtual address ranges). This is a dedicated
# scoring job with its own memory allocation, so default the cap to 64 GiB
# here (still a guard, just one that CUDA fits under). Explicit env wins.
os.environ.setdefault("FRONTIERCS_RESEARCH_EVAL_RLIMIT_GB", "64")

import frontiercs_research_eval as fre  # noqa: E402


def strip_think(response: str) -> str:
    """Byte-identical to verl.utils.reward_score.frontiercs.strip_think
    (reimplemented here so this job doesn't import the verl stack)."""
    if not response:
        return response
    _, sep, suffix = response.rpartition("</think>")
    return suffix if sep else response


# Positive control: correctness-passing (PyTorch) vector add. On the official
# 0..100 anchoring (0 = naive CPU, 100 = 2x PyTorch GPU) this scores ~50. If it
# scores 0 or infra-errors, THE SUBSTRATE IS BROKEN (VRAM squeeze / env) and
# every other 0 this job would produce is meaningless -- so abort loudly.
CONTROL_PROBLEM = "vector_addition/2_20"
CONTROL_SOLUTION = '''\
class Solution:
    def solve(self, spec_path: str = None) -> dict:
        return {"code": "import torch\\n\\ndef add(x, y):\\n    return x + y\\n"}
'''


def gpu_problem_ids() -> list[str]:
    return sorted(
        p for p in fre.list_research_problems() if fre.problem_needs_gpu(p)
    )


def preflight(python_exe: str) -> None:
    """Refuse to 'score' without a usable CUDA device (the silent-zero trap)."""
    probe = (
        "import torch,triton;assert torch.cuda.is_available(),'no CUDA device';"
        "print(torch.cuda.get_device_name(0))"
    )
    r = subprocess.run([python_exe, "-c", probe], capture_output=True, text=True,
                       timeout=300)
    if r.returncode != 0:
        sys.exit(
            f"PREFLIGHT FAILED: {python_exe} cannot see a CUDA GPU "
            f"(stderr: {r.stderr.strip()[-400:]}). Refusing to rescore -- every "
            f"GPU evaluator would fail. Run on a node with a free GPU."
        )
    print(f"[preflight] evaluator python OK on GPU: {r.stdout.strip()}")


def run_control(timeout: int) -> None:
    print(f"[control] scoring built-in torch add on {CONTROL_PROBLEM} ...")
    res = fre.evaluate_research_solution(
        CONTROL_PROBLEM, CONTROL_SOLUTION, timeout=timeout
    )
    score = float(res.get("score") or 0.0)
    print(f"[control] score={score:.2f} status={res.get('status')}")
    if score <= 0.0:
        sys.exit(
            f"CONTROL FAILED: a correctness-passing torch add scored "
            f"{score} on {CONTROL_PROBLEM} (expected ~50). The scoring "
            f"substrate is broken on this node (VRAM squeeze / env); aborting "
            f"so broken zeros are never recorded. raw tail: "
            f"{(res.get('raw') or '')[-400:]}"
        )


def collect_samples(inputs: list[str], gpu_ids: set[str],
                    n_samples: int) -> dict[tuple[str, int], dict]:
    files: list[Path] = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            found = sorted(p.rglob("samples.jsonl"))
            # shard files are the ground truth; merged/ dirs duplicate them
            shard = [f for f in found if "merged" not in f.parts]
            files.extend(shard or found)
        elif p.exists():
            files.append(p)
        else:
            sys.exit(f"input not found: {inp}")
    if not files:
        sys.exit("no samples.jsonl found under the given inputs")
    picked: dict[tuple[str, int], dict] = {}
    for f in files:
        for line in f.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("data_source") != "frontiercs_research":
                continue
            pid = str(r.get("ground_truth") or "")
            if pid not in gpu_ids:
                continue
            key = (pid, int(r.get("sample_idx", 0)))
            if key in picked and picked[key].get("text"):
                continue  # keep first occurrence that has text
            r["_source_file"] = str(f)
            picked[key] = r
    # keep the n_samples lowest sample_idx per problem
    by_pid: dict[str, list[int]] = {}
    for pid, si in picked:
        by_pid.setdefault(pid, []).append(si)
    keep = {
        (pid, si)
        for pid, sis in by_pid.items()
        for si in sorted(sis)[:n_samples]
    }
    return {k: v for k, v in picked.items() if k in keep}


def summarize(records: list[dict], gpu_ids: list[str]) -> dict:
    per: dict[str, dict] = {}
    for pid in gpu_ids:
        rs = [r for r in records if r["problem_id"] == pid]
        if not rs:
            continue
        scores = [float(r.get("score") or 0.0) for r in rs]
        per[pid] = {
            "n": len(rs),
            "mean": sum(scores) / len(scores),
            "best": max(scores),
            "scores": scores,
            "errors": sum(1 for r in rs if r.get("error")),
        }
    means = [v["mean"] for v in per.values()]
    bests = [v["best"] for v in per.values()]
    return {
        "n_problems": len(per),
        "n_samples": sum(v["n"] for v in per.values()),
        "n_errors": sum(v["errors"] for v in per.values()),
        "metrics": {
            "frontiercs_research_gpu": {
                "score": {
                    "mean@k": (sum(means) / len(means)) if means else None,
                    "best@k/mean": (sum(bests) / len(bests)) if bests else None,
                }
            }
        },
        "per_problem": per,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--samples", nargs="+", required=True,
                    help="samples.jsonl file(s) and/or eval output dir(s)")
    ap.add_argument("--tag", required=True, help="arm name for the output dir")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--problems", default=None,
                    help="comma list restricting to these GPU problem ids")
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--timeout", type=int,
                    default=int(os.environ.get("FRONTIERCS_RESEARCH_GPU_TIMEOUT",
                                               "2400")),
                    help="per-evaluator wall clock, seconds (2400 = the "
                         "campaign's research evaluator budget)")
    ap.add_argument("--skip-control", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--redo-errors", action="store_true",
                    help="with --resume: re-score samples whose previous "
                         "rescore ended in an infra error")
    args = ap.parse_args()

    gpu_ids_all = gpu_problem_ids()
    if len(gpu_ids_all) != 21:
        print(f"WARNING: expected 21 GPU problems, enumerated {len(gpu_ids_all)}",
              file=sys.stderr)
    gpu_ids = gpu_ids_all
    if args.problems:
        want = {x.strip() for x in args.problems.split(",") if x.strip()}
        unknown = want - set(gpu_ids_all)
        if unknown:
            sys.exit(f"not GPU research problems: {sorted(unknown)}")
        gpu_ids = [p for p in gpu_ids_all if p in want]

    out_dir = Path(args.out_dir) if args.out_dir else (
        PROJECT_ROOT / "outputs" / "research_gpu" / args.tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "samples_gpu.jsonl"
    out_summary = out_dir / "summary_gpu.json"

    python_exe = fre.RESEARCH_EVAL_PYTHON
    print(f"[rescore] evaluator python = {python_exe}")
    print(f"[rescore] timeout = {args.timeout}s; problems = {len(gpu_ids)}; "
          f"n_samples = {args.n_samples}")
    preflight(python_exe)
    if not args.skip_control:
        run_control(args.timeout)

    done: dict[tuple[str, int], dict] = {}
    if args.resume and out_jsonl.exists():
        for line in out_jsonl.open(encoding="utf-8"):
            try:
                r = json.loads(line)
                if args.redo_errors and r.get("error"):
                    continue
                done[(r["problem_id"], int(r["sample_idx"]))] = r
            except Exception:
                pass
        print(f"[rescore] resume: {len(done)} samples already rescored")
        if args.redo_errors:
            # rewrite the jsonl without the errored records we are redoing
            with out_jsonl.open("w", encoding="utf-8") as fh:
                for r in done.values():
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    picked = collect_samples(args.samples, set(gpu_ids), args.n_samples)
    todo = sorted(k for k in picked if k not in done)
    print(f"[rescore] {len(picked)} samples collected, {len(todo)} to score")
    missing = [p for p in gpu_ids if not any(k[0] == p for k in picked)]
    if missing:
        print(f"[rescore] WARNING: no stored samples for {len(missing)} "
              f"problems: {missing}", file=sys.stderr)

    records = list(done.values())
    with out_jsonl.open("a", encoding="utf-8") as fh:
        for i, key in enumerate(todo, 1):
            pid, si = key
            src = picked[key]
            text = src.get("text") or ""
            rec = {
                "problem_id": pid,
                "sample_idx": si,
                "tag": args.tag,
                "source_file": src.get("_source_file"),
                "score": 0.0,
                "score_unbounded": 0.0,
                "status": None,
                "error": None,
                "rescore_seconds": None,
                "rescored_at": _dt.datetime.now().isoformat(timespec="seconds"),
                "old_score": (src.get("metrics") or {}).get("score"),
                "old_error": (src.get("error") or None),
            }
            if not text:
                rec["error"] = "no stored generation text (run had --no-save-text?)"
            else:
                code = fre.extract_python_code(strip_think(text))
                if not code:
                    rec["status"] = "empty"
                else:
                    t0 = time.time()
                    try:
                        res = fre.evaluate_research_solution(
                            pid, code, timeout=args.timeout)
                        rec["score"] = float(res.get("score") or 0.0)
                        su = res.get("score_unbounded")
                        rec["score_unbounded"] = (
                            rec["score"] if su is None else float(su))
                        rec["status"] = res.get("status")
                        rec["message"] = (res.get("message") or None)
                    except fre.ResearchInfraError as exc:
                        rec["error"] = f"ResearchInfraError({str(exc)[:400]})"
                    rec["rescore_seconds"] = round(time.time() - t0, 2)
            records.append(rec)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[{i}/{len(todo)}] {pid} s{si} score={rec['score']:.2f} "
                  f"(old={rec['old_score']}) secs={rec['rescore_seconds']} "
                  f"err={rec['error'] or '-'}", flush=True)

    summary = summarize(records, gpu_ids)
    summary.update({
        "tag": args.tag,
        "scope": "frontiercs_research_gpu_21" if len(gpu_ids) == 21
                 else f"frontiercs_research_gpu_subset_{len(gpu_ids)}",
        "inputs": args.samples,
        "evaluator_python": python_exe,
        "timeout_seconds": args.timeout,
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "note": "SEPARATE Research-GPU column; do NOT merge into Research-64 "
                "history (口径不可追溯改变). EVAL-ONLY data.",
    })
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(json.dumps(summary["metrics"], indent=2))
    m = summary["metrics"]["frontiercs_research_gpu"]["score"]
    print(f"[rescore] DONE {summary['n_problems']} problems / "
          f"{summary['n_samples']} samples / {summary['n_errors']} errors -> "
          f"mean@k={m['mean@k']} best@k={m['best@k/mean']}")
    print(f"[rescore] summary -> {out_summary}")


if __name__ == "__main__":
    main()
