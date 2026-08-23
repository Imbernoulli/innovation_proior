#!/usr/bin/env python3
"""Measure evaluator-side score variance of the FrontierCS-Research scoring path.

Some official research evaluators are stochastic time-budgeted searches
(symbolic_regression/* runs PySR: the SAME solution has scored 0.04 and 100
across runs) and some score partly on measured runtime (vdb_pareto/*,
imagenet_pareto/* -> node-speed sensitive). This script quantifies that noise:
given a banked samples.jsonl from a research eval run, it re-scores each
distinct (problem, sample) K times through the exact production scoring path
(strip_think -> code extraction -> frontiercs_research_cpu_eval /
frontiercs_research_eval) and emits per-problem cross-rep variance stats with a
`lottery` flag (max per-sample cross-rep std > --threshold, default 5).

The output JSON (default data/research_scoring_variance.json) is the file
FRONTIERCS_RESEARCH_REPS_ONLY consumes (lottery_problems / per-problem lottery
flags), and the with/without-lottery aggregate policy in EVAL_ROBUSTNESS_zh.md
refers to it.

CPU-runnable. GPU (Triton) problems are SKIPPED unless --include-gpu (they need
a GPU node). Shard with --problems / --shard I/N; combine shard outputs with
--merge.

Examples
--------
# smoke (login node, 2 cheap problems, 1 sample each, 2 reps):
python scripts/measure_eval_variance.py \
    --samples outputs/cc_eval_q36_35bA3b_base_research_thinking_32k_vllm/merged/samples.jsonl \
    --problems cant_be_late/low_availability_tight_deadline_large_overhead,symbolic_regression/mccormick \
    --max-samples-per-problem 1 --reps 2 --out data/research_scoring_variance.smoke.json

# full CPU-family measurement (submit as a cpu-partition job, see report):
python scripts/measure_eval_variance.py \
    --samples outputs/cc_eval_q36_35bA3b_base_research_thinking_32k_vllm/merged/samples.jsonl \
    --reps 3 --out data/research_scoring_variance.json

# merge shard outputs:
python scripts/measure_eval_variance.py --merge data/var_shard0.json data/var_shard1.json \
    --out data/research_scoring_variance.json
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Measure SINGLE-run variance: neutralize the multi-rep median knob so an
# inherited environment cannot make each "rep" itself a median of N runs.
os.environ["FRONTIERCS_RESEARCH_SCORE_REPS"] = "1"

sys.path.insert(0, str(SCRIPT_DIR))
import frontiercs_research_cpu_eval as fcs_cpu  # noqa: E402
import frontiercs_research_eval as fcs_research  # noqa: E402

# Optional official C++ extractor (nbody_simulation); mirror the runner's
# fallback in eval_qwen35_base_vllm_request.py.
OFFICIAL_ROOT = PROJECT_ROOT / ".cache" / "Frontier-CS-official"
official_extract_cpp_code = None
if OFFICIAL_ROOT.exists():
    sys.path.insert(0, str(OFFICIAL_ROOT))
    sys.path.insert(0, str(OFFICIAL_ROOT / "src"))
    try:
        from algorithmic.scripts.generate_solutions import (  # type: ignore
            extract_cpp_code as official_extract_cpp_code,
        )
    except Exception:
        official_extract_cpp_code = None


def strip_think(response: str) -> str:
    """Verbatim copy of verl.utils.reward_score.frontiercs.strip_think (kept
    local so this stays CPU-light: no verl/pandas/torch import)."""
    if not response:
        return response
    _, sep, suffix = response.rpartition("</think>")
    return suffix if sep else response


def extract_code(problem_id: str, text: str) -> str:
    """Same extraction route as the production runner (strip_think first)."""
    stripped = strip_think(text)
    if (fcs_cpu.is_cpu_family(problem_id)
            and fcs_cpu.solution_language(problem_id) == "cpp"):
        if official_extract_cpp_code is not None:
            return official_extract_cpp_code(stripped)
    return fcs_research.extract_python_code(stripped)


def score_once(problem_id: str, code: str, timeout: Optional[int]) -> dict[str, Any]:
    kwargs = {} if timeout is None else {"timeout": timeout}
    if fcs_cpu.is_cpu_family(problem_id):
        return fcs_cpu.evaluate_cpu_research_solution(problem_id, code, **kwargs)
    return fcs_research.evaluate_research_solution(problem_id, code, **kwargs)


# ---------------------------------------------------------------------------
# samples.jsonl loading
# ---------------------------------------------------------------------------
def load_records(paths: list[str]) -> dict[tuple[str, int], dict[str, Any]]:
    """Dedupe on (problem, sample_idx): prefer records with error==null (resume
    reruns bank both an infra-error and a scored record for the same key);
    among equals the LAST one wins."""
    records: dict[tuple[str, int], dict[str, Any]] = {}
    for spec in paths:
        p = Path(spec)
        if p.is_dir():
            p = p / "samples.jsonl"
        if not p.is_file():
            sys.exit(f"ERROR: samples file not found: {p}")
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                pid = str(d.get("ground_truth") or "")
                if not pid or d.get("data_source") not in (None, "frontiercs_research"):
                    continue
                key = (pid, int(d.get("sample_idx", 0)))
                old = records.get(key)
                if old is not None and old.get("error") is None and d.get("error") is not None:
                    continue  # keep the scored record over an infra-error one
                records[key] = d
    return records


def parse_problem_filter(spec: Optional[str]) -> Optional[set]:
    if not spec:
        return None
    if spec.startswith("@"):
        entries = Path(spec[1:]).read_text(encoding="utf-8").split()
    else:
        entries = spec.split(",")
    return {e.strip() for e in entries if e.strip()}


def problem_matches(pid: str, flt: set) -> bool:
    return pid in flt or pid.split("/")[0] in flt


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
def measure(args: argparse.Namespace) -> dict[str, Any]:
    records = load_records(args.samples)
    flt = parse_problem_filter(args.problems)

    by_problem: dict[str, list[tuple[int, dict]]] = {}
    skipped = {"gpu": 0, "error": 0, "filtered": 0}
    for (pid, sidx), rec in sorted(records.items()):
        if flt is not None and not problem_matches(pid, flt):
            skipped["filtered"] += 1
            continue
        if not fcs_cpu.is_cpu_family(pid) and not args.include_gpu:
            skipped["gpu"] += 1
            continue
        if rec.get("error") is not None and not args.include_errors:
            skipped["error"] += 1
            continue
        by_problem.setdefault(pid, []).append((sidx, rec))

    pids = sorted(by_problem)
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        assert 0 <= i < n, f"bad --shard {args.shard} (expect I/N, 0-based I)"
        pids = pids[i::n]

    total = sum(min(len(by_problem[p]), args.max_samples_per_problem or 10**9)
                for p in pids)
    print(f"[measure] {len(pids)} problems, {total} (problem,sample) records, "
          f"{args.reps} reps each; skipped {skipped}", file=sys.stderr, flush=True)

    problems_out: dict[str, Any] = {}
    done = 0
    for pid in pids:
        samples = sorted(by_problem[pid])
        if args.max_samples_per_problem:
            samples = samples[: args.max_samples_per_problem]
        samples_out: dict[str, Any] = {}
        for sidx, rec in samples:
            banked = None
            try:
                banked = float((rec.get("metrics") or {}).get("score"))
            except (TypeError, ValueError):
                pass
            code = extract_code(pid, rec.get("text") or "")
            entry: dict[str, Any] = {
                "banked_score": banked,
                "rep_scores": [],
                "rep_errors": [],
                "rep_seconds": [],
            }
            if not code.strip():
                # deterministic 0 in the production path; nothing to rerun
                entry["empty_code"] = True
                entry["rep_scores"] = [0.0] * args.reps
                entry["rep_seconds"] = [0.0] * args.reps
            else:
                for _ in range(args.reps):
                    t0 = time.time()
                    try:
                        res = score_once(pid, code, args.timeout)
                        entry["rep_scores"].append(float(res.get("score") or 0.0))
                    except (fcs_cpu.ResearchInfraError,
                            fcs_research.ResearchInfraError) as exc:
                        entry["rep_errors"].append(f"infra: {exc}"[:300])
                    except Exception as exc:  # keep the sweep alive
                        entry["rep_errors"].append(
                            f"{type(exc).__name__}: {exc}"[:300])
                    entry["rep_seconds"].append(round(time.time() - t0, 2))
            ok = entry["rep_scores"]
            entry["median"] = float(statistics.median(ok)) if ok else None
            entry["range"] = (max(ok) - min(ok)) if ok else None
            entry["std"] = float(statistics.stdev(ok)) if len(ok) >= 2 else None
            samples_out[str(sidx)] = entry
            done += 1
            print(f"[measure] {done}/{total} {pid} s{sidx} "
                  f"scores={['%.2f' % s for s in ok]} "
                  f"errs={len(entry['rep_errors'])}", file=sys.stderr, flush=True)

        stds = [e["std"] for e in samples_out.values() if e["std"] is not None]
        ranges = [e["range"] for e in samples_out.values() if e["range"] is not None]
        drifts = [abs(e["median"] - e["banked_score"]) for e in samples_out.values()
                  if e["median"] is not None and e["banked_score"] is not None]
        unstable_infra = any(
            e["rep_errors"] and e["rep_scores"] for e in samples_out.values())
        max_std = max(stds) if stds else None
        problems_out[pid] = {
            "family": pid.split("/")[0],
            "n_samples": len(samples_out),
            "max_rep_std": max_std,
            "mean_rep_std": (sum(stds) / len(stds)) if stds else None,
            "max_rep_range": max(ranges) if ranges else None,
            "mean_abs_drift_from_banked": (sum(drifts) / len(drifts)) if drifts else None,
            "unstable_infra": unstable_infra,
            "lottery": bool(max_std is not None and max_std > args.threshold),
            "samples": samples_out,
        }

    return finalize(problems_out, meta={
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "reps": args.reps,
        "threshold_std": args.threshold,
        "samples_inputs": args.samples,
        "shard": args.shard,
        "include_gpu": args.include_gpu,
        "skipped": skipped,
        "argv": sys.argv,
    })


def finalize(problems: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    lottery = sorted(p for p, rec in problems.items() if rec.get("lottery"))
    meta = dict(meta)
    meta.update(n_problems=len(problems), n_lottery=len(lottery))
    return {"meta": meta, "problems": dict(sorted(problems.items())),
            "lottery_problems": lottery}


def merge(paths: list[str]) -> dict[str, Any]:
    problems: dict[str, Any] = {}
    inputs = []
    reps = None
    thr = None
    for p in paths:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        problems.update(data.get("problems") or {})
        m = data.get("meta") or {}
        inputs.append({"file": p, "meta": {k: m.get(k) for k in
                                           ("generated_utc", "hostname", "shard")}})
        reps = reps or m.get("reps")
        thr = thr or m.get("threshold_std")
    return finalize(problems, meta={
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "reps": reps, "threshold_std": thr,
        "merged_from": inputs, "argv": sys.argv,
    })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--samples", nargs="+", metavar="PATH",
                    help="banked samples.jsonl file(s) (or dirs containing one)")
    ap.add_argument("--reps", type=int, default=3,
                    help="re-score repetitions per (problem,sample) [3]")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "data" / "research_scoring_variance.json"))
    ap.add_argument("--problems", metavar="LIST",
                    help="comma list of problem ids / family prefixes, or @file")
    ap.add_argument("--shard", metavar="I/N",
                    help="deterministic problem-list shard (0-based I of N)")
    ap.add_argument("--max-samples-per-problem", type=int, default=0,
                    help="cap re-scored samples per problem (0 = all)")
    ap.add_argument("--threshold", type=float, default=5.0,
                    help="lottery flag: max per-sample cross-rep std > this [5]")
    ap.add_argument("--timeout", type=int, default=None,
                    help="per-evaluator-run timeout override (seconds)")
    ap.add_argument("--include-gpu", action="store_true",
                    help="also re-score GPU (Triton) problems -- needs a GPU node")
    ap.add_argument("--include-errors", action="store_true",
                    help="also re-score records whose banked run infra-errored")
    ap.add_argument("--merge", nargs="+", metavar="JSON",
                    help="merge shard variance JSONs into --out (no measuring)")
    args = ap.parse_args()

    if args.merge:
        out = merge(args.merge)
    else:
        if not args.samples:
            ap.error("--samples is required (unless --merge)")
        out = measure(args)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"[measure] wrote {out_path}  (problems={out['meta']['n_problems']}, "
          f"lottery={out['meta']['n_lottery']}: {out['lottery_problems']})",
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
