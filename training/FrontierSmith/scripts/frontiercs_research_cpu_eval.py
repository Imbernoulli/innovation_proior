#!/usr/bin/env python3
"""
Frontier-CS RESEARCH-track CPU-subset evaluation adapter (no Docker).

The base harness `frontiercs_research_eval.py` invokes every problem's
`evaluator.py` with the GPU/Triton convention
    evaluator.py --solution-path <sol> --spec-path <spec> --output-path result.json
and parses result.json with status=="success". That works for the 21 GPU
Triton-kernel problems and grammar_fuzzing/fuzzer (which has --solution-path
aliases), but the OTHER CPU families use DIFFERENT CLI conventions and output
shapes, established from each family's official run_evaluator.sh / evaluate.sh:

  family                         invocation (official)                               solution lang
  cant_be_late*                  --solution <sol> --spec <resources/submission_spec> python (Solution(Strategy))
  cloudcast                      --solution <sol> --spec <spec> --out <json>         python (search_algorithm via solve)
  grammar_fuzzing/fuzzer/sql     --solution-path <sol> --output-path <json>          python (fuzz)
  grammar_fuzzing/seed/sql       --solution <sol> --out <json>                       python (solve->list[str])
  imagenet_pareto/*              --solution <sol> --out <json>                       python (solve->nn.Module)
  llm_router                     --solution <sol> --out <json>                       python (solve route)
  llm_sql/*                      --solution <sol> --out <json>                       python (solve reorder)
  nbody_simulation/*             --solution-path <sol.cpp> --output-path <json>      C++ (Simulator + createSimulator)
  symbolic_regression/*          --solution-path <sol> --data-dir <resources/data>   python (solve->dict)
                                  --reference-path <resources/reference_metrics.json>
                                  --output-path <json>
  vdb_pareto/*                   --solution <sol> --out <json>  (cwd must have data/sift1M) python (add/search index)

EVERY CPU evaluator writes a JSON file with a top-level numeric "score" (0..100)
and, on failure, a top-level "error" string. Some success payloads use a
"combined_score" sentinel of -1e9 for hard failures (cant_be_late_multi). We
read the JSON, take "score" (clamped to [0,100]), and treat module/import
failures as INFRA errors (loud), not silent 0s -- mirroring the base harness's
ResearchInfraError contract.

This module is imported by the runner (frontiercs_research_eval.py exposes
`evaluate_research_solution`; for CPU problems the runner delegates here when
FRONTIERCS_CPU_ADAPTER=1, or callers import `evaluate_cpu_research_solution`
directly).

Env the compute node MUST set (provisioned offline on a login node):
  FRONTIERCS_RESEARCH_PYTHON = .../envs/research_overlay/bin/python   (triton 3.6 base + CPU deps)
  JULIA_DEPOT_PATH           = .../envs/research_overlay/julia_depot   (symbolic_regression / pysr)
  PYTHON_JULIAPKG_PROJECT    = .../envs/research_overlay/julia_env
"""

from __future__ import annotations

import json
import os
import re
import shutil
import statistics
import subprocess

def _rlimit_preexec():
    """Hard address-space cap for the evaluator subprocess (host-OOM guard, 2026-07-30).

    Two RL nodes were killed by Ray's memory monitor when research evaluator
    subprocesses stacked on the 288G optimizer offload. A concurrency semaphore in
    the reward adapter was not enough -- a single simulator can balloon. Cap each
    evaluator at FRONTIERCS_RESEARCH_EVAL_RLIMIT_GB (default 24 GiB) of address
    space; one that exceeds it dies with MemoryError -> ResearchInfraError ->
    fail-soft 0.0 for that sample, instead of the kernel/Ray killing the training
    node. Set the env to 0 to disable.
    """
    import os as _os, resource as _resource
    gb = float(_os.environ.get("FRONTIERCS_RESEARCH_EVAL_RLIMIT_GB", "24"))
    if gb > 0:
        lim = int(gb * (1 << 30))
        _resource.setrlimit(_resource.RLIMIT_AS, (lim, lim))
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OFFICIAL_ROOT = PROJECT_ROOT / ".cache" / "Frontier-CS-official"
RESEARCH_ROOT = OFFICIAL_ROOT / "research"
RESEARCH_PROBLEMS = RESEARCH_ROOT / "problems"

DEFAULT_OVERLAY = PROJECT_ROOT.parent / "envs" / "research_overlay"


def _overlay_python() -> str:
    env = os.environ.get("FRONTIERCS_RESEARCH_PYTHON")
    if env:
        return env
    p = DEFAULT_OVERLAY / "bin" / "python"
    if p.exists():
        return str(p)
    # fall back to sft_lf
    sft = PROJECT_ROOT.parent / "envs" / "sft_lf" / "bin" / "python"
    return str(sft if sft.exists() else (PROJECT_ROOT / ".venv" / "bin" / "python"))


CPU_EVAL_PYTHON = _overlay_python()
CPU_EVAL_TIMEOUT = int(os.environ.get("FRONTIERCS_RESEARCH_CPU_TIMEOUT", "2400"))

_INFRA_MARKERS = (
    # Fully-quoted TOP-LEVEL module names only (2026-08-01) -- see the twin
    # comment in frontiercs_research_eval.py: generic import markers turned
    # model-hallucinated submodule imports into fake infra errors.
    "no module named 'triton'",
    "no module named 'torch'",
    "no module named 'faiss'",
    "no module named 'flashinfer'",
    "no module named 'numpy'",
    "no module named 'sky_spot'",
    "no module named 'juliacall'",
    "no module named 'pysr'",
    "command not found",
    "no such file or directory",
    "g++: not found",
    "could not find julia",
    "juliacall",
    "permission denied",
    "disk quota exceeded",
)


class ResearchInfraError(RuntimeError):
    """Infra failure (missing dep / toolchain / data), not a model-side 0."""


# ---------------------------------------------------------------------------
# Per-family invocation spec.  Keyed by base family (first path component) with
# special-casing where leaves differ.
# ---------------------------------------------------------------------------
def _family_of(problem_id: str) -> str:
    return problem_id.split("/")[0]


def is_cpu_family(problem_id: str) -> bool:
    return _family_of(problem_id) in _CPU_FAMILIES


_CPU_FAMILIES = {
    "cant_be_late",
    "cant_be_late_multi",
    "cloudcast",
    "grammar_fuzzing",
    "imagenet_pareto",
    "llm_router",
    "llm_sql",
    "nbody_simulation",
    "symbolic_regression",
    "vdb_pareto",
}


def solution_language(problem_id: str) -> str:
    """'cpp' for nbody_simulation, else 'python'."""
    return "cpp" if _family_of(problem_id) == "nbody_simulation" else "python"


def _invocation(problem_id: str, sol_path: Path, out_path: Path) -> list[str]:
    """Argv (after the evaluator.py path) for this problem's official CLI."""
    fam = _family_of(problem_id)
    resources = "resources"  # relative to leaf cwd
    spec = f"{resources}/submission_spec.json"
    if fam in ("cant_be_late", "cant_be_late_multi"):
        # prints {"score":...} to stdout; --spec optional (default in resources)
        argv = ["--solution", str(sol_path)]
        if (RESEARCH_PROBLEMS / problem_id / resources / "submission_spec.json").exists():
            argv += ["--spec", spec]
        return argv
    if fam == "cloudcast":
        return ["--solution", str(sol_path), "--spec", spec, "--out", str(out_path)]
    if fam == "grammar_fuzzing" and "fuzzer" in problem_id:
        return ["--solution-path", str(sol_path), "--output-path", str(out_path)]
    if fam == "symbolic_regression":
        return [
            "--solution-path", str(sol_path),
            "--data-dir", f"{resources}/data",
            "--reference-path", f"{resources}/reference_metrics.json",
            "--output-path", str(out_path),
        ]
    if fam == "nbody_simulation":
        return ["--solution-path", str(sol_path), "--output-path", str(out_path)]
    # default: cloudcast/grammar-seed/imagenet/llm_router/llm_sql/vdb_pareto
    return ["--solution", str(sol_path), "--out", str(out_path)]


def _stdout_prints_json(problem_id: str) -> bool:
    """cant_be_late* print the result payload to stdout instead of a file."""
    return _family_of(problem_id) in ("cant_be_late", "cant_be_late_multi")


def _mirror_leaf(src: Path, dst: Path) -> None:
    """Create an isolated, writable working copy of a problem leaf at `dst`.

    Top-level entries are symlinked to the originals (cheap, shares the heavy
    SIFT1M / routerbench / dataset files), EXCEPT directories the evaluator
    writes into are real (copied) so concurrent runs stay isolated. evaluator.py
    is symlinked: its __file__ resolves to the original tree, so its relative
    sibling lookups (parent.parent/'common' for cant_be_late, resources/data for
    symbolic) still resolve against the read-only original -- which is what we
    want for shared read-only inputs. The evaluator's relative WRITES (./output_ans,
    ./output_program.py, ./data/index_cache) are CWD-relative -> land in `dst`.
    """
    import os as _os
    dst.mkdir(parents=True, exist_ok=True)
    # directories an evaluator may write into (must be private, not symlinked)
    WRITE_DIRS = {"data"}  # vdb_pareto writes data/index_cache
    for item in src.iterdir():
        if item.name == "__pycache__":
            continue
        target = dst / item.name
        if item.is_dir() and item.name in WRITE_DIRS:
            # real dir; symlink its children (the *.fvecs etc.) so the cache
            # subdir created at runtime is private to this workdir.
            target.mkdir(exist_ok=True)
            for child in item.iterdir():
                try:
                    _os.symlink(child.resolve(), target / child.name)
                except FileExistsError:
                    pass
        else:
            try:
                _os.symlink(item.resolve(), target)
            except FileExistsError:
                pass


# ---------------------------------------------------------------------------
# Score extraction
# ---------------------------------------------------------------------------
def _coerce_score(v: Any) -> Optional[float]:
    """float() the value, clamp to [0,100]; tolerate inf/-inf/nan."""
    try:
        s = float(v)
    except (TypeError, ValueError):
        return None
    if s != s:  # NaN
        return 0.0
    return max(0.0, min(100.0, s))


def _score_from_payload(payload: dict) -> tuple[Optional[float], Optional[str]]:
    """Return (score 0..100 or None, error-string or None)."""
    if not isinstance(payload, dict):
        return None, "evaluator output not a JSON object"
    err = payload.get("error")
    # cant_be_late_multi hard-failure sentinel
    cs = payload.get("combined_score")
    if cs is not None and isinstance(cs, (int, float)) and cs <= -1e8:
        return 0.0, (str(err) if err else "evaluator combined_score sentinel (failed)")
    # symbolic_regression nests the aggregate under summary.mean_score
    if "score" not in payload and isinstance(payload.get("summary"), dict):
        ms = payload["summary"].get("mean_score")
        if ms is not None:
            s = _coerce_score(ms)
            if s is not None:
                return s, (str(err) if err else None)
    if "score" in payload and payload["score"] is not None:
        s = _coerce_score(payload["score"])
        if s is None:
            return None, f"non-numeric score: {payload['score']!r}"
        return s, (str(err) if err else None)
    if err:
        return 0.0, str(err)
    return None, "no 'score' key in evaluator output"


def _grep_stdout_json(text: str) -> Optional[dict]:
    """Find the last top-level JSON object in stdout (cant_be_late prints it)."""
    for line in reversed((text or "").strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except Exception:
                continue
    return None


# ---------------------------------------------------------------------------
# Multi-rep scoring (evaluator-side randomness mitigation, 2026-08-03).
# Twin of the block in frontiercs_research_eval.py -- these two modules are
# deliberately import-independent, so the ~60 lines are duplicated (same
# pattern as _rlimit_preexec / _INFRA_MARKERS).
#   FRONTIERCS_RESEARCH_SCORE_REPS  (int, default 1): >1 => run evaluator N
#       times, record MEDIAN score, per-rep scores kept in `raw`.
#   FRONTIERCS_RESEARCH_REPS_ONLY   comma list of problem ids / family
#       prefixes, or path to data/research_scoring_variance.json (uses its
#       lottery_problems / per-problem lottery flags). Empty = all problems.
# ---------------------------------------------------------------------------
_REPS_ONLY_CACHE: dict[str, set] = {}


def _load_reps_only(spec: str) -> set:
    if spec in _REPS_ONLY_CACHE:
        return _REPS_ONLY_CACHE[spec]
    ids: set = set()
    p = Path(spec)
    if spec.endswith(".json") or p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            lp = data.get("lottery_problems")
            if isinstance(lp, list):
                ids.update(str(x) for x in lp)
            probs = data.get("problems")
            if isinstance(probs, dict):
                ids.update(pid for pid, rec in probs.items()
                           if isinstance(rec, dict) and rec.get("lottery"))
        except Exception as exc:
            print(f"[research-reps] WARNING: cannot read REPS_ONLY file "
                  f"{spec!r}: {exc}", file=sys.stderr)
    else:
        ids.update(x.strip() for x in spec.split(",") if x.strip())
    _REPS_ONLY_CACHE[spec] = ids
    return ids


def _score_reps_for(problem_id: str) -> int:
    try:
        reps = int(os.environ.get("FRONTIERCS_RESEARCH_SCORE_REPS", "1") or "1")
    except ValueError:
        reps = 1
    if reps <= 1:
        return 1
    only = (os.environ.get("FRONTIERCS_RESEARCH_REPS_ONLY") or "").strip()
    if not only:
        return reps
    ids = _load_reps_only(only)
    family = problem_id.split("/")[0]
    return reps if (problem_id in ids or family in ids) else 1


def _median_of_reps(problem_id: str, run_once, reps: int) -> dict[str, Any]:
    """Run `run_once()` `reps` times; return the rep closest to the median with
    score/score_unbounded replaced by the medians. Per-rep scores + statuses +
    infra errors are prepended to `raw`. All-reps-infra-fail re-raises."""
    results: list[dict[str, Any]] = []
    rep_errors: list[str] = []
    last_exc: Optional[BaseException] = None
    for _ in range(reps):
        try:
            results.append(run_once())
        except ResearchInfraError as exc:
            rep_errors.append(str(exc)[:300])
            last_exc = exc
    if not results:
        raise ResearchInfraError(
            f"{problem_id}: all {reps} scoring reps failed (last: {rep_errors[-1]})"
        ) from last_exc
    scores = [float(r.get("score") or 0.0) for r in results]
    med = float(statistics.median(scores))
    su_med = float(statistics.median(
        [float(r.get("score_unbounded") or 0.0) for r in results]))
    base = min(results, key=lambda r: abs(float(r.get("score") or 0.0) - med))
    rec = dict(base)
    rec["score"] = med
    rec["score_unbounded"] = su_med
    header = json.dumps({"multi_rep": {
        "reps": reps, "scores": scores, "median": med,
        "statuses": [r.get("status") for r in results],
        "rep_infra_errors": rep_errors,
    }})
    rec["raw"] = (f"[multi-rep] {header}\n" + (base.get("raw") or ""))[:8000]
    return rec


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def evaluate_cpu_research_solution(
    problem_id: str,
    solution_code: str,
    *,
    python_exe: str = CPU_EVAL_PYTHON,
    timeout: int = CPU_EVAL_TIMEOUT,
) -> dict[str, Any]:
    """Run a CPU-family evaluator.py in-place (cwd = the problem leaf dir) with
    the official CLI for that family. Returns:
        {score, score_unbounded, status, message, raw}
    score on 0..100. Raises ResearchInfraError on infrastructure failure.

    With FRONTIERCS_RESEARCH_SCORE_REPS>1 (optionally restricted via
    FRONTIERCS_RESEARCH_REPS_ONLY) the evaluator runs N times and the MEDIAN
    score is recorded; per-rep scores live in `raw`.
    """
    reps = _score_reps_for(problem_id)
    if reps <= 1:
        return _evaluate_cpu_research_solution_once(
            problem_id, solution_code, python_exe=python_exe, timeout=timeout)
    return _median_of_reps(
        problem_id,
        lambda: _evaluate_cpu_research_solution_once(
            problem_id, solution_code, python_exe=python_exe, timeout=timeout),
        reps,
    )


def _evaluate_cpu_research_solution_once(
    problem_id: str,
    solution_code: str,
    *,
    python_exe: str = CPU_EVAL_PYTHON,
    timeout: int = CPU_EVAL_TIMEOUT,
) -> dict[str, Any]:
    """Single evaluator run (the pre-2026-08-03 evaluate_cpu_research_solution)."""
    problem_path = RESEARCH_PROBLEMS / problem_id
    if not problem_path.exists():
        raise ResearchInfraError(f"research problem not found: {problem_path}")
    if not is_cpu_family(problem_id):
        raise ResearchInfraError(f"not a CPU family: {problem_id}")
    if not solution_code or not solution_code.strip():
        return {"score": 0.0, "score_unbounded": 0.0, "status": "empty",
                "message": "empty solution code", "raw": ""}

    lang = solution_language(problem_id)
    suffix = ".cpp" if lang == "cpp" else ".py"

    # Run each eval in an ISOLATED working copy of the leaf so concurrent samples
    # of the same problem don't clobber each other's artifacts (grammar_fuzzing
    # writes ./output_ans, cloudcast writes ./output_program.py, vdb writes
    # data/index_cache, etc). Mirror the leaf with symlinks (cheap for the large
    # SIFT1M / routerbench / BIRD data) so resources/ + data/ resolve, but the
    # cwd itself is private and writable.
    with tempfile.TemporaryDirectory(prefix="fcs_cpu_") as tmp:
        tmp = Path(tmp)
        sol_path = tmp / f"solution{suffix}"
        sol_path.write_text(solution_code, encoding="utf-8")
        out_path = tmp / "result.json"

        workdir = tmp / "leaf"
        _mirror_leaf(problem_path, workdir)

        argv = _invocation(problem_id, sol_path, out_path)
        cmd = [python_exe, "evaluator.py", *argv]

        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("CBL_LOG_LEVEL", "ERROR")
        # Julia depot for symbolic_regression / pysr
        env.setdefault("JULIA_DEPOT_PATH", str(DEFAULT_OVERLAY / "julia_depot"))
        env.setdefault("PYTHON_JULIAPKG_PROJECT", str(DEFAULT_OVERLAY / "julia_env"))
        # avoid CPU oversubscription on shared nodes
        env.setdefault("OMP_NUM_THREADS", env.get("FRONTIERCS_OMP", "8"))

        try:
            proc = subprocess.run(
                cmd, cwd=str(workdir), capture_output=True, text=True,
                timeout=timeout, env=env, preexec_fn=_rlimit_preexec,
            )
        except subprocess.TimeoutExpired as exc:
            raise ResearchInfraError(
                f"{problem_id}: evaluator timed out after {timeout}s") from exc

        logs = (proc.stdout or "") + "\n" + (proc.stderr or "")

        # 1) file output (most families)
        payload = None
        if out_path.exists():
            try:
                payload = json.loads(out_path.read_text(encoding="utf-8"))
            except Exception:
                payload = None
        # 2) cant_be_late* print JSON to stdout
        if payload is None and _stdout_prints_json(problem_id):
            payload = _grep_stdout_json(proc.stdout or "")

        if payload is not None:
            score, err = _score_from_payload(payload)
            if score is not None:
                # An error string that looks like a dependency/infra problem is
                # loud; a model-side runtime error (NameError, AttributeError on
                # the solution, syntax error in solution.py) is a legitimate 0.
                if err and any(m in err.lower() for m in _INFRA_MARKERS):
                    raise ResearchInfraError(f"{problem_id}: {err[:300]}")
                su = payload.get("score_unbounded", score)
                if isinstance(payload.get("summary"), dict) and "score_unbounded" not in payload:
                    su = payload["summary"].get("mean_score_unbounded", score)
                try:
                    su = float(su)
                    if su != su or su in (float("inf"), float("-inf")):
                        # keep unbounded finite for JSON/parquet; fall back to score
                        su = score
                except (TypeError, ValueError):
                    su = score
                return {
                    "score": float(score),
                    "score_unbounded": float(su),
                    "status": "success" if not err else "failed",
                    "message": (err[:300] if err else None),
                    "raw": logs[-4000:],
                }
            # payload present but unparseable score
            if any(m in logs.lower() for m in _INFRA_MARKERS):
                raise ResearchInfraError(f"{problem_id}: {err or 'unparseable score'}")
            return {"score": 0.0, "score_unbounded": 0.0, "status": "failed",
                    "message": (err or "unparseable score")[:300], "raw": logs[-4000:]}

        # 3) no parseable output at all -> infra failure
        low = logs.lower()
        if any(m in low for m in _INFRA_MARKERS) or proc.returncode != 0:
            tail = (logs.strip().splitlines() or [""])[-1]
            raise ResearchInfraError(
                f"{problem_id}: evaluator produced no result (rc={proc.returncode}): {tail[:300]}")
        return {"score": 0.0, "score_unbounded": 0.0, "status": "failed",
                "message": "no result.json and no parseable stdout", "raw": logs[-4000:]}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Frontier-CS CPU research eval adapter")
    ap.add_argument("--eval", metavar="PID", required=True)
    ap.add_argument("--solution", metavar="PATH", required=True)
    args = ap.parse_args()
    code = Path(args.solution).read_text(encoding="utf-8")
    try:
        res = evaluate_cpu_research_solution(args.eval, code)
        print(json.dumps({k: v for k, v in res.items() if k != "raw"}, indent=2))
    except ResearchInfraError as exc:
        print(f"INFRA-ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
