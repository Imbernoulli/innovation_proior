#!/usr/bin/env python3
"""
Frontier-CS 2.0 ("eng" / engineering / open-ended-optimization) track evaluation,
adapted for an HPC node WITHOUT Docker -- the direct sibling of
scripts/frontiercs_research_eval.py.

============================================================================
!! NAMING / SCOPE WARNING -- read before assuming this is "Frontier-Eng" !!
============================================================================
This module is the Frontier-CS *2.0* track (erdos / bboplace / vector_db /
generals_io / duckdb problems, single-shot solution.py generation scored by a
per-problem evaluator.py). It is NOT the standalone **Frontier-Engineering**
benchmark (arXiv 2604.12290, EinsiaLab/Frontier-Engineering: 47 iterative
generative-optimization tasks, `python -m frontier_eval task=unified ...`
driven by the OpenEvolve scaffold).

The two are DIFFERENT benchmarks that both happen to use the word "eng":
  * Frontier-CS 2.0 "eng" track   -> THIS FILE (data_source=frontiercs_eng,
                                     single-shot, no search loop).
  * Frontier-Engineering (v1)     -> /scratch/gpfs/CHIJ/bohan/Frontier-Engineering
                                     official repo; the FAITHFUL our-model harness
                                     lives there at scripts/ops/cc_ourmodel_eval.sh
                                     (+ submit_ourmodel_matrix.sh, kernel launcher
                                     cc_ourmodel_kernel_gpu.sh). Use THAT, not this,
                                     for Frontier-Engineering evaluation.

Do NOT wire this module into any "Frontier-Engineering" pipeline. See the
official repo's run.md / README.md for the real protocol.
============================================================================

The official `frontier eval 2.0 <problem_id> <solution>` / Harbor path runs each
problem's `evaluator.py` inside a per-problem Docker image (sometimes with a hidden
`judge_image` holding private benchmark data). Docker is NOT available on this
cluster (only apptainer/singularity), so this module runs each problem's
`evaluator.py` DIRECTLY on the compute node, reproducing the local-CLI workspace
the per-problem `evaluate.sh` expects:

    evaluate.sh: python evaluator.py /work/execution_env/solution_env/solution.py

i.e. the evaluator is invoked with a SINGLE positional argument -- the path to the
submitted solution (a .py file for the Python problems, a .json file for the
bboplace_direct_* problems) -- and prints `"<score> <score_unbounded>"` as its
last stdout line (the evaluator `main()` prints the human message to stderr and
the numeric pair to stdout). This is the 2.0 evaluator contract documented in
2.0/CONTRIBUTING.md:

    def evaluate(solution_path) -> (score, score_unbounded, message[, metrics])

Score is on a 0..100 scale (same continuous scoring philosophy as the rest of
Frontier-CS).

This module is import-safe: it does NOT import the official `frontier_cs` package
(whose __init__ pulls in google.generativeai and other LLM-gen deps). It only
reads the on-disk problem files and shells out to each `evaluator.py`.

RUNNABLE here (no Docker, no Harbor agent loop, deps already in envs/sft_lf):
  - erdos_demo            : pure-Python planar geometry, N=10, no external packages.
  - erdos_unit_distance   : pure-Python planar geometry, N=65536, no external packages.

BLOCKED here (documented, like the Research track's poc_generation Docker note):
  - bboplace_ispd2005 / bboplace_iccad2015 / bboplace_direct_ispd2005 /
    bboplace_direct_iccad2015 : the evaluator imports the BBOPlace-Bench runtime +
      ISPD2005/ICCAD2015 LEF/DEF benchmark data from BBOPLACE_ROOT (default
      /opt/bboplace-bench), which lives ONLY in the hidden judge image
      `ghcr.io/frontiercs/frontiercs-bboplace-data:*`. The README's "CPU-only, no
      DREAMPlace/GPU/Ray" claim is about the *runtime*; the benchmark *data* is
      judge-only and is not on this cluster -> evaluator raises
      "BBOPlace runtime not found at /opt/bboplace-bench".
  - vector_db_ann / vector_db_ann_relaxed : whole-`/app` Rust ANN service scored by
      a hidden SIFT1M-scale benchmark via Harbor (directory submission + judge
      sidecar). No standalone single-file solution; needs Harbor + hidden data.
  - generals_io_bot : patch-based bot scored against a hidden arena inside custom
      docker images (frontiercs/generals-io-bot-{agent,judge}); needs Docker + the
      hidden arena.
  - duckdb_e2e_query_optimization : DuckDB C++ source patch built + TPC-H timed
      inside custom experimental docker images; needs Docker-built judge image.

Provenance of reproduced logic (file:line in .cache/Frontier-CS-official):
  - evaluator contract / score line:  2.0/CONTRIBUTING.md (evaluate signature),
                                       2.0/problems/<id>/evaluator.py main()
  - local invocation:                 2.0/problems/<id>/evaluate.sh
  - prompt (readme-as-statement):     mirrors research/scripts gen_env + this file.
  - code extraction:                  same regex as frontiercs_research_eval.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

import yaml

# ----------------------------------------------------------------------------
# Locations
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OFFICIAL_ROOT = PROJECT_ROOT / ".cache" / "Frontier-CS-official"
ENG_ROOT = OFFICIAL_ROOT / "2.0"
ENG_PROBLEMS = ENG_ROOT / "problems"

# Problems that RUN DIRECTLY on this no-Docker cluster (deps already in envs/sft_lf:
# pure stdlib; erdos needs no external packages). bboplace_* and the service/game
# problems are BLOCKED (see module docstring) and intentionally NOT listed here.
RUNNABLE_PROBLEMS = (
    "erdos_demo",
    "erdos_unit_distance",
)

# Solution artifact kind per problem ("python" -> solution.py; "json" -> solution.json).
# Derived from config.yaml runtime.language / submission.kind; hard-coded for the
# runnable subset to keep this import-safe and explicit.
SOLUTION_KIND = {
    "erdos_demo": "python",
    "erdos_unit_distance": "python",
    # bboplace_direct_* would be "json"; bboplace_* would be "python" (blocked).
}

BLOCKED_PROBLEMS = {
    "bboplace_ispd2005": "needs BBOPlace-Bench runtime + ISPD2005 benchmark data "
    "(hidden frontiercs-bboplace-data judge image at /opt/bboplace-bench)",
    "bboplace_iccad2015": "needs BBOPlace-Bench runtime + ICCAD2015 benchmark data "
    "(hidden frontiercs-bboplace-data judge image at /opt/bboplace-bench)",
    "bboplace_direct_ispd2005": "needs BBOPlace-Bench runtime + ISPD2005 adaptec1 data "
    "(hidden frontiercs-bboplace-data judge image at /opt/bboplace-bench)",
    "bboplace_direct_iccad2015": "needs BBOPlace-Bench runtime + ICCAD2015 superblue1 data "
    "(hidden frontiercs-bboplace-data judge image at /opt/bboplace-bench)",
    "vector_db_ann": "whole-/app Rust ANN service scored by a hidden SIFT1M benchmark "
    "via Harbor (directory submission + judge sidecar); no single-file solution",
    "vector_db_ann_relaxed": "whole-/app Rust ANN service scored by a hidden SIFT1M "
    "benchmark via Harbor (directory submission + judge sidecar); no single-file solution",
    "generals_io_bot": "patch-based bot scored against a hidden arena inside custom "
    "docker images (frontiercs/generals-io-bot-{agent,judge}); needs Docker",
    "duckdb_e2e_query_optimization": "DuckDB C++ source patch built + TPC-H timed inside "
    "custom experimental docker images; needs Docker-built judge image",
}


# Default python used to run the evaluator. The runnable eng problems (erdos) are
# pure stdlib, so envs/sft_lf (or the project .venv) is sufficient; no overlay
# needed. Overridable via FRONTIERCS_ENG_PYTHON.
def _default_eng_python() -> str:
    env = os.environ.get("FRONTIERCS_ENG_PYTHON")
    if env:
        return env
    sft_lf = PROJECT_ROOT.parent / "envs" / "sft_lf" / "bin" / "python"
    if sft_lf.exists():
        return str(sft_lf)
    return str(PROJECT_ROOT / ".venv" / "bin" / "python")


ENG_EVAL_PYTHON = _default_eng_python()
# Per-problem hard wall-clock timeout (seconds) for the evaluator subprocess.
# erdos_unit_distance is N=65536 with a 10800s official budget; default generously.
ENG_EVAL_TIMEOUT = int(os.environ.get("FRONTIERCS_ENG_TIMEOUT", "1800"))


# ----------------------------------------------------------------------------
# Enumeration
# ----------------------------------------------------------------------------
def list_eng_problems() -> list[str]:
    """RUNNABLE 2.0 ("eng") problems on this no-Docker HPC harness."""
    return [p for p in RUNNABLE_PROBLEMS if (ENG_PROBLEMS / p / "evaluator.py").exists()]


def list_all_eng_problems() -> list[str]:
    """Every 2.0 problem directory on disk (runnable + blocked)."""
    if not ENG_PROBLEMS.exists():
        return []
    out = []
    for ev in ENG_PROBLEMS.glob("*/evaluator.py"):
        out.append(ev.parent.name)
    return sorted(set(out))


def solution_kind(problem_id: str) -> str:
    return SOLUTION_KIND.get(problem_id, "python")


def _load_config(problem_path: Path) -> dict:
    cfg = problem_path / "config.yaml"
    if not cfg.exists():
        return {}
    text = cfg.read_text(encoding="utf-8")
    try:
        return yaml.safe_load(text) or {}
    except Exception:
        try:
            return json.loads(text)
        except Exception:
            return {}


# ----------------------------------------------------------------------------
# Prompt construction (readme-as-statement; mirrors research build_messages)
# ----------------------------------------------------------------------------
_PY_TEMPLATE = """You are an expert programmer. Generate Python code for the given optimization problem.

EVALUATION ENVIRONMENT:
- CPU-only; no GPU. Python 3.11; standard library only (no external packages).

REQUIREMENTS:
1. Output ONLY Python code - no explanations, no markdown.
2. Define exactly the function/variable the Program Interface section asks for.
3. Use an efficient algorithm appropriate for the problem size.
4. The returned object must match the interface and validity constraints exactly.

Output ONLY the code, starting with imports."""


def read_readme(problem_id: str) -> str:
    pp = ENG_PROBLEMS / problem_id
    for name in ("readme", "README.md", "README", "readme.md"):
        f = pp / name
        if f.exists():
            return f.read_text(encoding="utf-8")
    raise FileNotFoundError(f"No readme in {pp}")


def build_eng_messages(problem_id: str) -> list[dict[str, str]]:
    """Canonical 2.0 generation prompt: system (env-aware) + user(readme)."""
    system_prompt = _PY_TEMPLATE
    readme = read_readme(problem_id)
    user_prompt = f"Problem:\n\n{readme}\n\nGenerate solution code:"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# ----------------------------------------------------------------------------
# Code extraction (same as frontiercs_research_eval, language=python)
# ----------------------------------------------------------------------------
def extract_python_code(text: str) -> str:
    code = (text or "").strip()
    pattern = r"```(?:python)?\s*\n(.*?)```"
    matches = re.findall(pattern, code, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()
    if code.startswith("```python"):
        code = code[len("```python"):].strip()
    if code.startswith("```"):
        code = code[3:].strip()
    if code.endswith("```"):
        code = code[:-3].strip()
    return code.strip()


# ----------------------------------------------------------------------------
# Evaluation (run evaluator.py with single positional arg; parse score line)
# ----------------------------------------------------------------------------
def _parse_score(output: str) -> tuple[Optional[float], Optional[float], Optional[str]]:
    """Parse the last numeric `"<score> <unbounded>"` stdout line. Identical
    contract to frontiercs_research_eval._parse_score."""
    lines = output.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("[") or "INFO" in line or "ERROR" in line:
            continue
        parts = line.split()
        if not parts:
            continue
        try:
            score = float(parts[0])
            score_unbounded = float(parts[1]) if len(parts) > 1 else score
            return score, score_unbounded, None
        except ValueError:
            continue
    for line in lines:
        if "Error" in line or "ERROR" in line:
            return None, None, line.strip()[:300]
    return None, None, "Could not parse score from evaluator output"


class EngInfraError(RuntimeError):
    """Raised when the 2.0 evaluator fails for INFRASTRUCTURE reasons (missing
    benchmark data / runtime, import/dependency failure, timeout, no parseable
    score) rather than a legitimate model-side 0 (solution ran but was invalid /
    did not beat the baseline). Loud on purpose so a blocked/broken substrate is
    not silently scored 0 (mirrors ResearchInfraError)."""


_INFRA_MARKERS = (
    "no module named",
    "modulenotfounderror",
    "importerror",
    "bboplace runtime not found",
    "judge image is missing",
    "could not parse score",
    "command not found",
    "no such file",
)


def evaluate_eng_solution(
    problem_id: str,
    solution_code: str,
    *,
    python_exe: str = ENG_EVAL_PYTHON,
    timeout: int = ENG_EVAL_TIMEOUT,
) -> dict[str, Any]:
    """Run the official 2.0 evaluator.py directly. Returns a dict:
        {score, score_unbounded, status, message, raw}
    score/score_unbounded are floats on 0..100 (or None on infra failure).
    Raises EngInfraError on infrastructure failure (blocked problem / missing
    data / crash) so callers record an error rather than a silent 0.
    """
    problem_path = ENG_PROBLEMS / problem_id
    if not problem_path.exists():
        raise EngInfraError(f"2.0 problem not found: {problem_path}")
    if problem_id in BLOCKED_PROBLEMS and problem_id not in RUNNABLE_PROBLEMS:
        raise EngInfraError(f"{problem_id} is BLOCKED on this harness: {BLOCKED_PROBLEMS[problem_id]}")
    if not solution_code or not solution_code.strip():
        return {
            "score": 0.0, "score_unbounded": 0.0, "status": "empty",
            "message": "empty solution code", "raw": "",
        }

    kind = solution_kind(problem_id)
    sol_name = "solution.json" if kind == "json" else "solution.py"

    with tempfile.TemporaryDirectory(prefix="fcs_eng_") as tmp:
        work = Path(tmp) / "work"
        # Reproduce the evaluate.sh workspace layout:
        #   /work/execution_env/solution_env/solution.{py,json}
        sol_dir = work / "execution_env" / "solution_env"
        sol_dir.mkdir(parents=True)
        sol_path = sol_dir / sol_name
        sol_path.write_text(solution_code, encoding="utf-8")

        # Run from a clean problem copy so result/temp artifacts never touch the
        # shared .cache tree (and __pycache__ is not written into it).
        prob_dst = work / "problem"
        shutil.copytree(problem_path, prob_dst, ignore=shutil.ignore_patterns("__pycache__"))

        cmd = [python_exe, "evaluator.py", str(sol_path)]
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        try:
            proc = subprocess.run(
                cmd, cwd=str(prob_dst), capture_output=True, text=True,
                timeout=timeout, env=env,
            )
        except subprocess.TimeoutExpired as exc:
            raise EngInfraError(
                f"2.0 evaluator timed out after {timeout}s for {problem_id}"
            ) from exc

        logs = (proc.stdout or "") + "\n" + (proc.stderr or "")
        score, su, perr = _parse_score(proc.stdout or "")
        if score is not None:
            return {
                "score": float(score), "score_unbounded": float(su),
                "status": "success", "message": None, "raw": logs[-4000:],
            }
        # No parseable score -> classify infra vs failed.
        low = logs.lower()
        if any(m in low for m in _INFRA_MARKERS) or proc.returncode != 0:
            last = (perr or (logs.strip().splitlines()[-1:] or [""])[0])
            raise EngInfraError(
                f"{problem_id}: evaluator produced no score (rc={proc.returncode}): {last}"[:300]
            )
        return {
            "score": 0.0, "score_unbounded": 0.0, "status": "failed",
            "message": perr or "no score", "raw": logs[-4000:],
        }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Frontier-CS 2.0 (eng) eval (no Docker)")
    ap.add_argument("--list", action="store_true", help="list runnable 2.0 problems")
    ap.add_argument("--list-all", action="store_true", help="list all 2.0 problems (runnable+blocked)")
    ap.add_argument("--prompt", metavar="PID", help="print the generation prompt for a problem")
    ap.add_argument("--eval", metavar="PID", help="evaluate a solution file for a problem")
    ap.add_argument("--solution", metavar="PATH", help="solution.{py,json} path (with --eval)")
    args = ap.parse_args()

    if args.list:
        for p in list_eng_problems():
            print(p)
    elif args.list_all:
        runnable = set(list_eng_problems())
        for p in list_all_eng_problems():
            tag = "RUNNABLE" if p in runnable else f"BLOCKED  ({BLOCKED_PROBLEMS.get(p, 'n/a')})"
            print(f"{p:32s} {tag}")
    elif args.prompt:
        for m in build_eng_messages(args.prompt):
            print(f"===== {m['role']} =====")
            print(m["content"])
    elif args.eval:
        code = Path(args.solution).read_text(encoding="utf-8")
        try:
            res = evaluate_eng_solution(args.eval, code)
            print(json.dumps({k: v for k, v in res.items() if k != "raw"}, indent=2))
        except EngInfraError as exc:
            print(f"INFRA-ERROR: {exc}", file=sys.stderr)
            sys.exit(2)
    else:
        ap.print_help()
