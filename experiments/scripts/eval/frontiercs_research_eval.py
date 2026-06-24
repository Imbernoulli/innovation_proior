#!/usr/bin/env python3
"""
Frontier-CS RESEARCH-track evaluation, adapted for an HPC node WITHOUT Docker.

The official `frontier eval research ...` path runs each problem's `evaluator.py`
inside a per-problem Docker image (GPU passthrough, per-problem deps). Docker is
NOT available on this cluster (only apptainer/singularity), so this module runs
the official `evaluator.py` DIRECTLY on the compute node, reproducing the Docker
workspace layout the evaluator expects:

    /work/research/<problem_id>/...        (copied problem dir + resources/)
    /work/execution_env/solution_env/solution.py
    cwd = the problem dir; run `python evaluator.py --solution-path ... \
          --spec-path resources/submission_spec.json --output-path result.json`

The evaluator prints `"<score> <score_unbounded>"` as its last stdout line and
also writes result.json. Score is on a 0..100 scale (same as the leaderboard's
per-problem score; the track Avg@k is the mean of these).

This module is import-safe: it does NOT import the official `frontier_cs` package
(whose __init__ pulls in google.generativeai and other LLM-gen deps). It only
reads the on-disk problem files, mirroring `SingleEvaluator.list_problems` and
`ResearchDockerRunner` enumeration/score-parsing logic.

Provenance of reproduced logic (file:line in .cache/Frontier-CS-official):
  - problem enumeration:  src/frontier_cs/single_evaluator.py:259-296
  - workspace layout:     src/frontier_cs/runner/research_docker.py:191-223, 295-349
  - score parsing:        src/frontier_cs/runner/research_docker.py:351-382
  - prompt (system+user): research/scripts/gen_env.py (PROMPT_TEMPLATES, get_system_prompt_for_problem)
                          research/scripts/generate_solutions.py:150 (user_prompt)
  - code extraction:      research/scripts/generate_solutions.py:228-254
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
RESEARCH_ROOT = OFFICIAL_ROOT / "research"
RESEARCH_PROBLEMS = RESEARCH_ROOT / "problems"

# Default python used to run the evaluator. Use envs/sft_lf (torch 2.10 +
# triton 3.6.0): the Triton-kernel problems' benchmark harness calls
# `triton.set_allocator`, a NEWER-triton API present in 3.6 but ABSENT in the
# project .venv's triton 3.2.0 -- so .venv import-errors on every GPU problem.
# (The official Docker image is a custom `triton-tlx` build, not stock 3.2.)
# Overridable via FRONTIERCS_RESEARCH_PYTHON.
def _default_research_python() -> str:
    env = os.environ.get("FRONTIERCS_RESEARCH_PYTHON")
    if env:
        return env
    sft_lf = PROJECT_ROOT.parent / "envs" / "sft_lf" / "bin" / "python"
    if sft_lf.exists():
        return str(sft_lf)
    return str(PROJECT_ROOT / ".venv" / "bin" / "python")


RESEARCH_EVAL_PYTHON = _default_research_python()
# Per-problem hard wall-clock timeout (seconds) for the evaluator subprocess.
RESEARCH_EVAL_TIMEOUT = int(os.environ.get("FRONTIERCS_RESEARCH_TIMEOUT", "1200"))

POC_SUBCATS = [
    "poc_generation/heap_buffer_overflow",
    "poc_generation/heap_use_after_free",
    "poc_generation/stack_buffer_overflow",
    "poc_generation/uninitialized_value",
]


def list_research_problems_official() -> list[str]:
    """The official 68-problem leaderboard set (poc_generation collapsed to 4
    category entries). Mirrors SingleEvaluator.list_problems research branch."""
    if not RESEARCH_PROBLEMS.exists():
        return []
    problems: list[str] = []
    if (RESEARCH_PROBLEMS / "poc_generation").exists():
        problems.extend(POC_SUBCATS)
    for evaluator in RESEARCH_PROBLEMS.rglob("evaluator.py"):
        if "poc_generation" in str(evaluator):
            continue
        problems.append(str(evaluator.parent.relative_to(RESEARCH_PROBLEMS)))
    return sorted(set(problems))


# ----------------------------------------------------------------------------
# Enumeration  (mirrors single_evaluator.list_problems research branch)
# ----------------------------------------------------------------------------
def list_research_problems() -> list[str]:
    """RUNNABLE research problems on this no-Docker HPC harness.

    This is the official 68-problem set MINUS the 4 poc_generation security
    categories. poc_generation is excluded because: (a) each of the 4 category
    entries is not a single runnable problem -- it aggregates ~20 per-CVE
    `arvo_*`/`oss_fuzz_*` variant subdirs (no category-level readme/evaluator);
    and (b) every variant sets docker `dind: true` (Docker-in-Docker to build
    the sanitizer target), which is impossible without Docker. So 64 problems
    are runnable here. See list_research_problems_official() for the full 68.
    """
    out: list[str] = []
    for evaluator in RESEARCH_PROBLEMS.rglob("evaluator.py"):
        if "poc_generation" in str(evaluator):
            continue
        out.append(str(evaluator.parent.relative_to(RESEARCH_PROBLEMS)))
    return sorted(set(out))


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


def problem_needs_gpu(problem_id: str) -> bool:
    c = _load_config(RESEARCH_PROBLEMS / problem_id)
    rt = c.get("runtime", {}) or {}
    docker = rt.get("docker", {}) or {}
    res = rt.get("resources", {}) or {}
    return bool(docker.get("gpu") or rt.get("requires_gpu") or res.get("accelerators"))


# ----------------------------------------------------------------------------
# Prompt construction  (mirrors gen_env.py + generate_solutions.py)
# ----------------------------------------------------------------------------
_GPU_SPECS = {
    "L4": ("NVIDIA L4", "24GB"),
    "A10G": ("NVIDIA A10G", "24GB"),
    "A100": ("NVIDIA A100", "40GB"),
    "A100-40GB": ("NVIDIA A100", "40GB"),
    "A100-80GB": ("NVIDIA A100", "80GB"),
    "H100": ("NVIDIA H100", "80GB"),
    "V100": ("NVIDIA V100", "16GB"),
    "T4": ("NVIDIA T4", "16GB"),
}

_PY_TEMPLATE = """You are an expert programmer. Generate Python code for the given problem.

{environment_section}
REQUIREMENTS:
1. Output ONLY Python code - no explanations, no markdown
2. Implement ALL required classes/functions from the API section
3. Use efficient algorithms appropriate for the evaluation environment
4. Final class name must match the API specification exactly

Output ONLY the code, starting with imports."""


def _fmt_spec(spec: str) -> str:
    return f"{spec[:-1]}+ (or more)" if spec.endswith("+") else spec


def _effective_gpu_type(rt: dict) -> Optional[str]:
    res = rt.get("resources", {}) or {}
    acc = res.get("accelerators")
    if acc:
        return str(acc).split(":")[0]
    docker = rt.get("docker", {}) or {}
    if docker.get("gpu"):
        return "L4"
    if rt.get("requires_gpu"):
        return "L4"
    return None


def _environment_section(problem_id: str) -> str:
    c = _load_config(RESEARCH_PROBLEMS / problem_id)
    rt = c.get("runtime", {}) or {}
    res = rt.get("resources", {}) or {}
    env_desc = rt.get("environment")
    gpu_type = _effective_gpu_type(rt)
    lines = ["EVALUATION ENVIRONMENT:"]
    if res.get("instance_type"):
        lines.append(f"- Instance: {res['instance_type']}")
    if gpu_type:
        name, vram = _GPU_SPECS.get(gpu_type, _GPU_SPECS["L4"])
        lines.append(f"- GPU: {name} ({vram} VRAM)")
        if res.get("cpus") or res.get("memory"):
            lines.append(
                f"- CPU: {_fmt_spec(str(res.get('cpus','8')))} vCPUs, "
                f"{_fmt_spec(str(res.get('memory','32')))}GB RAM"
            )
    else:
        lines.append(
            f"- CPU-only: {_fmt_spec(str(res.get('cpus','8')))} vCPUs, "
            f"{_fmt_spec(str(res.get('memory','16')))}GB RAM (NO GPU)"
        )
    if res.get("disk_size"):
        lines.append(f"- Disk: {res['disk_size']}GB")
    if env_desc:
        lines.append(f"- {env_desc}")
    return "\n".join(lines)


def read_readme(problem_id: str) -> str:
    pp = RESEARCH_PROBLEMS / problem_id
    for name in ("readme", "README.md", "README", "readme.md"):
        f = pp / name
        if f.exists():
            return f.read_text(encoding="utf-8")
    raise FileNotFoundError(f"No README in {pp}")


def build_research_messages(problem_id: str) -> list[dict[str, str]]:
    """Canonical research generation prompt: system (env-aware) + user(readme)."""
    system_prompt = _PY_TEMPLATE.format(environment_section=_environment_section(problem_id))
    readme = read_readme(problem_id)
    user_prompt = f"Problem:\n\n{readme}\n\nGenerate solution code:"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# ----------------------------------------------------------------------------
# Code extraction  (mirrors generate_solutions.py:228-254, language=python)
# ----------------------------------------------------------------------------
def extract_python_code(text: str) -> str:
    code = (text or "").strip()
    pattern = r"```(?:python)?\s*\n(.*?)```"
    matches = re.findall(pattern, code, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()
    # Fallback: strip stray fences
    if code.startswith("```python"):
        code = code[len("```python"):].strip()
    if code.startswith("```"):
        code = code[3:].strip()
    if code.endswith("```"):
        code = code[:-3].strip()
    return code.strip()


# ----------------------------------------------------------------------------
# Evaluation (mirrors ResearchDockerRunner workspace + score parse, no Docker)
# ----------------------------------------------------------------------------
def _parse_score(output: str) -> tuple[Optional[float], Optional[float], Optional[str]]:
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


class ResearchInfraError(RuntimeError):
    """Raised when the research evaluator fails for INFRASTRUCTURE reasons
    (missing GPU, import/dependency failure, timeout) rather than a legitimate
    model-side 0 (solution ran but failed correctness / produced low speedup).
    Loud on purpose so a broken substrate is not silently scored 0."""


_INFRA_MARKERS = (
    "no module named",
    "modulenotfounderror",
    "importerror",
    "cuda error",
    "no cuda",
    "gpu required",
    "out of memory",
    "could not parse score",
    "command not found",
    "no such file",
    "traceback (most recent call last)",
)


def evaluate_research_solution(
    problem_id: str,
    solution_code: str,
    *,
    python_exe: str = RESEARCH_EVAL_PYTHON,
    timeout: int = RESEARCH_EVAL_TIMEOUT,
) -> dict[str, Any]:
    """Run the official evaluator.py directly. Returns a dict:
        {score, score_unbounded, status, passed_tests, total_tests, message, raw}
    score/score_unbounded are floats on 0..100 (or None on infra failure).
    Raises ResearchInfraError on infrastructure failure (so callers can record
    it as an error rather than a silent 0).
    """
    problem_path = RESEARCH_PROBLEMS / problem_id
    if not problem_path.exists():
        raise ResearchInfraError(f"research problem not found: {problem_path}")
    if not solution_code or not solution_code.strip():
        return {
            "score": 0.0, "score_unbounded": 0.0, "status": "empty",
            "message": "empty solution code", "raw": "",
        }

    with tempfile.TemporaryDirectory(prefix="fcs_research_") as tmp:
        work = Path(tmp) / "work"
        prob_dst = work / "research" / problem_id
        prob_dst.mkdir(parents=True)
        # Copy problem dir (files + subdirs e.g. resources/), skip __pycache__
        for item in problem_path.iterdir():
            if item.name == "__pycache__":
                continue
            if item.is_file():
                shutil.copy2(item, prob_dst / item.name)
            elif item.is_dir():
                shutil.copytree(item, prob_dst / item.name)
        # Copy parent-level `common/` dirs (some problems import from them)
        parts = problem_id.split("/")
        for i in range(1, len(parts)):
            parent = "/".join(parts[:i])
            common = RESEARCH_PROBLEMS / parent / "common"
            if common.is_dir():
                dst = work / "research" / parent / "common"
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    shutil.copytree(common, dst)
        # Place solution at the Docker-expected path
        sol_dir = work / "execution_env" / "solution_env"
        sol_dir.mkdir(parents=True)
        sol_path = sol_dir / "solution.py"
        sol_path.write_text(solution_code, encoding="utf-8")

        spec = prob_dst / "resources" / "submission_spec.json"
        cmd = [python_exe, "evaluator.py", "--solution-path", str(sol_path)]
        if spec.exists():
            cmd += ["--spec-path", str(spec)]
        cmd += ["--output-path", "./result.json"]

        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        # Make /work absolute path resolvable for evaluate.sh-style scripts that
        # reference /work/execution_env -- we don't use evaluate.sh (it hardcodes
        # /work) and call evaluator.py directly, so EXEC_ROOT is passed via flag.
        try:
            proc = subprocess.run(
                cmd, cwd=str(prob_dst), capture_output=True, text=True,
                timeout=timeout, env=env,
            )
        except subprocess.TimeoutExpired as exc:
            raise ResearchInfraError(
                f"research evaluator timed out after {timeout}s for {problem_id}"
            ) from exc

        logs = (proc.stdout or "") + "\n" + (proc.stderr or "")

        # Prefer the structured result.json if present (authoritative).
        result_json = prob_dst / "result.json"
        if result_json.exists():
            try:
                rj = json.loads(result_json.read_text(encoding="utf-8"))
            except Exception:
                rj = None
            if isinstance(rj, dict):
                status = rj.get("status")
                if status == "success" and "score" in rj:
                    score = float(rj.get("score") or 0.0)
                    su = rj.get("score_unbounded", score)
                    return {
                        "score": score,
                        "score_unbounded": float(su if su is not None else score),
                        "status": "success",
                        "passed_tests": rj.get("passed_tests"),
                        "total_tests": rj.get("total_tests"),
                        "geometric_mean_speedup": rj.get("geometric_mean_speedup"),
                        "message": rj.get("error"),
                        "raw": logs[-4000:],
                    }
                # status == "error" / failed: evaluator decided this is a real fail.
                # Treat as legitimate 0 (model side) unless it looks like infra.
                msg = str(rj.get("error", "") or "")
                if any(m in msg.lower() for m in _INFRA_MARKERS):
                    raise ResearchInfraError(f"{problem_id}: {msg[:300]}")
                return {
                    "score": 0.0, "score_unbounded": 0.0, "status": "failed",
                    "message": msg[:300] or "evaluator status!=success", "raw": logs[-4000:],
                }

        # No result.json -> fall back to parsing the last numeric line of stdout.
        score, su, perr = _parse_score(logs)
        if score is not None:
            return {
                "score": float(score), "score_unbounded": float(su),
                "status": "success", "message": None, "raw": logs[-4000:],
            }
        # No score at all -> infra failure (import error / GPU missing / crash).
        low = logs.lower()
        if any(m in low for m in _INFRA_MARKERS) or proc.returncode != 0:
            raise ResearchInfraError(
                f"{problem_id}: evaluator produced no score (rc={proc.returncode}): "
                f"{(perr or logs.strip().splitlines()[-1:] or [''])[0]}"[:300]
            )
        return {
            "score": 0.0, "score_unbounded": 0.0, "status": "failed",
            "message": perr or "no score", "raw": logs[-4000:],
        }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Frontier-CS research eval (no Docker)")
    ap.add_argument("--list", action="store_true", help="list research problems")
    ap.add_argument("--list-cpu", action="store_true", help="list CPU-only research problems")
    ap.add_argument("--list-gpu", action="store_true", help="list GPU research problems")
    ap.add_argument("--prompt", metavar="PID", help="print the generation prompt for a problem")
    ap.add_argument("--eval", metavar="PID", help="evaluate a solution file for a problem")
    ap.add_argument("--solution", metavar="PATH", help="solution.py path (with --eval)")
    args = ap.parse_args()

    if args.list:
        probs = list_research_problems()
        print(f"# {len(probs)} research problems")
        for p in probs:
            print(p)
    elif args.list_cpu:
        for p in list_research_problems():
            if not problem_needs_gpu(p):
                print(p)
    elif args.list_gpu:
        for p in list_research_problems():
            if problem_needs_gpu(p):
                print(p)
    elif args.prompt:
        for m in build_research_messages(args.prompt):
            print(f"===== {m['role']} =====")
            print(m["content"])
    elif args.eval:
        code = Path(args.solution).read_text(encoding="utf-8")
        try:
            res = evaluate_research_solution(args.eval, code)
            print(json.dumps({k: v for k, v in res.items() if k != "raw"}, indent=2))
        except ResearchInfraError as exc:
            print(f"INFRA-ERROR: {exc}", file=sys.stderr)
            sys.exit(2)
    else:
        ap.print_help()
