# Copyright 2025 - frontiersmith_synth reward for VERL RL training
"""frontiersmith_synth reward: run a candidate solution through the colleague's
frontiersmith_synth harness and return a 0-100 score.

MIRRORS the FrontierCS reward (frontiercs.py) in *interface* -- same
compute_score(data_source, solution_str, ground_truth, extra_info, ...) -> float
in [0, 100], same fail-soft-on-infra contract -- but scores OFFLINE with NO judge
server: it reuses the frontiersmith_synth harness (harness/validate_problem.py for
gen+checker problems, harness/validate_pyproblem.py for evaluator/program-mode
problems, harness/isorun.py for the bwrap sandbox) to run the candidate against the
problem's generated test cases and average the per-case Ratio into [0, 1], then
scales to [0, 100].

Two scoring shapes (see the harness docstrings):
  * gen_checker  (formats A/C/D): candidate is a stdin->stdout PROGRAM (C++ or Py).
                 The harness compiles gen/checker (testlib for .cpp/.cc), generates
                 n_cases inputs, runs the candidate OS-sandboxed on each, and the
                 checker prints `Ratio: <float in [0,1]>`. Score = mean ratio.
  * evaluator    (formats B/E): candidate is a Python program (stdin JSON -> stdout
                 JSON). Score = `python3 evaluator.py <candidate.py>` -> `Ratio:`.

Determinism/infra:
  * A returned 0.0 is a legitimate model-side 0 (no code / non-compiling / infeasible
    / timeout). Genuine *infrastructure* failures (missing problem dir, missing g++,
    harness crash, bwrap unavailable when required) raise SynthInfraError; compute_score
    wraps it fail-soft (default ON) so one bad problem never kills the whole RL run,
    exactly like frontiercs.py.
  * No network. The candidate runs under bwrap (via the harness) which hides the synth
    tree and the parent /proc, so an adversarial rollout cannot read the checker source.
"""

import importlib.util
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---- locate the frontiersmith_synth tree (colleague's, OUTSIDE FrontierSmith) ----
# Override with FRONTIERSMITH_SYNTH_ROOT. Default mirrors the repo layout:
#   <fs>/FrontierSmith/verl/verl/utils/reward_score/frontiersmith_synth.py
#   <fs>/innovation_prior/frontiersmith_synth/
_THIS = Path(__file__).resolve()
_DEFAULT_SYNTH_ROOT = _THIS.parents[5] / "innovation_prior" / "frontiersmith_synth"


def _synth_root() -> Path:
    env = os.environ.get("FRONTIERSMITH_SYNTH_ROOT")
    return Path(env).resolve() if env else _DEFAULT_SYNTH_ROOT


# Per-candidate wall-time budget for the WHOLE harness score (compile + generate +
# run all cases + check). Generous because gen/compile can be slow; overridable.
MAX_WALL = float(os.environ.get("FRONTIERSMITH_SYNTH_MAX_WALL", "300"))
# Fail-soft on infra errors (default ON) -- same knob shape as frontiercs.
FAIL_SOFT = os.environ.get("FRONTIERSMITH_SYNTH_FAIL_SOFT", "1") not in ("0", "false", "False", "")

# ---- host-RAM guard (2026-08-10) ------------------------------------------------
# Same failure mode frontiercs_research hit on 07-27/07-30: verl's agent-loop path
# streams every finished rollout straight into compute_score with no application
# bound -- 8 RewardLoopWorkers x ~32 executor threads = up to ~256 concurrent
# scoring forests (apptainer + compile + generated cases), which OOM-killed the
# rlv3 training nodes at 450G AND 650G ~1h into step-1 (first completions wave).
# Cap concurrent scorings PER WORKER PROCESS: 8 workers x default 1 = <=8 in
# flight node-wide. Override: FRONTIERSMITH_SYNTH_MAX_CONC.
import threading as _threading

_SCORE_SEM = _threading.Semaphore(int(os.environ.get("FRONTIERSMITH_SYNTH_MAX_CONC", "1")))

RATIO_RE = re.compile(r"Ratio:\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")

# Cache the dynamically-loaded harness modules (keyed by file path).
_HARNESS_MODS: dict[str, Any] = {}


class SynthInfraError(RuntimeError):
    """Raised on infrastructure failure (missing problem/harness, no compiler,
    harness crash). LOUD on purpose: an infra fault must not be silently mapped to
    reward 0.0 (indistinguishable from genuine model weakness). compute_score wraps
    it fail-soft by default."""


def _load_harness_module(name: str):
    """Dynamically import a harness module (validate_problem / validate_pyproblem /
    isorun) from the synth tree. Cached. Reuses the harness's OWN scoring helpers so
    we never re-implement a checker."""
    if name in _HARNESS_MODS:
        return _HARNESS_MODS[name]
    path = _synth_root() / "harness" / f"{name}.py"
    if not path.exists():
        raise SynthInfraError(f"frontiersmith_synth harness module missing: {path}")
    spec = importlib.util.spec_from_file_location(f"_fsx_{name}", str(path))
    mod = importlib.util.module_from_spec(spec)
    # isorun imports nothing special; validate_pyproblem needs harness dir importable.
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    _HARNESS_MODS[name] = mod
    return mod


# ------------------------------------------------------------------ code extraction
def strip_think(response: str) -> str:
    if not response:
        return response
    _, sep, suffix = response.rpartition("</think>")
    return suffix if sep else response


def extract_code(response_text: str, lang: str) -> str:
    """Extract candidate source from a model response. `lang` in {cpp, py}.
    Mirrors frontiercs.extract_cpp but language-aware (fenced block preferred)."""
    if not response_text:
        return ""
    text = strip_think(response_text)
    if not text:
        return ""
    text = text.strip()
    if lang == "cpp":
        fences = r'```(?:cpp|c\+\+|C\+\+)?\s*\n(.*?)```'
    else:
        fences = r'```(?:python|py|python3)?\s*\n(.*?)```'
    matches = re.findall(fences, text, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()
    # Fallback: strip a leading/trailing bare fence.
    for pre in ("```cpp", "```c++", "```python", "```py", "```"):
        if text.startswith(pre):
            text = text[len(pre):].strip()
            break
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


# ------------------------------------------------------------------ shape 1: gen+checker
def _run_checker(vp, chk_cmd, chk_is_py, inf: Path, outf: Path, work: Path) -> float:
    """Run the problem's checker on one (in,out) pair -> ratio in [0,1]. This is the
    same logic as validate_problem.main's inner `score_checker` closure (which is not
    importable at module level): it invokes the harness-resolved checker command and
    parses `Ratio:` via the harness's own vp.parse_ratio. Reuses the compiled/checker
    process and parser -- no re-implementation of the checker itself."""
    ans = work / (inf.stem + ".ans")
    crc, cout, cerr, cto = vp.run_capture(
        chk_cmd(str(inf), str(outf), str(ans)), timeout=120, cwd=work)
    if cto:
        return 0.0
    if crc is not None and crc < 0:          # checker killed by a signal
        return 0.0
    if chk_is_py and crc not in (0, None):   # py checker must exit 0 to be trusted
        return 0.0
    blob = (cerr or "") + "\n" + (cout or "")
    r, ok2 = vp.parse_ratio(blob)
    return r if ok2 else 0.0


def _score_gen_checker(pdir: Path, code: str, lang: str) -> float:
    """Reuse validate_problem.py's helpers to score ONE candidate program over the
    problem's n_cases. Returns mean ratio in [0,1]. Raises SynthInfraError on infra."""
    vp = _load_harness_module("validate_problem")
    cfg = vp.parse_config(pdir)
    n = cfg["n_cases"]

    work = Path(tempfile.mkdtemp(prefix="fsx_gc_"))
    try:
        # Resolve + compile generator and checker using the harness's own routines.
        gen_cmd, g_ok, g_err = vp.resolve_generator(pdir, work)
        chk_cmd, c_ok, c_err, chk_is_py = vp.resolve_checker(pdir, work, cfg["checker"])
        if not g_ok:
            raise SynthInfraError(f"generator failed to build for {pdir.name}: {g_err[:400]}")
        if not c_ok:
            raise SynthInfraError(f"checker failed to build for {pdir.name}: {c_err[:400]}")

        # Write the candidate and build its run command (compile if C++).
        if lang == "cpp":
            src = work / "cand.cpp"
            src.write_text(code)
            binp = work / "cand.bin"
            ok, err = vp.compile_cpp(src, binp, with_testlib=False)
            if not ok:
                return 0.0  # non-compiling model code == legitimate model-side 0
            cand_cmd = [str(binp)]
        else:
            src = work / "cand.py"
            src.write_text(code)
            # cheap syntax gate: a non-parsing program is a model-side 0
            rc, _, _, _ = vp.run_capture(["python3", "-c",
                                          f"import ast;ast.parse(open('{src}').read())"], 30)
            if rc != 0:
                return 0.0
            cand_cmd = ["python3", str(src)]

        # Generate cases + run + check, exactly like validate_problem.main's inner loop.
        total = 0.0
        for i in range(1, n + 1):
            inf = work / f"{i}.in"
            with open(inf, "wb") as fo:
                rc, err, to = vp.run(gen_cmd(i), timeout=40.0, stdout=fo, cwd=work)
            if to or rc != 0 or inf.stat().st_size == 0:
                raise SynthInfraError(f"gen case {i} failed for {pdir.name} (rc={rc}, to={to})")
            (work / f"{i}.ans").write_text("")  # scorer problems ignore ans
            outf = work / f"{i}.cand.out"
            # candidate runs OS-sandboxed (bwrap) -- can't read the co-located source.
            r, to2 = vp.sandbox_run_solution(cand_cmd, inf, outf,
                                             cfg["time_s"] + 2.0, work)
            if to2 or r != 0:
                total += 0.0  # TLE / runtime error on this case -> 0
                continue
            total += _run_checker(vp, chk_cmd, chk_is_py, inf, outf, work)
        return total / n if n else 0.0
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ------------------------------------------------------------------ shape 2: evaluator
def _score_evaluator(pdir: Path, code: str) -> float:
    """Reuse validate_pyproblem.py's score_solution to score ONE candidate .py via the
    problem's evaluator.py. Returns ratio in [0,1]. Raises SynthInfraError on infra."""
    vpp = _load_harness_module("validate_pyproblem")
    cfg = vpp.parse_config(pdir)
    evaluator = pdir / cfg["checker"]
    if not evaluator.exists():
        raise SynthInfraError(f"evaluator missing for {pdir.name}: {evaluator}")

    work = Path(tempfile.mkdtemp(prefix="fsx_ev_"))
    try:
        src = work / "cand.py"
        src.write_text(code)
        # syntax gate -> model-side 0 (not infra)
        rc, out, err, to = vpp.run_capture(["python3", "-c",
                                            f"import ast;ast.parse(open('{src}').read())"], 30)
        if rc != 0:
            return 0.0
        ratio, vec, status = vpp.score_solution(evaluator, src, cfg["time_s"] + 10.0)
        # score_solution already maps TLE/crash/no-ratio/oob -> 0.0 (model-side).
        return float(ratio)
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ------------------------------------------------------------------ dispatch
def _classify(pdir: Path) -> tuple[str, str]:
    """Return (scoring_shape, lang) for a problem dir. Prefer extra_info when given
    by the caller; this is the fallback that reads config.yaml, matching the parquet
    builder's classify()."""
    txt = (pdir / "config.yaml").read_text()
    m = re.search(r"checker:\s*(\S+)", txt)
    checker = m.group(1) if m else ""
    has_gen = (pdir / "gen.cpp").exists() or (pdir / "gen.py").exists()
    if checker == "evaluator.py" and (pdir / "evaluator.py").exists() and not has_gen:
        return "evaluator", "py"
    sol_dir = pdir / "solutions"
    sols = (list(sol_dir.glob("*.cpp")) + list(sol_dir.glob("*.py"))) if sol_dir.exists() else []
    n_cpp = sum(1 for s in sols if s.suffix == ".cpp")
    n_py = sum(1 for s in sols if s.suffix == ".py")
    lang = "cpp" if n_cpp >= n_py else "py"
    return "gen_checker", lang


def _compute_score_inner(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> float:
    if data_source != "frontiersmith_synth":
        raise ValueError(f"data_source must be 'frontiersmith_synth', got {data_source}")

    pid = str(ground_truth)
    pdir = _synth_root() / "problems" / pid
    if not pdir.is_dir():
        raise SynthInfraError(f"frontiersmith_synth problem dir missing: {pdir}")

    ei = extra_info or {}
    shape = ei.get("scoring_shape")
    lang = ei.get("lang")
    if shape not in ("gen_checker", "evaluator") or lang not in ("cpp", "py"):
        shape, lang = _classify(pdir)

    code = extract_code(solution_str, lang)
    if not code:
        return 0.0  # no code produced -> real model-side 0

    with _SCORE_SEM:
        if shape == "evaluator":
            ratio = _score_evaluator(pdir, code)
        else:
            ratio = _score_gen_checker(pdir, code, lang)

    # harness returns [0,1]; scale to [0,100] to match the frontiercs reward range.
    ratio = max(0.0, min(1.0, float(ratio)))
    return ratio * 100.0


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> float:
    """Score a frontiersmith_synth rollout in [0, 100]. Fail-soft on infra by default
    (an infra fault on ONE problem is logged + scored 0.0 rather than killing the run),
    mirroring frontiercs.compute_score. Set FRONTIERSMITH_SYNTH_FAIL_SOFT=0 to raise."""
    if not FAIL_SOFT:
        return _compute_score_inner(data_source, solution_str, ground_truth, extra_info, **kwargs)
    try:
        return _compute_score_inner(data_source, solution_str, ground_truth, extra_info, **kwargs)
    except SynthInfraError as exc:
        logger.warning(
            "frontiersmith_synth infra error on problem %s -> scoring 0.0 and continuing "
            "(FAIL_SOFT). %s", str(ground_truth), exc,
        )
        return 0.0
