#!/usr/bin/env bash
# jiaolab MLS-Bench CPU eval preflight -- everything that can be wrong BEFORE a
# GPU is claimed. eval_mlsbench_local.sh calls this first (SKIP_PREFLIGHT=1 to
# bypass); run it standalone any time to check the box is still eval-ready:
#
#   bash scripts/jiaolab/mlsbench_preflight.sh            # the 22 CPU tasks
#   TASKS="ml-symbolic-regression" bash scripts/jiaolab/mlsbench_preflight.sh
#   BUILD_LOCAL=1 bash scripts/jiaolab/mlsbench_preflight.sh   # + warm the envs
#
# It checks, in order:
#   1. MLSBENCH_ROOT is a checkout AND still carries the view edit contract
#      (VIEW_SCHEMA) + --use-replace. A fresh MLS-Bench clone fails here on
#      purpose: it is NOT the FrontierSmith scoring regime.
#   2. vendor/{data,external_packages,workspace} are REAL populated dirs. On
#      gpublaze these are symlinks into the /srv MLS-Bench dev checkout; jiaolab
#      has no such checkout, so the port materialises them inside the harness.
#   3. MLSBENCH_PY imports the PATCHED mlsbench (not some other copy) + openai.
#   4. Every package the requested tasks need has its `mlsbench-<pkg>` conda env
#      and that env's python imports. container_runtime=local means the tasks run
#      in these envs; a missing one is a silent per-task failure at step ~1.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_jiaolab.sh"

MLSBENCH_ROOT="${MLSBENCH_ROOT:-$FS_ROOT/.cache/mlsbench-eval}"
MLSBENCH_PY="${MLSBENCH_PY:-/home/bohan/miniconda3/envs/mlsbench-driver/bin/python}"
[ -x "$MLSBENCH_PY" ] || MLSBENCH_PY="$(command -v python3)"
CONDA_EXE_BIN="${CONDA_EXE_BIN:-/home/bohan/miniconda3/condabin/conda}"
[ -x "$CONDA_EXE_BIN" ] || CONDA_EXE_BIN="$(command -v conda || true)"
CONDA_ENVS_DIR="${CONDA_ENVS_DIR:-/home/bohan/miniconda3/envs}"

fail=0
ok()   { echo "[preflight] OK    $*"; }
bad()  { echo "[preflight] FAIL  $*" >&2; fail=1; }

# ---- 1. harness identity + patch layers --------------------------------------
if [ ! -d "$MLSBENCH_ROOT/src/mlsbench" ]; then
  bad "MLSBENCH_ROOT=$MLSBENCH_ROOT is not an MLS-Bench checkout"
else
  n_view="$(grep -c 'VIEW_SCHEMA' "$MLSBENCH_ROOT/src/mlsbench/agent/tools.py" 2>/dev/null || echo 0)"
  if [ "${n_view:-0}" -ge 1 ]; then
    ok "view edit contract present (VIEW_SCHEMA x$n_view)"
  else
    bad "$MLSBENCH_ROOT has NO view edit contract (mlsbench_edit_contract.diff not applied)."
    bad "  A fresh clone is not a valid harness -- rsync the patched tree from gpublaze."
  fi
  if grep -rq -- "use-replace" "$MLSBENCH_ROOT/src/mlsbench" 2>/dev/null; then
    ok "--use-replace supported"
  else
    bad "$MLSBENCH_ROOT has no --use-replace support; every task would fail in <1s"
  fi
  echo "[preflight] harness $(git -C "$MLSBENCH_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')@$(git -C "$MLSBENCH_ROOT" rev-parse --short HEAD 2>/dev/null || echo '?')"
fi

# ---- 2. vendor trees ----------------------------------------------------------
for d in data external_packages workspace; do
  p="$MLSBENCH_ROOT/vendor/$d"
  if [ -L "$p" ] && [ ! -e "$p" ]; then
    bad "vendor/$d is a DANGLING symlink -> $(readlink "$p") (gpublaze layout copied verbatim?)"
  elif [ ! -d "$p" ]; then
    bad "vendor/$d missing"
  elif [ "$d" != "workspace" ] && [ -z "$(ls -A "$p" 2>/dev/null)" ]; then
    bad "vendor/$d is empty"
  else
    ok "vendor/$d ($(du -sh "$p" 2>/dev/null | cut -f1))"
  fi
done

# ---- 3. driver python ---------------------------------------------------------
if PYTHONPATH="$MLSBENCH_ROOT/src" "$MLSBENCH_PY" - "$MLSBENCH_ROOT" <<'PY'
import sys, pathlib
root = pathlib.Path(sys.argv[1]).resolve()
import mlsbench, openai, yaml  # noqa: F401
got = pathlib.Path(mlsbench.__file__).resolve()
assert str(got).startswith(str(root)), f"mlsbench resolved to {got}, NOT {root}"
print(f"mlsbench<-{got}  openai {openai.__version__}")
PY
then ok "driver python $MLSBENCH_PY resolves the patched mlsbench + openai"
else bad "driver python $MLSBENCH_PY cannot import the patched mlsbench / openai (see above)"; fi

# ---- 3b. conda-backed local runtime is really active ---------------------------
# MLS-Bench's _has_conda_support() decides between `conda run -n mlsbench-<pkg>`
# and a PIP_TARGET site-packages fallback. The fallback is silent and is NOT the
# gpublaze regime, so assert the wrapper really emits a conda run.
if PATH="$(dirname "$CONDA_EXE_BIN"):$PATH" PYTHONPATH="$MLSBENCH_ROOT/src" "$MLSBENCH_PY" - <<'PYCONDA'
from mlsbench.cli import find_conda_exe, wrap_with_conda
exe = find_conda_exe()
assert exe, "find_conda_exe() returned None -> local runtime would fall back to PIP_TARGET"
w = wrap_with_conda(["bash", "-c", "true"], {}, pkg_name="scikit-learn")
assert w[:3] == [exe, "run", "--no-capture-output"] and "mlsbench-scikit-learn" in w, w
print(f"conda={exe}  wrap={' '.join(w[:5])} ...")
PYCONDA
then ok "container_runtime=local will use per-package conda envs"
else bad "conda-backed local runtime NOT active -- MLS would silently run in a PIP_TARGET dir"; fi

# ---- 4. per-package conda envs -------------------------------------------------
TASK_LIST="${TASKS:-${SMOKE_TASK:-}}"
PKGS="$(MLSBENCH_ROOT="$MLSBENCH_ROOT" TASK_LIST="$TASK_LIST" "$MLSBENCH_PY" - <<'PY'
import json, os, pathlib, sys
sys.path.insert(0, str(pathlib.Path(os.environ["MLSBENCH_ROOT"]).parent))
root = pathlib.Path(os.environ["MLSBENCH_ROOT"])
DEFAULT = """causal-discovery-discrete causal-observational-linear-gaussian
causal-observational-linear-non-gaussian causal-observational-nonlinear
causal-treatment-effect llm-scaling-law-discovery ml-active-learning
ml-anomaly-detection ml-calibration ml-clustering-algorithm
ml-dimensionality-reduction ml-ensemble-boosting ml-missing-data-imputation
ml-selective-deferral ml-subgroup-calibration-shift ml-symbolic-regression
mlsys-moe-load-balance optimization-evolution-strategy
optimization-hyperparameter-search optimization-multi-objective
optimization-nas optimization-online-bandit""".split()
tasks = os.environ.get("TASK_LIST", "").split() or DEFAULT
pkgs = set()
for t in tasks:
    cfg = root / "tasks" / t / "config.json"
    if not cfg.exists():
        print(f"MISSINGTASK:{t}", file=sys.stderr)
        continue
    for c in json.loads(cfg.read_text()).get("test_cmds", []):
        if c.get("package"):
            pkgs.add(c["package"])
print(" ".join(sorted(pkgs)))
PY
)" || bad "could not map tasks -> packages"

if [ -n "${PKGS// /}" ]; then
  for pkg in $PKGS; do
    envdir="$CONDA_ENVS_DIR/mlsbench-$pkg"
    if [ ! -x "$envdir/bin/python" ]; then
      bad "conda env mlsbench-$pkg MISSING ($envdir) -- tasks using '$pkg' cannot run under container_runtime=local"
    elif "$envdir/bin/python" -c 'import sys; sys.exit(0)' >/dev/null 2>&1; then
      ok "conda env mlsbench-$pkg ($("$envdir/bin/python" -V 2>&1))"
    else
      bad "conda env mlsbench-$pkg python is broken (relocation not applied?)"
    fi
    if [ -n "$(ls -A "$MLSBENCH_ROOT/vendor/external_packages/$pkg" 2>/dev/null)" ]; then
      :
    else
      bad "vendor/external_packages/$pkg missing or empty"
    fi
  done
fi

# ---- 5. optional: warm the local runtime (BUILD_LOCAL=1) -----------------------
# MLS-Bench keys its "already prepared" stamp on a fingerprint that includes the
# ABSOLUTE pkg_dir and data_root, so a tree shipped from gpublaze is always
# considered unbuilt here and the FIRST task of the first run pays for
# build_local_package (re-running each package's install_cmds inside its conda
# env; mostly "already satisfied", but scikit-learn also re-fetches adbench).
# Doing it up front keeps that cost -- and any network failure -- out of the eval.
#
# NOTE the deliberate HF_HUB_OFFLINE=0 below. This is SETUP, not the eval: the
# eval itself always runs offline (eval_mlsbench_local.sh exports
# HF_HUB_OFFLINE=1, same as gpublaze). scaling-law-lab's install_cmds end in a
# prepare_data.py that unconditionally re-pulls `pkuHaowei/sldbench` from the
# Hub -- it is not idempotent, so a shipped vendor/data/scaling_law does not stop
# it -- and it would otherwise fail here and again on the first eval. The
# dataset revision is PINNED in that script, and the re-pull was verified
# byte-identical (md5) to the files shipped from gpublaze, so provenance is
# unchanged. If this box ever loses network, pre-seed the stamp instead.
if [ "$fail" = 0 ] && [ "${BUILD_LOCAL:-0}" = "1" ] && [ -n "${PKGS// /}" ]; then
  echo "[preflight] warming local runtime for: $PKGS"
  PATH="$(dirname "$CONDA_EXE_BIN"):$PATH" PYTHONPATH="$MLSBENCH_ROOT/src" \
  HF_HUB_OFFLINE="${BUILD_HF_OFFLINE:-0}" HF_DATASETS_OFFLINE="${BUILD_HF_OFFLINE:-0}" \
  MLSBENCH_ROOT="$MLSBENCH_ROOT" PKGS="$PKGS" "$MLSBENCH_PY" - <<'PYBUILD' || fail=1
import json, os, pathlib, sys
root = pathlib.Path(os.environ["MLSBENCH_ROOT"]).resolve()
import mlsbench.cli as cli
cli.PROJECT_ROOT = root
gcfg = {"container_runtime": "local", "data_root": str(root / "vendor" / "data")}
rc = 0
for pkg in os.environ["PKGS"].split():
    cfg = json.loads((root / "vendor" / "pkg_configs" / pkg / "config.json").read_text())
    pkg_dir = root / "vendor" / "external_packages" / pkg
    try:
        cli.build_local_package(pkg, cfg, pkg_dir, gcfg, force=False)
        print(f"[build] {pkg}: ready")
    except Exception as e:
        print(f"[build] {pkg}: FAILED {e}", file=sys.stderr)
        rc = 1
sys.exit(rc)
PYBUILD
  [ "$fail" = 0 ] && ok "local runtime warmed (vendor/images/local/*.json stamps written)"
fi

if [ "$fail" = 0 ]; then echo "[preflight] ALL CHECKS PASSED"; else echo "[preflight] FAILED" >&2; fi
exit "$fail"
