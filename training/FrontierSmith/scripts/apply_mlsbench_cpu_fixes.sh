#!/usr/bin/env bash
# =============================================================================
# apply_mlsbench_cpu_fixes.sh
#
# Applies every [apply now] fix for the 6 MLS-Bench CPU tasks that scored a
# constant 0.0 for all 55 banked runs, to a fresh MLS-Bench checkout.
#
#   A. ml-anomaly-detection, ml-missing-data-imputation
#      budget_check.py runs INSIDE the task container and imports
#      edits/mid_edit.py, which does `import dgp`. holdout/<task>/dgp.py is
#      deliberately never bind-mounted, so the budget check always died with
#      ModuleNotFoundError and every test was reported FAILED.
#      FIX: read the template straight from edits/custom_template.py (which is
#      byte-identical to mid_edit's create-op content) instead of importing
#      mid_edit.
#
#   B. ml-selective-deferral, ml-subgroup-calibration-shift
#      vendor/images/scikit-learn.sif (built 2026-04-11) predates the pkg_config
#      that added aif360==0.6.1, and the AIF360 fairness data was never staged
#      (prepare_data.py swallows the failure with a WARNING).
#      FIX: a read-mostly "sidecar" directory bound at /opt/fairness carrying
#      (1) aif360 installed --no-deps (so the image's pinned scikit-learn 1.5.2
#      / numpy / pandas are NOT shadowed) and (2) a WRITABLE, pre-populated
#      adult/compas/law_school cache. No SIF rebuild (impossible: unprivileged
#      user namespaces are disabled cluster-wide, so apptainer build --fakeroot
#      cannot run).
#
#   C. optimization-multi-objective
#      holdout/.../dgp.py marshals a code object with the HOST python (3.12) but
#      tasks/.../edits/custom_template.py unmarshals it inside deap.sif
#      (python 3.11). marshal does not validate the producing version: the load
#      succeeds and the first call executes 3.12 opcodes on the 3.11 eval loop
#      -> SIGSEGV (exit 139) before any agent edit runs.
#      FIX: produce the blob with the CONTAINER's interpreter (apptainer exec
#      deap.sif python), and record its magic number so a future mismatch raises
#      a clear error instead of segfaulting.
#
#   D. causal-observational-linear-non-gaussian (and every task with a
#      numpy-using host-side parser)
#      tools.py::_run_parallel_cmds runs each test_cmd in a ThreadPoolExecutor
#      worker, and each worker then parses its output HOST-side. That parse
#      calls dgp.truth() -> np.linalg.solve. numpy 2.4.6 here is linked against
#      scipy-openblas 0.3.31, which crashes when several threads enter LAPACK
#      concurrently on this 128-core node -> the `mlsbench agent` process itself
#      dies with SIGSEGV.
#      FIX: pin the HOST BLAS/OpenMP thread pools to 1. These variables are NOT
#      in mlsbench's PASSTHROUGH_ENV_VARS, so they never reach the containers --
#      in-container compute keeps full parallelism.
#
# Usage:
#   ./apply_mlsbench_cpu_fixes.sh [--root <MLSBENCH_ROOT>]
#                                 [--sidecar <DIR>]
#                                 [--runner <cc_eval_mlsbench_cpu_ailab.sh>]
#                                 [--skip-sidecar-build] [--dry-run]
#
# NOTE: --sidecar must be on SHARED storage (compute nodes cannot see /tmp of
# the login node), and building it needs INTERNET, so run this on a login node.
# Everything is idempotent; re-running is safe.
# =============================================================================
set -euo pipefail

ROOT="${MLSBENCH_ROOT:-/scratch/gpfs/CHIJ/bohan/MLS-Bench-dev}"
SIDECAR="${MLSBENCH_FAIRNESS_SIDECAR:-/scratch/gpfs/CHIJ/bohan/mlsbench_fairness}"
RUNNER="${MLSBENCH_RUNNER:-/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/slurm/cc_eval_mlsbench_cpu_ailab.sh}"
SKIP_SIDECAR=0
DRY_RUN=0

while [ $# -gt 0 ]; do
  case "$1" in
    --root)    ROOT="$2"; shift 2 ;;
    --sidecar) SIDECAR="$2"; shift 2 ;;
    --runner)  RUNNER="$2"; shift 2 ;;
    --skip-sidecar-build) SKIP_SIDECAR=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '2,60p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

PY="${MLSBENCH_PY:-/home/bl3615/miniconda3/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

say() { echo "[fix] $*"; }
[ -d "$ROOT/src/mlsbench" ] || { echo "ERROR: $ROOT is not an MLS-Bench checkout" >&2; exit 1; }
say "root    = $ROOT"
say "sidecar = $SIDECAR"
say "runner  = $RUNNER"
[ "$DRY_RUN" = 1 ] && say "DRY RUN - nothing will be written"

# -----------------------------------------------------------------------------
# Fix A - budget_check.py must not import mid_edit.py (which imports holdout dgp)
# -----------------------------------------------------------------------------
say "=== A: budget_check.py -> read edits/custom_template.py directly ==="
ROOT="$ROOT" DRY_RUN="$DRY_RUN" "$PY" - <<'PYEOF'
import os, pathlib, sys
root = pathlib.Path(os.environ["ROOT"]); dry = os.environ["DRY_RUN"] == "1"
OLD = '''mid_edit = load_module(os.path.join(TASK_DIR, "edits", "mid_edit.py"), "_mid_edit")
config = json.loads(open(os.path.join(TASK_DIR, "config.json")).read())
editable_file = None
for f in config.get("files", []):
    if f.get("edit"):
        editable_file = f["filename"]
        break

template_content = None
for op in mid_edit.OPS:
    if op.get("op") == "create" and op.get("file") == editable_file:
        template_content = op["content"]
        break
'''
NEW = '''# NOTE: do NOT import edits/mid_edit.py here. This file runs INSIDE the task
# container, and mid_edit.py does `import dgp`, whose module lives in
# holdout/<task>/dgp.py -- deliberately never bind-mounted into the container.
# Importing it therefore always raises ModuleNotFoundError in-container and the
# budget check fails before it ever counts a parameter. The only thing we need
# from mid_edit is the `create` op content for the editable file, which is
# byte-identical to edits/custom_template.py, so read that directly.
config = json.loads(open(os.path.join(TASK_DIR, "config.json")).read())
editable_file = None
for f in config.get("files", []):
    if f.get("edit"):
        editable_file = f["filename"]
        break

template_content = None
_template_path = os.path.join(TASK_DIR, "edits", "custom_template.py")
if os.path.exists(_template_path):
    with open(_template_path) as _tf:
        template_content = _tf.read()
else:
    # Fallback (host-side only, where holdout/ is importable).
    mid_edit = load_module(os.path.join(TASK_DIR, "edits", "mid_edit.py"), "_mid_edit")
    for op in mid_edit.OPS:
        if op.get("op") == "create" and op.get("file") == editable_file:
            template_content = op["content"]
            break
'''
for t in ("ml-anomaly-detection", "ml-missing-data-imputation"):
    p = root / "tasks" / t / "budget_check.py"
    s = p.read_text()
    if "do NOT import edits/mid_edit.py here" in s:
        print(f"  [skip] {t}: already patched"); continue
    if OLD not in s:
        print(f"  [WARN] {t}: anchor not found, NOT patched", file=sys.stderr); continue
    if not dry:
        p.write_text(s.replace(OLD, NEW, 1))
    print(f"  [ok]   {t}: budget_check.py patched")
PYEOF

# -----------------------------------------------------------------------------
# Fix B - aif360 sidecar (package + writable fairness data cache)
# -----------------------------------------------------------------------------
say "=== B: aif360 sidecar + config wiring ==="
SIF="$ROOT/vendor/images/scikit-learn.sif"
if [ "$SKIP_SIDECAR" = 0 ] && [ "$DRY_RUN" = 0 ]; then
  if [ ! -f "$SIF" ]; then
    echo "  [WARN] $SIF missing - skipping sidecar build" >&2
  elif [ -f "$SIDECAR/data/lsac.sas7bdat" ] && [ -d "$SIDECAR/pysite/aif360" ]; then
    say "  [skip] sidecar already built at $SIDECAR"
  else
    mkdir -p "$SIDECAR/pysite" "$SIDECAR/data"
    say "  installing aif360==0.6.1 (--no-deps, so the image's pinned sklearn/numpy/pandas win)"
    apptainer exec --bind "$SIDECAR:$SIDECAR" "$SIF" \
      pip install --no-cache-dir --no-deps --target="$SIDECAR/pysite" "aif360==0.6.1" >/dev/null
    # Seed the OpenML cache from the prepared data_root (adult == data_id 1590).
    DR="$(grep -oP '(?<=^data_root: ).*' "$ROOT/configs/config.yaml" 2>/dev/null || true)"
    DR="${DR:-/scratch/gpfs/CHIJ/st3812/projects/MLS-Bench/vendor/data}"
    [ -d "$DR/sklearn/openml" ] && rsync -a "$DR/sklearn/openml" "$SIDECAR/data/" && chmod -R u+w "$SIDECAR/data"
    say "  pre-fetching adult / compas / law_school into $SIDECAR/data (needs internet)"
    apptainer exec --bind "$SIDECAR:/opt/fairness" --env PYTHONPATH=/opt/fairness/pysite "$SIF" \
      python -c "
import warnings; warnings.filterwarnings('ignore')
from aif360.sklearn.datasets import fetch_adult, fetch_compas, fetch_lawschool_gpa
for nm, fn in (('adult',fetch_adult),('compas',fetch_compas),('law_school',fetch_lawschool_gpa)):
    fn(data_home='/opt/fairness/data', cache=True, binary_race=True, dropna=True)
    print('    cached', nm)
"
  fi
fi

ROOT="$ROOT" SIDECAR="$SIDECAR" DRY_RUN="$DRY_RUN" "$PY" - <<'PYEOF'
import json, os, pathlib
root = pathlib.Path(os.environ["ROOT"]); side = os.environ["SIDECAR"]
dry = os.environ["DRY_RUN"] == "1"

p = root / "vendor/pkg_configs/scikit-learn/config.json"
c = json.loads(p.read_text())
flags = c.setdefault("apptainer_flags", [])
bind = f"{side}:/opt/fairness"
# drop any stale sidecar bind from an earlier run, then re-add
cleaned, i = [], 0
while i < len(flags):
    if flags[i] == "--bind" and i + 1 < len(flags) and flags[i + 1].endswith(":/opt/fairness"):
        i += 2; continue
    cleaned.append(flags[i]); i += 1
cleaned += ["--bind", bind]
c["apptainer_flags"] = cleaned
c.setdefault("env", {})["PYTHONPATH"] = "/opt/fairness/pysite"
if not dry:
    p.write_text(json.dumps(c, indent=2) + "\n")
print(f"  [ok]   pkg_configs/scikit-learn: bind {bind}, PYTHONPATH=/opt/fairness/pysite")

for t in ("ml-selective-deferral", "ml-subgroup-calibration-shift"):
    tp = root / "tasks" / t / "config.json"
    tc = json.loads(tp.read_text())
    for e in tc["test_cmds"]:
        e.setdefault("env", {})["SKLEARN_DATA_HOME"] = "/opt/fairness/data"
    if not dry:
        tp.write_text(json.dumps(tc, indent=2) + "\n")
    print(f"  [ok]   {t}: SKLEARN_DATA_HOME=/opt/fairness/data on {len(tc['test_cmds'])} test_cmds")
PYEOF

# -----------------------------------------------------------------------------
# Fix C - marshal the MOEA objective with the CONTAINER's python
# -----------------------------------------------------------------------------
say "=== C: optimization-multi-objective marshal version mismatch ==="
ROOT="$ROOT" DRY_RUN="$DRY_RUN" "$PY" - <<'PYEOF'
import os, pathlib, sys
root = pathlib.Path(os.environ["ROOT"]); dry = os.environ["DRY_RUN"] == "1"

# --- C1: holdout dgp.py -------------------------------------------------------
p = root / "holdout/optimization-multi-objective/dgp.py"
s = p.read_text()
if "_marshal_in_container" in s:
    print("  [skip] dgp.py: already patched")
else:
    OLD_IMPORTS = "import base64\nimport marshal\n\nimport numpy as np\n"
    NEW_IMPORTS = ("import base64\nimport importlib.util\nimport inspect\nimport marshal\n"
                   "import os\nimport shutil\nimport subprocess\nimport sys\nimport textwrap\n"
                   "from pathlib import Path\n\nimport numpy as np\n")
    OLD_BLOB = '''def _evaluator_blob() -> str:
    """Base64 of the marshalled black-box objective code object."""
    return base64.b64encode(marshal.dumps(_objective_kernel.__code__)).decode("ascii")
'''
    NEW_BLOB = '''# marshal is CPython-VERSION-SPECIFIC and completely unvalidated: a code object
# dumped by 3.12 and loaded by 3.11 does NOT raise -- marshal.loads happily
# rebuilds it, and the older eval loop then executes 3.12 opcodes as garbage and
# SIGSEGVs on the first call. This spec is generated on the HOST (whatever
# interpreter runs `mlsbench`, currently CPython 3.12) but consumed INSIDE the
# task container (deap.sif == python:3.11-slim, CPython 3.11), so the blob must
# be produced by the CONTAINER's interpreter. We therefore ship the kernel
# SOURCE into the container image's python and marshal it there. Source never
# reaches the workspace, so the agent-visible opacity is unchanged.

_CONTAINER_IMAGE = Path(__file__).resolve().parents[2] / "vendor" / "images" / "deap.sif"

_MARSHAL_BOOTSTRAP = (
    "import base64, importlib.util, marshal, sys\\n"
    "src = sys.stdin.read()\\n"
    "ns = {}\\n"
    "exec(compile(src, '<objective_kernel>', 'exec'), ns)\\n"
    "blob = base64.b64encode(marshal.dumps(ns['_objective_kernel'].__code__)).decode('ascii')\\n"
    "sys.stdout.write(importlib.util.MAGIC_NUMBER.hex() + ' ' + blob)\\n"
)


def _kernel_source() -> str:
    return textwrap.dedent(inspect.getsource(_objective_kernel))


def _marshal_in_container() -> tuple[str, str] | None:
    """Return (py_magic_hex, b64_blob) marshalled by the task container's python.

    Returns None when the container cannot be used (no apptainer, no image, or
    the exec failed), so the caller can fall back to a local marshal.
    """
    image = os.environ.get("MLSBENCH_MOEA_IMAGE", str(_CONTAINER_IMAGE))
    runtime = shutil.which("apptainer") or shutil.which("singularity")
    if not runtime or not os.path.exists(image):
        return None
    try:
        out = subprocess.run(
            [runtime, "exec", image, "python", "-c", _MARSHAL_BOOTSTRAP],
            input=_kernel_source(), capture_output=True, text=True, timeout=300,
        )
    except Exception:
        return None
    if out.returncode != 0 or " " not in out.stdout:
        return None
    magic, _, blob = out.stdout.strip().partition(" ")
    if not blob:
        return None
    return magic, blob


def _evaluator_blob() -> tuple[str, str]:
    """Return (py_magic_hex, base64 of the marshalled black-box objective)."""
    got = _marshal_in_container()
    if got is not None:
        return got
    # Fallback: marshal locally. The py_magic we record is OUR magic, so a
    # container running a different CPython raises a clear error in
    # custom_template._load_objective_code instead of segfaulting.
    return (
        importlib.util.MAGIC_NUMBER.hex(),
        base64.b64encode(marshal.dumps(_objective_kernel.__code__)).decode("ascii"),
    )
'''
    OLD_CFG = '    cfg = PROBLEMS[problem]\n    return {\n        "alias": ALIASES[problem],'
    NEW_CFG = ('    cfg = PROBLEMS[problem]\n    _py_magic, _blob = _evaluator_blob()\n'
               '    return {\n        "alias": ALIASES[problem],')
    OLD_SPEC = '        "kind": _KIND_ORDER.index(problem),\n        "evaluator": _evaluator_blob(),\n    }'
    NEW_SPEC = '        "kind": _KIND_ORDER.index(problem),\n        "py_magic": _py_magic,\n        "evaluator": _blob,\n    }'
    for old in (OLD_IMPORTS, OLD_BLOB, OLD_CFG, OLD_SPEC):
        if old not in s:
            print("  [WARN] dgp.py: anchor missing, NOT patched", file=sys.stderr); sys.exit(0)
    for old, new in ((OLD_IMPORTS, NEW_IMPORTS), (OLD_BLOB, NEW_BLOB), (OLD_CFG, NEW_CFG), (OLD_SPEC, NEW_SPEC)):
        s = s.replace(old, new, 1)
    if not dry:
        p.write_text(s)
    print("  [ok]   dgp.py: evaluator marshalled by deap.sif's python + py_magic recorded")

# --- C2: agent-facing template ------------------------------------------------
# CRITICAL: config.json and every edits/*.edit.py replace custom_template.py
# lines 159-303 by ABSOLUTE index, so this patch must not move any line <= 303.
# The one changed line is a 1:1 swap; the helper is appended after the editable
# region, just above the __main__ guard.
p = root / "tasks/optimization-multi-objective/edits/custom_template.py"
s = p.read_text()
if "_load_objective_code" in s:
    print("  [skip] custom_template.py: already patched")
else:
    before = s.splitlines()
    OLD = '    code = marshal.loads(base64.b64decode(spec["evaluator"]))\n'
    NEW = '    code = _load_objective_code(spec)\n'
    GUARD = 'if __name__ == "__main__":\n    main()\n'
    HELPER = '''def _load_objective_code(spec):
    """Unmarshal the black-box objective code object, version-checked.

    marshal does NOT validate the CPython version that produced a blob: a code
    object dumped by a different interpreter loads without error and then
    SIGSEGVs the process on the first call. Fail loudly instead.
    """
    import importlib.util

    want = spec.get("py_magic")
    have = importlib.util.MAGIC_NUMBER.hex()
    if want and want != have:
        raise RuntimeError(
            "objective spec was marshalled by a different CPython bytecode "
            f"version (spec magic {want}, this interpreter {have}); calling it "
            "would segfault. Regenerate the spec with this container's python."
        )
    return marshal.loads(base64.b64decode(spec["evaluator"]))


'''
    if s.count(OLD) != 1 or s.count(GUARD) != 1:
        print("  [WARN] custom_template.py: anchors missing, NOT patched", file=sys.stderr); sys.exit(0)
    s = s.replace(OLD, NEW, 1).replace(GUARD, HELPER + GUARD, 1)
    after = s.splitlines()
    # The editable region (lines 159-303, 1-based) must be byte-identical and at
    # the same line numbers, or every baseline's `replace 159-303` op corrupts
    # the fixed code above/below it.
    assert before[158:303] == after[158:303], \
        "line drift in the 159-303 editable region -- would break every edit op"
    if not dry:
        p.write_text(s)
    print(f"  [ok]   custom_template.py: guarded loader ({len(before)} -> {len(after)} lines, 1-303 unchanged)")
PYEOF

# -----------------------------------------------------------------------------
# Fix D - pin host BLAS/OpenMP threads (never reaches the containers)
# -----------------------------------------------------------------------------
say "=== D: host BLAS thread pinning ==="
ENVSNIP='export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1'
if [ -f "$RUNNER" ]; then
  if grep -q "OPENBLAS_NUM_THREADS" "$RUNNER"; then
    say "  [skip] runner already pins OPENBLAS_NUM_THREADS"
  elif [ "$DRY_RUN" = 1 ]; then
    say "  [dry]  would insert into $RUNNER: $ENVSNIP"
  else
    RUNNER="$RUNNER" "$PY" - <<'PYEOF'
import os, pathlib
p = pathlib.Path(os.environ["RUNNER"]); s = p.read_text()
anchor = 'export TOKENIZERS_PARALLELISM=false\n'
block = (anchor +
 "\n# MLS-Bench parses each test_cmd's output HOST-side inside a ThreadPoolExecutor\n"
 "# worker (tools.py::_run_parallel_cmds). Several of those parsers call into\n"
 "# numpy/LAPACK (e.g. causal-observational-*'s dgp.truth -> np.linalg.solve).\n"
 "# numpy's scipy-openblas build SIGSEGVs when multiple threads enter LAPACK\n"
 "# concurrently on this 128-core node, killing the whole `mlsbench agent`\n"
 "# process (returncode -11). Pin the HOST pools to 1; these vars are NOT in\n"
 "# mlsbench's PASSTHROUGH_ENV_VARS, so in-container compute keeps full threads.\n"
 "export OPENBLAS_NUM_THREADS=1\n"
 "export OMP_NUM_THREADS=1\n"
 "export MKL_NUM_THREADS=1\n"
 "export NUMEXPR_NUM_THREADS=1\n")
assert anchor in s, "anchor 'export TOKENIZERS_PARALLELISM=false' not found in runner"
p.write_text(s.replace(anchor, block, 1))
print("  [ok]   runner patched with host BLAS thread pinning")
PYEOF
  fi
else
  say "  [WARN] runner $RUNNER not found - export these yourself before 'mlsbench agent':"
  echo "         $ENVSNIP"
fi

say "DONE."
say "Sanity check (one baseline per fixed group, ~2-6 min each):"
cat <<EOS
  cd $ROOT
  export MLSBENCH_NO_PREBUILT=1 MLSBENCH_SCHEDULER_MANAGED=1 PYTHONPATH=$ROOT/src
  $ENVSNIP
  $PY -m mlsbench baseline ml-anomaly-detection            --name isolation_forest        --config configs/config.yaml --seed 42
  $PY -m mlsbench baseline ml-missing-data-imputation      --name mean_impute             --config configs/config.yaml --seed 42
  $PY -m mlsbench baseline ml-selective-deferral           --name confidence_thresholding --config configs/config.yaml --seed 42
  $PY -m mlsbench baseline ml-subgroup-calibration-shift   --name temperature_scaling     --config configs/config.yaml --seed 42
  $PY -m mlsbench baseline optimization-multi-objective    --name nsga2                   --config configs/config.yaml --seed 42
  $PY -m mlsbench baseline causal-observational-linear-non-gaussian --name icalingam       --config configs/config.yaml --seed 42
EOS
