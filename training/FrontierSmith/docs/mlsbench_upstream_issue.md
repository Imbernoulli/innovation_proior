# Upstream issue draft — MLS-Bench

Target repo: `Imbernoulli/MLS-Bench` (public). Verified against `main` @
`ab3951ac2f92b25ccebcce05db10864065810679` (2026-08-12).

**Do not post automatically.** Command to post is at the bottom of this file.

---

## Title

```
optimization-multi-objective: the objective spec is marshalled by the host interpreter and unmarshalled inside python:3.11, so the task SIGSEGVs whenever the host is not CPython 3.11
```

---

## Body (paste below this line)

## Summary

`optimization-multi-objective` ships its black-box objective to the agent
container as a **marshalled CPython code object**. The blob is produced by the
**host** interpreter (whatever python runs `mlsbench`) and consumed by the
**container** interpreter, which the package config pins to `python:3.11-slim`.

`marshal` does not record or validate the CPython version that produced a code
object. When the two interpreters differ, `marshal.loads()` **succeeds
silently** and returns a corrupt code object; the interpreter then executes
foreign opcodes on the first call and dies with `SIGSEGV` (exit 139).

Because `pyproject.toml` declares `requires-python = ">=3.10"`, any user who
installed `mlsbench` under CPython 3.10, 3.12, or 3.13 hits this. The task
crashes before the agent's edit is ever exercised, so it records a score of
`0.0` for every model and every seed — indistinguishable from a legitimate
failure to improve.

This affects the **native** (`mlsbench agent` / `mlsbench baseline`) path only.
The Harbor path is immune; see "Why Harbor is unaffected" below.

## Root cause

**Producer — runs on the host.** `mid_edit.py` imports the held-out `dgp`
module and writes the spec into the workspace at setup time:

- [`tasks/optimization-multi-objective/edits/mid_edit.py:33`](https://github.com/Imbernoulli/MLS-Bench/blob/ab3951ac2f92b25ccebcce05db10864065810679/tasks/optimization-multi-objective/edits/mid_edit.py#L33)
  — `import dgp  # host-only ...`
- [`tasks/optimization-multi-objective/edits/mid_edit.py:69-72`](https://github.com/Imbernoulli/MLS-Bench/blob/ab3951ac2f92b25ccebcce05db10864065810679/tasks/optimization-multi-objective/edits/mid_edit.py#L69-L72)
  — writes `deap/_moea_specs/{alias}_seed{seed}.json.b64`
- The marshalling itself (held-out `dgp.py`, mirrored in the Harbor bundle):
  [`harbor/tasks/mls-bench__optimization-multi-objective/tests/meta/dgp.py:334`](https://github.com/Imbernoulli/MLS-Bench/blob/ab3951ac2f92b25ccebcce05db10864065810679/harbor/tasks/mls-bench__optimization-multi-objective/tests/meta/dgp.py#L334)
  — `return base64.b64encode(marshal.dumps(ns["_f"].__code__)).decode("ascii")`

`mid_edit.py` is executed in the harness process on the host, so the marshal is
done by the host interpreter:

- [`src/mlsbench/agent/tools.py:4267-4286`](https://github.com/Imbernoulli/MLS-Bench/blob/ab3951ac2f92b25ccebcce05db10864065810679/src/mlsbench/agent/tools.py#L4267-L4286)
  — `load_mid_edit_ops()` loads the module via `importlib` and returns `OPS`
- [`src/mlsbench/agent/base.py:206-209`](https://github.com/Imbernoulli/MLS-Bench/blob/ab3951ac2f92b25ccebcce05db10864065810679/src/mlsbench/agent/base.py#L206-L209)
  — `setup_workspace()` applies those ops

**Consumer — runs in the container**, with no version check:

- [`tasks/optimization-multi-objective/edits/custom_template.py:142`](https://github.com/Imbernoulli/MLS-Bench/blob/ab3951ac2f92b25ccebcce05db10864065810679/tasks/optimization-multi-objective/edits/custom_template.py#L142)
  — `code = marshal.loads(base64.b64decode(spec["evaluator"]))`

**The two interpreters are independently chosen:**

- [`vendor/pkg_configs/deap/config.json:2`](https://github.com/Imbernoulli/MLS-Bench/blob/ab3951ac2f92b25ccebcce05db10864065810679/vendor/pkg_configs/deap/config.json#L2)
  — `"base_image": "python:3.11-slim"`
- [`pyproject.toml:4`](https://github.com/Imbernoulli/MLS-Bench/blob/ab3951ac2f92b25ccebcce05db10864065810679/pyproject.toml#L4)
  — `requires-python = ">=3.10"`

Nothing constrains the host interpreter to match the image.

## Minimal reproduction

Standalone, no MLS-Bench checkout required. It reproduces the exact
producer/consumer pair (host 3.12 → container 3.11):

```python
# dump.py  — run with the HOST interpreter
import base64, marshal, sys, importlib.util
def _objective_kernel(individual, n_obj):
    x = list(individual)
    return [sum(v * v for v in x) + i for i in range(n_obj)]
print(sys.version.split()[0], importlib.util.MAGIC_NUMBER.hex())
open("blob.b64", "w").write(
    base64.b64encode(marshal.dumps(_objective_kernel.__code__)).decode()
)
```

```python
# load.py  — run INSIDE the deap image
import base64, marshal, sys, types, importlib.util
print("loader:", sys.version.split()[0], importlib.util.MAGIC_NUMBER.hex(), flush=True)
code = marshal.loads(base64.b64decode(open("blob.b64").read()))
print("marshal.loads OK -> no version error raised", flush=True)
f = types.FunctionType(code, {"__builtins__": __builtins__}, "objective")
print("calling...", flush=True)
print("result:", f([1.0, 2.0], 2), flush=True)
```

```
$ python dump.py                        # host CPython 3.12.7
3.12.7 cb0d0d0a

$ apptainer exec vendor/images/deap.sif python load.py
loader: 3.11.15 a70d0d0a
marshal.loads OK -> no version error raised
calling...
Segmentation fault
$ echo $?
139
```

Note the two magic numbers differ (`cb0d0d0a` vs `a70d0d0a`) and `marshal.loads`
still returns without raising. The crash lands on the **first call**, not on the
load.

The same crash occurs through the real task path; the core file confirms the
faulting process is the container interpreter running the generated spec:

```
Core was generated by `/usr/local/bin/python .../repro_moo.py
    .../deap/_moea_specs/p0_seed42.json.b64'.
Program terminated with signal SIGSEGV, Segmentation fault.
#0  0x00001527a0ce495c in ?? ()
#1  0x3c731504d03a8823 in ?? ()
```

(The unsymbolised garbage frames are the interpreter having jumped into
mis-decoded bytecode.)

An end-to-end reproduction on a checkout, with any host python other than 3.11:

```bash
python -m mlsbench baseline optimization-multi-objective \
    --name nsga2 --config configs/config.yaml --seed 42
```

## Suggested fix

Any one of these removes the failure mode; they are listed cheapest-first.

1. **Validate the version (minimum viable, turns a segfault into an error).**
   Record the producer's `importlib.util.MAGIC_NUMBER.hex()` alongside the blob
   in `gen_problem()`, and check it in `_build_objective()` before
   `marshal.loads`, raising a clear `RuntimeError` on mismatch. This does not
   make the task runnable on a mismatched host, but it converts a silent
   `0.0` into a diagnosable error.

2. **Marshal with the container's interpreter.** Have the producer shell out to
   the task image (`apptainer exec deap.sif python -c ...` /
   `docker run ... python -c ...`) to compile and marshal the kernel, so the
   blob is always produced by the interpreter that will load it.

3. **Ship source instead of bytecode (recommended).** The stated goal of the
   marshalled blob is opacity about *which* held-out benchmark problem is being
   optimised, and `_KERNEL_SRC` is already a name-free, problem-specific source
   string. Putting that string in the spec and `compile()`-ing it in the
   container is version-independent by construction and preserves the same
   opacity property, since the source carries no problem name and no `kind`.

Option 1 is worth doing regardless of which of 2/3 is chosen: an unguarded
cross-version `marshal.loads` is a segfault waiting to happen anywhere it
appears.

## Why Harbor is unaffected

In the Harbor bundle the generator runs *inside* the task container at eval
time, so producer and consumer are the same interpreter:

- [`harbor/tasks/mls-bench__optimization-multi-objective/tests/eval/_inputgen/apply.py:4`](https://github.com/Imbernoulli/MLS-Bench/blob/ab3951ac2f92b25ccebcce05db10864065810679/harbor/tasks/mls-bench__optimization-multi-objective/tests/eval/_inputgen/apply.py#L4)
  — "Runs at eval time INSIDE the task container"

Only the native `mlsbench agent` / `mlsbench baseline` path is affected. This
also means fix option 2 above is essentially "do what Harbor already does".

## Scope

`optimization-multi-objective` is the only task affected: it is the sole task
that marshals a code object across the host/container boundary.

```
$ grep -rn "marshal" --include=*.py tasks/ src/ vendor/
tasks/optimization-multi-objective/edits/custom_template.py:23:import marshal
tasks/optimization-multi-objective/edits/custom_template.py:142:    code = marshal.loads(...)
tasks/optimization-multi-objective/edits/mid_edit.py:6: ... a marshalled black-box
```

`optimization-evolution-strategy` uses the same `deap` image but does not
marshal, so it is unaffected.

## Environment

| | |
|---|---|
| MLS-Bench | `main` @ `ab3951ac2f92b25ccebcce05db10864065810679` (2026-08-12) |
| Host interpreter | CPython 3.12.7 (magic `cb0d0d0a`) |
| Container interpreter | CPython 3.11.15 (magic `a70d0d0a`), from `python:3.11-slim` |
| Container runtime | Apptainer 1.x (`vendor/images/deap.sif`); the same mismatch applies to the Docker path |
| Task | `optimization-multi-objective` (package `deap`) |
| Observed | exit 139 / `SIGSEGV`, recorded as score `0.0` for every model and seed |

---

## Notes for us — findings deliberately EXCLUDED from the issue above

Three further root causes were investigated for the same batch of
constant-`0.0` CPU tasks. None of them belongs upstream, for the reasons below.
This section is internal and must **not** be pasted into the issue.

### A. `budget_check.py` → `mid_edit.py` → `import dgp` — ALREADY FIXED UPSTREAM

Claim: `budget_check.py` runs in-container and imports `edits/mid_edit.py`,
which does a bare `import dgp`; `holdout/<task>/dgp.py` is never bind-mounted,
so the check dies with `ModuleNotFoundError` and every test is reported FAILED
(`ml-anomaly-detection`, `ml-missing-data-imputation`).

The mechanism was real on our checkout (`MLS-Bench-dev` @ `322b92236`,
2026-06-24), which has the unguarded import. It is **fixed on current
upstream**: `mid_edit.py` now wraps the import and falls back to `dgp = None`,
with a comment that names `budget_check.py` as the motivating caller.

- Fixed in `fd8a2343a` (2026-07-04, PR #54, "fix(oracle-leakage): close
  held-out-answer leakage in 26 tasks").
- Current upstream `tasks/ml-anomaly-detection/edits/mid_edit.py:25-33`.
- `_CUSTOM_PY` is read directly from `custom_template.py` at line 35, so `OPS[0]`
  (the `create` op that `budget_check.py` needs) is always present regardless of
  `dgp`.

The unguarded-import → in-container-failure combination is exactly two tasks
upstream (verified by intersecting "budget_check imports mid_edit" with
"mid_edit imports dgp"), and both are now guarded. Nothing to file.
**Confidence: high.** Our local patch is redundant with upstream; we should
rebase onto current `main` rather than keep carrying it.

### B. `aif360` missing for `ml-selective-deferral` / `ml-subgroup-calibration-shift` — LOCAL

Claim: `vendor/images/scikit-learn.sif` predates the `pkg_config` that added
`aif360==0.6.1`, and the fairness data is never staged.

- **Stale image: purely ours.** Upstream
  `vendor/pkg_configs/scikit-learn/config.json:4` and `:14` have carried
  `aif360==0.6.1` in `install_cmds`/`local_install_cmds` since the initial
  public release (`b37a7e71f`, 2026-05-11). Our `.sif` was built 2026-04-11,
  i.e. before the public repo existed. Rebuilding the image fixes it. Nothing
  upstream.
- `host_data_prepare_requirements` already lists `aif360==0.6.1`
  (`config.json:26-32`), added in `1a8cfa2da` — that was issue #12, already
  filed by someone else and closed.
- **Data staging: could not demonstrate an upstream failure.** With `aif360`
  importable and network available, `fetch_adult` and `fetch_compas` both
  succeed against current upstream code; `fetch_lawschool_gpa` failed once with
  a transient `RemoteDisconnected`. Our sidecar already holds a
  successfully-downloaded `lsac.sas7bdat`, so the URL does work.

The only arguably-upstream residue is that
`vendor/data_scripts/scikit-learn/prepare_data.py:89-90` catches the AIF360
failure, prints `WARNING`, and `main()` still returns 0 — so
`mlsbench data scikit-learn` reports success while two tasks' data is missing
(22 of 30 `prepare_data.py` scripts do hard-fail, so this one is somewhat of an
outlier). That is a real robustness gap, but we never observed it *cause* a
failure upstream — our blocker was the stale image — so filing it would mean
reporting a code smell as a bug. Left out.
**Confidence: high that the stale image is ours; medium that the WARNING
swallow is worth reporting at all.**

### C'. Host BLAS SIGSEGV in `_run_parallel_cmds` — LOCAL, AND OUR STATED ROOT CAUSE WAS WRONG

Claim (from `apply_mlsbench_cpu_fixes.sh`): `tools.py::_run_parallel_cmds`
parses test output host-side inside a `ThreadPoolExecutor` worker; that parse
calls `dgp.truth()` → `np.linalg.solve`; numpy's scipy-openblas build "crashes
when several threads enter LAPACK concurrently on this 128-core node".

The crash is real and reproduces with the actual
`causal-observational-linear-non-gaussian` `dgp.truth()` (exit 139). **But the
stated mechanism is false on all three counts**, established by controls:

| Control | Result |
|---|---|
| 3 worker threads (as the harness runs it) | SIGSEGV |
| **1** worker thread (no concurrency at all) | **SIGSEGV** |
| Main thread only, no `ThreadPoolExecutor` | **OK** |
| Restricted to 8 cores via `taskset -c 0-7` | SIGSEGV |
| `threading.stack_size(64MB)` before spawning | **OK** |
| `OPENBLAS_NUM_THREADS=1` | **OK** |
| `ulimit -s 8192` (bounded stack rlimit) | **OK** |

So it is neither concurrency nor core count. The real trigger is our shell's
`ulimit -s unlimited`: when `RLIMIT_STACK` is unlimited, glibc gives newly
spawned threads a small fixed default stack instead of the rlimit value, while
the main thread still grows dynamically. OpenBLAS's per-thread buffers overflow
that small stack, so **any** numpy LAPACK call from **any** non-main Python
thread segfaults. Calling numpy from a worker thread is entirely legitimate;
MLS-Bench is not doing anything wrong.

This is a local environment artifact (our `ulimit -s unlimited` × the
`scipy-openblas 0.3.31.188.0` build in numpy 2.4.6). Not upstream.
**Confidence: high** — `ulimit -s 8192` alone fixes it with no other change.

Follow-up for our own runner: the current fix pins
`OPENBLAS_NUM_THREADS=1`, which works but needlessly serialises host-side
parsing. `ulimit -s 8192` is the better one-line fix and keeps BLAS threading.
The fix script's claim that these variables are not in
`PASSTHROUGH_ENV_VARS` is correct and was verified (`src/mlsbench/cli.py:57-84`
passes through only proxy and API-key variables), so the pin does not leak into
containers.

---

## Pre-flight before posting

- `gh` is authenticated as **`Imbernoulli`**, which is the owner of the target
  repo. Confirm this is the account you want as the issue author.
- Existing issues searched (`budget_check`, `aif360`, `marshal`, `segfault`,
  `ModuleNotFoundError`, `dgp`, `SIGSEGV`, `score 0`): no existing report of
  the marshal/version-mismatch bug. #12 (`[bug]package scikit-learn`, closed)
  covers the `aif360` requirement, which is why finding B is excluded.
- The repo has **no** `CONTRIBUTING.md` and **no** issue templates
  (`.github/` contains only three `sync-*.yml` workflows), so there is no
  required format. The body above follows the de-facto house style of the most
  recent issues (#79, #82): `## Summary`, blob-permalink `file:line` citations
  pinned to a SHA, then consequences / options.

## Command

The postable body has already been extracted to
`docs/mlsbench_upstream_issue_body.md` (this file is **not** postable as-is — it
contains the internal notes section). Run:

```bash
gh issue create \
  --repo Imbernoulli/MLS-Bench \
  --title "optimization-multi-objective: the objective spec is marshalled by the host interpreter and unmarshalled inside python:3.11, so the task SIGSEGVs whenever the host is not CPython 3.11" \
  --body-file /scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/docs/mlsbench_upstream_issue_body.md
```

If `docs/mlsbench_upstream_issue.md` is edited, regenerate the body with:

```bash
python - <<'EOF'
import pathlib
p = pathlib.Path("/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/docs/mlsbench_upstream_issue.md")
s = p.read_text()
body = s[s.index("## Summary", s.index("## Body (paste below this line)")):s.index("## Notes for us")]
body = body.rstrip().rstrip("-").rstrip()
assert "Notes for us" not in body
pathlib.Path(str(p).replace(".md", "_body.md")).write_text(body + "\n")
EOF
```
