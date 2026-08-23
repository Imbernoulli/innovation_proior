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
