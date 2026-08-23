#!/usr/bin/env python3
"""
mlsbench_contract_eval.py — settle which file-editing contract the MLS-Bench
agent should use, MEASURED ON TASK SCORE.

Why this exists
---------------
The earlier A/B (outputs/cc_mls_editab/) compared four edit-tool arms on
EDIT-LEVEL metrics only (first-edit acceptance 19.3/30.1/33.7/43.4%). Over 1,281
banked runs, edit rejection is ~uncorrelated with task score (r = -0.029), so an
edit-metric win does not imply a score win. This runner measures the thing the
decision actually depends on: the 20-task MLS-Bench mean, with enough replicates
that the per-arm standard error is smaller than the effect we hope to detect.

Design decisions that are NOT negotiable (each one fixes a real defect):

  * FIXED DENOMINATOR 20. `summary.json`'s own `mean_score` divides by
    `n_scored`, which inflates any run that lost tasks to failures. We always
    divide by the number of tasks in the panel.

  * ONE LEADERBOARD IDENTITY PER (arm, rep). `mlsbench score` groups
    tasks/<task>/leaderboard.csv rows BY MODEL STRING and returns a single row
    per model across the whole file's history (scoring/evaluate.py
    evaluate_task: pick max by (completeness, timestamp) within a tier). Running
    the same task twice under one `--model` string therefore re-reads the older
    replicate's row instead of the new one — replicates would silently collide.
    Every cell gets its own model string, `vllm/<TAG>__<arm>__r<NN>`, so the
    grouping key isolates it. vLLM must register all of them via
    --served-model-name (see the `served-names` subcommand).

  * NO SILENT INFRA ZEROS. A task that produced no leaderboard row is not
    automatically a 0. It is classified: agent-completed-without-a-submission is
    a genuine 0; a timeout / API error / OOM / SIGSEGV is infrastructure and is
    RETRIED. Unresolved infra failures leave the cell INCOMPLETE and the cell is
    excluded from the headline, never quoted. (Both the strict number and an
    infra->0 sensitivity number are printed, so nothing is hidden.)

  * RESUMABLE, AND NEVER QUOTES A PARTIAL RUN. Per-task records are written
    atomically as they finish; `cell.json` appears only when all 20 tasks are
    terminal. `report` reads cell.json files only.

  * DISK. /scratch/gpfs/CHIJ is at 97% of both its size and its inode quota, and
    MLS-Bench's default vendor/workspace is already 120 GB (optimization-nas
    alone is ~560 MB and ~690 files PER RUN). Each cell-task gets a private
    workspace via `mlsbench agent --workspace` and it is deleted as soon as the
    score and the instrumentation have been harvested.

Design and power
----------------
Unit of analysis: one CELL = one arm run once over the whole 20-task panel. Its
statistic is the 20-task mean with a FIXED denominator of 20.

Where the variance is, and what pairing can and cannot buy.

  X[a,r] = mu_a + (1/20) sum_t tau_t + eps[a,r]

Every run scores the SAME 20 tasks under the SAME data seed (seeds: [42]), so
the task-difficulty term is a constant shared by every cell in every arm and is
differenced out exactly. The panel mean IS the task-blocked estimator; there is
no further "pair on task" gain to collect for this estimand. (What the fixed
panel does buy is large and already banked: per-task means span 0.00-0.35 with
SD ~0.106 across the 20 tasks, so RESAMPLING tasks per run would add
0.106^2/20 = 5.6e-4 to the variance, taking sigma from ~0.021 to ~0.032 and
costing 2.3x more replicates. Never resample the panel.)

Pairing on REPLICATE INDEX can only buy the common-random-number correlation
rho between arm A rep r and arm B rep r: Var(D) = s_A^2 + s_B^2 - 2 rho s_A s_B,
so the power gain is 1/(1-rho) at equal s. The runner gives matched cells a
matched MLSBENCH_SAMPLING_SEED, but the arms present different tool schemas from
turn 1, so the trajectories diverge immediately and rho is expected to be near
zero. The design therefore does NOT rely on pairing for its power; `report`
prints the realised rho and pairing_variance_ratio so the assumption is measured
rather than believed.

Replicates needed (two-arm, alpha=0.05 two-sided, 80% power):

    n_per_arm = 2 (1.96 + 0.8416)^2 sigma^2 / delta^2  =  15.7 sigma^2 / delta^2

At sigma = 0.021 (the working prior: a true replicate of identical weights gave
0.038 vs 0.068, and the cross-model SD of the 20-task mean over 29 banked runs
is 0.0206 -- two independent routes agreeing):

    delta 0.030 -> n =   8      delta 0.015 -> n =  31
    delta 0.020 -> n =  18      delta 0.010 -> n =  69

Comparing three arms against one reference costs a multiplicity correction:
Dunnett at alpha=0.05 needs roughly 1.3x these n.

The sigma prior itself rests on 1 degree of freedom, which is why the pilot runs
first. The pilot estimates sigma two ways: directly from between-replicate
spread (few df), and as sqrt(sum_t s_t^2)/20 from the per-task run SDs, which
uses ~20x more df under the assumption that tasks within a run are independent.
`report` prints both, and their agreement is the check on that assumption.

Usage
-----
  # names vLLM must serve (feed to --served-model-name)
  python scripts/mlsbench_contract_eval.py served-names --tag q35 --reps 4

  # run work items (resumable; re-run the same command after a preemption)
  python scripts/mlsbench_contract_eval.py run --out OUT --tag q35 --reps 4 \
      --root /scratch/gpfs/CHIJ/bohan/MLS-Bench-dev --config CFG --concurrency 24

  # aggregate; prints per-arm mean +- SE on the 20-task mean and paired tests
  python scripts/mlsbench_contract_eval.py report --out OUT
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# --------------------------------------------------------------------------
# The panel. Frozen: the 20 non-agent CPU tasks, identical in every arm and
# every replicate. Because every run scores the same 20 tasks, the task-identity
# variance is already differenced out of the 20-task mean — see the power note
# in `report`.
# --------------------------------------------------------------------------
CPU_TASKS = [
    "causal-discovery-discrete",
    "causal-observational-linear-gaussian",
    "causal-observational-linear-non-gaussian",
    "causal-observational-nonlinear",
    "causal-treatment-effect",
    "ml-active-learning",
    "ml-anomaly-detection",
    "ml-calibration",
    "ml-clustering-algorithm",
    "ml-dimensionality-reduction",
    "ml-ensemble-boosting",
    "ml-missing-data-imputation",
    "ml-selective-deferral",
    "ml-subgroup-calibration-shift",
    "ml-symbolic-regression",
    "mlsys-moe-load-balance",
    "optimization-evolution-strategy",
    "optimization-hyperparameter-search",
    "optimization-multi-objective",
    "optimization-nas",
]

# --------------------------------------------------------------------------
# Arms. `use_replace` chooses the tool schema; `env` selects the contract
# variant inside MLS-Bench. All four are reachable from ONE checkout, so no arm
# needs a different working tree (which would confound the comparison).
#
#   linerange        public contract: edit(op in create/insert/replace) by line
#                    number. MLSBENCH_USE_REPLACE unset -> no --use-replace.
#   replace_strict   current/upstream str_replace: byte-exact matcher, original
#                    prompt wording, no view tool.
#   replace_fx       patched matcher (tools.resolve_old_str ladder) + its prompt
#                    guidance, but view() withheld — isolates the matcher.
#   replace_fx_view  patched matcher + view(). This is what the checkout does by
#                    default today, i.e. the current eval path.
#
# A FIFTH SLOT is deliberately left open for the new contract another agent is
# implementing: drop a JSON file with the same shape and pass --arms-file. No
# code change here is needed as long as the new contract is env-selectable.
# --------------------------------------------------------------------------
DEFAULT_ARMS: dict[str, dict] = {
    "linerange": {
        "use_replace": False,
        # The checkout's line-range schema carries a PATCHED op description that
        # spells out each op's required companion arguments (upstream's is one
        # line, and guided decoding then emits op='replace' with no line numbers,
        # which is auto-rejected). That patch is itself a contract improvement,
        # so folding it into the control would credit str_replace for a win that
        # belongs to the line-range op. The control is the genuinely public one.
        "env": {"MLSBENCH_LINERANGE_SCHEMA": "public"},
        "desc": "public line-range edit(create/insert/replace) — the control",
    },
    "linerange_fx": {
        "use_replace": False,
        "env": {},
        "desc": "line-range with the patched op-description (not run by default; "
                "add it if the verdict turns on the control's schema wording)",
    },
    "replace_strict": {
        "use_replace": True,
        "env": {"MLSBENCH_STRICT_STR_REPLACE": "1"},
        "desc": "upstream str_replace: byte-exact, original prompt, no view",
    },
    # NOTE the explicit MLSBENCH_REWRITE_OP=0 on both patched-str_replace arms.
    # The incoming rewrite-op contract (scripts/mlsbench_edit_contract.diff)
    # defaults that knob to 1, so the moment it lands in MLS-Bench-dev these two
    # arms would silently acquire a whole extra operation. Naming every knob an
    # arm depends on -- rather than relying on a default -- is what keeps an arm
    # meaning the same thing before and after somebody else's patch lands.
    "replace_fx": {
        "use_replace": True,
        "env": {"MLSBENCH_VIEW_TOOL": "0", "MLSBENCH_REWRITE_OP": "0"},
        "desc": "patched str_replace matcher, no view tool",
    },
    "replace_fx_view": {
        "use_replace": True,
        "env": {"MLSBENCH_VIEW_TOOL": "1", "MLSBENCH_REWRITE_OP": "0"},
        "desc": "patched str_replace matcher + view tool",
    },
    # The NEW contract (rewrite + str_replace + view). Requires
    # scripts/mlsbench_edit_contract.diff to be applied to MLSBENCH_ROOT;
    # without it MLSBENCH_REWRITE_OP is inert and this arm collapses onto
    # replace_fx_view, which the run preflight refuses to let happen silently.
    "rewrite_view": {
        "use_replace": True,
        "env": {"MLSBENCH_VIEW_TOOL": "1", "MLSBENCH_REWRITE_OP": "1"},
        "desc": "rewrite + str_replace + view (new contract)",
    },
}

# Log-tail markers that mean "the harness/infrastructure broke", not "the agent
# failed the task". These are RETRIED; everything else is taken at face value.
INFRA_PATTERNS = [
    r"APIConnectionError", r"APITimeoutError", r"Connection error",
    r"Connection refused", r"Read timed out", r"Remote end closed connection",
    r"ServiceUnavailableError", r"InternalServerError", r"HTTP/1\.1 5\d\d",
    r"CUDA out of memory", r"No space left on device", r"Disk quota exceeded",
    r"Killed\b", r"Segmentation fault", r"returncode -11", r"returncode -9",
    r"vLLM .*(died|exited)", r"BrokenPipeError", r"OSError: \[Errno 28\]",
]
INFRA_RE = re.compile("|".join(INFRA_PATTERNS))

SYNTAX_ERR_RE = re.compile(
    r"SyntaxError|IndentationError|TabError|unexpected indent|invalid syntax"
)

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(f"[contract-eval] {msg}", flush=True)


def atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


def cell_model(tag: str, arm: str, rep: int) -> str:
    """Leaderboard/API identity of one (arm, rep) cell. MUST be unique per cell:
    see the module docstring on evaluate_task's model-string grouping."""
    return f"{tag}__{arm}__r{rep:02d}"


def load_arms(arms_file: str | None) -> dict[str, dict]:
    if not arms_file:
        return dict(DEFAULT_ARMS)
    spec = json.loads(Path(arms_file).read_text())
    if not isinstance(spec, dict) or not spec:
        raise SystemExit(f"--arms-file {arms_file}: expected a non-empty object")
    for name, a in spec.items():
        if "use_replace" not in a:
            raise SystemExit(f"arm '{name}' missing 'use_replace'")
        a.setdefault("env", {})
        a.setdefault("desc", "")
    return spec


# ==========================================================================
# Preflight: prove the arms are distinct, and pin the harness they ran against
# ==========================================================================
#
# This eval shares MLS-Bench-dev with other work. If someone lands a contract
# patch while a campaign is in flight, arms silently change meaning: the new
# rewrite-op patch, for instance, defaults MLSBENCH_REWRITE_OP=1, which would
# hand `replace_fx` and `replace_fx_view` a whole extra operation halfway
# through. Two guards:
#
#   1. an arm-CONTRACT fingerprint -- what the model is actually shown under
#      that arm's env -- checked pairwise before any GPU time is spent, so an
#      arm that has collapsed onto another cannot masquerade as a null result;
#   2. a harness SOURCE fingerprint stored in every cell, so `report` can refuse
#      to pool cells that were produced by different harnesses.

_FINGERPRINT_SRC = ("agent/tools.py", "agent/interactive.py", "agent/models.py")

_ARM_FP_SNIPPET = """
import hashlib, json, sys
from unittest import mock
from mlsbench.agent import interactive as I
def _init(self, task_name, global_config, workspace_root=None):
    self.task_name = task_name
with mock.patch.object(I.BaseAgent, "__init__", _init), \
     mock.patch.object(I, "build_client", lambda cfg: object()):
    a = I.InteractiveAgent("t", {"use_replace": USE_REPLACE})
blob = json.dumps(a._tool_schemas, sort_keys=True, default=str) + "\\x00" + a.system_prompt
sys.stdout.write(hashlib.sha256(blob.encode()).hexdigest())
"""


def harness_fingerprint(root: Path) -> str:
    h = hashlib.sha256()
    for rel in _FINGERPRINT_SRC:
        p = root / "src" / "mlsbench" / rel
        h.update(p.read_bytes() if p.exists() else b"<missing>")
    return h.hexdigest()[:16]


def arm_contract_fingerprint(root: Path, spec: dict, python_exe: str,
                             env_base: dict) -> str:
    """Hash of exactly what the model is shown under this arm's env."""
    env = dict(env_base)
    for k in ("MLSBENCH_STRICT_STR_REPLACE", "MLSBENCH_VIEW_TOOL",
              "MLSBENCH_LINERANGE_SCHEMA", "MLSBENCH_REWRITE_OP",
              "MLSBENCH_SYNTAX_GATE"):
        env.pop(k, None)
    env.update({k: str(v) for k, v in spec.get("env", {}).items()})
    env["PYTHONPATH"] = f"{root}/src:" + env.get("PYTHONPATH", "")
    code = _ARM_FP_SNIPPET.replace("USE_REPLACE", repr(bool(spec.get("use_replace"))))
    out = subprocess.run([python_exe, "-c", code], cwd=str(root), env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, timeout=180)
    if out.returncode != 0 or not out.stdout.strip():
        raise SystemExit("arm fingerprint failed:\n" + (out.stderr or "")[-2000:])
    return out.stdout.strip()[:16]


def preflight(root: Path, arms: dict, arm_names: list[str], python_exe: str,
              env_base: dict) -> dict:
    fps = {a: arm_contract_fingerprint(root, arms[a], python_exe, env_base)
           for a in arm_names}
    log(f"harness fingerprint: {harness_fingerprint(root)}")
    for a in arm_names:
        log(f"  arm {a:18s} contract={fps[a]}  env={arms[a].get('env')}")
    dupes = [(a, b) for i, a in enumerate(arm_names)
             for b in arm_names[i + 1:] if fps[a] == fps[b]]
    if dupes:
        raise SystemExit(
            "PREFLIGHT FAILED: these arms are the SAME contract — the model is "
            "shown identical tools and an identical prompt, so comparing them "
            "would produce a null result that looks like evidence:\n  "
            + "\n  ".join(f"{a} == {b}" for a, b in dupes)
            + "\nUsually this means a harness patch that an arm's env var was "
              "supposed to select has not been applied to "
            f"{root}, or its default flipped.")
    return fps


# ==========================================================================
# Instrumentation harvest
# ==========================================================================

def _iter_messages(agent_dir: Path) -> list[dict]:
    p = agent_dir / "messages.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _classify_edit_rejection(result: str) -> str:
    for pat, name in (
        ("old_str not found", "old_str_not_found"),
        ("matches", "ambiguous"),
        ("not unique", "ambiguous"),
        ("exceed the editable range", "outside_editable_range"),
        ("allow_create", "allow_create_false"),
        ("required for op", "missing_argument"),
        ("must not be empty", "empty_old_str"),
        ("not in allowed packages", "bad_path"),
        ("File not editable", "file_not_editable"),
        ("File not found", "file_not_found"),
    ):
        if pat in result:
            return name
    return "other"


def harvest_instrumentation(root: Path, task: str, log_label: str,
                            workspace_root: Path) -> dict:
    """Post-hoc metrics for one agent run. Pure reads; never re-runs anything.

    Everything here comes from MLS-Bench's own RunLogger output
    (logs/<task>/<provider>/<exp>__<label>/agent/) plus the surviving workspace
    file, so it costs no GPU time and works on already-banked runs.
    """
    m: dict = {
        "n_steps": None, "n_tests": None, "agent_done": None,
        "n_edit_calls": 0, "n_edit_accepted": 0, "n_edit_rejected": 0,
        "n_accepted_left_file_broken": 0,
        "n_view_calls": 0, "n_undo_calls": 0, "n_test_calls": 0,
        "n_tests_run": 0, "n_tests_wasted_on_syntax": 0,
        "reject_reasons": {},
        "first_edit_accepted": None,
        "submitted_unmodified_template": None,
        "final_equals_template": None,
        "completion_tokens": None,
        "instrumented": False,
    }

    hits = sorted((root / "logs" / task).glob(f"*/*__{log_label}/agent"))
    if not hits:
        return m
    agent_dir = hits[-1]
    m["agent_log_dir"] = str(agent_dir)

    sp = agent_dir / "summary.json"
    if sp.exists():
        try:
            s = json.loads(sp.read_text())
        except json.JSONDecodeError:
            s = {}
        m["n_steps"] = s.get("steps")
        m["n_tests"] = s.get("tests")
        m["agent_done"] = s.get("done")
        m["completion_tokens"] = (s.get("tokens") or {}).get("completion_tokens")
        m["prompt_tokens"] = (s.get("tokens") or {}).get("prompt_tokens")
        # Fallback only — see the messages.jsonl pass below, which also works
        # for runs that crashed before summary.json was written.
        m["_syntax_waste_from_summary"] = sum(
            1 for e in (s.get("test_history") or [])
            if SYNTAX_ERR_RE.search(e.get("feedback") or ""))

    msgs = _iter_messages(agent_dir)
    # RunLogger's `_meta` row carries the WORKSPACE directory name (which differs
    # from the log dir name: the log dir has the label appended). Use it to pin
    # the template check to THIS run's workspace — globbing would otherwise pick
    # up a neighbouring run's file whenever the workspace root is shared.
    exp_name = next((r.get("exp_name") for r in msgs if r.get("role") == "_meta"), None)
    results_by_step: dict[str, str] = {}
    for r in msgs:
        if r.get("role") == "tool_result":
            results_by_step[str(r.get("step"))] = str(r.get("result") or "")
    for r in msgs:
        if r.get("role") != "assistant":
            continue
        tool = r.get("tool_name")
        step = str(r.get("step"))
        res = results_by_step.get(step, "")
        if tool == "view":
            m["n_view_calls"] += 1
        elif tool == "undo":
            m["n_undo_calls"] += 1
        elif tool == "test":
            m["n_test_calls"] += 1
            # Only a result carrying "[Test #" actually consumed budget; a call
            # made after the budget is exhausted comes back as a bare ERROR.
            # "tests wasted on syntax errors" means a test out of a budget of
            # THREE spent discovering the file does not parse -- an
            # editing-contract failure, not a scientific one.
            if "[Test #" in res:
                m["n_tests_run"] += 1
                if SYNTAX_ERR_RE.search(res):
                    m["n_tests_wasted_on_syntax"] += 1
        elif tool == "edit":
            m["n_edit_calls"] += 1
            ok = not res.lstrip().startswith("ERROR")
            if m["first_edit_accepted"] is None:
                m["first_edit_accepted"] = bool(ok)
            if ok:
                m["n_edit_accepted"] += 1
                # The harness's own syntax gate is the authoritative signal that
                # an ACCEPTED edit left the file unparseable; fall back to
                # compiling the post-edit snapshot when the marker is absent.
                broken = "no longer parses as Python" in res
                if not broken:
                    for snap in (agent_dir / "files").glob(f"step_{step}_*"):
                        if snap.suffix != ".py":
                            continue
                        try:
                            ast.parse(snap.read_text(errors="replace"))
                        except SyntaxError:
                            broken = True
                        except Exception:  # noqa: BLE001
                            pass
                        break
                if broken:
                    m["n_accepted_left_file_broken"] += 1
            else:
                m["n_edit_rejected"] += 1
                k = _classify_edit_rejection(res)
                m["reject_reasons"][k] = m["reject_reasons"].get(k, 0) + 1
    m["instrumented"] = bool(msgs)
    if not msgs and m.get("_syntax_waste_from_summary"):
        m["n_tests_wasted_on_syntax"] = m["_syntax_waste_from_summary"]
        m["n_tests_run"] = m.get("n_tests") or 0
    m.pop("_syntax_waste_from_summary", None)
    if m["n_steps"] is None:
        # summary.json is only written on a clean exit; a run killed by the
        # context-overflow crash has none, and dropping those runs from the
        # step counts would bias exactly the arms that burn the most context.
        m["n_steps"] = sum(1 for r in msgs if r.get("role") == "assistant") or None

    # Submission of the unmodified template. Read the surviving workspace copy
    # of the editable file and compare against the task's own template. Done
    # BEFORE the workspace is deleted; None if the file could not be located.
    try:
        cfg = json.loads((root / "tasks" / task / "config.json").read_text())
        editable = next((f["filename"] for f in cfg.get("files", []) if f.get("edit")), None)
        tpl = root / "tasks" / task / "edits" / "custom_template.py"
        if editable and tpl.exists():
            cands = []
            if exp_name:
                p = workspace_root / task / exp_name / editable
                if p.exists():
                    cands = [p]
            if not cands:
                # Only trust a glob when it is unambiguous; a shared workspace
                # root would otherwise hand us someone else's file.
                g = sorted(workspace_root.glob(f"{task}/*/{editable}"))
                cands = g if len(g) == 1 else []
            if cands:
                same = cands[0].read_text(errors="replace") == tpl.read_text(errors="replace")
                m["final_equals_template"] = same
                m["submitted_unmodified_template"] = bool(same or m["n_edit_accepted"] == 0)
        if m["submitted_unmodified_template"] is None:
            m["submitted_unmodified_template"] = (m["n_edit_accepted"] == 0)
    except Exception as e:  # noqa: BLE001
        m["template_check_error"] = str(e)
    return m


# ==========================================================================
# One (arm, rep, task) work item
# ==========================================================================

def score_task(root: Path, task: str, model: str, python_exe: str,
               env: dict, task_log: Path) -> tuple[float | None, list, str | None]:
    """`mlsbench score` for one task+model. Returns (score, settings, error)."""
    cmd = [python_exe, "-m", "mlsbench", "score", task,
           "--model", model, "--format", "json"]
    try:
        out = subprocess.run(cmd, cwd=str(root), env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             timeout=900, text=True)
    except Exception as e:  # noqa: BLE001
        return None, [], f"score_subprocess: {e}"
    with open(task_log, "a") as lf:
        lf.write("\n### SCORE\n# " + " ".join(cmd) + "\n")
        lf.write(out.stdout or "")
        if out.stderr:
            lf.write("\n### SCORE (stderr)\n" + out.stderr)
    # stdout must be pure JSON: `mlsbench score` emits calibration UserWarnings
    # on stderr, and merging the streams used to corrupt the JSON and silently
    # discard real scores. Keep them split, but still recover a JSON object if
    # a future banner ever leaks onto stdout.
    s = out.stdout or ""
    # `mlsbench score --format json` prints an ENGLISH SENTENCE on stdout and
    # exits 0 when the model has no leaderboard row -- it ignores --format on
    # that path (scoring/__init__.py::_cmd_score_task; fixed in this checkout,
    # but handle the old text too so the runner works against an unpatched
    # tree). Parsing that as JSON fails, and treating the failure as
    # infrastructure would RETRY a run that in fact completed and simply never
    # submitted anything -- burning GPU, and then dropping a genuine zero out of
    # the mean when the retries also "fail". That biases the comparison against
    # exactly the arms that fail to produce a submission, which is the effect we
    # are here to measure. Recognise it as "no row" instead.
    if s.lstrip().startswith("No results for task"):
        return None, [], None

    def _why() -> str:
        # Distinguish "the scorer never ran" from "the scorer printed garbage".
        # Both used to surface as score_json_unparseable, which is unactionable.
        tail = (out.stderr or "").strip().splitlines()[-1:] or [""]
        return (f"score_rc={out.returncode} stdout_len={len(s)} "
                f"stderr_tail={tail[0][:200]!r}")
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        i, j = s.find("{"), s.rfind("}")
        if i == -1 or j <= i:
            return None, [], f"score_json_unparseable ({_why()})"
        try:
            data = json.loads(s[i:j + 1])
        except json.JSONDecodeError:
            return None, [], f"score_json_unparseable ({_why()})"
    entries = data.get(task, [])
    # The model string is unique per cell, so an exact match is the ONLY
    # acceptable row. Never fall back to entries[-1]: that is how a neighbouring
    # cell's score leaks into this one.
    match = next((e for e in entries if e.get("model") == model), None)
    if match is None:
        return None, [], None
    settings = [{"name": x.get("name"), "score": x.get("score")}
                for x in match.get("settings", [])]
    return match.get("task_score"), settings, None


def run_item(*, task: str, arm: str, rep: int, arms: dict, root: Path,
             config: str, tag: str, out_dir: Path, ws_root: Path,
             python_exe: str, timeout: int, env_base: dict,
             seed_base: int, keep_workspace: bool, attempt: int) -> dict:
    """Run one agent episode + score + harvest. Returns the per-task record."""
    model = f"vllm/{cell_model(tag, arm, rep)}"
    label = f"ce-{arm}-r{rep:02d}-{task}"
    cell_dir = out_dir / "cells" / arm / f"r{rep:02d}"
    task_log = cell_dir / "task_logs" / f"{task}.log"
    task_log.parent.mkdir(parents=True, exist_ok=True)
    # Private workspace per item: deleted after harvest so the campaign cannot
    # walk the shared fileset off its size/inode quota.
    item_ws = ws_root / arm / f"r{rep:02d}" / task

    env = dict(env_base)
    env.update({k: str(v) for k, v in arms[arm].get("env", {}).items()})
    env["MLSBENCH_LOG_LABEL"] = label
    # Deterministic per-cell sampling seed. Same seed for the same (rep, task)
    # across arms, so the arms are drawn from matched noise wherever the
    # trajectories have not yet diverged; different across reps.
    env["MLSBENCH_SAMPLING_SEED"] = str(
        (seed_base + 1_000_003 * rep + 10_007 * CPU_TASKS.index(task)) % (2 ** 31 - 1)
    )

    rec: dict = {
        "task": task, "arm": arm, "rep": rep, "model": model,
        "attempt": attempt, "status": "unknown", "score": None,
        "sampling_seed": env["MLSBENCH_SAMPLING_SEED"],
    }
    t0 = time.time()

    cmd = [python_exe, "-m", "mlsbench", "agent", task,
           "--model", model, "--config", config, "--workspace", str(item_ws)]
    if arms[arm].get("use_replace"):
        cmd.append("--use-replace")

    item_ws.mkdir(parents=True, exist_ok=True)
    with open(task_log, "w") as lf:
        lf.write(f"### AGENT {task} arm={arm} rep={rep} attempt={attempt}\n"
                 f"# {' '.join(cmd)}\n"
                 f"# arm env: {arms[arm].get('env')}\n"
                 f"# sampling seed: {env['MLSBENCH_SAMPLING_SEED']}\n\n")
        lf.flush()
        try:
            proc = subprocess.run(cmd, cwd=str(root), env=env,
                                  stdout=lf, stderr=subprocess.STDOUT,
                                  timeout=timeout)
            rec["agent_returncode"] = proc.returncode
        except subprocess.TimeoutExpired:
            rec["agent_returncode"] = None
            rec["status"] = "timeout"
            lf.write(f"\n### TIMEOUT after {timeout}s\n")

    score, settings, serr = score_task(root, task, model, python_exe, env, task_log)
    rec["score"] = score
    rec["settings"] = settings
    if serr:
        rec["score_error"] = serr

    rec["metrics"] = harvest_instrumentation(root, task, label, item_ws)

    # ---- classify: genuine result vs infrastructure failure -----------------
    tail = ""
    try:
        with open(task_log, errors="replace") as lf:
            tail = lf.read()[-20000:]
    except OSError:
        pass
    # CONTEXT OVERFLOW. MLS-Bench does not trim history: when the conversation
    # passes max-model-len the OpenAI client raises 400 and the agent process
    # dies mid-episode, keeping whatever it had already submitted. This happened
    # in 3 of 20 tasks on the reference run. It is NOT infrastructure — it is a
    # real consequence of the contract, and the arms differ precisely in how
    # much context they burn (view() dumps files; str_replace echoes old_str and
    # new_str; the line-range op sends only integers). Count it per arm, because
    # an arm can lose score by running out of context rather than by editing
    # badly, and that distinction changes what we should ship.
    ctx = bool(re.search(r"maximum context length|context_length_exceeded|"
                         r"reduce the length of the (input )?prompt", tail))
    rec["metrics"]["context_overflow"] = ctx
    rec["metrics"]["agent_crashed"] = bool(rec.get("agent_returncode"))

    infra_hit = INFRA_RE.search(tail)
    if isinstance(score, (int, float)):
        rec["status"] = "scored"
    elif rec["status"] == "timeout" or serr or infra_hit:
        rec["status"] = "infra_failed"
        rec["infra_reason"] = (rec.get("score_error")
                               or (infra_hit.group(0) if infra_hit else "timeout"))
    elif rec.get("agent_returncode") is not None and rec["metrics"].get("instrumented"):
        # The agent ran to completion and simply never produced a scorable
        # submission. That is a real failure of the contract, worth 0 — not an
        # infrastructure excuse.
        rec["status"] = "no_submission"
        rec["score"] = 0.0
    else:
        rec["status"] = "infra_failed"
        rec["infra_reason"] = "agent produced no log"

    rec["elapsed_s"] = round(time.time() - t0, 1)
    if not keep_workspace:
        shutil.rmtree(item_ws, ignore_errors=True)
    return rec


# ==========================================================================
# run
# ==========================================================================

def terminal(rec: dict | None) -> bool:
    return bool(rec) and rec.get("status") in ("scored", "no_submission")


def seal_cells(out_dir: Path) -> int:
    """Seal every cell whose tasks are ALL terminal. Idempotent.

    Called after each finished item, at the end of a run, AND at the start of
    `report`. Sealing only at the end of a run would mean a preempted job left
    nothing quotable even when whole cells had finished -- and jobs here get
    preempted, which is the entire reason for banking per-task records.
    """
    meta_p = out_dir / "arms.json"
    if not meta_p.exists():
        return 0
    try:
        meta = json.loads(meta_p.read_text())
    except json.JSONDecodeError:
        return 0
    tasks = meta.get("tasks") or CPU_TASKS
    tag = meta.get("tag", "")
    fps = meta.get("arm_contract_fingerprints") or {}
    sealed = 0
    for d in sorted(out_dir.glob("cells/*/r*")):
        if (d / "cell.json").exists():
            continue
        arm, rep = d.parent.name, int(d.name.lstrip("r"))
        recs = {}
        for t in tasks:
            p = d / "tasks" / f"{t}.json"
            if not p.exists():
                break
            try:
                recs[t] = json.loads(p.read_text())
            except json.JSONDecodeError:
                break
        if len(recs) != len(tasks) or not all(terminal(recs[t]) for t in tasks):
            continue
        atomic_write_json(d / "cell.json", {
            "arm": arm, "rep": rep, "tag": tag,
            "model": f"vllm/{cell_model(tag, arm, rep)}",
            "tasks": tasks, "denominator": len(tasks), "complete": True,
            "harness_fingerprint": meta.get("harness_fingerprint"),
            "arm_contract_fingerprint": fps.get(arm),
            "mean_score": sum(float(recs[t]["score"] or 0.0) for t in tasks) / len(tasks),
            "records": [recs[t] for t in tasks],
            "sealed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        sealed += 1
    return sealed


def cmd_run(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    out_dir = Path(args.out).resolve()
    arms = load_arms(args.arms_file)
    arm_names = [a for a in (args.arms or list(arms)) if a in arms]
    if not arm_names:
        raise SystemExit("no arms selected")
    reps = list(range(args.rep_start, args.rep_start + args.reps))
    tasks = list(args.tasks) if args.tasks else list(CPU_TASKS)
    ws_root = Path(args.workspace_root).resolve() if args.workspace_root else out_dir / "ws"

    out_dir.mkdir(parents=True, exist_ok=True)

    env_base = dict(os.environ)
    env_base["PYTHONPATH"] = f"{root}/src:" + env_base.get("PYTHONPATH", "")
    env_base["MLSBENCH_SCHEDULER_MANAGED"] = "1"
    env_base.setdefault("MLSBENCH_NO_PREBUILT", "1")
    env_base.setdefault("HF_HUB_OFFLINE", "1")
    env_base.setdefault("TRANSFORMERS_OFFLINE", "1")
    # Host-side BLAS pinning: MLS-Bench parses test output in a thread pool and
    # some parsers enter LAPACK, which SIGSEGVs the whole agent process when
    # several threads do it at once on this node.
    for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
              "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env_base[v] = "1"
    # MLSBENCH_USE_REPLACE is consumed by the older worker; here the arm spec is
    # authoritative, so drop it to avoid a stale value confusing anything.
    env_base.pop("MLSBENCH_USE_REPLACE", None)
    for k in ("MLSBENCH_STRICT_STR_REPLACE", "MLSBENCH_VIEW_TOOL"):
        env_base.pop(k, None)

    # Preflight BEFORE any GPU time: prove every selected arm is a genuinely
    # different contract, and pin the harness they will run against.
    arm_fps = preflight(root, arms, arm_names, args.python, env_base)
    harness_fp = harness_fingerprint(root)
    atomic_write_json(out_dir / "arms.json", {
        "arms": {k: arms[k] for k in arm_names},
        "tasks": tasks, "reps": reps, "tag": args.tag,
        "denominator": len(tasks), "root": str(root),
        "seed_base": args.seed_base,
        "harness_fingerprint": harness_fp,
        "arm_contract_fingerprints": arm_fps,
    })

    # Work order: (rep, task, arm). Arms advance in lockstep, so a job killed
    # part-way still leaves the arms balanced, and whole cells finish together
    # instead of one arm finishing while another has nothing.
    items = [(rep, t, arm) for rep in reps for t in tasks for arm in arm_names]

    done_count = {"n": 0, "skipped": 0}
    total = len(items)
    lock = threading.Lock()

    def work(item) -> None:
        rep, task, arm = item
        cell_dir = out_dir / "cells" / arm / f"r{rep:02d}"
        rec_path = cell_dir / "tasks" / f"{task}.json"
        prior = None
        if rec_path.exists():
            try:
                prior = json.loads(rec_path.read_text())
            except json.JSONDecodeError:
                prior = None
        if terminal(prior):
            with lock:
                done_count["skipped"] += 1
                done_count["n"] += 1
            return
        attempt = int((prior or {}).get("attempt", 0))
        # `rec` must never be None below: a resumed item whose banked attempts
        # are already exhausted skips the loop entirely, and the old code then
        # dereferenced None and killed the pool.
        rec = prior or {"task": task, "arm": arm, "rep": rep,
                        "status": "attempts_exhausted", "score": None}
        while attempt < args.max_attempts:
            attempt += 1
            try:
                rec = run_item(task=task, arm=arm, rep=rep, arms=arms, root=root,
                               config=args.config, tag=args.tag, out_dir=out_dir,
                               ws_root=ws_root, python_exe=args.python,
                               timeout=args.timeout, env_base=env_base,
                               seed_base=args.seed_base,
                               keep_workspace=args.keep_workspace, attempt=attempt)
            except Exception as e:  # noqa: BLE001
                # One item blowing up must not take the other 159 with it; the
                # whole point of banking per-task records is that the campaign
                # survives partial failure.
                rec = {"task": task, "arm": arm, "rep": rep, "attempt": attempt,
                       "status": "infra_failed", "score": None,
                       "infra_reason": f"runner exception: {e}"}
            atomic_write_json(rec_path, rec)
            if terminal(rec):
                break
            log(f"RETRY {arm}/r{rep:02d}/{task}: {rec.get('status')} "
                f"({rec.get('infra_reason')}) attempt {attempt}/{args.max_attempts}")
            time.sleep(min(60, 5 * attempt))
        with lock:
            done_count["n"] += 1
            n = done_count["n"]
            # Seal opportunistically: a job killed at minute 400 should still
            # leave every finished cell quotable.
            if seal_cells(out_dir):
                log("sealed a cell")
        log(f"[{n}/{total}] {arm}/r{rep:02d}/{task:42s} "
            f"status={rec.get('status'):14s} score={rec.get('score')} "
            f"({rec.get('elapsed_s')}s)")

    log(f"root={root} out={out_dir}")
    log(f"arms={arm_names} reps={reps} tasks={len(tasks)} items={total} "
        f"concurrency={args.concurrency}")
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        for fut in [ex.submit(work, it) for it in items]:
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001
                log(f"WORKER ERROR (continuing): {e}")

    sealed = seal_cells(out_dir)
    log(f"DONE. {done_count['skipped']} items already banked; {sealed} complete cells on disk.")
    if not args.keep_workspace:
        shutil.rmtree(ws_root, ignore_errors=True)
    return 0


# ==========================================================================
# report
# ==========================================================================

def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _sd(xs):
    n = len(xs)
    if n < 2:
        return float("nan")
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def _t_crit(df: int) -> float:
    """Two-sided 95% t critical value (table + normal tail)."""
    tbl = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
           7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
           13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
           19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000,
           120: 1.980}
    if df <= 0:
        return float("nan")
    if df in tbl:
        return tbl[df]
    keys = sorted(tbl)
    if df > keys[-1]:
        return 1.96
    for k in keys:
        if df < k:
            return tbl[k]
    return 1.96


def _t_sf(t: float, df: int) -> float:
    """Two-sided p-value for Student t, via the incomplete beta (no scipy)."""
    if df <= 0 or not math.isfinite(t):
        return float("nan")
    x = df / (df + t * t)
    return _betainc(df / 2.0, 0.5, x)


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a,b) by continued fraction."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    if x >= (a + 1) / (a + b + 2):
        return 1.0 - _betainc(b, a, 1 - x)
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 300):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        c = 1.0 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        f *= c * d
        if abs(1.0 - c * d) < 1e-12:
            break
    return front * (f - 1.0)


def load_cells(out_dir: Path) -> tuple[list[dict], list[str]]:
    """Complete cells only. Anything partial is listed, never averaged."""
    cells, partial = [], []
    for cell_json in sorted(out_dir.glob("cells/*/r*/cell.json")):
        try:
            c = json.loads(cell_json.read_text())
        except json.JSONDecodeError:
            partial.append(f"{cell_json} (unreadable)")
            continue
        recs = c.get("records", [])
        n_terminal = sum(1 for r in recs if terminal(r))
        if not c.get("complete") or n_terminal != c.get("denominator", len(recs)):
            partial.append(f"{c.get('arm')}/r{c.get('rep')} ({n_terminal} terminal)")
            continue
        cells.append(c)
    # Cells that exist on disk but were never sealed.
    for d in sorted(out_dir.glob("cells/*/r*")):
        if not (d / "cell.json").exists():
            n = len(list((d / "tasks").glob("*.json"))) if (d / "tasks").exists() else 0
            partial.append(f"{d.parent.name}/{d.name} (unsealed, {n} task records)")
    return cells, partial


def sensitivity_infra_zero(out_dir: Path, tasks: list[str]) -> dict:
    """What the arm means would be if every unresolved infrastructure failure
    were counted as a 0 instead of leaving its cell unsealed.

    Excluding failed cells is the right default (an infra zero is not a model
    result), but excluding them can itself bias the comparison if one arm fails
    more often. Printing both makes that visible instead of arguing about it.
    """
    by_arm: dict[str, list[float]] = {}
    for d in sorted(out_dir.glob("cells/*/r*")):
        arm = d.parent.name
        recs = {}
        for p in (d / "tasks").glob("*.json"):
            try:
                r = json.loads(p.read_text())
            except json.JSONDecodeError:
                continue
            recs[r.get("task")] = r
        if len(recs) != len(tasks):
            continue
        by_arm.setdefault(arm, []).append(
            sum(float(recs[t].get("score") or 0.0) for t in tasks) / len(tasks))
    return {a: {"n": len(v), "mean": _mean(v), "sd": _sd(v)} for a, v in by_arm.items()}


def arm_stats(cells: list[dict]) -> dict:
    by_arm: dict[str, list[dict]] = {}
    for c in cells:
        by_arm.setdefault(c["arm"], []).append(c)
    out = {}
    for arm, cs in by_arm.items():
        cs = sorted(cs, key=lambda c: c["rep"])
        means = [c["mean_score"] for c in cs]
        n = len(means)
        sd = _sd(means)
        se = sd / math.sqrt(n) if n >= 2 else float("nan")
        out[arm] = {
            "n_cells": n, "reps": [c["rep"] for c in cs],
            "mean": _mean(means), "sd": sd, "se": se,
            "ci95": (_mean(means) - _t_crit(n - 1) * se,
                     _mean(means) + _t_crit(n - 1) * se) if n >= 2 else (float("nan"),) * 2,
            "cell_means": means,
        }
    return out


def paired_diff(cells: list[dict], a: str, b: str) -> dict:
    """Paired on replicate index. Also reports the unpaired (Welch) comparison
    and the between-arm correlation, so the value of pairing is measured, not
    assumed."""
    ma = {c["rep"]: c["mean_score"] for c in cells if c["arm"] == a}
    mb = {c["rep"]: c["mean_score"] for c in cells if c["arm"] == b}
    common = sorted(set(ma) & set(mb))
    d = [ma[r] - mb[r] for r in common]
    n = len(d)
    res = {"arm_a": a, "arm_b": b, "n_pairs": n, "reps": common,
           "diff": _mean(d) if n else float("nan")}
    if n >= 2:
        sd = _sd(d)
        se = sd / math.sqrt(n)
        t = _mean(d) / se if se > 0 else float("nan")
        res.update({"sd_diff": sd, "se_diff": se, "t": t, "df": n - 1,
                    "p": _t_sf(t, n - 1),
                    "ci95": (_mean(d) - _t_crit(n - 1) * se,
                             _mean(d) + _t_crit(n - 1) * se)})
        xa = [ma[r] for r in common]
        xb = [mb[r] for r in common]
        sa, sb = _sd(xa), _sd(xb)
        if sa > 0 and sb > 0:
            mua, mub = _mean(xa), _mean(xb)
            cov = sum((x - mua) * (y - mub) for x, y in zip(xa, xb)) / (n - 1)
            rho = cov / (sa * sb)
            res["rho_between_arms"] = rho
            # Variance ratio unpaired:paired = (sa^2+sb^2)/(sa^2+sb^2-2*rho*sa*sb).
            den = sa * sa + sb * sb - 2 * rho * sa * sb
            res["pairing_variance_ratio"] = ((sa * sa + sb * sb) / den) if den > 0 else float("inf")
            se_unp = math.sqrt(sa * sa / n + sb * sb / n)
            res["se_diff_unpaired"] = se_unp
    return res


def per_task_table(cells: list[dict]) -> dict:
    """Per (arm, task) mean and per-task run SD pooled over arms. The per-task
    SD is what tells you where the noise actually lives."""
    acc: dict[tuple[str, str], list[float]] = {}
    for c in cells:
        for r in c["records"]:
            acc.setdefault((c["arm"], r["task"]), []).append(float(r.get("score") or 0.0))
    tasks = sorted({k[1] for k in acc})
    arms = sorted({k[0] for k in acc})
    out = {"tasks": tasks, "arms": arms, "cells": {}, "task_sd": {}}
    for t in tasks:
        pooled = []
        for a in arms:
            xs = acc.get((a, t), [])
            out["cells"][f"{a}|{t}"] = {"n": len(xs), "mean": _mean(xs), "sd": _sd(xs)}
            if len(xs) >= 2:
                pooled.append((len(xs) - 1, _sd(xs) ** 2))
        if pooled:
            df = sum(w for w, _ in pooled)
            out["task_sd"][t] = math.sqrt(sum(w * v for w, v in pooled) / df) if df else float("nan")
    return out


def edit_vs_template_table(cells: list[dict]) -> dict:
    """Per task: score when the agent submitted the UNMODIFIED template vs when
    it actually changed the code.

    This exists because of a measurement discovered in the pilot's very first
    cell: ml-anomaly-detection scored 0.5016 after 4 steps and ZERO edits, by
    submitting the stub it was handed. MLS-Bench calibrates each task so the
    reference baseline maps to ~0.5, and on the tasks whose template already IS
    a competent baseline, editing can only lose points. The panel mean is 0.057,
    so one such task is worth 0.025 -- larger than sigma.

    The consequence for THIS comparison is direct: an arm can win on task score
    by editing LESS, which is the opposite of what we would want to ship. So the
    contract verdict must be read next to this table, not instead of it.
    """
    acc: dict[tuple[str, bool], list[float]] = {}
    for c in cells:
        for r in c["records"]:
            m = r.get("metrics") or {}
            unmod = m.get("submitted_unmodified_template")
            if unmod is None:
                continue
            acc.setdefault((r["task"], bool(unmod)), []).append(float(r.get("score") or 0.0))
    out = {}
    for t in sorted({k[0] for k in acc}):
        tpl, ed = acc.get((t, True), []), acc.get((t, False), [])
        out[t] = {"n_template": len(tpl), "template_mean": _mean(tpl) if tpl else None,
                  "n_edited": len(ed), "edited_mean": _mean(ed) if ed else None,
                  "edit_delta": (_mean(ed) - _mean(tpl)) if (tpl and ed) else None}
    return out


def instrumentation_table(cells: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for c in cells:
        a = out.setdefault(c["arm"], {
            "n_runs": 0, "edit_calls": 0, "edit_accepted": 0, "edit_rejected": 0,
            "accepted_broken": 0, "view_calls": 0, "undo_calls": 0,
            "tests": 0, "tests_run": 0, "tests_wasted_syntax": 0,
            "steps": 0, "n_steps_obs": 0,
            "first_edit_accepted": 0, "first_edit_obs": 0,
            "unmodified_template": 0, "unmodified_obs": 0,
            "no_submission": 0, "context_overflow": 0, "agent_crashed": 0,
            "prompt_tokens": 0, "completion_tokens": 0, "token_obs": 0,
            "reject_reasons": {},
        })
        for r in c["records"]:
            a["n_runs"] += 1
            if r.get("status") == "no_submission":
                a["no_submission"] += 1
            m = r.get("metrics") or {}
            a["edit_calls"] += m.get("n_edit_calls", 0)
            a["edit_accepted"] += m.get("n_edit_accepted", 0)
            a["edit_rejected"] += m.get("n_edit_rejected", 0)
            a["accepted_broken"] += m.get("n_accepted_left_file_broken", 0)
            a["view_calls"] += m.get("n_view_calls", 0)
            a["undo_calls"] += m.get("n_undo_calls", 0)
            a["tests"] += m.get("n_test_calls", 0)
            a["tests_run"] += m.get("n_tests_run", 0)
            a["tests_wasted_syntax"] += m.get("n_tests_wasted_on_syntax", 0)
            a["context_overflow"] += 1 if m.get("context_overflow") else 0
            a["agent_crashed"] += 1 if m.get("agent_crashed") else 0
            if isinstance(m.get("n_steps"), int):
                a["steps"] += m["n_steps"]
                a["n_steps_obs"] += 1
            if isinstance(m.get("prompt_tokens"), int):
                a["prompt_tokens"] += m["prompt_tokens"]
                a["completion_tokens"] += m.get("completion_tokens") or 0
                a["token_obs"] += 1
            if m.get("first_edit_accepted") is not None:
                a["first_edit_obs"] += 1
                a["first_edit_accepted"] += 1 if m["first_edit_accepted"] else 0
            if m.get("submitted_unmodified_template") is not None:
                a["unmodified_obs"] += 1
                a["unmodified_template"] += 1 if m["submitted_unmodified_template"] else 0
            for k, v in (m.get("reject_reasons") or {}).items():
                a["reject_reasons"][k] = a["reject_reasons"].get(k, 0) + v
    return out


def mde(sd: float, n_per_arm: int, paired_ratio: float = 1.0) -> float:
    """Minimum detectable difference, 80% power, alpha=0.05 two-sided."""
    if n_per_arm < 2 or not math.isfinite(sd):
        return float("nan")
    return (1.96 + 0.8416) * math.sqrt(2.0 / (n_per_arm * paired_ratio)) * sd


def n_for(sd: float, delta: float, paired_ratio: float = 1.0) -> float:
    if delta <= 0 or not math.isfinite(sd):
        return float("nan")
    return 2.0 * ((1.96 + 0.8416) ** 2) * (sd ** 2) / (delta ** 2) / paired_ratio


def cmd_report(args: argparse.Namespace) -> int:
    out_dir = Path(args.out).resolve()
    seal_cells(out_dir)          # idempotent; recovers cells from a killed job
    cells, partial = load_cells(out_dir)
    stats = arm_stats(cells)
    L: list[str] = []
    L.append("=" * 78)
    L.append("MLS-Bench edit-contract comparison — TASK SCORE")
    L.append("=" * 78)
    L.append(f"complete cells: {len(cells)}   excluded (partial/unsealed): {len(partial)}")
    for p in partial[:20]:
        L.append(f"    EXCLUDED  {p}")
    if len(partial) > 20:
        L.append(f"    ... +{len(partial) - 20} more")
    if not cells:
        L.append("\nNo complete cell yet — nothing to report. (A partial run is never quoted.)")
        print("\n".join(L))
        return 0

    fps = {c.get("harness_fingerprint") for c in cells}
    if len(fps) > 1:
        L.append("\n*** HARNESS CHANGED MID-CAMPAIGN — DO NOT POOL THESE CELLS ***")
        for fp in sorted(fps, key=lambda x: str(x)):
            who = sorted({f"{c['arm']}/r{c['rep']:02d}" for c in cells
                          if c.get("harness_fingerprint") == fp})
            L.append(f"    harness {fp}: {len(who)} cells — {', '.join(who[:8])}"
                     + (" ..." if len(who) > 8 else ""))
        L.append("    Re-run the affected arms against one harness before "
                 "quoting any difference.")
    for arm in sorted({c["arm"] for c in cells}):
        afps = {c.get("arm_contract_fingerprint") for c in cells if c["arm"] == arm}
        if len(afps) > 1:
            L.append(f"*** arm {arm} ran under {len(afps)} DIFFERENT contracts "
                     f"{sorted(map(str, afps))} — its cells are not comparable ***")

    den = cells[0].get("denominator", 20)
    L.append(f"\n20-task mean uses a FIXED denominator of {den} "
             f"(never n_scored).")
    L.append("")
    L.append(f"{'arm':18s} {'n':>3} {'mean':>8} {'sd':>8} {'se':>8}   95% CI")
    for arm in sorted(stats):
        s = stats[arm]
        L.append(f"{arm:18s} {s['n_cells']:>3} {s['mean']:>8.4f} {s['sd']:>8.4f} "
                 f"{s['se']:>8.4f}   [{s['ci95'][0]:.4f}, {s['ci95'][1]:.4f}]")

    sens = sensitivity_infra_zero(out_dir, cells[0]["tasks"])
    if any(sens[a]["n"] != stats.get(a, {}).get("n_cells") for a in sens):
        L.append("\nsensitivity — same table with unresolved infra failures counted as 0")
        L.append("(if this disagrees with the headline, one arm is failing more "
                 "often and the exclusion is doing work):")
        for arm in sorted(sens):
            s = sens[arm]
            L.append(f"    {arm:18s} n={s['n']:>3} mean={s['mean']:.4f} sd={s['sd']:.4f}")

    ref = args.reference if args.reference in stats else sorted(stats)[0]
    L.append(f"\npaired differences vs {ref} (pairing on replicate index):")
    L.append(f"{'arm':18s} {'pairs':>5} {'diff':>9} {'se':>8} {'t':>7} {'p':>8}   95% CI"
             "   rho  paired-var-gain")
    for arm in sorted(stats):
        if arm == ref:
            continue
        d = paired_diff(cells, arm, ref)
        if d["n_pairs"] < 2:
            L.append(f"{arm:18s} {d['n_pairs']:>5}   (need >=2 paired reps)")
            continue
        L.append(f"{arm:18s} {d['n_pairs']:>5} {d['diff']:>9.4f} {d['se_diff']:>8.4f} "
                 f"{d['t']:>7.2f} {d['p']:>8.4f}   "
                 f"[{d['ci95'][0]:.4f}, {d['ci95'][1]:.4f}]   "
                 f"{d.get('rho_between_arms', float('nan')):.2f}  "
                 f"{d.get('pairing_variance_ratio', float('nan')):.2f}x")

    # --- power ---------------------------------------------------------------
    sds = [stats[a]["sd"] for a in stats if math.isfinite(stats[a]["sd"])]
    ns = [stats[a]["n_cells"] for a in stats]
    if sds:
        sd_pool = math.sqrt(_mean([s * s for s in sds]))
        L.append(f"\npooled single-run SD of the {den}-task mean: sigma = {sd_pool:.4f} "
                 f"(from n={ns})")
        L.append("power (alpha=0.05 two-sided, 80%), reps PER ARM needed:")
        for delta in (0.030, 0.020, 0.015, 0.010, 0.005):
            L.append(f"    detect delta={delta:.3f} -> n = {n_for(sd_pool, delta):.1f} per arm")
        for n in (4, 8, 12, 16, 20, 24, 32):
            L.append(f"    at n={n:2d} per arm -> MDE = {mde(sd_pool, n):.4f}")

    # --- per-task variance ---------------------------------------------------
    pt = per_task_table(cells)
    if pt["task_sd"]:
        L.append("\nwhere the noise lives (pooled per-task run SD, contribution to "
                 "var of the panel mean):")
        tot = sum(v * v for v in pt["task_sd"].values()) / (den ** 2)
        rows = sorted(pt["task_sd"].items(), key=lambda kv: -kv[1])
        for t, s in rows:
            share = (s * s / (den ** 2)) / tot * 100 if tot > 0 else float("nan")
            L.append(f"    {t:44s} sd={s:.4f}  {share:5.1f}% of panel-mean variance")
        L.append(f"    implied sigma of the panel mean from per-task SDs: "
                 f"{math.sqrt(tot):.4f}")

    # --- does editing even help? ---------------------------------------------
    evt = edit_vs_template_table(cells)
    rows = [(t, v) for t, v in evt.items() if v["edit_delta"] is not None]
    if rows:
        L.append("\ndoes editing help, per task? (score when the UNMODIFIED template "
                 "was submitted vs when the code was changed)")
        L.append(f"{'task':44s} {'n_tpl':>5} {'template':>9} {'n_edit':>6} "
                 f"{'edited':>9} {'delta':>9}")
        for t, v in sorted(rows, key=lambda kv: kv[1]["edit_delta"]):
            L.append(f"{t:44s} {v['n_template']:>5} {v['template_mean']:>9.4f} "
                     f"{v['n_edited']:>6} {v['edited_mean']:>9.4f} "
                     f"{v['edit_delta']:>+9.4f}")
        hurt = [t for t, v in rows if v["edit_delta"] < 0]
        if hurt:
            L.append(f"    editing LOWERS the score on {len(hurt)}/{len(rows)} tasks — "
                     f"on these an arm is rewarded for editing less, so read the "
                     f"contract verdict together with the unmod-template rate above.")
    tpl_only = [(t, v) for t, v in evt.items()
                if v["edit_delta"] is None and v["template_mean"]]
    if tpl_only:
        L.append("    template-only observations (no edited run to compare): "
                 + ", ".join(f"{t}={v['template_mean']:.3f}" for t, v in tpl_only))

    # --- instrumentation -----------------------------------------------------
    inst = instrumentation_table(cells)
    L.append("\ninstrumentation (per arm, pooled over all runs in complete cells):")
    hdr = (f"{'arm':18s} {'runs':>5} {'1st edit ok':>11} {'edit acc':>9} "
           f"{'rejected':>9} {'acc->broken':>11} {'tests':>6} {'wasted syn':>10} "
           f"{'steps/run':>9} {'views':>6} {'undo':>5} {'unmod tpl':>9} {'no submit':>9}")
    L.append(hdr)
    for arm in sorted(inst):
        a = inst[arm]
        pct = lambda k, d: (100.0 * a[k] / a[d]) if a[d] else float("nan")  # noqa: E731
        L.append(
            f"{arm:18s} {a['n_runs']:>5} "
            f"{pct('first_edit_accepted','first_edit_obs'):>10.1f}% "
            f"{pct('edit_accepted','edit_calls'):>8.1f}% "
            f"{pct('edit_rejected','edit_calls'):>8.1f}% "
            f"{(100.0*a['accepted_broken']/a['edit_accepted'] if a['edit_accepted'] else float('nan')):>10.1f}% "
            f"{a['tests_run']:>6} "
            f"{(100.0*a['tests_wasted_syntax']/a['tests_run'] if a['tests_run'] else float('nan')):>9.1f}% "
            f"{(a['steps']/a['n_steps_obs'] if a['n_steps_obs'] else float('nan')):>9.1f} "
            f"{a['view_calls']:>6} {a['undo_calls']:>5} "
            f"{pct('unmodified_template','unmodified_obs'):>8.1f}% "
            f"{(100.0*a['no_submission']/a['n_runs']):>8.1f}%")
    L.append("\ncontext pressure (the arms differ in how much context the contract "
             "costs, and MLS-Bench does not trim history — it 400s and the episode "
             "dies mid-run):")
    L.append(f"{'arm':18s} {'ctx overflow':>12} {'agent crashed':>13} "
             f"{'prompt tok/run':>14} {'completion tok/run':>18}")
    for arm in sorted(inst):
        a = inst[arm]
        L.append(f"{arm:18s} "
                 f"{(100.0*a['context_overflow']/a['n_runs']):>11.1f}% "
                 f"{(100.0*a['agent_crashed']/a['n_runs']):>12.1f}% "
                 f"{(a['prompt_tokens']/a['token_obs'] if a['token_obs'] else float('nan')):>14.0f} "
                 f"{(a['completion_tokens']/a['token_obs'] if a['token_obs'] else float('nan')):>18.0f}")
    for arm in sorted(inst):
        rr = inst[arm]["reject_reasons"]
        if rr:
            L.append(f"    {arm:18s} rejections: " +
                     "  ".join(f"{k}={v}" for k, v in
                               sorted(rr.items(), key=lambda kv: -kv[1])))

    txt = "\n".join(L)
    print(txt)
    (out_dir / "report.txt").write_text(txt + "\n")
    atomic_write_json(out_dir / "report.json", {
        "n_complete_cells": len(cells), "excluded": partial,
        "denominator": den, "arms": stats,
        "paired_vs_reference": {a: paired_diff(cells, a, ref)
                                for a in stats if a != ref},
        "reference": ref,
        "per_task": pt, "instrumentation": inst,
        "edit_vs_template": evt,
    })
    return 0


def cmd_reharvest(args: argparse.Namespace) -> int:
    """Recompute instrumentation for banked cells from the agent logs.

    Instrumentation is a pure function of MLS-Bench's RunLogger output, which
    survives long after the run. So an improvement to what we measure can be
    applied to cells that are already on disk, with no GPU time at all --
    important here, because a campaign takes many hours and a metric bug found
    at hour six should not cost the first six hours of runs.

    Fields that can only be read from the (deleted) workspace --
    final_equals_template, submitted_unmodified_template -- are carried over
    from the banked record rather than recomputed as None.
    """
    root = Path(args.root).resolve()
    out_dir = Path(args.out).resolve()
    carry = ("final_equals_template", "submitted_unmodified_template",
             "context_overflow", "agent_crashed", "template_check_error")
    n_cells = n_recs = 0
    for cell_json in sorted(out_dir.glob("cells/*/r*/cell.json")):
        c = json.loads(cell_json.read_text())
        for r in c.get("records", []):
            old_m = r.get("metrics") or {}
            log_dir = old_m.get("agent_log_dir")
            if not log_dir:
                continue
            label = Path(log_dir).parent.name.split("__", 1)[-1]
            new_m = harvest_instrumentation(root, r["task"], label, Path("/nonexistent"))
            if not new_m.get("instrumented"):
                continue
            for k in carry:
                if k in old_m:
                    new_m[k] = old_m[k]
            r["metrics"] = new_m
            n_recs += 1
        # per-task json files too, so a resumed run stays consistent with the seal
        for r in c.get("records", []):
            atomic_write_json(
                cell_json.parent / "tasks" / f"{r['task']}.json", r)
        atomic_write_json(cell_json, c)
        n_cells += 1
    log(f"reharvested {n_recs} records across {n_cells} sealed cells "
        f"(scores untouched)")
    return 0


def cmd_served_names(args: argparse.Namespace) -> int:
    """Emit every model name vLLM must register.

    `mlsbench` sends the model string with its `vllm/` routing prefix stripped
    (agent/models.py build_client), so vLLM has to know the bare name; older
    checkouts send the prefixed one. Register both, for every cell.
    """
    arms = load_arms(args.arms_file)
    names = []
    for arm in (args.arms or list(arms)):
        if arm not in arms:
            continue
        for rep in range(args.rep_start, args.rep_start + args.reps):
            n = cell_model(args.tag, arm, rep)
            names += [n, f"vllm/{n}"]
    print(" ".join(names))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--tag", required=True, help="model tag (matches vLLM served names)")
        p.add_argument("--arms", nargs="*", default=None)
        p.add_argument("--arms-file", default=None,
                       help="JSON {arm: {use_replace, env, desc}} — how a NEW "
                            "contract joins the comparison without a code change")
        p.add_argument("--reps", type=int, default=2)
        p.add_argument("--rep-start", type=int, default=0)

    p = sub.add_parser("run")
    common(p)
    p.add_argument("--out", required=True)
    p.add_argument("--root", default=os.environ.get(
        "MLSBENCH_ROOT", "/scratch/gpfs/CHIJ/bohan/MLS-Bench-dev"))
    p.add_argument("--config", required=True)
    p.add_argument("--tasks", nargs="*", default=None)
    p.add_argument("--concurrency", type=int, default=20)
    p.add_argument("--timeout", type=int, default=5400)
    p.add_argument("--max-attempts", type=int, default=2,
                   help="retries for INFRASTRUCTURE failures only")
    p.add_argument("--python", default=os.environ.get(
        "MLSBENCH_PY", "/home/bl3615/miniconda3/bin/python"))
    p.add_argument("--workspace-root", default=None)
    p.add_argument("--keep-workspace", action="store_true")
    p.add_argument("--seed-base", type=int, default=20260808)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("report")
    p.add_argument("--out", required=True)
    p.add_argument("--reference", default="linerange")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("reharvest")
    p.add_argument("--out", required=True)
    p.add_argument("--root", default=os.environ.get(
        "MLSBENCH_ROOT", "/scratch/gpfs/CHIJ/bohan/MLS-Bench-dev"))
    p.set_defaults(func=cmd_reharvest)

    p = sub.add_parser("served-names")
    common(p)
    p.set_defaults(func=cmd_served_names)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
