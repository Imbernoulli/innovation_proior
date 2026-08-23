#!/usr/bin/env python3
"""Offline checks for the edit-CONTRACT A/B (no GPU, no containers, seconds).

Two things must hold before any GPU time is spent:

  1. The four arms really are four different contracts. An arm that silently
     collapses onto another one would produce a null result that looks like
     evidence. So assert, per arm, on the tool schemas the model is shown AND
     the system prompt text, plus the matcher behaviour behind them.

  2. The statistics in `report` are right, including the fixed denominator and
     the refusal to quote a partial cell.

Run:  python scripts/test_mlsbench_contract_arms.py [--root <MLSBENCH_ROOT>]
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

FAILS: list[str] = []


def check(name: str, cond, extra: str = "") -> None:
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}  {extra}")
        FAILS.append(name)


def arm_view(root: Path, use_replace: bool, env: dict) -> tuple[set[str], str]:
    """Instantiate InteractiveAgent under one arm's env and return
    (tool names shown to the model, system prompt)."""
    for k in ("MLSBENCH_STRICT_STR_REPLACE", "MLSBENCH_VIEW_TOOL",
              "MLSBENCH_LINERANGE_SCHEMA"):
        os.environ.pop(k, None)
    os.environ.update({k: str(v) for k, v in env.items()})

    sys.path.insert(0, str(root / "src"))
    for m in [m for m in list(sys.modules) if m.startswith("mlsbench")]:
        del sys.modules[m]
    interactive = importlib.import_module("mlsbench.agent.interactive")

    def fake_base_init(self, task_name, global_config, workspace_root=None):
        self.task_name = task_name

    with mock.patch.object(interactive.BaseAgent, "__init__", fake_base_init), \
         mock.patch.object(interactive, "build_client", lambda cfg: object()):
        agent = interactive.InteractiveAgent("t", {"use_replace": use_replace})
    names = {s["name"] for s in agent._tool_schemas}
    return names, agent.system_prompt, list(agent._tool_schemas)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get(
        "MLSBENCH_ROOT", "/scratch/gpfs/CHIJ/bohan/MLS-Bench-dev"))
    args = ap.parse_args()
    root = Path(args.root).resolve()

    ce = importlib.import_module("mlsbench_contract_eval")

    print("arms are genuinely distinct contracts:")
    seen: dict[str, tuple] = {}
    for arm, spec in ce.DEFAULT_ARMS.items():
        names, prompt, agent_schemas = arm_view(root, spec["use_replace"], spec["env"])
        has_view = "view" in names
        # Which edit op vocabulary is the model shown?
        ops = "str_replace" if "str_replace" in prompt else "linerange"
        edit_desc = next((json.dumps(sc) for sc in agent_schemas
                          if sc["name"] == "edit"), "")
        seen[arm] = (ops, has_view, prompt, edit_desc)
        print(f"    {arm:18s} ops={ops:11s} view={has_view}  tools={sorted(names)}")

    check("linerange shows the line-range op, no view",
          seen["linerange"][0] == "linerange" and not seen["linerange"][1])
    check("replace_strict shows str_replace and NO view",
          seen["replace_strict"][0] == "str_replace" and not seen["replace_strict"][1])
    check("replace_fx shows str_replace and NO view",
          seen["replace_fx"][0] == "str_replace" and not seen["replace_fx"][1])
    check("replace_fx_view shows str_replace AND view",
          seen["replace_fx_view"][0] == "str_replace" and seen["replace_fx_view"][1])
    check("strict restores the upstream prompt wording",
          "must match exactly" in seen["replace_strict"][2]
          and "Keep old_str SHORT" not in seen["replace_strict"][2])
    check("patched arms carry the short-anchor guidance",
          "Keep old_str SHORT" in seen["replace_fx"][2]
          and "Keep old_str SHORT" in seen["replace_fx_view"][2])
    check("no-view arm does not advertise view() in the prompt",
          "view(filename" not in seen["replace_fx"][2])
    # Distinctness must include the edit schema itself: two arms whose prompts
    # match but whose tool schemas differ are still two contracts, and two arms
    # that match on BOTH would be one arm counted twice.
    core = ["linerange", "replace_strict", "replace_fx", "replace_fx_view"]
    check("the four default arms differ pairwise",
          len({seen[a] for a in core}) == len(core))

    # rewrite_view needs scripts/mlsbench_edit_contract.diff applied to the
    # checkout. Until then MLSBENCH_REWRITE_OP is inert and the arm collapses
    # onto replace_fx_view. State which world we are in rather than failing:
    # the runner's own preflight is what refuses to spend GPU time on a
    # collapsed arm.
    rewrite_live = seen["rewrite_view"] != seen["replace_fx_view"]
    print(f"\n  rewrite_view is {'LIVE' if rewrite_live else 'INERT'} in {root}"
          f" ({'diff applied' if rewrite_live else 'apply mlsbench_edit_contract.diff to enable'})")
    check("the line-range CONTROL uses the public one-line op description",
          "REQUIRED companion" not in seen["linerange"][3]
          and "REQUIRED companion" in seen["linerange_fx"][3],
          "control schema is not the public one")

    print("\nmatcher behind the arms:")
    sys.path.insert(0, str(root / "src"))
    for m in [m for m in list(sys.modules) if m.startswith("mlsbench")]:
        del sys.modules[m]
    tools = importlib.import_module("mlsbench.agent.tools")
    FILE = "class M:\n    def fit(self, X):\n        return X\n"
    ANCHOR = "    def fit(self,  X):"          # doubled internal space
    os.environ["MLSBENCH_STRICT_STR_REPLACE"] = "1"
    check("strict matcher rejects a whitespace-normalised anchor",
          not tools.resolve_old_str(FILE, ANCHOR)["ok"])
    os.environ.pop("MLSBENCH_STRICT_STR_REPLACE")
    check("patched matcher accepts it",
          tools.resolve_old_str(FILE, ANCHOR)["ok"])

    print("\nreport statistics:")
    # fixed denominator: 20 tasks, 4 non-zero -> mean must be sum/20, not sum/4
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        tasks = ce.CPU_TASKS
        truth = {}
        for arm, base in (("A", 0.05), ("B", 0.08)):
            for rep in range(4):
                recs = []
                for i, t in enumerate(tasks):
                    s = (base + 0.01 * rep) * 20 / 4 if i < 4 else 0.0
                    recs.append({"task": t, "status": "scored", "score": s,
                                 "metrics": {"n_edit_calls": 2, "n_edit_accepted": 1,
                                             "n_edit_rejected": 1, "n_steps": 12,
                                             "n_accepted_left_file_broken": 0,
                                             "n_test_calls": 3,
                                             "n_tests_run": 3,
                                             "n_tests_wasted_on_syntax": 1,
                                             "first_edit_accepted": True,
                                             "submitted_unmodified_template": False,
                                             "reject_reasons": {"ambiguous": 1}}})
                mean = sum(r["score"] for r in recs) / len(tasks)
                truth[(arm, rep)] = mean
                ce.atomic_write_json(out / "cells" / arm / f"r{rep:02d}" / "cell.json", {
                    "arm": arm, "rep": rep, "tag": "t", "tasks": tasks,
                    "denominator": len(tasks), "complete": True,
                    "mean_score": mean, "records": recs})
        # one partial cell that must be excluded, not averaged
        ce.atomic_write_json(
            out / "cells" / "A" / "r09" / "tasks" / f"{tasks[0]}.json",
            {"task": tasks[0], "status": "scored", "score": 9.9})

        cells, partial = ce.load_cells(out)
        check("partial cell excluded", len(cells) == 8 and len(partial) == 1,
              f"{len(cells)} cells, {len(partial)} partial")
        st = ce.arm_stats(cells)
        check("fixed denominator honoured",
              abs(st["A"]["mean"] - sum(truth[("A", r)] for r in range(4)) / 4) < 1e-12,
              f"{st['A']['mean']}")
        check("mean is sum/20 not sum/n_nonzero",
              abs(st["A"]["mean"] - (0.05 + 0.06 + 0.07 + 0.08) / 4) < 1e-12,
              f"{st['A']['mean']}")
        d = ce.paired_diff(cells, "B", "A")
        check("paired diff = +0.03", abs(d["diff"] - 0.03) < 1e-12, str(d["diff"]))
        check("perfectly correlated arms -> pairing kills the variance",
              d["se_diff"] < 1e-9 and d.get("rho_between_arms", 0) > 0.999,
              f"se={d['se_diff']} rho={d.get('rho_between_arms')}")
        inst = ce.instrumentation_table(cells)
        check("instrumentation counts all runs", inst["A"]["n_runs"] == 4 * len(tasks))
        check("wasted-syntax tests tallied",
              inst["A"]["tests_wasted_syntax"] == 4 * len(tasks))

    # p-value sanity against a known value: t=2.101, df=18 -> p ~= 0.05
    check("t-distribution p-value is calibrated",
          abs(ce._t_sf(2.101, 18) - 0.05) < 0.002, f"{ce._t_sf(2.101, 18):.4f}")
    check("n_for matches the closed form",
          abs(ce.n_for(0.021, 0.02) - 2 * (1.96 + 0.8416) ** 2 * 0.021 ** 2 / 0.02 ** 2) < 1e-9)

    print(f"\n{'ALL PASS' if not FAILS else str(len(FAILS)) + ' FAILURES: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
