#!/usr/bin/env python3
"""Flatten an OpenEvolve / ThetaEvolve run directory into a single summary.json.

OpenEvolve writes the best evolved program's metrics to
  <run_dir>/best/best_program_info.json
with at least {"metrics": {"combined_score": ..., "objective_value": ...}}.
For ThetaEvolve tasks, combined_score == objective_value == the task's reported
score (sum of radii for circle packing, C-bound for autocorrelation tasks, ...).

This script reads that file and emits a small, eval-harness-friendly summary so
callers can grab one number without knowing OpenEvolve internals.

Usage:
  python parse_openevolve_best.py --run-dir <dir> --task <task> --tag <tag> \
      --out <dir>/summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_all_programs(run_dir: Path) -> dict:
    """Collect every evaluated program from the latest checkpoint's programs_bulk.

    Returns {program_id: program_dict}. Empty dict if none found.
    """
    cps = sorted((run_dir / "checkpoints").glob("checkpoint_*/programs_bulk.json"),
                 key=lambda p: int(p.parent.name.split("_")[-1]))
    if not cps:
        return {}
    try:
        return json.loads(cps[-1].read_text())
    except Exception:
        return {}


def _discrimination(run_dir: Path) -> dict:
    """Compute whether the reported best came from the MODEL or is just the seed.

    The eval only faithfully measures discovery ability if a model-produced
    program beats the iteration-0 seed. If the best is the seed (the model never
    produced a valid, superior diff), the headline score is model-independent and
    must NOT be read as a discovery signal. We surface that explicitly.
    """
    progs = _load_all_programs(run_dir)
    out = {
        "seed_score": None,            # combined_score of iteration-0 / parent-less program
        "best_model_score": None,      # best combined_score among model-produced programs
        "best_is_seed": None,          # True if reported best == the seed (no model gain)
        "model_beat_seed": None,       # True if any valid model program > seed
        "model_improvement_over_seed": None,
        "num_programs": len(progs) or None,
        "num_valid_model_programs": None,   # model programs with validity==1.0
        "num_model_programs": None,         # model programs that parsed+ran at all
    }
    if not progs:
        return out

    def cs(p):
        return (p.get("metrics") or {}).get("combined_score")

    # Seed = generation 0 and/or parent_id None and/or iteration_found 0.
    seed = None
    model_scores = []
    n_valid = 0
    n_model = 0
    for p in progs.values():
        is_seed = (p.get("generation") == 0 and p.get("parent_id") in (None, "")) \
            or p.get("iteration_found") == 0 and p.get("parent_id") in (None, "")
        if is_seed and seed is None:
            seed = p
            continue
        n_model += 1
        sc = cs(p)
        if sc is not None:
            model_scores.append(sc)
        if (p.get("metrics") or {}).get("validity") == 1.0:
            n_valid += 1

    out["num_model_programs"] = n_model
    out["num_valid_model_programs"] = n_valid
    if seed is not None:
        out["seed_score"] = cs(seed)
    if model_scores:
        out["best_model_score"] = max(model_scores)
    if out["seed_score"] is not None and out["best_model_score"] is not None:
        out["model_improvement_over_seed"] = out["best_model_score"] - out["seed_score"]
        out["model_beat_seed"] = out["best_model_score"] > out["seed_score"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--task", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    best = args.run_dir / "best" / "best_program_info.json"
    if not best.is_file():
        # Fall back to the latest checkpoint's best if the top-level one is absent.
        cps = sorted((args.run_dir / "checkpoints").glob("checkpoint_*/best_program_info.json"))
        if cps:
            best = cps[-1]
    if not best.is_file():
        print(f"ERROR: no best_program_info.json under {args.run_dir}", file=sys.stderr)
        return 2

    info = json.loads(best.read_text())
    metrics = info.get("metrics", {})

    # Faithfulness instrumentation: did the MODEL drive the best, or is it the seed?
    disc = _discrimination(args.run_dir)
    # The reported best is the seed iff it is iteration 0 / generation 0 with no parent.
    best_is_seed = (info.get("generation") == 0 and info.get("parent_id") in (None, "")) \
        or info.get("iteration") == 0 and info.get("parent_id") in (None, "")
    disc["best_is_seed"] = bool(best_is_seed)

    summary = {
        "tag": args.tag,
        "task": args.task,
        "run_dir": str(args.run_dir),
        "best_program_info": str(best),
        "iteration": info.get("iteration"),
        "generation": info.get("generation"),
        # The headline number to read:
        "best_combined_score": metrics.get("combined_score"),
        "best_objective_value": metrics.get("objective_value"),
        "validity": metrics.get("validity"),
        "rl_normalized_reward": metrics.get("rl_normalized_reward"),
        # Discovery-faithfulness signal. If model_beat_seed is False (or
        # best_is_seed is True), the headline score is the model-independent
        # SEED and does NOT measure this model's discovery ability.
        "discrimination": disc,
        "metrics": metrics,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    if disc.get("best_is_seed") or disc.get("model_beat_seed") is False:
        print(
            "WARNING: reported best is the SEED program (model did not beat it); "
            "this score is model-independent and must not be read as a discovery signal.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
