#!/usr/bin/env python3
"""Prepare VERL parquet data for the new ThetaEvolve + TTT-Discover RL tasks.

These are single-instance optimization problems, so each task is one prompt
replicated across N seeds (the seed is injected into the prompt + extra_info so
rollouts differ and GRPO has a usable train set). Schema mirrors the existing
frontiercs/alebench parquets:
    prompt        : list[{"role": "user", "content": ...}]
    reward_model  : {"ground_truth": <data_source>}
    data_source   : str  (thetaevolve_ac3 | thetaevolve_circle | ttt_erdos)
    extra_info    : {"seed": int, "task": str}   (extra; ignored by VERL if unused)

Outputs under data/thetaevolve_ttt/:
    train_thetaevolve_ttt.parquet            (all 3 tasks mixed)
    val_thetaevolve_ttt.parquet              (1 row per task)
    train_thetaevolve_ttt_smoke.parquet      (small, for the smoke job)

Usage:
    python scripts/prepare_thetaevolve_ttt_parquet.py
    python scripts/prepare_thetaevolve_ttt_parquet.py --n 24 --smoke-n 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "thetaevolve_ttt"


# --------------------------- prompts (thinking ok) --------------------------- #
THETAEVOLVE_AC3_PROMPT = """You are an expert in harmonic analysis, extremal problems, and numerical optimization. Think step by step about the construction, then write the program.

# Task: Third autocorrelation inequality (minimize the C3 constant)

Design a Python program that constructs a discrete function f on the domain [-1/4, 1/4], represented as an array of `heights`, to MINIMIZE

    C3 = 2 * n * max(|conv(f, f)|) / (sum(f))^2

where `n = len(heights)` and `conv(f, f) = numpy.convolve(heights, heights)`. Lower C3 is better. The current state of the art is C3 = 1.4557; a Gaussian-with-oscillation baseline gives about 3.16, so there is large headroom.

## Output contract
Write a complete Python program inside a single ```python code fence. When run as a script (under `if __name__ == "__main__":`) it MUST save its candidate solution by calling:

    from openevolve.modular_utils.file_io_controller import save_search_results
    save_search_results(best_perfect_solution=None,
                        current_solution={'data': heights_list},
                        target_value=1.4557)

where `heights_list` is your `heights` array as a Python list (`heights.tolist()`). Do NOT print anything else as the answer; the saved `current_solution['data']` is what gets scored. Use only numpy / standard library. Keep total runtime under 60 seconds (random seed for this instance: __SEED__).
"""

THETAEVOLVE_CIRCLE_PROMPT = """You are an expert in computational geometry and packing optimization. Reason about the arrangement, then write the program.

# Task: Circle packing in the unit square, n = 26 (maximize sum of radii)

Design a Python program that places 26 non-overlapping circles inside the unit square [0,1] x [0,1] to MAXIMIZE the sum of their radii. Circles must stay inside the square and must not overlap. Higher sum-of-radii is better; the AlphaEvolve target is 2.635, and a naive concentric-rings baseline reaches about 0.96, so there is large headroom.

## Output contract
Write a complete Python program inside a single ```python code fence. When run as a script (under `if __name__ == "__main__":`) it MUST save its candidate by calling:

    from openevolve.modular_utils.file_io_controller import save_search_results
    save_search_results(best_perfect_solution=None,
                        current_solution={'data': (centers_list, radii_list)},
                        n_circles=26, target_value=2.635)

where `centers_list` is a list of 26 [x, y] pairs (`centers.tolist()`) and `radii_list` is a list of 26 radii (`radii.tolist()`). The saved `current_solution['data'] = (centers, radii)` is what gets scored (overlaps / boundary violations score 0). Use only numpy / standard library. Keep runtime under 60 seconds (random seed for this instance: __SEED__).
"""

TTT_ERDOS_PROMPT = """You are an expert in harmonic analysis, numerical optimization, and mathematical discovery. Reason carefully, then give your answer.

# Task: Erdos minimum-overlap problem (minimize the C5 bound)

Find a step function h: [0, 2] -> [0, 1] that MINIMIZES the overlap integral

    C5 = max_k integral of h(x) * (1 - h(x+k)) dx

Constraints: 0 <= h(x) <= 1 for all x, and integral over [0,2] of h(x) dx = 1.

Discretize h as `n_points` samples over [0, 2] with dx = 2.0 / n_points, so that:
  - 0 <= h[i] <= 1 for all i
  - sum(h) == n_points / 2   (equivalently sum(h) * dx == 1)
The evaluation computes C5 = max(numpy.correlate(h, 1 - h, mode="full") * dx). Lower C5 is better. The current record is about C5 = 0.3808; a constant h = 0.5 gives C5 = 0.5, so there is meaningful headroom. Smaller sequences (a few hundred samples or fewer) are preferred.

## Output contract
Give your final answer as EITHER:
  (a) the sequence h directly, as a JSON array of floats inside a single ```json code fence, e.g. ```json\n[0.5, 0.49, ...]\n```; OR
  (b) a Python program inside a single ```python code fence that defines `run(seed=42, budget_s=1000)` returning `(h_values, c5_bound, n_points)` (or that assigns a 1D `sequence`/`h` array). It will be executed (numpy/scipy only, no file/network IO) and the produced sequence will be scored.

Either way only the numeric sequence is scored, and the objective is recomputed independently. Random seed for this instance: __SEED__.
"""

TTT_AC1_PROMPT = """You are an expert in harmonic analysis, numerical optimization, and mathematical discovery. Reason carefully, then give your answer.

# Task: First autocorrelation inequality (minimize the C1 upper bound)

Find a non-negative sequence `xs` (a discretized step function f >= 0) that MINIMIZES the autocorrelation upper-bound constant

    C1 = 2 * n * max(conv(xs, xs)) / (sum(xs))^2

where `n = len(xs)` and `conv(xs, xs) = numpy.convolve(xs, xs)` (full mode). Each entry must satisfy 0 <= xs[i] <= 1000. Lower C1 is better. The current record is about C1 = 1.5029; a flat/box sequence gives about C1 = 2.0, so there is meaningful headroom. Smaller sequences are preferred.

## Output contract
Give your final answer as EITHER:
  (a) the sequence `xs` directly, as a JSON array of non-negative floats inside a single ```json code fence, e.g. ```json\\n[1.0, 0.99, ...]\\n```; OR
  (b) a Python program inside a single ```python code fence that defines `run(seed=42, budget_s=1000)` returning the sequence (or that assigns a 1D `sequence`/`xs` array). It will be executed (numpy/scipy only, no file/network IO) and the produced sequence will be scored.

Either way only the numeric sequence is scored, and C1 is recomputed independently. Random seed for this instance: __SEED__.
"""

TTT_AC2_PROMPT = """You are an expert in harmonic analysis, numerical optimization, and mathematical discovery. Reason carefully, then give your answer.

# Task: Second autocorrelation inequality (maximize the C2 lower bound)

Find a non-negative sequence `xs` (a discretized step function f >= 0) that MAXIMIZES the autoconvolution lower-bound ratio

    C2 = ||conv||_2^2 / (||conv||_1 * ||conv||_inf)

where `conv = numpy.convolve(xs, xs)` (full mode), the L2 norm-squared is the Simpson piecewise-linear integral of conv^2 over [-1/2, 1/2] (with zero padding at both ends), `||conv||_1 = sum(|conv|) / (len(conv) + 1)`, and `||conv||_inf = max(|conv|)`. Each entry must satisfy 0 <= xs[i] <= 1000. Higher C2 is better. The current record is about C2 = 0.9592; a trivial box gives about C2 = 0.80, so there is meaningful headroom.

## Output contract
Give your final answer as EITHER:
  (a) the sequence `xs` directly, as a JSON array of non-negative floats inside a single ```json code fence, e.g. ```json\\n[1.0, 0.98, ...]\\n```; OR
  (b) a Python program inside a single ```python code fence that defines `run(seed=42, budget_s=1000)` returning the sequence (or that assigns a 1D `sequence`/`xs` array). It will be executed (numpy/scipy only, no file/network IO) and the produced sequence will be scored.

Either way only the numeric sequence is scored, and C2 is recomputed independently. Random seed for this instance: __SEED__.
"""

_TASKS = {
    "thetaevolve_ac3": THETAEVOLVE_AC3_PROMPT,
    "thetaevolve_circle": THETAEVOLVE_CIRCLE_PROMPT,
    "ttt_erdos": TTT_ERDOS_PROMPT,
    "ttt_ac1": TTT_AC1_PROMPT,
    "ttt_ac2": TTT_AC2_PROMPT,
}


def _row(data_source: str, prompt_tmpl: str, seed: int) -> dict:
    return {
        "prompt": [{"role": "user", "content": prompt_tmpl.replace("__SEED__", str(seed))}],
        "reward_model": {"ground_truth": data_source},
        "data_source": data_source,
        "extra_info": {"seed": seed, "task": data_source},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24, help="rows per task in the train set")
    ap.add_argument("--smoke-n", type=int, default=4, help="rows per task in the smoke set")
    ap.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_rows, val_rows, smoke_rows = [], [], []
    for ds, tmpl in _TASKS.items():
        for s in range(args.n):
            train_rows.append(_row(ds, tmpl, s))
        val_rows.append(_row(ds, tmpl, 10_000))
        for s in range(args.smoke_n):
            smoke_rows.append(_row(ds, tmpl, s))

    for name, rows in [
        ("train_thetaevolve_ttt", train_rows),
        ("val_thetaevolve_ttt", val_rows),
        ("train_thetaevolve_ttt_smoke", smoke_rows),
    ]:
        df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
        out = args.output_dir / f"{name}.parquet"
        df.to_parquet(out, index=False)
        counts = df["data_source"].value_counts().to_dict()
        print(f"Saved {len(df)} rows -> {out}  ({counts})")


if __name__ == "__main__":
    main()
