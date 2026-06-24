#!/usr/bin/env python3
"""
Prepare Frontier-CS problems as Parquet for VERL RL training / eval.

ALGORITHMIC track (default): numeric-ID C++ problems -> data_source="frontiercs".
  Creates train.parquet / val.parquet / full.parquet.

RESEARCH track (--research): the 68 research problems (Python `Solution` class,
  scored by the official evaluator.py) -> data_source="frontiercs_research".
  Creates research.parquet. ground_truth is the research problem id (the nested
  path under research/problems, e.g. "flash_attn" or "gemm_optimization/squares").

Usage:
  python scripts/prepare_frontiercs_parquet.py                 # algorithmic train/val
  python scripts/prepare_frontiercs_parquet.py --full-for-both # algorithmic full.parquet
  python scripts/prepare_frontiercs_parquet.py --research      # research.parquet (all 68)
  python scripts/prepare_frontiercs_parquet.py --research --research-cpu-only  # 47 CPU-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
PROBLEMS_DIR = PROJECT_ROOT / "Frontier-CS" / "algorithmic" / "problems"
DEFAULT_OUT = PROJECT_ROOT / "data" / "frontiercs"


def build_prompt(statement: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": f"""You are a competitive programmer. Solve the following problem in C++. Output ONLY the C++ code wrapped in ```cpp and ```. No explanation.

{statement}

Generate solution code:""",
        }
    ]


def build_research_rows(cpu_only: bool, gpu_only: bool = False) -> list[dict]:
    """Build research-track parquet rows. ground_truth = research problem id.

    Reuses scripts/frontiercs_research_eval.py for enumeration + the canonical
    (system + readme) prompt, so the parquet prompt matches exactly what the
    request script / official generator would send.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from frontiercs_research_eval import (
        build_research_messages,
        list_research_problems,
        problem_needs_gpu,
    )

    rows: list[dict] = []
    for pid in list_research_problems():
        gpu = problem_needs_gpu(pid)
        if cpu_only and gpu:
            continue
        if gpu_only and not gpu:
            continue
        rows.append(
            {
                "prompt": build_research_messages(pid),
                "reward_model": {"ground_truth": pid},
                "data_source": "frontiercs_research",
                "extra_info": {"needs_gpu": bool(problem_needs_gpu(pid))},
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Fraction for validation")
    parser.add_argument("--full-for-both", action="store_true", help="Use all problems for both train and val (creates full.parquet)")
    parser.add_argument("--research", action="store_true", help="Build the RESEARCH-track parquet (research.parquet) instead of algorithmic")
    parser.add_argument("--research-cpu-only", action="store_true", help="With --research, include only CPU-only research problems")
    parser.add_argument("--research-gpu-only", action="store_true", help="With --research, include only GPU research problems (self-contained Triton)")
    parser.add_argument("--research-out", type=str, default="research.parquet", help="Filename for the research parquet")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.research:
        rows = build_research_rows(cpu_only=args.research_cpu_only, gpu_only=args.research_gpu_only)
        if not rows:
            print("No research problems found.")
            return
        df = pd.DataFrame(rows)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out = args.output_dir / args.research_out
        df.to_parquet(out, index=False)
        n_gpu = sum(1 for r in rows if r["extra_info"]["needs_gpu"])
        print(f"Saved {len(df)} research problems ({n_gpu} GPU, {len(df)-n_gpu} CPU) -> {out}")
        return

    if args.full_for_both:
        args.val_ratio = 0.0

    rows = []
    for pid_dir in sorted(PROBLEMS_DIR.iterdir()):
        if not pid_dir.is_dir():
            continue
        if not pid_dir.name.isdigit():
            continue
        stmt = pid_dir / "statement.txt"
        if not stmt.exists():
            continue
        statement = stmt.read_text(encoding="utf-8")
        rows.append(
            {
                "prompt": build_prompt(statement),
                "reward_model": {"ground_truth": pid_dir.name},
                "data_source": "frontiercs",
            }
        )

    if not rows:
        print("No numeric-ID problems found.")
        return

    df = pd.DataFrame(rows)
    if args.val_ratio > 0:
        df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
        n_val = max(1, int(len(df) * args.val_ratio))
        df_val = df.iloc[:n_val]
        df_train = df.iloc[n_val:]
    else:
        df_train = df
        df_val = df.iloc[:0]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.parquet"
    val_path = args.output_dir / "val.parquet"

    df_train.to_parquet(train_path, index=False)
    print(f"Saved {len(df_train)} train -> {train_path}")
    if len(df_val) > 0:
        df_val.to_parquet(val_path, index=False)
        print(f"Saved {len(df_val)} val -> {val_path}")

    if args.full_for_both:
        full_path = args.output_dir / "full.parquet"
        df.to_parquet(full_path, index=False)
        print(f"Saved {len(df)} full (use for both train & val) -> {full_path}")


if __name__ == "__main__":
    main()
