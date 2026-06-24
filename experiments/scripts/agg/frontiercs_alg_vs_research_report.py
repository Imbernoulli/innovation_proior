#!/usr/bin/env python3
"""
Report Algorithm-only vs Algorithm+Research Frontier-CS scores side by side.

Reads, per model:
  - the ALGORITHM-track summary (metrics.frontiercs.score: mean@5 / best@5/mean)
  - the RESEARCH-track summary (metrics.frontiercs_research.score: mean@N / best@N/mean)
and prints them as SEPARATE tracks (never hard-averaged), matching the
leaderboard convention (Algorithmic Avg@k / Score@k and Research Avg@k / Score@k
are reported independently).

Usage:
  python scripts/frontiercs_alg_vs_research_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

# logical model -> (algorithm summary dir, research summary dir)
MODELS = {
    "q35_inst_start (Qwen3.5-9B-bf16)": (
        "cc_eval_q35_inst_start_thinking_32k_both_vllm",
        "cc_eval_q35_inst_start_research_thinking_32k_vllm",
    ),
    "q35_sft (sft_q35_a100_method)": (
        "cc_eval_q35_a100_method_thinking_32k_both_vllm",
        "cc_eval_q35_sft_research_thinking_32k_vllm",
    ),
    "q35_soup10 (soup_q35_a100_method_soupa10)": (
        "cc_eval_q35_a100_method_soupa10_thinking_32k_both_vllm",
        "cc_eval_q35_soup10_research_thinking_32k_vllm",
    ),
    "q3_inst_start (Qwen3-8B)": (
        "cc_eval_q3_inst_start_thinking_32k_both_vllm",
        "cc_eval_q3_inst_start_research_thinking_32k_vllm",
    ),
    "q3_sft (sft_q3_a100_method)": (
        "cc_eval_q3_a100_method_thinking_32k_both_vllm",
        "cc_eval_q3_sft_research_thinking_32k_vllm",
    ),
    "q3_soup10 (soup_q3_a100_method_soupa10)": (
        "cc_eval_q3_a100_method_soupa10_thinking_32k_both_vllm",
        "cc_eval_q3_soup10_research_thinking_32k_vllm",
    ),
}


def _score_block(summary_path: Path, source: str):
    if not summary_path.exists():
        return None
    try:
        s = json.loads(summary_path.read_text())
    except Exception:
        return None
    sc = s.get("metrics", {}).get(source, {}).get("score", {})
    if not sc:
        return None
    mean = next((sc[k] for k in sc if k.startswith("mean@")), None)
    best = next((sc[k] for k in sc if k.startswith("best@") and k.endswith("/mean")), None)
    n = s.get("complete_problem_count")
    return {"mean@k": mean, "best@k": best, "n_problems": n}


def main() -> None:
    print(f"{'model':<42} {'ALG mean@5':>11} {'ALG best@5':>11} "
          f"{'RES mean@N':>11} {'RES best@N':>11} {'RES n':>6}")
    print("-" * 100)
    for name, (alg_dir, res_dir) in MODELS.items():
        alg = _score_block(OUT / alg_dir / "summary.json", "frontiercs")
        res = _score_block(OUT / res_dir / "summary.json", "frontiercs_research")
        am = f"{alg['mean@k']:.3f}" if alg and alg["mean@k"] is not None else "--"
        ab = f"{alg['best@k']:.3f}" if alg and alg["best@k"] is not None else "--"
        rm = f"{res['mean@k']:.3f}" if res and res["mean@k"] is not None else "--"
        rb = f"{res['best@k']:.3f}" if res and res["best@k"] is not None else "--"
        rn = f"{res['n_problems']}" if res and res["n_problems"] is not None else "--"
        print(f"{name:<42} {am:>11} {ab:>11} {rm:>11} {rb:>11} {rn:>6}")
    print("\nNote: ALG = algorithmic track (172 C++ problems, full.parquet).")
    print("RES = research track GPU subset (21 Triton problems) unless run with the")
    print("full research.parquet. Tracks are reported SEPARATELY (not averaged).")


if __name__ == "__main__":
    main()
