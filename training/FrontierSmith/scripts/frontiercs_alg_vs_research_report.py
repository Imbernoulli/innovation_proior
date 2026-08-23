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

# logical model -> {alg, res_gpu, res_cpu} output-dir names under outputs/.
# res_gpu = prior 21-Triton run; res_cpu = the 43 CPU-problem run (research_cpu.parquet).
# "all research" is computed by pooling the per-problem scores from both summaries.
MODELS = {
    "q35_inst_start (Qwen3.5-9B-bf16)": {
        "alg": "cc_eval_q35_inst_start_thinking_32k_both_vllm",
        "res_gpu": "cc_eval_q35_inst_start_research_research_thinking_32k_vllm",
        "res_cpu": "cc_eval_q35_inst_start_researchcpu_thinking_32k_vllm",
    },
    "q35_sft (sft_q35_a100_method)": {
        "alg": "cc_eval_q35_a100_method_thinking_32k_both_vllm",
        "res_gpu": "cc_eval_q35_sft_research_research_thinking_32k_vllm",
        "res_cpu": "cc_eval_q35_sft_researchcpu_thinking_32k_vllm",
    },
    "q35_soup10 (soup_q35_a100_method_soupa10)": {
        "alg": "cc_eval_q35_a100_method_soupa10_thinking_32k_both_vllm",
        "res_gpu": "cc_eval_q35_soup10_research_research_thinking_32k_vllm",
        "res_cpu": "cc_eval_q35_soup10_researchcpu_thinking_32k_vllm",
    },
    "q3_inst_start (Qwen3-8B)": {
        "alg": "cc_eval_q3_inst_start_thinking_32k_both_vllm",
        "res_gpu": "cc_eval_q3_inst_start_research_research_thinking_32k_vllm",
        "res_cpu": "cc_eval_q3_inst_start_researchcpu_thinking_32k_vllm",
    },
    "q3_sft (sft_q3_a100_method)": {
        "alg": "cc_eval_q3_a100_method_thinking_32k_both_vllm",
        "res_gpu": "cc_eval_q3_sft_research_research_thinking_32k_vllm",
        "res_cpu": "cc_eval_q3_sft_researchcpu_thinking_32k_vllm",
    },
    "q3_soup10 (soup_q3_a100_method_soupa10)": {
        "alg": "cc_eval_q3_a100_method_soupa10_thinking_32k_both_vllm",
        "res_gpu": "cc_eval_q3_soup10_research_research_thinking_32k_vllm",
        "res_cpu": "cc_eval_q3_soup10_researchcpu_thinking_32k_vllm",
    },
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


def _fmt(v) -> str:
    return f"{v:.3f}" if v is not None else "--"


def _weighted(a, b, na, nb):
    """Problem-count-weighted combine of two subset means (= true pooled mean@k
    because each subset value is itself a per-problem mean over n problems)."""
    if a is None and b is None:
        return None, 0
    parts, n = 0.0, 0
    if a is not None and na:
        parts += a * na; n += na
    if b is not None and nb:
        parts += b * nb; n += nb
    return (parts / n if n else None), n


def main() -> None:
    hdr = (f"{'model':<42} {'ALG m@5':>8} {'ALG b@5':>8} | "
           f"{'GPU m':>7} {'GPU b':>7} {'GPUn':>4} | "
           f"{'CPU m':>7} {'CPU b':>7} {'CPUn':>4} | "
           f"{'ALL m':>7} {'ALL b':>7} {'ALLn':>4}")
    print(hdr)
    print("-" * len(hdr))
    for name, d in MODELS.items():
        alg = _score_block(OUT / d["alg"] / "summary.json", "frontiercs")
        gpu = _score_block(OUT / d["res_gpu"] / "summary.json", "frontiercs_research")
        cpu = _score_block(OUT / d["res_cpu"] / "summary.json", "frontiercs_research")
        gn = (gpu or {}).get("n_problems") or 0
        cn = (cpu or {}).get("n_problems") or 0
        all_m, all_n = _weighted((gpu or {}).get("mean@k"), (cpu or {}).get("mean@k"), gn, cn)
        all_b, _ = _weighted((gpu or {}).get("best@k"), (cpu or {}).get("best@k"), gn, cn)
        print(
            f"{name:<42} "
            f"{_fmt((alg or {}).get('mean@k')):>8} {_fmt((alg or {}).get('best@k')):>8} | "
            f"{_fmt((gpu or {}).get('mean@k')):>7} {_fmt((gpu or {}).get('best@k')):>7} {gn:>4} | "
            f"{_fmt((cpu or {}).get('mean@k')):>7} {_fmt((cpu or {}).get('best@k')):>7} {cn:>4} | "
            f"{_fmt(all_m):>7} {_fmt(all_b):>7} {all_n:>4}"
        )
    print("\nm = mean@k (avg of per-problem score), b = best@k/mean (oracle best-of-k mean).")
    print("ALG = algorithmic track (172 C++, full.parquet). GPU = 21 Triton research")
    print("problems; CPU = 43 CPU research problems; ALL = problem-count-weighted pool")
    print("of GPU+CPU (the true pooled research mean@k). Tracks reported SEPARATELY,")
    print("never hard-averaged across ALG vs RES. '--' = summary.json not yet present.")


if __name__ == "__main__":
    main()
