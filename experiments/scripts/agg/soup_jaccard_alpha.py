#!/usr/bin/env python3
"""Jaccard-similarity-to-baseline vs alpha, REUSING the 附录A protocol verbatim
(run_similarity_analysis.py functions; only LOG_DIRS is swapped for the clean-arm
alpha sweep MLS task logs)."""
import importlib.util, statistics, math, json
from pathlib import Path

SRC = "/scratch/gpfs/CHIJ/bohan/fs/innovation_prior/experiments/run_similarity_analysis.py"
spec = importlib.util.spec_from_file_location("simana", SRC)
m = importlib.util.module_from_spec(spec)
import sys as _sys
_sys.modules["simana"] = m
spec.loader.exec_module(m)

OUT = Path("/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/outputs")
DIRS = {
    "base_a0": "q35_start_devfix",
    "nom_a5": "clean_clean_nomaintain_wd01_a5",
    "nom_a10": "clean_clean_nomaintain_wd01_a10",
    "nom_a20": "clean_clean_nomaintain_wd01_a20",
    "nom_a30": "clean_clean_nomaintain_wd01_a30",
    "nom_a50": "clean_clean_nomaintain_wd01_a50",
    "nom_sft_a100": "clean_nom_wd01_sft_devfix",
    "full_a5": "clean_clean_full_wd01_a5",
    "full_a10": "clean_clean_full_wd01_a10",
    "full_a20": "clean_clean_full_wd01_a20",
    "full_a30": "clean_clean_full_wd01_a30",
    "full_sft_a100": "clean_full_wd01_sft_devfix",
    "wd03_a10": "clnom_wd03_a10",
    "newmt_a10": "clnom_newmt_a10",
}

tasks = sorted(p.name for p in m.TASKS_DIR.iterdir() if p.is_dir())
res = {}
for name, tag in DIRS.items():
    log_dir = OUT / f"cc_mlsbench_cpu_{tag}" / "task_logs"
    scores, beyond, fallback, navail = [], 0, 0, 0
    pertask = {}
    for task in tasks:
        desc_path = m.TASKS_DIR / task / "task_description.md"
        if not desc_path.exists() or not (log_dir / f"{task}.log").exists():
            continue
        baselines, _ = m.extract_baselines(desc_path.read_text(encoding="utf-8", errors="replace"))
        r = m.analyze_model(task, name, log_dir, baselines)
        if r.status == "log_missing":
            continue
        navail += 1
        if r.jaccard is not None:
            scores.append(r.jaccard)
            pertask[task] = round(r.jaccard, 3)
        if r.beyond:
            beyond += 1
        if r.status == "fallback":
            fallback += 1
    res[name] = {
        "mean_jaccard": statistics.mean(scores) if scores else math.nan,
        "n_scored": len(scores), "n_avail": navail,
        "beyond_baseline": beyond, "fallback": fallback,
        "pertask": pertask,
    }
    print(f"{name:14s} mean_jaccard={res[name]['mean_jaccard']:.3f} (n={len(scores)}/{navail})"
          f" beyond={beyond} fallback={fallback}")
json.dump(res, open("jaccard_alpha_results.json", "w"), indent=1)
