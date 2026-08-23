#!/usr/bin/env python3
"""Aggregate the {model x task} ThetaEvolve / TTT-Discover OpenEvolve eval matrix
into per-task x per-model tables. Reads every
  ThetaEvolve/outputs/cc_eval_theta_<tag>_<task>/job_*/summary.json
and reports best_combined_score, the engine's seed_score (floor), and the
model-over-seed improvement. Splits Theta (tag has no ttt_ prefix) vs TTT (ttt_).
"""
import json, glob, os, re, sys
from collections import defaultdict

OUT_ROOT = "/scratch/gpfs/CHIJ/bohan/fs/ThetaEvolve/outputs"
TASKS = ["circle_packing_modular", "first_autocorr_inequality",
         "second_autocorr_inequality", "third_autocorr_inequality",
         "hadamard_matrix"]
MODEL_ORDER = ["q35_start", "q35_method_sft", "q35_method_soup10",
               "q35_methodtraj_sft", "q35_methodtraj_soup10"]

def newest_summary(d):
    cands = sorted(glob.glob(os.path.join(d, "job_*", "summary.json")),
                   key=os.path.getmtime)
    return cands[-1] if cands else None

# results[(bench, model, task)] = dict
results = {}
for task in TASKS:
    for model in MODEL_ORDER:
        for bench, tag in (("theta", model), ("ttt", f"ttt_{model}")):
            d = os.path.join(OUT_ROOT, f"cc_eval_theta_{tag}_{task}")
            s = newest_summary(d) if os.path.isdir(d) else None
            if not s:
                results[(bench, model, task)] = None
                continue
            try:
                j = json.load(open(s))
            except Exception as e:
                results[(bench, model, task)] = {"error": str(e)}
                continue
            disc = j.get("discrimination") or {}
            results[(bench, model, task)] = {
                "best": j.get("best_objective_value", j.get("best_combined_score")),  # RAW objective; combined_score is inverted 1/(obj+EPS) for minimization tasks (first/third_autocorr, AC1)
                "seed": disc.get("seed_score"),
                "impr": disc.get("model_improvement_over_seed"),
                "beat": disc.get("model_beat_seed"),
                "iter": j.get("iteration"),
                "summary": s,
            }

def fmt(x, w=8, p=4):
    if x is None: return "  --  ".ljust(w)
    if isinstance(x, bool): return ("yes" if x else "no").ljust(w)
    try: return f"{x:.{p}f}".ljust(w)
    except Exception: return str(x).ljust(w)

for bench, label in (("theta", "THETAEVOLVE"), ("ttt", "TTT-DISCOVER (via OpenEvolve adapter)")):
    bench_tasks = TASKS if bench == "theta" else \
        ["first_autocorr_inequality", "second_autocorr_inequality",
         "circle_packing_modular"]  # third_autocorr is NOT a TTT-Discover task (no AC3)
    print(f"\n========== {label}: best_objective_value (raw task metric; min-tasks first/third_autocorr lower=better) ==========")
    header = "model".ljust(24) + "".join(t[:18].ljust(20) for t in bench_tasks)
    print(header)
    for model in MODEL_ORDER:
        row = model.ljust(24)
        for task in bench_tasks:
            r = results.get((bench, model, task))
            if r is None: row += "  pending/–".ljust(20)
            elif "error" in r: row += "  ERR".ljust(20)
            else: row += fmt(r["best"], w=20)
        print(row)
    # coverage
    done = sum(1 for model in MODEL_ORDER for task in bench_tasks
               if results.get((bench, model, task)) and "best" in (results[(bench,model,task)] or {}))
    total = len(MODEL_ORDER) * len(bench_tasks)
    print(f"  coverage: {done}/{total} cells have a summary.json")

# detailed long-form for everything finished
print("\n========== DETAIL (finished cells) bench/model/task: best | seed_floor | impr | beat ==========")
for (bench, model, task), r in sorted(results.items()):
    if r and "best" in r:
        print(f"{bench:5s} {model:24s} {task:28s} best={fmt(r['best'])} seed={fmt(r['seed'])} impr={fmt(r['impr'])} beat={fmt(r['beat'],6,0)} it={r['iter']}")

# remaining
print("\n========== REMAINING (no summary yet) ==========")
rem = [(b,m,t) for (b,m,t),r in results.items() if r is None or "best" not in (r or {})]
for b,m,t in sorted(rem):
    print(f"  {b:5s} {m:24s} {t}")
print(f"  total remaining: {len(rem)}")
