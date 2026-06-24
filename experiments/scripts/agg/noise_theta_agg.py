#!/usr/bin/env python3
"""Aggregate multi-seed Theta/TTT noise re-runs into per-model mean+-std over seeds,
and compute between-model effect sizes (Cohen's d) + whether CIs separate.

Reads ThetaEvolve/outputs/cc_eval_theta_noise_*/job_*/summary.json (theta)
  and ThetaEvolve/outputs/cc_eval_theta_ttt_noise_*/job_*/summary.json (ttt).
Tags: noise_theta_<model>_s<seed> / noise_ttt_<model>_s<seed>.
"""
import json, glob, re, collections, itertools
import numpy as np

TH = "/scratch/gpfs/CHIJ/bohan/fs/ThetaEvolve/outputs"

def collect():
    # bench -> model -> {seed: score}
    data = collections.defaultdict(lambda: collections.defaultdict(dict))
    raw = []
    for p in glob.glob(f"{TH}/cc_eval_theta_*/job_*/summary.json"):
        top = p.split("/outputs/")[1].split("/")[0]
        m = re.match(r"cc_eval_theta_(ttt_)?noise_(theta|ttt)_(\w+?)_s(\d+)_", top)
        if not m:
            continue
        bench = "TTT" if m.group(1) else "THETA"
        model = m.group(3); seed = int(m.group(4))
        try:
            d = json.load(open(p))
        except Exception:
            continue
        sc = d.get("best_combined_score")
        if sc is None:
            continue
        disc = d.get("discrimination") or {}
        data[bench][model][seed] = sc
        raw.append((bench, model, seed, sc, disc.get("best_is_seed")))
    return data, raw

def cohend(a, b):
    a = np.array(a); b = np.array(b)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    sp = np.sqrt(((na-1)*a.var(ddof=1) + (nb-1)*b.var(ddof=1)) / (na+nb-2))
    if sp == 0:
        return float("inf") if a.mean() != b.mean() else 0.0
    return (a.mean() - b.mean()) / sp

if __name__ == "__main__":
    data, raw = collect()
    print(f"collected {len(raw)} noise runs")
    SEED_FLOOR = {"THETA": 0.9598, "TTT": 0.3166}
    for bench in ("THETA", "TTT"):
        print(f"\n===== {bench} (seed floor {SEED_FLOOR[bench]}) =====")
        models = data.get(bench, {})
        stats = {}
        for model, sd in sorted(models.items()):
            vals = list(sd.values())
            arr = np.array(vals)
            stats[model] = arr
            mean = arr.mean(); std = arr.std(ddof=1) if len(arr) > 1 else float("nan")
            se = std/np.sqrt(len(arr)) if len(arr) > 1 else float("nan")
            ci = (mean-1.96*se, mean+1.96*se) if len(arr) > 1 else (float("nan"),)*2
            print(f"  {model:12s} n={len(arr)} seeds={sorted(sd)} "
                  f"mean={mean:.4f} std={std:.4f} 95%CI=[{ci[0]:.4f},{ci[1]:.4f}] "
                  f"vals={[round(v,3) for v in vals]}")
        # pairwise effect sizes
        print(f"  --- pairwise (Cohen's d, Welch t-ish via CI overlap) ---")
        for a, b in itertools.combinations(sorted(stats), 2):
            d = cohend(stats[a], stats[b])
            if d is None:
                print(f"    {a} vs {b}: insufficient seeds")
                continue
            ma, mb = stats[a].mean(), stats[b].mean()
            # CI separation
            sa = stats[a].std(ddof=1)/np.sqrt(len(stats[a]))
            sb = stats[b].std(ddof=1)/np.sqrt(len(stats[b]))
            sep = abs(ma-mb) > 1.96*np.sqrt(sa**2+sb**2)
            print(f"    {a:11s} ({ma:.3f}) vs {b:11s} ({mb:.3f}): Δ={ma-mb:+.3f} d={d:+.2f} "
                  f"{'SEPARATED' if sep else 'overlap (NOISE)'}")
