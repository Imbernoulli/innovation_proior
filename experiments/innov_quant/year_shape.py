#!/usr/bin/env python3
"""Test-time YEAR sweep: shape, not a two-point contrast.

The hypothesis under test (stated before looking at the numbers): performance peaks
around 2022-2026 -- far past years describe a world where the idea did not exist yet,
far future years are out of distribution -- so the curve should be an inverted U.

Two tests, both far more powerful than the single 2000-vs-2026 contrast used earlier:
  T1  NEAR = {2025, 2026} vs FAR = {<=2010} U {>=2050}, paired per problem.
  T2  quadratic fit of the per-year arm mean; an inverted U means a NEGATIVE t^2 term.
Cells = (arm x bench); aggregated by sign test + Stouffer.

Noise floor does NOT come from this script any more. Its only same-year re-runs were
y1950/y1950r2 and y2025/y2025r2, and section 28 showed both first runs were assembled
after an 8h walltime TIMEOUT, which selects on exactly the variable being measured. The
honest floor is the six `_y2026` replicate pairs, run through the identical machinery in
year_completion.py.
"""
import json, glob, os, re, collections, math
import numpy as np
from scipy.stats import wilcoxon, binomtest, norm

OUT = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi/outputs"
NEAR, FARLO, FARHI = {2025, 2026}, 2010, 2050

def read(dirpat):
    """-> {bench: {(problem, sample_idx): score}}"""
    got = collections.defaultdict(dict)
    for f in glob.glob(os.path.join(OUT, dirpat, "shard_*", "samples.jsonl")):
        for ln in open(f):
            try: r = json.loads(ln)
            except Exception: continue
            if r.get("error") is not None: continue
            m = r.get("metrics") or {}
            s = m.get("score")
            if s is None: continue
            gt = r.get("ground_truth")
            gt = gt if isinstance(gt, str) else json.dumps(gt, sort_keys=True)
            got[r.get("data_source")][(gt, r.get("sample_idx"))] = float(s)
    return got

# rlv5_lo32nm_a10_s20's FIRST y1950 and y2025 runs were assembled after an 8h walltime
# TIMEOUT (jobs 13769934, 13769964) and are survivor-biased toward short generations -- a
# request only got written if none of its five draws ran to the cap. Their r2 re-runs are
# single clean jobs of the identical protocol, so those are the points used here. This
# matters beyond hygiene: y2025 is a NEAR point, and the bias inflates it.
CLEAN = {("rlv5_lo32nm_a10_s20", 1950): "rlv5_lo32nm_a10_s20_y1950r2",
         ("rlv5_lo32nm_a10_s20", 2025): "rlv5_lo32nm_a10_s20_y2025r2"}


def arm_year(arm, year):
    tag = CLEAN.get((arm, year)) or (f"{arm}_y{year}" if year != "main" else arm)
    return read(f"cc_eval_{tag}_thinking_32k_both_vllm")

def perprob(d):
    by = collections.defaultdict(list)
    for (p, _), s in d.items(): by[p].append(s)
    return {p: float(np.mean(v)) for p, v in by.items()}

def paired(a, b):
    ks = sorted(set(a) & set(b))
    if len(ks) < 8: return None
    d = np.array([a[k] - b[k] for k in ks]); nz = d[d != 0]
    if len(nz) < 5: return None
    _, p = wilcoxon(nz)
    if not np.isfinite(p): return None
    z = (1.0 if np.median(nz) > 0 else -1.0) * abs(norm.ppf(max(p, 1e-12) / 2))
    return dict(n=len(d), mean=float(d.mean()), pos=int((d > 0).sum()),
                neg=int((d < 0).sum()), p=float(p), z=float(z))

def agg(cells, label):
    cells = [c for c in cells if c]
    if not cells:
        print(f"| {label} | — | — | — | — | — |"); return
    k, n = sum(1 for c in cells if c["mean"] > 0), len(cells)
    Z = sum(c["z"] for c in cells) / math.sqrt(n)
    print(f"| **{label}** | **{n}** | **{k}/{n} 正** | **{binomtest(k,n,0.5).pvalue:.4f}** | "
          f"**{Z:+.2f}** | **{2*(1-norm.cdf(abs(Z))):.4f}** |")

# arms with enough year points to define a curve
CURVE = {
 "base9b_v2c":                [1700,1800,1900,1950,1975,2000,2010,2025,2026,2050,2075,2100],
 "lo32nm_a10":                [1700,1800,1900,1950,1975,2000,2010,2025,2026,2050,2075,2100],
 "rlv5_lo32nm_a10_s20":       [1800,1900,1950,1975,2000,2010,2025,2026,2050,2075,2100],
 "base4b":                    [1900,2000,2010,2026,2100],
 "4b_lo32nm_a10":             [1900,2000,2010,2026,2100],
 "rlv5_4b_lo32nm_a10_s20":    [1900,2000,2010,2026,2100],
 # the ft01mix line and the two rlv5_base arms were submitted as one year batch
 # with exactly these four points, so none of their points comes from the main run.
 "ft01mix_a10":               [2000,2025,2050,2075],
 "rlv5_base_s20":             [2000,2025,2050,2075],
 "rlv5_ft01mix_a10_s20":      [2000,2025,2050,2075],
 "4b_ft01mix_a10":            [2000,2025,2050,2075],
 "rlv5_4b_base_s20":          [2000,2025,2050,2075],
 "rlv5_4b_ft01mix_a10_s20":   [2000,2025,2050,2075],
}
# Which year sweep an arm belongs to.  These are NOT one experiment: the OLD sweep has a
# 5-to-12 point grid with NEAR={2025,2026} (4B arms only have 2026), the NEW one is the
# main ft01mix line plus the two rlv5_base arms submitted as a single batch with exactly
# four points and NEAR={2025}.  Pooling them into one 24-cell Stouffer hides that, so the
# aggregate is also reported per sweep.
SWEEP = {a: ("新四点批次" if years == [2000, 2025, 2050, 2075] else "旧扫描")
         for a, years in CURVE.items()}
BEN = ["frontiercs", "frontiercs_research", "alebench"]
CACHE = {}
def get(arm, y):
    k = (arm, y)
    if k not in CACHE: CACHE[k] = arm_year(arm, y)
    return CACHE[k]

print("# 年份扫描:倒 U 形检验\n")
print("假设(先写后看):成绩在 2025/2026 附近最高,越早越低(那时这些想法还不存在)、")
print("越晚越低(离分布太远)。NEAR={2025,2026},FAR={<=2010} ∪ {>=2050}。\n")

print("## T1 逐题配对:NEAR − FAR\n")
print("| arm | bench | n题 | Δ均值 | +/− | Wilcoxon p |")
print("|---|---|---|---|---|---|")
cells = collections.defaultdict(list)
for arm, years in CURVE.items():
    near = [y for y in years if y in NEAR]
    far  = [y for y in years if y <= FARLO or y >= FARHI]
    if not near or not far: continue
    for b in BEN:
        na, fa = collections.defaultdict(list), collections.defaultdict(list)
        for y in near:
            for p, s in perprob(get(arm, y).get(b, {})).items(): na[p].append(s)
        for y in far:
            for p, s in perprob(get(arm, y).get(b, {})).items(): fa[p].append(s)
        A = {p: float(np.mean(v)) for p, v in na.items()}
        B = {p: float(np.mean(v)) for p, v in fa.items()}
        c = paired(A, B)
        if not c: continue
        cells[b].append(c); cells["ALL"].append(c)
        cells["9B" if not arm.startswith(("base4b", "4b_", "rlv5_4b")) else "4B"].append(c)
        cells[SWEEP[arm]].append(c)
        print(f"| {arm} | {b} | {c['n']} | {c['mean']:+.3f} | {c['pos']}/{c['neg']} | {c['p']:.4f} |")
print("\n| 聚合 | 格子 | 方向 | 符号检验 p | Stouffer Z | p |")
print("|---|---|---|---|---|---|")
for k in ("ALL", "新四点批次", "旧扫描", "9B", "4B",
          "frontiercs", "frontiercs_research", "alebench"):
    agg(cells[k], f"T1 {k}")

print("\n## T2 二次项(倒 U ⇒ 系数为负)\n")
print("| arm | bench | 峰值年 | t² 系数 | 形状 |")
print("|---|---|---|---|---|")
neg = tot = 0
for arm, years in CURVE.items():
    for b in BEN:
        xs, ys = [], []
        for y in years:
            pp = perprob(get(arm, y).get(b, {}))
            if len(pp) >= 20: xs.append(y); ys.append(float(np.mean(list(pp.values()))))
        if len(xs) < 5: continue
        t = (np.array(xs, float) - 2000) / 100.0
        c2, c1, c0 = np.polyfit(t, np.array(ys), 2)
        peak = 2000 + 100 * (-c1 / (2 * c2)) if c2 != 0 else float("nan")
        tot += 1; neg += (c2 < 0)
        print(f"| {arm} | {b} | {peak:.0f} | {c2:+.3f} | {'∩ 倒U' if c2 < 0 else '∪ 正U'} |")
print(f"\n**{neg}/{tot} 个 (臂×bench) 的二次项为负(倒 U);符号检验 p="
      f"{binomtest(neg, tot, 0.5).pvalue:.4f}**" if tot else "")

print("\n## 噪声底:同年复跑(什么都没改)\n")
print("| 对比 | bench | n题 | Δ均值 | +/− | Wilcoxon p |")
print("|---|---|---|---|---|---|")
# nothing left to compare here: the only same-year re-runs this arm has are the two
# timeout-contaminated pairs, now resolved in favour of the clean side by CLEAN above.
# The honest floor for this analysis is the `_y2026` replicate set in year_completion.py.
REPS = []
rc = []
for arm, y, r2 in REPS:
    for b in BEN:
        c = paired(perprob(read(f"cc_eval_{arm}_y{r2}_thinking_32k_both_vllm").get(b, {})),
                   perprob(get(arm, y).get(b, {})))
        if not c: continue
        rc.append(c)
        print(f"| {arm} y{y} 复跑 | {b} | {c['n']} | {c['mean']:+.3f} | {c['pos']}/{c['neg']} | {c['p']:.4f} |")
print("\n| 聚合 | 格子 | 方向 | 符号检验 p | Stouffer Z | p |")
print("|---|---|---|---|---|---|")
agg(rc, "噪声底(同年复跑)")
