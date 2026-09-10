#!/usr/bin/env python3
"""Diagnostics for the 'no arm differs' null on the Tang & Yang measures.
1. Paired between-arm tests (same seed set / same area) instead of overlapping per-arm CIs.
2. Noise floor: sample-0 vs sample-1 of the SAME arm, paired the same way.
3. Sensitivity: base9b vs base4b (different models) paired.
4. Prompt dominance: cosine sim of ideas across arms for the same seed vs within arm vs across seeds.
5. Arm identifiability: nearest-centroid classification of idea embeddings, leave-one-area-out.
Run with OMP_NUM_THREADS=6.
"""
import json, os, sys, collections, random
import numpy as np
S = os.path.dirname(os.path.abspath(__file__)); DD = os.path.join(S, "data")
rng = np.random.default_rng(7); B = 4000
J = lambda fn: [json.loads(l) for l in open(fn)]
ANN = [f"{DD}/ann_papers_0.jsonl", f"{DD}/ann_papers_1.jsonl", f"{DD}/ann_papers_2.jsonl", f"{DD}/ann_ideas.jsonl", f"{DD}/ann_redo.jsonl"]
ann = {}
for fn in ANN:
    for r in J(fn):
        if not r.get("unparsed"): ann[r["doc_id"]] = r
E = np.load(f"{DD}/emb.npy"); ids = json.load(open(f"{DD}/emb.ids.json")); row = {d: i for i, d in enumerate(ids)}
seedsets = J(f"{DD}/seedsets.jsonl"); sset = {s["seed_id"]: s for s in seedsets}
follow = J(f"{DD}/followons.jsonl"); fol_by_seed = collections.defaultdict(list)
for f in follow: fol_by_seed[f["seed_id"]].append(f["id"])
pool = J(f"{DD}/impact_pool.jsonl"); pool_by_area = collections.defaultdict(list)
for p in pool: pool_by_area[p["area"]].append(p)

ideas = collections.defaultdict(dict)  # tag -> seed -> {sample_idx: doc}
for d in ann:
    if ann[d]["kind"] == "generated_idea" and d in row:
        tag, sid, k = d.split("|"); ideas[tag][sid] = ideas[tag].get(sid, {}); ideas[tag][sid][int(k)] = d
tags = sorted(ideas)
cent = {}
for sid, s in sset.items():
    sv = [E[row[p]] for p in s["seeds"] if p in row]
    if sv: c = np.sum(sv, 0); cent[sid] = c / np.linalg.norm(c)

# impact machinery (same as analyze.py)
s_p = {}
for area, ps in pool_by_area.items():
    by_year = collections.defaultdict(list)
    for p in ps: by_year[p["year"]].append(p)
    for y, lst in by_year.items():
        logs = np.array([np.log1p(p["cites"]) for p in lst])
        for p, l in zip(lst, logs): s_p[p["id"]] = float(l - (logs.sum() - l) / max(1, len(logs) - 1)) if len(logs) > 1 else 0.0
_pm = {}
for area, ps in pool_by_area.items():
    ps2 = [p for p in ps if p["id"] in row and p["id"] in s_p]
    if ps2: _pm[area] = (np.stack([E[row[p["id"]]] for p in ps2]), np.array([s_p[p["id"]] for p in ps2]), np.array([p["year"] for p in ps2]))
def impact(v, area, t):
    if area not in _pm: return np.nan
    M, sv, yr = _pm[area]; m = yr <= t
    if m.sum() < 5: return np.nan
    sims = M[m] @ v; top = np.argsort(-sims)[:20]; return float(sv[m][top].mean())

def per_seed(tag, which=None):
    """distance & impact per seed set for an arm; which=None uses all samples, else only that sample index."""
    dist, imp = {}, {}
    for sid, dd in ideas[tag].items():
        if sid not in cent: continue
        docs = [d for k, d in dd.items() if which is None or k == which]
        if not docs: continue
        vs = np.stack([E[row[d]] for d in docs]); s = sset[sid]
        dist[sid] = float(np.mean(1 - vs @ cent[sid]))
        imp[sid] = float(np.nanmean([impact(v, s["area"], s["t"]) for v in vs]))
    return dist, imp

def paired(a, b, label):
    keys = sorted(set(a) & set(b)); x = np.array([a[k] - b[k] for k in keys])
    bs = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)])
    lo, hi = np.percentile(bs, [2.5, 97.5]); star = "*" if (lo > 0 or hi < 0) else " "
    print(f"  {label:58s} n={len(x):3d} diff={x.mean():+.4f} CI[{lo:+.4f},{hi:+.4f}] {star}  P(>0)={np.mean(bs>0):.3f}")
    return x.mean(), lo, hi

print("=" * 100); print("1. PAIRED between-arm differences (per seed set), distance then impact")
PS = {t: per_seed(t) for t in tags}
groups = {"9B": ("base9b_v2c", ["ft01mix_a10", "ft03nm_a20", "lo32nm_a10", "rlv5_base_s20", "rlv5_ft01mix_a10_s20", "rlv5_ft03nm_a20_s20", "rlv5_lo32nm_a10_s20"]),
          "4B": ("base4b", ["4b_ft01mix_a10", "4b_lo32nm_a10", "rlv5_4b_base_s20", "rlv5_4b_ft01mix_a10_s20"])}
for m, idx in (("distance", 0), ("impact", 1)):
    print(f"-- {m}")
    for g, (base, arms) in groups.items():
        for a in arms: paired(PS[a][idx], PS[base][idx], f"{a} - {base}")
    paired(PS["rlv5_ft01mix_a10_s20"][idx], PS["ft01mix_a10"][idx], "rlv5_ft01mix_a10_s20 - ft01mix_a10 (RL effect, 9B)")
    paired(PS["rlv5_lo32nm_a10_s20"][idx], PS["lo32nm_a10"][idx], "rlv5_lo32nm_a10_s20 - lo32nm_a10 (RL effect, 9B)")
    paired(PS["rlv5_ft01mix_a10_s20"][idx], PS["rlv5_base_s20"][idx], "rlv5_ft01mix_a10_s20 - rlv5_base_s20 (ours vs ctrl, 9B)")
    paired(PS["rlv5_4b_ft01mix_a10_s20"][idx], PS["4b_ft01mix_a10"][idx], "rlv5_4b_ft01mix_a10_s20 - 4b_ft01mix_a10 (RL effect, 4B)")
    paired(PS["rlv5_4b_ft01mix_a10_s20"][idx], PS["rlv5_4b_base_s20"][idx], "rlv5_4b_ft01mix_a10_s20 - rlv5_4b_base_s20 (ours vs ctrl, 4B)")
    print("  -- sensitivity / noise floor")
    paired(PS["base4b"][idx], PS["base9b_v2c"][idx], "base4b - base9b_v2c (different base models)")
    for t in ("base9b_v2c", "rlv5_ft01mix_a10_s20", "base4b"):
        a0, a1 = per_seed(t, 0), per_seed(t, 1)
        paired(a0[idx], a1[idx], f"{t}: sample0 - sample1 (same arm, noise floor)")
    # human follow-ons vs AI for scale
    hu = {}
    for sid in PS["base9b_v2c"][idx]:
        s = sset[sid]; fo = [E[row[p]] for p in fol_by_seed[sid] if p in row]
        if not fo: continue
        vs = np.stack(fo)
        hu[sid] = float(np.mean(1 - vs @ cent[sid])) if idx == 0 else float(np.nanmean([impact(v, s["area"], s["t"]) for v in vs]))
    paired(hu, PS["base9b_v2c"][idx], "human follow-ons - base9b_v2c (scale reference)")

print("=" * 100); print("2. PAIRED breadth per area (45 areas)")
def breadth_area(tag):
    out = {}
    by_area = collections.defaultdict(list)
    for sid, dd in ideas[tag].items(): by_area[sset[sid]["area"]] += [E[row[d]] for d in dd.values()]
    for a, vs in by_area.items():
        M = np.stack(vs); G = M @ M.T; n = len(M); out[a] = float(1 - (G.sum() - np.trace(G)) / (n * (n - 1)))
    return out
BR = {t: breadth_area(t) for t in tags}
for g, (base, arms) in groups.items():
    for a in arms: paired(BR[a], BR[base], f"breadth {a} - {base}")
paired(BR["rlv5_ft01mix_a10_s20"], BR["ft01mix_a10"], "breadth rlv5_ft01mix_a10_s20 - ft01mix_a10 (RL effect)")
paired(BR["base4b"], BR["base9b_v2c"], "breadth base4b - base9b_v2c")

print("=" * 100); print("3. PROMPT DOMINANCE: cosine similarity between idea embeddings")
def sims(pairs):
    v = np.array([float(E[row[a]] @ E[row[b]]) for a, b in pairs]); return v.mean(), np.percentile(v, 50), len(v)
same_arm_same_seed, diff_arm_same_seed, same_arm_diff_seed = [], [], []
t9 = groups["9B"][1] + ["base9b_v2c"]
rs = random.Random(3)
sids = sorted(set.intersection(*[set(ideas[t]) for t in t9]))
for sid in sids:
    for t in t9:
        dd = ideas[t][sid]
        if 0 in dd and 1 in dd: same_arm_same_seed.append((dd[0], dd[1]))
    for i in range(len(t9)):
        for j in range(i + 1, len(t9)):
            a, b = ideas[t9[i]][sid], ideas[t9[j]][sid]
            if 0 in a and 0 in b: diff_arm_same_seed.append((a[0], b[0]))
for t in t9:
    for _ in range(600):
        s1, s2 = rs.sample(sids, 2)
        if 0 in ideas[t][s1] and 0 in ideas[t][s2]: same_arm_diff_seed.append((ideas[t][s1][0], ideas[t][s2][0]))
for lab, pr in (("same arm, same seed (two samples)", same_arm_same_seed), ("DIFFERENT arm, same seed", diff_arm_same_seed), ("same arm, different seed", same_arm_diff_seed)):
    m, med, n = sims(pr); print(f"  {lab:40s} mean cos = {m:.3f}  median = {med:.3f}  (n={n})")
# idea vs its own seed centroid, and human follow-on vs centroid, for reference
print(f"  idea vs seed centroid (base9b): {np.mean([1-v for v in PS['base9b_v2c'][0].values()]):.3f} sim;  follow-on vs centroid: {1-np.mean(list(hu.values())) if False else 'see distance above'}")

print("=" * 100); print("4. ARM IDENTIFIABILITY: nearest-centroid, leave-one-area-out, 9B arms (chance = 1/8 = 12.5%)")
areas = sorted(set(s["area"] for s in seedsets))
X, y, ar = [], [], []
for ti, t in enumerate(t9):
    for sid, dd in ideas[t].items():
        for d in dd.values(): X.append(E[row[d]]); y.append(ti); ar.append(sset[sid]["area"])
X = np.stack(X); y = np.array(y); ar = np.array(ar)
correct = 0; tot = 0; conf = np.zeros((len(t9), len(t9)), int)
for a in areas:
    tr = ar != a; te = ~tr
    C = np.stack([X[tr & (y == k)].mean(0) for k in range(len(t9))]); C /= np.linalg.norm(C, axis=1, keepdims=True)
    pred = np.argmax(X[te] @ C.T, 1); correct += (pred == y[te]).sum(); tot += te.sum()
    for p_, t_ in zip(pred, y[te]): conf[t_, p_] += 1
print(f"  accuracy = {correct/tot:.3f} (n={tot})")
for k, t in enumerate(t9): print(f"  {t:24s} " + " ".join(f"{c:4d}" for c in conf[k]))
# same for SFT-vs-base and RL-vs-base binary, and base9b vs base4b
def binacc(ta, tb):
    Xa = np.stack([E[row[d]] for dd in ideas[ta].values() for d in dd.values()]); Xb = np.stack([E[row[d]] for dd in ideas[tb].values() for d in dd.values()])
    ara = np.array([sset[sid]["area"] for sid, dd in ideas[ta].items() for _ in dd]); arb = np.array([sset[sid]["area"] for sid, dd in ideas[tb].items() for _ in dd])
    c = n = 0
    for a in areas:
        ca = Xa[ara != a].mean(0); cb = Xb[arb != a].mean(0); ca /= np.linalg.norm(ca); cb /= np.linalg.norm(cb)
        pa = (Xa[ara == a] @ ca > Xa[ara == a] @ cb).sum(); pb = (Xb[arb == a] @ cb > Xb[arb == a] @ ca).sum()
        c += pa + pb; n += (ara == a).sum() + (arb == a).sum()
    return c / n
for ta, tb in (("ft01mix_a10", "base9b_v2c"), ("rlv5_ft01mix_a10_s20", "base9b_v2c"), ("rlv5_ft01mix_a10_s20", "rlv5_base_s20"), ("base4b", "base9b_v2c"), ("rlv5_4b_ft01mix_a10_s20", "base4b"), ("rlv5_4b_base_s20", "base4b")):
    print(f"  binary {ta} vs {tb}: acc = {binacc(ta, tb):.3f} (chance 0.5)")
