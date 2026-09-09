#!/usr/bin/env python3
"""Tang & Yang four measures, per arm, with human references and bootstrap CIs (paper S1.3).

Inputs (all under data/ or given): seedsets/followons/impact_pool/frontier_corpus/papers, annotations
(keywords) for papers and ideas, one embedding matrix covering every annotated doc_id.
  breadth(X_a)  = mean pairwise cosine distance within area a; human = seed papers of area a, one seed
                  sampled per idea so |human| == |AI| (paper 4.1); averaged over areas.
  dist(x; r)    = 1 - e_x . c_r, c_r = normalized centroid of the 5 seeds; human = follow-on papers of run r.
  alignment(g)  = |K_g ∩ F_{f,t+1}| / |F_{f,t+1}|, F = top-10% keywords among field-f human papers in t+1
                  (frontier corpus, follow-ons excluded); K_g pooled over the group's docs; computed per (field, t), averaged.
  impact(x)     = mean over the k=20 nearest same-area human papers published <= t of s_p = log(1+c_p) - LOO mean(area, year).
Bootstrap: resample research areas (breadth, alignment) / seed sets (distance, impact), 2000 draws, 95% percentile CI.
Keyword matching: lowercase, strip punctuation, singularize trailing 's' -- identical for both sides (paper does not specify).
"""
import json, os, re, sys, collections, random
import numpy as np
S = os.path.dirname(os.path.abspath(__file__)); DD = os.path.join(S, "data")
EMB = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DD, "emb")
ANN = sys.argv[2:] if len(sys.argv) > 2 else [os.path.join(DD, "ann_papers.jsonl"), os.path.join(DD, "ann_ideas.jsonl")]
rng = random.Random(7); B = 2000

J = lambda fn: [json.loads(l) for l in open(fn)]
seedsets = J(f"{DD}/seedsets.jsonl"); follow = J(f"{DD}/followons.jsonl"); pool = J(f"{DD}/impact_pool.jsonl"); frontier = J(f"{DD}/frontier_corpus.jsonl")
papers = {p["id"]: p for p in J(f"{DD}/papers.jsonl")}
ann = {}
for fn in ANN:
    for r in J(fn):
        if not r.get("unparsed"): ann[r["doc_id"]] = r
E = np.load(EMB + ".npy"); ids = json.load(open(EMB + ".ids.json")); row = {d: i for i, d in enumerate(ids)}
def emb(d): return E[row[d]] if d in row else None
def norm_kw(k):
    k = re.sub(r"[^a-z0-9 ]", " ", k.lower()); k = re.sub(r"\s+", " ", k).strip()
    return k[:-1] if k.endswith("s") and len(k) > 4 else k
kws = {d: {norm_kw(k) for k in r.get("keywords", [])} - {""} for d, r in ann.items()}

# ideas: doc_id = tag|seed_id|sample_idx
ideas = collections.defaultdict(list)   # tag -> list of (seed_id, doc_id)
for d in ann:
    if ann[d]["kind"] == "generated_idea":
        tag, sid, _ = d.split("|"); ideas[tag].append((sid, d))
sset = {s["seed_id"]: s for s in seedsets}
fol_by_seed = collections.defaultdict(list)
for f in follow: fol_by_seed[f["seed_id"]].append(f["id"])
fol_ids = {f["id"] for f in follow}
pool_by_area = collections.defaultdict(list)
for p in pool: pool_by_area[p["area"]].append(p)

def ci(vals):
    v = np.array(vals, dtype=float); v = v[~np.isnan(v)]
    if len(v) == 0: return (float("nan"), float("nan"), float("nan"), 0)
    bs = [np.mean(rng.choices(list(v), k=len(v))) for _ in range(B)]
    return (float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(v))
def pair_dist(M):
    if len(M) < 2: return float("nan")
    G = M @ M.T; n = len(M); return float((n * n - np.trace(G) - (G.sum() - np.trace(G)) * 0) and (1 - (G.sum() - np.trace(G)) / (n * (n - 1))))

# ---- impact scores for human pool papers (LOO by area-year)
s_p = {}
for area, ps in pool_by_area.items():
    by_year = collections.defaultdict(list)
    for p in ps: by_year[p["year"]].append(p)
    for y, lst in by_year.items():
        logs = np.array([np.log1p(p["cites"]) for p in lst])
        for p, l in zip(lst, logs):
            s_p[p["id"]] = float(l - (logs.sum() - l) / max(1, len(logs) - 1)) if len(logs) > 1 else 0.0
_pool_mat = {}
for area, ps in pool_by_area.items():
    ps2 = [p for p in ps if p["id"] in row and p["id"] in s_p]
    if ps2: _pool_mat[area] = (np.stack([E[row[p["id"]]] for p in ps2]), np.array([s_p[p["id"]] for p in ps2]), np.array([p["year"] for p in ps2]), np.array([p["id"] for p in ps2]))
def impact(x_vec, area, t, exclude=()):
    """exclude: paper ids removed from the neighbour pool (used to drop the seed set itself -- seeds were picked by citation count)."""
    if area not in _pool_mat: return float("nan")
    M, sv, yr, pid = _pool_mat[area]; m = yr <= t
    if exclude: m = m & ~np.isin(pid, list(exclude))
    if m.sum() < 5: return float("nan")
    sims = M[m] @ x_vec; top = np.argsort(-sims)[:20]
    return float(sv[m][top].mean())

# ---- frontier sets per (field, t)
F = {}
for (field, t), grp in collections.defaultdict(list, {k: [] for k in set((f["field"], f["t"]) for f in frontier)}).items(): pass
fc = collections.defaultdict(list)
for f in frontier:
    if f["id"] not in fol_ids and f["id"] in kws: fc[(f["field"], f["t"])].append(f["id"])
for key, docs in fc.items():
    cnt = collections.Counter(k for d in docs for k in kws[d]); n = max(1, int(round(0.1 * len(cnt))))
    F[key] = {k for k, _ in cnt.most_common(n)}

def group_measures(name, items):
    """items: list of (seed_id, doc_id) for one comparison group (an arm, or the human follow-on set)."""
    out = {}
    # breadth per area (equal n: one random seed paper per idea for the human side)
    by_area = collections.defaultdict(list)
    for sid, d in items:
        if d in row: by_area[sset[sid]["area"]].append((sid, d))
    br_ai, br_hu = [], []
    for area, lst in by_area.items():
        M = np.stack([E[row[d]] for _, d in lst]); br_ai.append(pair_dist(M))
        hs = []
        for sid, _ in lst:
            cands = [p for p in sset[sid]["seeds"] if p in row]
            if cands: hs.append(E[row[rng.choice(cands)]])
        br_hu.append(pair_dist(np.stack(hs)) if len(hs) > 1 else float("nan"))
    out["breadth"] = (ci(br_ai), ci(br_hu))
    # distance to seed centroid, per seed set (AI mean vs follow-on mean)
    d_ai, d_hu = [], []
    by_seed = collections.defaultdict(list)
    for sid, d in items:
        if d in row: by_seed[sid].append(d)
    for sid, ds in by_seed.items():
        sv = [E[row[p]] for p in sset[sid]["seeds"] if p in row]
        if not sv: continue
        c = np.sum(sv, 0); c /= np.linalg.norm(c)
        d_ai.append(float(np.mean([1 - E[row[d]] @ c for d in ds])))
        fo = [E[row[p]] for p in fol_by_seed[sid] if p in row]
        d_hu.append(float(np.mean([1 - v @ c for v in fo])) if fo else float("nan"))
    out["distance"] = (ci(d_ai), ci(d_hu))
    # frontier alignment per (field, t): pooled keywords of the group's docs vs follow-on docs
    al_ai, al_hu, al_hu_eq = [], [], []
    by_ft = collections.defaultdict(list)
    for sid, d in items: by_ft[(sset[sid]["field"], sset[sid]["t"])].append((sid, d))
    for key, lst in by_ft.items():
        if key not in F or not F[key]: continue
        K = set().union(*[kws.get(d, set()) for _, d in lst]); al_ai.append(len(K & F[key]) / len(F[key]))
        Kh = set().union(*[kws.get(p, set()) for sid, _ in lst for p in fol_by_seed[sid]]); al_hu.append(len(Kh & F[key]) / len(F[key]))
        # equal-n human side: per seed set, sample as many follow-ons as the group has ideas for that seed
        per_seed = collections.Counter(sid for sid, _ in lst); hs = []
        for sid, n in per_seed.items():
            fo = [p for p in fol_by_seed[sid] if p in kws]; hs += rng.sample(fo, min(n, len(fo)))
        Kh2 = set().union(*[kws.get(p, set()) for p in hs]) if hs else set(); al_hu_eq.append(len(Kh2 & F[key]) / len(F[key]))
    out["alignment"] = (ci(al_ai), ci(al_hu))
    out["alignment_eqn"] = (ci(al_ai), ci(al_hu_eq))
    # impact per seed set
    i_ai, i_hu = [], []
    for sid, ds in by_seed.items():
        s = sset[sid]
        i_ai.append(float(np.nanmean([impact(E[row[d]], s["area"], s["t"]) for d in ds])))
        fo = [impact(E[row[p]], s["area"], s["t"]) for p in fol_by_seed[sid] if p in row]
        i_hu.append(float(np.nanmean(fo)) if fo else float("nan"))
    out["impact"] = (ci(i_ai), ci(i_hu))
    # variant: seed papers removed from the neighbour pool (they were selected by citation count, so a doc that sits
    # next to them inherits their high residual; AI ideas sit closer to the seeds than follow-ons do)
    i_ai2, i_hu2 = [], []
    for sid, ds in by_seed.items():
        s = sset[sid]; ex = set(s["seeds"])
        i_ai2.append(float(np.nanmean([impact(E[row[d]], s["area"], s["t"], ex) for d in ds])))
        fo = [impact(E[row[p]], s["area"], s["t"], ex) for p in fol_by_seed[sid] if p in row]
        i_hu2.append(float(np.nanmean(fo)) if fo else float("nan"))
    out["impact_noseed"] = (ci(i_ai2), ci(i_hu2))
    # keyword count parity (paper: 11.80 vs 11.88)
    out["kw_per_doc"] = (float(np.mean([len(kws.get(d, ())) for _, d in items])),
                         float(np.mean([len(kws.get(p, ())) for sid, _ in items for p in fol_by_seed[sid] if p in kws] or [float("nan")])))
    out["n_docs"] = len([1 for _, d in items if d in row])
    return out

res = {tag: group_measures(tag, items) for tag, items in sorted(ideas.items())}
json.dump(res, open(f"{DD}/results.json", "w"), indent=1)
def fmt(c): return f"{c[0]:.3f} [{c[1]:.3f}, {c[2]:.3f}] (n={c[3]})"
for tag, r in res.items():
    print(f"\n== {tag}  ideas={r['n_docs']}  kw/doc AI={r['kw_per_doc'][0]:.2f} human={r['kw_per_doc'][1]:.2f}")
    for m in ("breadth", "distance", "alignment", "alignment_eqn", "impact", "impact_noseed"):
        a, h = r[m]; d = a[0] - h[0]
        print(f"  {m:10s} AI {fmt(a)}   human {fmt(h)}   AI-human {d:+.3f}")
