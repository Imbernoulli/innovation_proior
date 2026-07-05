# FROZEN evaluator -- do not ship to solvers as editable.
# Pandemic contact-net community detection: the candidate is a standalone clustering
# ALGORITHM. For each seeded "contact-exposure" dataset it receives ONLY the public
# 2-D exposure coordinates plus the number of outbreak clusters k, and must return a
# cluster label for every individual. The evaluator recomputes the Adjusted Rand Index
# (ARI) against the HIDDEN ground-truth community labels and aggregates by GEOMETRIC
# MEAN across datasets, so a method that overfits one geometry (e.g. round household
# blobs) but fails another (ring / chain transmission) is heavily penalized.
# Deterministic + isolated: the candidate runs in its own subprocess (isorun) and never
# sees the hidden labels.
import os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")
import sys, json, math
import numpy as np
import isorun

N_INST = 10


# --------------------------------------------------------------------------- #
# Deterministic dataset family (skinned as pandemic contact exposures).       #
# Each generator returns (X, y) with X = float coords, y = int community id.   #
# --------------------------------------------------------------------------- #
def _gen_blobs(rng, n, k, spread, sep):
    ang = np.linspace(0.0, 2 * np.pi, k, endpoint=False)
    centers = np.stack([sep * np.cos(ang), sep * np.sin(ang)], axis=1)
    per = n // k
    Xs, ys = [], []
    for j in range(k):
        m = per if j < k - 1 else n - per * (k - 1)
        pts = centers[j] + rng.normal(0.0, spread, size=(m, 2))
        Xs.append(pts); ys.append(np.full(m, j))
    return np.vstack(Xs), np.concatenate(ys)


def _gen_aniso(rng, n, k, spread, sep):
    X, y = _gen_blobs(rng, n, k, spread, sep)
    T = np.array([[1.0, 0.75], [0.15, 0.55]])   # fixed anisotropic shear
    return X @ T.T, y


def _gen_moons(rng, n, noise):
    m1 = n // 2; m2 = n - m1
    t1 = np.linspace(0.0, np.pi, m1)
    t2 = np.linspace(0.0, np.pi, m2)
    o = np.stack([np.cos(t1), np.sin(t1)], axis=1)
    i = np.stack([1.0 - np.cos(t2), 1.0 - np.sin(t2) - 0.5], axis=1)
    X = np.vstack([o, i]) + rng.normal(0.0, noise, size=(n, 2))
    y = np.concatenate([np.zeros(m1, dtype=int), np.ones(m2, dtype=int)])
    return X, y


def _gen_circles(rng, n, k, noise):
    per = n // k
    Xs, ys = [], []
    for j in range(k):
        m = per if j < k - 1 else n - per * (k - 1)
        r = 1.0 + 1.6 * j
        th = rng.uniform(0.0, 2 * np.pi, size=m)
        pts = np.stack([r * np.cos(th), r * np.sin(th)], axis=1)
        pts = pts + rng.normal(0.0, noise, size=(m, 2))
        Xs.append(pts); ys.append(np.full(m, j))
    return np.vstack(Xs), np.concatenate(ys)


def make_instances():
    """Ten seeded contact-exposure datasets with deliberately varied transmission
    geometry and a difficulty gradient (later instances noisier / more clusters)."""
    specs = [
        ("blobs",   190, 3, dict(spread=0.55, sep=6.0)),   # 0 easy households
        ("blobs",   220, 4, dict(spread=0.9,  sep=6.0)),   # 1 moderate overlap
        ("blobs",   200, 3, dict(spread=1.55, sep=4.5)),   # 2 heavy overlap (hard)
        ("aniso",   210, 3, dict(spread=0.55, sep=5.5)),   # 3 elongated chains
        ("moons",   200, 2, dict(noise=0.09)),             # 4 ring transmission
        ("moons",   220, 2, dict(noise=0.16)),             # 5 noisy rings (hard)
        ("circles", 210, 2, dict(noise=0.10)),             # 6 concentric spread
        ("circles", 240, 3, dict(noise=0.14)),             # 7 three rings (hard)
        ("aniso",   240, 4, dict(spread=0.7,  sep=5.5)),   # 8 four chains (hard)
        ("moons",   200, 2, dict(noise=0.22)),             # 9 very noisy (held-out hard)
    ]
    out = []
    for s, (kind, n, k, kw) in enumerate(specs):
        rng = np.random.RandomState(2027 + 7 * s)
        if kind == "blobs":
            X, y = _gen_blobs(rng, n, k, **kw)
        elif kind == "aniso":
            X, y = _gen_aniso(rng, n, k, **kw)
        elif kind == "moons":
            X, y = _gen_moons(rng, n, **kw)
        elif kind == "circles":
            X, y = _gen_circles(rng, n, k, **kw)
        else:
            raise ValueError(kind)
        # deterministic shuffle so label order carries no information
        perm = rng.permutation(len(y))
        X = X[perm]; y = y[perm]
        out.append({
            "public": {"X": [[float(a), float(b)] for a, b in X], "k": int(k)},
            "hidden": {"y": [int(v) for v in y]},
        })
    return out


# --------------------------------------------------------------------------- #
# Adjusted Rand Index (deterministic, numpy).                                 #
# --------------------------------------------------------------------------- #
def _ari(labels_true, labels_pred):
    lt = np.asarray(labels_true)
    lp = np.asarray(labels_pred)
    tu = {v: i for i, v in enumerate(np.unique(lt))}
    pu = {v: i for i, v in enumerate(np.unique(lp))}
    r, c = len(tu), len(pu)
    cont = np.zeros((r, c), dtype=np.int64)
    for a, b in zip(lt, lp):
        cont[tu[a], pu[b]] += 1

    def comb2(x):
        x = np.asarray(x, dtype=np.float64)
        return x * (x - 1.0) / 2.0

    sum_ij = comb2(cont).sum()
    a_i = comb2(cont.sum(axis=1)).sum()
    b_j = comb2(cont.sum(axis=0)).sum()
    n = lt.shape[0]
    tot = comb2(np.array([n])).sum()
    if tot <= 0:
        return 0.0
    expected = a_i * b_j / tot
    maxi = 0.5 * (a_i + b_j)
    denom = maxi - expected
    if abs(denom) < 1e-12:
        # both trivial partitions -> perfectly matching only if identical structure
        return 1.0 if sum_ij == maxi else 0.0
    return float((sum_ij - expected) / denom)


def baseline(inst):
    """Trivial construction the evaluator computes itself: put everyone in ONE
    outbreak cluster -> ARI 0. Reported for reference; scoring floors valid answers."""
    y = inst["hidden"]["y"]
    return _ari(y, [0] * len(y))


def score(inst, ans):
    """Strictly validate the candidate labeling, then recompute ARI. A structurally
    valid labeling earns a normalized score in [0.1, 1.0] via 0.1 + 0.9*max(0,ARI);
    anything malformed scores 0."""
    y = inst["hidden"]["y"]
    n = len(y)
    if not isinstance(ans, dict):
        return False, 0.0
    labels = ans.get("labels")
    if not isinstance(labels, list) or len(labels) != n:
        return False, 0.0
    clean = []
    for v in labels:
        if isinstance(v, bool):
            return False, 0.0
        if not isinstance(v, (int, float)):
            return False, 0.0
        fv = float(v)
        if not math.isfinite(fv) or fv != int(fv):
            return False, 0.0
        clean.append(int(fv))
    ari = _ari(y, clean)
    if not math.isfinite(ari):
        return False, 0.0
    norm = 0.1 + 0.9 * max(0.0, min(1.0, ari))
    return True, float(norm)


def main():
    cand = sys.argv[1]
    insts = make_instances()
    vec = []
    for inst in insts:
        ans, st = isorun.run_candidate(cand, inst["public"], timeout=25)
        if st != "OK":
            vec.append(0.0); continue
        try:
            ok, obj = score(inst, ans)
        except Exception:
            ok = False
        if not ok:
            vec.append(0.0); continue
        vec.append(obj if (obj == obj and 0.0 <= obj <= 1.0) else 0.0)
    # geometric mean: a single collapsed dataset drags the whole score down,
    # rewarding transferable community detection over per-geometry overfitting.
    if all(v > 0.0 for v in vec):
        gm = math.exp(sum(math.log(v) for v in vec) / len(vec))
    else:
        gm = 0.0
    print("Ratio: %.6f" % gm)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


main()
