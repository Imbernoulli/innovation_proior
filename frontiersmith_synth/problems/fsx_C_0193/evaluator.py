#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_C_0193 -- "City Signal Grid: Intersection Regime Clustering"
(family: classical-ml-algorithm; eval_form: quality-metric; MLS-Bench ml-* shape).

A traffic-management authority runs the signal grid of a large city.  Every
intersection streams a flow signature -- a d-dimensional feature vector summarising
its demand pattern (approach volumes, turn ratios, peak-hour skew, queue spillback,
pedestrian phase load, ...).  Intersections belong to a small number of latent
OPERATING REGIMES (arterial corridors, downtown grid, residential feeders,
industrial gateways, ...); intersections in the same regime should share a signal
timing plan.  The regime label of every intersection is known to THIS evaluator but
HIDDEN from the candidate.

The candidate designs an UNSUPERVISED CLUSTERING heuristic.  It is run as an
ISOLATED subprocess (isorun): it reads ONE JSON "public instance" from stdin and
writes ONE JSON answer (a length-N list of integer cluster ids) to stdout.  It never
sees the true regime labels, held-out data, or this evaluator's memory.

Public instance JSON (what the candidate reads on stdin):
    {
      "X":    [[float, ...], ...],   # N x d flow-signature matrix (a fresh copy)
      "n":    int,                   # number of intersections N
      "d":    int,                   # number of features
      "k":    int,                   # number of regimes to recover (a hint)
      "seed": int                    # per-instance seed the candidate MAY use for its own RNG
    }

Answer JSON (what the candidate writes on stdout):
    [int, int, ..., int]             # length N; label[i] = cluster id of intersection i
    (a JSON object {"labels": [...]} is also accepted).  Cluster ids are arbitrary
    integers; only the PARTITION they induce matters (the metric is label-invariant).

Quality is the Adjusted Rand Index (ARI) between the candidate's partition and the
hidden regime labels, recomputed deterministically here (contingency form; no
sklearn, no wall-time).  The city family is intentionally DIVERSE -- compact
isotropic regimes (plain k-means friendly), sheared/anisotropic regimes
(whitening/PCA friendly), wildly-scaled feature units with large-scale nuisance
channels (standardization friendly), and unequal-size / unequal-spread regimes.
No single off-the-shelf recipe wins them all, and the final score is a GEOMETRIC
MEAN over per-instance normalized scores, so a method that overfits one city
structure and collapses on another is heavily penalized.  This rewards a clustering
algorithm that GENERALIZES.

Per-instance normalization is an affine anchor against the NULL clustering (every
intersection in one group, ARI = 0):

    r = clamp( 0.1 + 0.9 * ari_cand, 0, 1 )

so a candidate that only reproduces the null partition maps to ~0.1 and a perfect
recovery (ARI = 1) maps to 1.0.  Valid instances are floored to a small positive
value so the geometric mean stays defined; an instance where the candidate raises,
returns the wrong shape, or emits non-finite / non-integer ids scores exactly 0.0
(dragging the geometric mean to 0).

CLI:  python3 evaluator.py <candidate.py>
Prints:
  Ratio: <geometric mean of per-instance r, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import os
# Cap BLAS/OMP threads so numpy imports cleanly inside isorun's memory-capped child
# (OpenBLAS otherwise reserves per-thread buffers that blow the address space).
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import sys, json, math
import numpy as np
import isorun

VALID_FLOOR = 0.02     # floor for VALID instances so gmean stays defined
CAND_TIMEOUT = 20


# ============================ city family (instances) ======================
def _counts(rng, n, k, varsize):
    if varsize:
        w = rng.uniform(0.6, 2.4, size=k)
        w = w / w.sum()
        c = np.floor(w * n).astype(np.int64)
        c = np.maximum(c, 6)
        # fix rounding so the counts sum to exactly n
        diff = int(n - c.sum())
        i = 0
        while diff != 0:
            j = i % k
            if diff > 0:
                c[j] += 1; diff -= 1
            elif c[j] > 6:
                c[j] -= 1; diff += 1
            i += 1
    else:
        base = n // k
        c = np.full(k, base, dtype=np.int64)
        c[: n - base * k] += 1
    return c


def _base_blobs(rng, n, d, k, spread, std, varsize=False):
    """k isotropic Gaussian regimes; return X (n,d), y (n,)."""
    centers = rng.normal(0.0, spread, size=(k, d))
    counts = _counts(rng, n, k, varsize)
    parts = []
    ys = []
    for c in range(k):
        s = std
        if varsize:
            s = std * float(rng.uniform(0.6, 1.8))
        parts.append(centers[c] + rng.normal(0.0, s, size=(int(counts[c]), d)))
        ys.append(np.full(int(counts[c]), c, dtype=np.int64))
    X = np.vstack(parts)
    y = np.concatenate(ys)
    return _shuffle(rng, X, y)


def _gen_blobs(seed, n, d, k, spread, std=1.0):
    rng = np.random.default_rng(seed)
    return _base_blobs(rng, n, d, k, spread, std)


def _gen_varsize(seed, n, d, k, spread, std=1.0):
    rng = np.random.default_rng(seed)
    return _base_blobs(rng, n, d, k, spread, std, varsize=True)


def _gen_aniso(seed, n, d, k, spread, std=1.0):
    """Compact regimes passed through a shared ill-conditioned linear map so the
    clusters become elongated / correlated -- raw k-means is distorted; PCA
    whitening (decorrelation) recovers the geometry."""
    rng = np.random.default_rng(seed)
    X, y = _base_blobs(rng, n, d, k, spread, std)
    A = rng.normal(0.0, 1.0, size=(d, d))
    # bias toward anisotropy: stretch along a few random axes
    U, _, Vt = np.linalg.svd(A)
    s = np.exp(rng.uniform(np.log(0.25), np.log(4.0), size=d))
    s[0] *= 6.0            # one strongly stretched direction
    M = (U * s) @ Vt
    Xa = X @ M
    return Xa.astype(np.float64), y


def _gen_scaled(seed, n, d, k, spread, std=1.0):
    """Informative regimes live in (d-2) features on a small scale; TWO features
    are large-scale NUISANCE channels carrying no regime information.  Raw
    distance is dominated by the nuisance channels (k-means clusters noise);
    per-feature standardization restores the informative geometry."""
    rng = np.random.default_rng(seed)
    d_inf = d - 2
    X, y = _base_blobs(rng, n, d_inf, k, spread, std)
    # two large-scale nuisance channels (identical across regimes -> pure noise)
    noise = rng.normal(0.0, 1.0, size=(n, 2))
    X = np.hstack([X, noise])
    scales = np.ones(d)
    scales[:d_inf] = np.exp(rng.uniform(np.log(0.05), np.log(0.4), size=d_inf))
    scales[d_inf:] = np.exp(rng.uniform(np.log(20.0), np.log(60.0), size=2))
    perm = rng.permutation(d)           # hide which columns are nuisance
    Xs = (X * scales)[:, perm]
    return Xs.astype(np.float64), y


def _shuffle(rng, X, y):
    perm = rng.permutation(X.shape[0])
    return X[perm].astype(np.float64), y[perm].astype(np.int64)


def _build_instances():
    specs = [
        ("blobs",   dict(seed=301, n=480, d=6,  k=5, spread=1.9)),   # compact, mild overlap
        ("blobs",   dict(seed=302, n=600, d=8,  k=6, spread=1.4)),   # closer regimes (overlap)
        ("aniso",   dict(seed=303, n=540, d=6,  k=5, spread=1.9)),   # sheared
        ("aniso",   dict(seed=304, n=720, d=8,  k=6, spread=1.7)),   # held-out: bigger + tighter
        ("scaled",  dict(seed=305, n=500, d=6,  k=5, spread=2.6)),   # nuisance-scale channels
        ("scaled",  dict(seed=306, n=660, d=8,  k=6, spread=2.1)),
        ("varsize", dict(seed=307, n=560, d=6,  k=5, spread=2.0)),   # unequal size/spread
        ("varsize", dict(seed=308, n=780, d=7,  k=7, spread=2.1)),   # held-out: more regimes
        ("blobs",   dict(seed=309, n=640, d=10, k=8, spread=1.7)),   # many regimes, high-d
        ("aniso",   dict(seed=310, n=840, d=10, k=8, spread=1.8)),   # large held-out sheared
    ]
    gens = {"blobs": _gen_blobs, "aniso": _gen_aniso,
            "scaled": _gen_scaled, "varsize": _gen_varsize}
    out = []
    for kind, p in specs:
        X, y = gens[kind](**p)
        out.append({"name": f"{kind}{p['seed']}", "X": X, "y": y, "k": int(p["k"])})
    return out


# ============================ Adjusted Rand Index ==========================
def _ari(y_true, y_pred):
    """Deterministic Adjusted Rand Index via the contingency table."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = y_true.shape[0]
    if n < 2:
        return 0.0
    _, it = np.unique(y_true, return_inverse=True)
    _, ip = np.unique(y_pred, return_inverse=True)
    nt = it.max() + 1
    npd = ip.max() + 1
    cont = np.zeros((int(nt), int(npd)), dtype=np.float64)
    np.add.at(cont, (it, ip), 1.0)

    def comb2(x):
        return x * (x - 1.0) / 2.0

    sum_ij = comb2(cont).sum()
    a = cont.sum(axis=1)
    b = cont.sum(axis=0)
    sum_a = comb2(a).sum()
    sum_b = comb2(b).sum()
    total = comb2(float(n))
    expected = (sum_a * sum_b) / total if total > 0 else 0.0
    max_index = 0.5 * (sum_a + sum_b)
    denom = max_index - expected
    if denom <= 0.0:
        # both partitions trivial/identical -> perfect agreement by convention
        return 1.0 if sum_ij == max_index else 0.0
    return float((sum_ij - expected) / denom)


# ============================ candidate answer handling ====================
def _valid_labels(ans, n):
    """Return an int64 array of shape (n,) or None if the answer is invalid."""
    if isinstance(ans, dict):
        ans = ans.get("labels", None)
    if ans is None:
        return None
    if not isinstance(ans, list):
        return None
    if len(ans) != n:
        return None
    try:
        arr = np.asarray(ans, dtype=np.float64)
    except Exception:
        return None
    if arr.ndim != 1 or arr.shape[0] != n:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    # labels must be (near-)integer valued
    if not np.all(np.abs(arr - np.round(arr)) < 1e-9):
        return None
    return np.round(arr).astype(np.int64)


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <candidate.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        X, y = inst["X"], inst["y"]
        n = X.shape[0]

        public = {
            "X": X.tolist(),
            "n": int(n),
            "d": int(X.shape[1]),
            "k": int(inst["k"]),
            "seed": int(20240193 + n),
        }
        ans, st = isorun.run_candidate(cand, public, timeout=CAND_TIMEOUT)
        if st != "OK":
            vec.append(0.0)
            continue

        lab = _valid_labels(ans, n)
        if lab is None:
            vec.append(0.0)
            continue

        try:
            ari = _ari(y, lab)
        except Exception:
            vec.append(0.0)
            continue

        r = 0.1 + 0.9 * ari
        if r < 0.0:
            r = 0.0
        elif r > 1.0:
            r = 1.0
        if r < VALID_FLOOR:
            r = VALID_FLOOR
        vec.append(float(r))

    if any(v <= 0.0 for v in vec):
        ratio = 0.0
    else:
        ratio = math.exp(sum(math.log(v) for v in vec) / len(vec))

    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(v, 6) for v in vec]))


if __name__ == "__main__":
    main()
