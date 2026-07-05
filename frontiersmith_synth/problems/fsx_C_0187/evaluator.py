#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_C_0187 -- "Ski Resort Lift Traffic: Skier-Segment Clustering"
(family: classical-ml-algorithm; eval_form: quality-metric; MLS-Bench ml-* shape).

A large alpine resort logs, for each guest-day, a feature vector describing how a
guest used the mountain (mean lift-ride altitude, run-difficulty preference, number
of lift rides, distance skied, chairlift-vs-gondola ratio, rest-stop count, ...).
Marketing wants to recover the latent GUEST SEGMENTS ("first-timers", "powder
hounds", "family cruisers", "park rats", "apres-ski socialites", ...) from these
usage vectors WITHOUT any labels.  The true segment of every guest-day is known to
this evaluator but HIDDEN from the candidate.

The candidate designs an UNSUPERVISED CLUSTERING heuristic.  It is run as an
ISOLATED subprocess (isorun): it reads ONE JSON "public instance" from stdin and
writes ONE JSON answer (a length-N list of integer cluster labels) to stdout.  It
never sees the true segments, the hidden data, or this evaluator's memory.

Public instance JSON (what the candidate reads on stdin):
    {
      "X":    [[float, ...], ...],   # N x d guest-day usage matrix (a fresh copy)
      "n":    int,                   # number of guest-days N
      "d":    int,                   # number of usage features
      "k":    int,                   # number of segments to recover (given)
      "seed": int                    # a per-instance seed the candidate MAY use for its own RNG
    }

Answer JSON (what the candidate writes on stdout):
    [int, int, ..., int]            # length N; label[i] = segment id assigned to guest-day i
                                    # (ids are arbitrary; any partition into <= k groups is scored
                                    #  permutation-invariantly)

Quality is measured by the ADJUSTED RAND INDEX (ARI) between the candidate's
partition and the hidden segments, recomputed deterministically here from the
contingency table (no sklearn, no wall-time).  The resort family is intentionally
DIVERSE -- compact well-separated segments (centroid/k-means friendly), interleaving
"beginner-vs-expert" crescents and concentric ability rings (non-convex; needs a
connectivity/graph method), anisotropically stretched segments, wildly unequal
segment spread, and high-dimensional profiles padded with noise features.  No single
off-the-shelf clustering wins them all, and the final score is a GEOMETRIC MEAN over
per-instance normalized scores, so a method that overfits one geometry and collapses
on another is heavily penalized.  This rewards a clustering algorithm that
GENERALIZES.

Per-instance normalization is an affine anchor against the evaluator's own internal
baseline (a rank-quantile split of the FIRST feature into k contiguous groups):

    r = clamp( 0.1 + 0.9 * (ari_cand - ari_base) / max(1 - ari_base, MIN_DENOM), 0, 1 )

so a candidate that merely reproduces the baseline maps to ~0.1 and a perfect
partition (ARI = 1) maps to 1.0.  Valid instances are floored to a small positive
value so the geometric mean stays defined; an instance where the candidate raises,
returns the wrong shape, or emits non-finite / non-integer labels scores exactly
0.0 (dragging the geometric mean to 0).

CLI:  python3 evaluator.py <candidate.py>
Prints:
  Ratio: <geometric mean of per-instance r, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import os
# Cap BLAS/OMP threads so numpy imports cleanly inside isorun's memory-capped child.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import sys, json, math
import numpy as np
import isorun

VALID_FLOOR = 0.02     # floor for VALID instances so gmean stays defined
MIN_DENOM = 0.15       # cap on 1/(1-ari_base) so an easy instance still leaves headroom
CAND_TIMEOUT = 20


# ============================ resort family (instances) ====================
def _shuffle(rng, X, y):
    perm = rng.permutation(X.shape[0])
    return X[perm].astype(np.float64), y[perm].astype(np.int64)


def _gen_blobs(seed, n, d, k, sep, spread):
    """Compact, well-separated Gaussian segments (centroid/k-means friendly)."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, 1.0, size=(k, d))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12
    centers *= sep
    assign = rng.integers(0, k, size=n)
    X = centers[assign] + rng.normal(0.0, spread, size=(n, d))
    return _shuffle(rng, X, assign)


def _gen_moons(seed, n, noise):
    """Two interleaving crescents (beginner vs expert flow); k=2, non-convex:
    k-means slices them in half, a connectivity/graph method separates them."""
    rng = np.random.default_rng(seed)
    n0 = n // 2
    n1 = n - n0
    t0 = np.linspace(0.0, math.pi, n0)
    t1 = np.linspace(0.0, math.pi, n1)
    X0 = np.stack([np.cos(t0), np.sin(t0)], axis=1)
    X1 = np.stack([1.0 - np.cos(t1), 1.0 - np.sin(t1) - 0.5], axis=1)
    X = np.vstack([X0, X1]) + rng.normal(0.0, noise, size=(n, 2))
    y = np.array([0] * n0 + [1] * n1, dtype=np.int64)
    return _shuffle(rng, X, y)


def _gen_rings(seed, n, k, noise):
    """k concentric ability rings; k-means carves pie slices, radius/connectivity
    structure recovers the rings."""
    rng = np.random.default_rng(seed)
    per = [n // k] * k
    per[-1] += n - sum(per)
    Xs, ys = [], []
    for c in range(k):
        m = per[c]
        radius = 1.0 + 2.2 * c
        ang = rng.uniform(0.0, 2.0 * math.pi, size=m)
        r = radius + rng.normal(0.0, noise, size=m)
        Xs.append(np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1))
        ys.append(np.full(m, c, dtype=np.int64))
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    return _shuffle(rng, X, y)


def _gen_aniso(seed, n, d, k, sep, spread, shear):
    """Well-separated blobs then a random linear (shear+scale) transform, so the
    segments become long thin diagonals -- isotropic Euclidean k-means struggles."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, 1.0, size=(k, d)) * sep
    assign = rng.integers(0, k, size=n)
    X = centers[assign] + rng.normal(0.0, spread, size=(n, d))
    T = np.eye(d) + shear * rng.normal(0.0, 1.0, size=(d, d))
    X = X @ T
    return _shuffle(rng, X, assign)


def _gen_varied(seed, n, d, k, sep):
    """Segments with wildly UNEQUAL spread (a tight family-cruiser core plus a
    diffuse powder-hound cloud); equal-variance k-means over-claims the diffuse one."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, 1.0, size=(k, d))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12
    centers *= sep
    spreads = np.exp(rng.uniform(np.log(0.18), np.log(1.4), size=k))
    assign = rng.integers(0, k, size=n)
    X = centers[assign] + rng.normal(0.0, 1.0, size=(n, d)) * spreads[assign][:, None]
    return _shuffle(rng, X, assign)


def _gen_highdim(seed, n, d_sig, d_noise, k, sep, spread):
    """Signal blobs in d_sig dims padded with d_noise pure-noise features on a
    larger scale -- distance is diluted; standardization + the right features matter."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, 1.0, size=(k, d_sig)) * sep
    assign = rng.integers(0, k, size=n)
    Xs = centers[assign] + rng.normal(0.0, spread, size=(n, d_sig))
    Xn = rng.normal(0.0, 2.0, size=(n, d_noise))
    X = np.hstack([Xs, Xn])
    return _shuffle(rng, X, assign)


def _build_instances():
    specs = [
        ("blobs",   dict(seed=1871, n=200, d=5, k=4, sep=6.0, spread=1.0)),
        ("blobs",   dict(seed=1872, n=240, d=6, k=5, sep=5.5, spread=1.1)),   # harder: tighter/more k
        ("moons",   dict(seed=1873, n=200, noise=0.10)),
        ("moons",   dict(seed=1874, n=260, noise=0.14)),                      # harder: noisier
        ("rings",   dict(seed=1875, n=220, k=3, noise=0.16)),
        ("rings",   dict(seed=1876, n=240, k=2, noise=0.20)),                 # held-out: k=2
        ("aniso",   dict(seed=1877, n=210, d=4, k=4, sep=5.0, spread=0.9, shear=0.6)),
        ("aniso",   dict(seed=1878, n=250, d=5, k=4, sep=4.5, spread=1.0, shear=0.8)),  # harder
        ("varied",  dict(seed=1879, n=210, d=4, k=4, sep=6.0)),
        ("varied",  dict(seed=1880, n=250, d=5, k=5, sep=6.5)),               # held-out: more k
        ("highdim", dict(seed=1881, n=220, d_sig=4, d_noise=10, k=4, sep=5.5, spread=1.0)),
        ("highdim", dict(seed=1882, n=250, d_sig=5, d_noise=16, k=5, sep=5.0, spread=1.0)),  # held-out
    ]
    gens = {"blobs": _gen_blobs, "moons": _gen_moons, "rings": _gen_rings,
            "aniso": _gen_aniso, "varied": _gen_varied, "highdim": _gen_highdim}
    out = []
    for kind, p in specs:
        X, y = gens[kind](**p)
        k = int(len(np.unique(y)))
        out.append({"name": f"{kind}{p['seed']}", "X": X, "y": y, "k": k})
    return out


# ============================ Adjusted Rand Index ==========================
def _ari(y_true, y_pred):
    """Deterministic ARI from the contingency table (permutation-invariant)."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    n = y_true.shape[0]
    _, t = np.unique(y_true, return_inverse=True)
    _, p = np.unique(y_pred, return_inverse=True)
    nt = int(t.max()) + 1
    npd = int(p.max()) + 1
    cont = np.zeros((nt, npd), dtype=np.int64)
    np.add.at(cont, (t, p), 1)

    def comb2(x):
        x = np.asarray(x, dtype=np.float64)
        return x * (x - 1.0) / 2.0

    sum_ij = float(comb2(cont).sum())
    a = cont.sum(axis=1)
    b = cont.sum(axis=0)
    sum_a = float(comb2(a).sum())
    sum_b = float(comb2(b).sum())
    total = n * (n - 1) / 2.0
    if total <= 0:
        return 0.0
    expected = sum_a * sum_b / total
    max_index = 0.5 * (sum_a + sum_b)
    denom = max_index - expected
    if abs(denom) < 1e-12:
        # both partitions trivial (e.g. all-in-one) -> no information
        return 0.0
    return (sum_ij - expected) / denom


# ============================ internal baseline ============================
def _baseline_labels(X, k):
    """Rank-quantile split of the FIRST feature into k contiguous groups.
    A weak generic partition that the trivial candidate reproduces exactly."""
    f = np.asarray(X[:, 0], dtype=np.float64)
    n = f.shape[0]
    order = np.argsort(f, kind="mergesort")
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n, dtype=np.int64)
    lab = (rank * k) // n
    np.clip(lab, 0, k - 1, out=lab)
    return lab


# ============================ candidate answer handling ====================
def _valid_labels(ans, n):
    """Return an int64 array of shape (n,) or None if the answer is invalid."""
    if isinstance(ans, dict):
        ans = ans.get("labels", None)
    if ans is None:
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
    if not np.all(np.abs(arr - np.round(arr)) < 1e-6):
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
        X, y, k = inst["X"], inst["y"], inst["k"]
        n = X.shape[0]
        ari_base = _ari(y, _baseline_labels(X, k))
        denom = max(1.0 - ari_base, MIN_DENOM)

        public = {
            "X": X.tolist(),
            "n": int(n),
            "d": int(X.shape[1]),
            "k": int(k),
            "seed": int(20240187 + n),
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
            ari_cand = _ari(y, lab)
        except Exception:
            vec.append(0.0)
            continue

        r = 0.1 + 0.9 * (ari_cand - ari_base) / denom
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
