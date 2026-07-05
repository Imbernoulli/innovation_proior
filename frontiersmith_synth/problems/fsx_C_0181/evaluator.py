#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_C_0181 -- "Warehouse AGV Fleet: Fault-Telemetry Anomaly Scoring"
(family: classical-ml-algorithm; eval_form: quality-metric; MLS-Bench ml-* shape).

A fleet of autonomous guided vehicles (AGVs) in a warehouse continuously reports
telemetry snapshots.  Each snapshot is a d-dimensional feature vector (motor
current, wheel-slip, battery draw, vibration, path deviation, ...).  Most rows are
NOMINAL operation; a small unknown fraction are FAULTS (bearing wear, wheel slip,
sensor drift, ...).  The ground-truth fault label of every row is known to this
evaluator but HIDDEN from the candidate.

The candidate designs an UNSUPERVISED anomaly-scoring heuristic.  It is run as an
ISOLATED subprocess (isorun): it reads ONE JSON "public instance" from stdin and
writes ONE JSON answer (a length-N list of real-valued anomaly scores, higher ==
more anomalous) to stdout.  It never sees the labels, the hidden data, or this
evaluator's memory.

Public instance JSON (what the candidate reads on stdin):
    {
      "X":    [[float, ...], ...],   # N x d telemetry matrix (a fresh copy)
      "n":    int,                   # number of rows N
      "d":    int,                   # number of features
      "contamination": float,        # approximate fault fraction (a hint; AUC needs no threshold)
      "seed": int                    # per-instance seed the candidate MAY use for its own RNG
    }

Answer JSON (what the candidate writes on stdout):
    [float, float, ..., float]       # length N; score[i] = anomaly score of row i (higher = worse)

Quality is measured by the ROC-AUC between the candidate's scores and the hidden
fault labels, recomputed deterministically here (rank/Mann-Whitney form, average
ranks for ties; no sklearn, no wall-time).  The fleet family is intentionally
DIVERSE -- global far-outliers (distance-friendly), correlation-manifold breakers
(Mahalanobis/PCA-friendly), multi-modal operating regimes (density/kNN-friendly),
wildly-scaled features (standardization-friendly), and local-density outliers
(LOF-friendly).  No single off-the-shelf detector wins them all, and the final
score is a GEOMETRIC MEAN over per-instance normalized scores, so a method that
overfits one fleet structure and collapses on another is heavily penalized.  This
rewards an anomaly-scoring algorithm that GENERALIZES.

Per-instance normalization is an affine anchor against the evaluator's own
internal baseline (raw Euclidean distance to the global mean):

    r = clamp( 0.1 + 0.9 * (auc_cand - auc_base) / (1 - auc_base), 0, 1 )

so a candidate that merely reproduces the baseline maps to ~0.1 and a perfect
detector (AUC = 1) maps to 1.0.  Valid instances are floored to a small positive
value so the geometric mean stays defined; an instance where the candidate raises,
returns the wrong shape, or emits non-finite scores scores exactly 0.0 (dragging
the geometric mean to 0).

CLI:  python3 evaluator.py <candidate.py>
Prints:
  Ratio: <geometric mean of per-instance r, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import os
# Cap BLAS/OMP threads so numpy imports cleanly inside isorun's RLIMIT_AS-capped
# child (OpenBLAS otherwise reserves per-thread buffers that blow the address
# space).  isorun copies this environment into the candidate subprocess.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import sys, json, math
import numpy as np
import isorun

VALID_FLOOR = 0.02     # floor for VALID instances so gmean stays defined
MIN_DENOM = 0.05       # cap on 1/(1-auc_base) so an already-strong baseline still leaves headroom
CAND_TIMEOUT = 20


# ============================ fleet family (instances) =====================
def _counts(n, contam):
    n_anom = max(3, int(round(n * contam)))
    n_anom = min(n_anom, n // 3)
    return n - n_anom, n_anom


def _gen_global(seed, n, d, contam, radius):
    """Unimodal nominal cloud + far isotropic outliers (raw-distance friendly)."""
    rng = np.random.default_rng(seed)
    n_norm, n_anom = _counts(n, contam)
    Xn = rng.normal(0.0, 1.0, size=(n_norm, d))
    dirs = rng.normal(0.0, 1.0, size=(n_anom, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    rad = rng.uniform(radius, radius * 1.5, size=(n_anom, 1))
    Xa = dirs * rad + rng.normal(0.0, 1.1, size=(n_anom, d))
    X = np.vstack([Xn, Xa])
    y = np.array([0] * n_norm + [1] * n_anom, dtype=np.int64)
    return _shuffle(rng, X, y)


def _gen_correlated(seed, n, d, contam):
    """Nominals live on a low-rank correlated manifold; faults break the
    correlation (same marginal ranges, independent) -> raw distance-to-mean
    is blind, Mahalanobis / PCA-residual / local sparsity detect them."""
    rng = np.random.default_rng(seed)
    n_norm, n_anom = _counts(n, contam)
    rank = 2
    W = rng.normal(0.0, 1.0, size=(rank, d))
    Z = rng.normal(0.0, 1.0, size=(n_norm, rank))
    Xn = Z @ W + rng.normal(0.0, 0.25, size=(n_norm, d))
    # per-feature std of nominals, so faults match marginals but not the joint structure
    std = Xn.std(axis=0) + 1e-9
    Xa = rng.normal(0.0, 1.0, size=(n_anom, d)) * std
    X = np.vstack([Xn, Xa])
    y = np.array([0] * n_norm + [1] * n_anom, dtype=np.int64)
    return _shuffle(rng, X, y)


def _gen_multimodal(seed, n, d, contam, modes):
    """Several well-separated nominal operating regimes; faults fall in the
    gaps.  Distance-to-(global)mean flags the far regimes, not the faults;
    kNN / local density detects the gap points."""
    rng = np.random.default_rng(seed)
    n_norm, n_anom = _counts(n, contam)
    centers = rng.normal(0.0, 1.0, size=(modes, d))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12
    centers *= 5.0
    assign = rng.integers(0, modes, size=n_norm)
    Xn = centers[assign] + rng.normal(0.0, 1.0, size=(n_norm, d))
    lo, hi = X_bounds = (centers.min(axis=0) - 2.0, centers.max(axis=0) + 2.0)
    Xa = rng.uniform(lo, hi, size=(n_anom, d))
    X = np.vstack([Xn, Xa])
    y = np.array([0] * n_norm + [1] * n_anom, dtype=np.int64)
    return _shuffle(rng, X, y)


def _gen_scaled(seed, n, d, contam):
    """Features on wildly different scales; the fault deviates in a SMALL-scale
    feature.  Raw distance is dominated by the big-scale features and misses it;
    per-feature standardization recovers it."""
    rng = np.random.default_rng(seed)
    n_norm, n_anom = _counts(n, contam)
    scales = np.exp(rng.uniform(np.log(0.05), np.log(40.0), size=d))
    Xn = rng.normal(0.0, 1.0, size=(n_norm, d)) * scales
    Xa = rng.normal(0.0, 1.0, size=(n_anom, d)) * scales
    small = int(np.argmin(scales))
    # push faults out along the smallest-scale axis (invisible in raw distance)
    bump = rng.choice(np.array([-1.0, 1.0]), size=n_anom) * rng.uniform(2.4, 3.4, size=n_anom)
    Xa[:, small] = bump * scales[small]
    X = np.vstack([Xn, Xa])
    y = np.array([0] * n_norm + [1] * n_anom, dtype=np.int64)
    return _shuffle(rng, X, y)


def _gen_local(seed, n, d, contam):
    """Two nominal clusters of different density; faults are LOCAL outliers
    sitting just outside the DENSE cluster (closer to the global mean than the
    far sparse cluster).  Global distance ranks the sparse cluster as anomalous;
    local density detects the true faults."""
    rng = np.random.default_rng(seed)
    n_norm, n_anom = _counts(n, contam)
    n_dense = int(n_norm * 0.65)
    n_sparse = n_norm - n_dense
    c_dense = np.zeros(d)
    c_sparse = np.zeros(d); c_sparse[0] = 9.0
    Xd = c_dense + rng.normal(0.0, 0.4, size=(n_dense, d))
    Xs = c_sparse + rng.normal(0.0, 1.4, size=(n_sparse, d))
    # local outliers: a moderate radius off the DENSE cluster centre
    dirs = rng.normal(0.0, 1.0, size=(n_anom, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    Xa = c_dense + dirs * rng.uniform(2.0, 3.0, size=(n_anom, 1))
    X = np.vstack([Xd, Xs, Xa])
    y = np.array([0] * (n_dense + n_sparse) + [1] * n_anom, dtype=np.int64)
    return _shuffle(rng, X, y)


def _shuffle(rng, X, y):
    perm = rng.permutation(X.shape[0])
    return X[perm].astype(np.float64), y[perm].astype(np.int64)


def _build_instances():
    specs = [
        ("global", dict(seed=201, n=200, d=6, contam=0.10, radius=2.6)),
        ("global", dict(seed=202, n=260, d=6, contam=0.12, radius=2.2)),   # harder: smaller radius
        ("correlated", dict(seed=203, n=220, d=6, contam=0.10)),
        ("correlated", dict(seed=204, n=280, d=8, contam=0.10)),           # held-out: higher d
        ("multimodal", dict(seed=205, n=240, d=5, contam=0.12, modes=4)),
        ("multimodal", dict(seed=206, n=300, d=5, contam=0.10, modes=5)),  # held-out: more modes
        ("scaled", dict(seed=207, n=200, d=6, contam=0.10)),
        ("scaled", dict(seed=208, n=260, d=6, contam=0.12)),
        ("local", dict(seed=209, n=220, d=4, contam=0.12)),
        ("local", dict(seed=210, n=300, d=4, contam=0.10)),                # held-out: larger
    ]
    gens = {"global": _gen_global, "correlated": _gen_correlated,
            "multimodal": _gen_multimodal, "scaled": _gen_scaled, "local": _gen_local}
    out = []
    for kind, p in specs:
        X, y = gens[kind](**p)
        out.append({"name": f"{kind}{p['seed']}", "X": X, "y": y,
                    "contam": float(p["contam"])})
    return out


# ============================ ROC-AUC (deterministic) ======================
def _auc(y, s):
    """Rank-based ROC-AUC; average ranks for ties. y in {0,1}, higher s = anomaly."""
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    n = y.shape[0]
    n_pos = int((y == 1).sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0           # 1-indexed average rank
        ranks[order[i:j + 1]] = avg
        i = j + 1
    sum_pos = float(ranks[y == 1].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


# ============================ internal baseline ============================
def _baseline_scores(X):
    """Raw Euclidean distance to the global mean (a weak, generic detector)."""
    mu = X.mean(axis=0)
    return np.sqrt(np.sum((X - mu) ** 2, axis=1))


# ============================ candidate answer handling ====================
def _valid_scores(ans, n):
    """Return a float64 array of shape (n,) or None if the answer is invalid."""
    if isinstance(ans, dict):
        ans = ans.get("scores", None)
    if ans is None:
        return None
    try:
        s = np.asarray(ans, dtype=np.float64)
    except Exception:
        return None
    if s.ndim != 1 or s.shape[0] != n:
        return None
    if not np.all(np.isfinite(s)):
        return None
    return s


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <candidate.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()
    n_inst = len(instances)

    vec = []
    for inst in instances:
        X, y = inst["X"], inst["y"]
        n = X.shape[0]
        auc_base = _auc(y, _baseline_scores(X))
        denom = max(1.0 - auc_base, MIN_DENOM)

        public = {
            "X": X.tolist(),
            "n": int(n),
            "d": int(X.shape[1]),
            "contamination": inst["contam"],
            "seed": int(20240181 + n),
        }
        ans, st = isorun.run_candidate(cand, public, timeout=CAND_TIMEOUT)
        if st != "OK":
            vec.append(0.0)
            continue

        s = _valid_scores(ans, n)
        if s is None:
            vec.append(0.0)
            continue

        try:
            auc_cand = _auc(y, s)
        except Exception:
            vec.append(0.0)
            continue

        r = 0.1 + 0.9 * (auc_cand - auc_base) / denom
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
