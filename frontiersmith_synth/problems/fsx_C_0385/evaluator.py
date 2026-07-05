#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_C_0385 -- "Rooftop Garden Sensor Grid: Missing-Reading
Reconstruction" (family: classical-ml-algorithm; eval_form: quality-metric;
MLS-Bench ml-* shape).

A city runs a network of instrumented ROOFTOP GARDENS.  Each garden logs, per
observation window, a vector of environmental sensor readings (soil moisture,
canopy temperature, PAR light, air humidity, substrate pH, wind, CO2, leaf-wetness,
...).  Cheap sensors drop out constantly, so the raw matrix is riddled with holes.
Operations wants the missing readings RECONSTRUCTED from the surviving ones so the
irrigation controller can act on complete vectors.

The candidate designs a MISSING-VALUE IMPUTATION heuristic.  It is run as an
ISOLATED subprocess (isorun): it reads ONE JSON "public instance" from stdin (the
sensor matrix with holes) and writes ONE JSON answer (its fill for each hole) to
stdout.  It NEVER sees the true (withheld) readings, held-out data, or this
evaluator's memory.

Public instance JSON (stdin):
    {
      "X":       [[float|null, ...], ...],  # N x d matrix; null = a dropped reading
      "n":       int,                       # number of observation windows N
      "d":       int,                       # number of sensor channels
      "missing": [[i, j], ...],             # the (row, col) holes to fill, in a FIXED order
      "n_miss":  int,                       # == len(missing)
      "seed":    int                        # a per-instance seed the candidate MAY use
    }
  Observed entries are finite floats; every hole is `null` in X and appears exactly
  once in `missing`.

Answer JSON (stdout):
    [float, float, ...]                     # length n_miss; fill[t] imputes hole missing[t]
  (a dict {"fill": [...]} is also accepted).  Any non-finite value, wrong length, or
  non-numeric entry scores 0 on that instance.

Quality is the coefficient of determination R^2 between the candidate's fills and the
withheld true readings, recomputed deterministically here (no sklearn, no wall-time).
The garden family is intentionally DIVERSE in reconstructability: some gardens are
strongly LOW-RANK (a few latent drivers -- sun, watering cycle -- so a channel is
largely predictable from its neighbours) and some are near-FULL-RANK + noisy (little
cross-channel structure, so even an ideal linear model can recover only a sliver).
Missing fractions and grid sizes also vary.  The final score is a GEOMETRIC MEAN over
per-instance normalized scores, so a method tuned to one regime that collapses on
another is heavily penalized -- this rewards an imputer that GENERALIZES.

Per-instance normalization is an affine anchor against the evaluator's own internal
baseline (per-CHANNEL mean imputation, i.e. fill each hole with the observed mean of
its column):

    r = clamp( 0.1 + 0.9 * (r2_cand - r2_base) / max(1 - r2_base, MIN_DENOM), 0, 1 )

so a candidate that merely reproduces mean imputation maps to ~0.1 and a perfect
reconstruction (R^2 = 1) maps to 1.0.  Valid instances are floored to a small positive
value so the geometric mean stays defined; an instance where the candidate raises,
returns the wrong shape, or emits non-finite values scores exactly 0.0.

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
MIN_DENOM = 0.15       # cap on 1/(1-r2_base) so an easy instance still leaves headroom
CAND_TIMEOUT = 20


# ============================ garden family (instances) ====================
def _gen_instance(seed, n, d, rank, noise, miss_frac):
    """Low-rank sensor field X = Z W (a few latent drivers) plus channel noise,
    standardized per channel; then holes are punched MCAR.  `rank` small + `noise`
    small  -> strongly reconstructable; `rank` ~ d + `noise` large -> nearly
    unreconstructable (only mean-level info survives)."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, rank))
    W = rng.standard_normal((rank, d))
    X = Z @ W
    X = X + noise * rng.standard_normal((n, d))
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-9
    X = (X - mu) / sd

    # MCAR mask: True = missing
    M = rng.random((n, d)) < miss_frac
    # guard: each channel keeps >= min_obs observed (so a column mean/regression exists)
    min_obs = max(4, int(0.4 * n))
    for j in range(d):
        obs = np.flatnonzero(~M[:, j])
        if obs.size < min_obs:
            need = min_obs - obs.size
            missrows = np.flatnonzero(M[:, j])          # sorted -> deterministic
            M[missrows[:need], j] = False
    # guard: each row keeps >= 1 observed (so row-based methods have an anchor)
    for i in range(n):
        if M[i].all():
            M[i, 0] = False
    # guard: at least a handful of holes to score
    if M.sum() < 8:
        flat = rng.permutation(n * d)[:8]
        for f in flat:
            M[f // d, f % d] = True

    miss = [[int(i), int(j)] for i in range(n) for j in range(d) if M[i, j]]
    y_true = np.array([X[i, j] for i, j in miss], dtype=np.float64)
    return X, M, miss, y_true


def _build_instances():
    # (seed, n, d, rank, noise, miss_frac)  -- mixed reconstructability + held-out regimes
    specs = [
        (30851, 140, 8,  2, 0.10, 0.20),   # very low-rank, clean -> easy
        (30852, 160, 10, 3, 0.20, 0.25),   # low-rank, mild noise
        (30853, 150, 8,  4, 0.30, 0.22),   # medium
        (30854, 180, 12, 5, 0.30, 0.28),   # medium, larger grid / more holes
        (30855, 140, 8,  6, 0.45, 0.25),   # near-full-rank + noisy -> hard
        (30856, 170, 12, 10, 0.55, 0.30),  # full-rank + noisy -> very hard (headroom)
        (30857, 200, 9,  3, 0.15, 0.35),   # low-rank but heavy missingness (held-out)
        (30858, 130, 7,  2, 0.25, 0.20),   # low-rank, small grid
        (30859, 160, 11, 4, 0.35, 0.24),   # medium-high dim
        (30860, 150, 10, 7, 0.50, 0.26),   # hard, high rank
        (30861, 190, 14, 4, 0.25, 0.30),   # low-rank, wide grid (held-out)
        (30862, 145, 8,  5, 0.40, 0.32),   # medium-hard, heavy missingness
    ]
    out = []
    for (seed, n, d, rank, noise, miss_frac) in specs:
        X, M, miss, y_true = _gen_instance(seed, n, d, rank, noise, miss_frac)
        out.append({"name": f"g{seed}", "X": X, "M": M, "miss": miss,
                    "y_true": y_true, "seed": seed})
    return out


# ============================ R^2 ==========================================
def _r2(y_true, y_hat):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_hat = np.asarray(y_hat, dtype=np.float64)
    ybar = float(y_true.mean())
    sst = float(np.sum((y_true - ybar) ** 2))
    if sst <= 1e-12:
        return 0.0
    sse = float(np.sum((y_true - y_hat) ** 2))
    return 1.0 - sse / sst


# ============================ internal baseline ============================
def _baseline_fill(X, M, miss):
    """Per-channel mean imputation: fill each hole with the observed mean of its
    column.  A weak generic imputer that the trivial candidate reproduces exactly."""
    d = X.shape[1]
    col_mean = np.zeros(d, dtype=np.float64)
    for j in range(d):
        obs = X[~M[:, j], j]
        col_mean[j] = float(obs.mean()) if obs.size > 0 else 0.0
    return np.array([col_mean[j] for _, j in miss], dtype=np.float64)


# ============================ candidate answer handling ====================
def _valid_fill(ans, n_miss):
    """Return a float64 array of shape (n_miss,) or None if the answer is invalid."""
    if isinstance(ans, dict):
        ans = ans.get("fill", ans.get("values", None))
    if ans is None or isinstance(ans, (str, bytes)):
        return None
    try:
        arr = np.asarray(ans, dtype=np.float64)
    except Exception:
        return None
    if arr.ndim != 1 or arr.shape[0] != n_miss:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _public_X(X, M):
    """Build the stdin matrix: observed entries as floats, holes as null (None)."""
    n, d = X.shape
    out = []
    for i in range(n):
        row = []
        for j in range(d):
            row.append(None if M[i, j] else float(X[i, j]))
        out.append(row)
    return out


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <candidate.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        X, M, miss, y_true = inst["X"], inst["M"], inst["miss"], inst["y_true"]
        n, d = X.shape
        n_miss = len(miss)

        base_fill = _baseline_fill(X, M, miss)
        r2_base = _r2(y_true, base_fill)
        denom = max(1.0 - r2_base, MIN_DENOM)

        public = {
            "X": _public_X(X, M),
            "n": int(n),
            "d": int(d),
            "missing": [[int(i), int(j)] for i, j in miss],
            "n_miss": int(n_miss),
            "seed": int(20240385 + inst["seed"]),
        }
        ans, st = isorun.run_candidate(cand, public, timeout=CAND_TIMEOUT)
        if st != "OK":
            vec.append(0.0)
            continue

        fill = _valid_fill(ans, n_miss)
        if fill is None:
            vec.append(0.0)
            continue

        try:
            r2_cand = _r2(y_true, fill)
        except Exception:
            vec.append(0.0)
            continue

        r = 0.1 + 0.9 * (r2_cand - r2_base) / denom
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
