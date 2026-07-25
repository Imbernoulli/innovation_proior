#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic scorer for 'shared-basis signal probes'.

Reads the instance from <in>, the participant probe matrix from <out>.
Validates feasibility strictly; on any violation prints Ratio: 0.0 and exits 0.

Score: hidden test signals (T per family, seeded by a fixed constant combined with
the instance parameters -- never printed anywhere) are measured through the probe
matrix P with additive Gaussian noise (sigma fixed), then reconstructed by a FIXED
OMP decoder run with each family's own basis. F = maximum over families of the mean
relative reconstruction error (minimization). Baseline B = F of the naive
construction P[j, j % n] = pmax. Prints Ratio: min(1, 0.1 * B / F).

Bit-for-bit deterministic: single fixed seed; no wall-time; no dict/set iteration
order dependence.
"""
import sys
import math
import numpy as np

T = 24               # hidden test signals per family
SIGMA = 0.2          # measurement noise std
SIG_SEED = 987123    # fixed constant; combined with instance parameters below
SCALE = 65.0         # score multiplier: Ratio = min(1, (SCALE/1000) * B / F)


def fail(msg):
    print("infeasible: %s" % msg, file=sys.stderr)
    print("Ratio: 0.000000")
    sys.exit(0)


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    pos = 0
    try:
        n = int(toks[pos]); m = int(toks[pos + 1]); K = int(toks[pos + 2])
        s = int(toks[pos + 3]); pmax = int(toks[pos + 4])
        pos += 5
        bases = []
        for _ in range(K):
            d = int(toks[pos]); pos += 1
            vals = [float(x) for x in toks[pos:pos + n * d]]
            pos += n * d
            B = np.array(vals, dtype=float).reshape(n, d)
            bases.append(B)
    except (IndexError, ValueError) as e:
        fail("cannot parse instance: %s" % e)
    return n, m, K, s, pmax, bases


def read_output(path, m, n, pmax):
    try:
        with open(path) as f:
            lines = [ln.split() for ln in f.read().splitlines() if ln.strip()]
    except OSError as e:
        fail("cannot read output: %s" % e)
    if len(lines) != m:
        fail("expected %d probe rows, got %d" % (m, len(lines)))
    P = np.zeros((m, n), dtype=float)
    for j, ln in enumerate(lines):
        if len(ln) != n:
            fail("row %d: expected %d entries, got %d" % (j, n, len(ln)))
        for i, tok in enumerate(ln):
            try:
                v = float(tok)
            except ValueError:
                fail("row %d col %d: not a number: %r" % (j, i, tok))
            if not math.isfinite(v):
                fail("row %d col %d: non-finite value" % (j, i))
            if abs(v - round(v)) > 1e-9:
                fail("row %d col %d: non-integer value %r" % (j, i, tok))
            v = round(v)
            if abs(v) > pmax:
                fail("row %d col %d: |%d| exceeds pmax=%d" % (j, i, v, pmax))
            P[j, i] = v
    return P


def omp_decode(P, B, y, s):
    """FIXED decoder: s iterations of orthogonal matching pursuit with
    column-norm-normalized correlation and least-squares refit. Do not change."""
    D = P @ B  # m x d
    colnorm = np.linalg.norm(D, axis=0) + 1e-12
    supp = []
    r = y.copy()
    a = np.zeros(B.shape[1])
    coef = np.zeros(0)
    for _ in range(s):
        corr = np.abs(D.T @ r) / colnorm
        for j in supp:
            corr[j] = -1.0
        j = int(np.argmax(corr))
        supp.append(j)
        Ds = D[:, supp]
        coef, *_ = np.linalg.lstsq(Ds, y, rcond=None)
        r = y - Ds @ coef
    a[np.array(supp)] = coef
    return B @ a


def family_max_error(P, bases, s, sig_seed):
    rng = np.random.default_rng(sig_seed)
    m = P.shape[0]
    fam_err = []
    for B in bases:
        d = B.shape[1]
        errs = []
        for _ in range(T):
            supp = rng.choice(d, s, replace=False)
            coef = rng.uniform(0.5, 1.5, s) * (rng.integers(0, 2, s) * 2 - 1)
            a = np.zeros(d)
            a[supp] = coef
            x = B @ a
            y = P @ x + rng.normal(0.0, SIGMA, m)
            xh = omp_decode(P, B, y, s)
            errs.append(float(np.linalg.norm(x - xh) / (np.linalg.norm(x) + 1e-12)))
        fam_err.append(sum(errs) / len(errs))
    return max(fam_err)


def main():
    if len(sys.argv) < 3:
        fail("usage: verify.py <in> <out> <ans>")
    n, m, K, s, pmax, bases = read_instance(sys.argv[1])
    P = read_output(sys.argv[2], m, n, pmax)

    dims = [B.shape[1] for B in bases]
    # deterministic per-instance seed for the hidden test signals
    sig_seed = SIG_SEED * 1000003 + n * 100000 + K * 1000 + sum(dims) * 7 + s

    F = family_max_error(P, bases, s, sig_seed)

    # internal baseline: naive direct probes P[j, j % n] = pmax
    Pb = np.zeros((m, n), dtype=float)
    for j in range(m):
        Pb[j, j % n] = pmax
    B = family_max_error(Pb, bases, s, sig_seed)

    sc = min(1000.0, SCALE * B / max(F, 1e-9))
    print("max-family mean relative error: %.6f (baseline %.6f)" % (F, B), file=sys.stderr)
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
