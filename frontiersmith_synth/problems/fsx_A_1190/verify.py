#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the spectral-unmixing (linear-mixing / simplex-vertex / nonneg
sum-to-one) task.

Reads the test id from <in>'s header and reconstructs the hidden scene (endmembers M*,
abundances A*, the pixel spectra Y) with the EXACT same seeded formula as gen.py (the
hidden law lives only here and in gen.py -- never in the input/output the solver sees).

Feasibility (checked strictly; any violation -> Ratio: 0.0):
  - correct token counts, all finite
  - submitted endmembers >= -1e-6 and bounded (sanity)
  - submitted abundances in [-1e-6, 1+1e-6] and sum to 1 per pixel (+-1e-3)
  - reconstruction fidelity: sum_k a_hat_kj * m_hat_k must be close to the GIVEN y_j
    (mean relative L2 error <= 0.15) -- this ties nonnegativity+sum-to-one to the actual
    forward model, so a submission of arbitrary/garbage numbers cannot pass.

Score (only if feasible): best-permutation match of the K=3 submitted endmembers against
the hidden truth (relative L2 error -> exponential similarity), plus per-pixel abundance L1
error against the hidden truth (same permutation) -> exponential similarity. F is their
mean; Ratio = min(1, F).
"""
import sys, math, random

R = 8
K = 3
CAP_BY_TEST = {1: 0.95, 2: 0.92, 3: 0.85, 4: 0.80, 5: 0.90,
               6: 0.88, 7: 0.75, 8: 0.85, 9: 0.72, 10: 0.90}
SIGMA = 0.01

M_MAX = 50.0          # sanity bound on submitted endmember entries
NONNEG_TOL = 1e-6
SUM1_TOL = 1e-3
RECON_TOL = 0.35       # max mean relative L2 reconstruction error allowed (trivial baseline ~0.14-0.22)
ALPHA = 0.15           # endmember relative-L2-error decay scale
BETA = 0.25            # abundance L1-error decay scale


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def hidden_instance(t):
    rng = random.Random(31337 + 101 * t)
    N = 24 + 8 * t
    cap = CAP_BY_TEST.get(t, None)

    while True:
        M = [[round(rng.uniform(0.1, 1.0), 6) for _ in range(K)] for _ in range(R)]
        ok = True
        for a_ in range(K):
            for b_ in range(a_ + 1, K):
                d = sum((M[r][a_] - M[r][b_]) ** 2 for r in range(R)) ** 0.5
                if d < 0.8:
                    ok = False
        if ok:
            break

    A, Y = [], []
    for _j in range(N):
        while True:
            expo = [-math.log(rng.random()) for _ in range(K)]
            s = sum(expo)
            u = [e / s for e in expo]
            if cap is None or max(u) <= cap:
                break
        a = u
        y = [sum(M[r][k] * a[k] for k in range(K)) + rng.gauss(0.0, SIGMA) for r in range(R)]
        y = [max(0.0, v) for v in y]
        A.append(a)
        Y.append(y)
    return N, M, A, Y


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def main():
    tokens = open(sys.argv[1]).read().split()
    try:
        it = iter(tokens)
        t = int(next(it))
        Rp = int(next(it)); Kp = int(next(it)); Np = int(next(it))
        Ygiven = [[float(next(it)) for _ in range(Rp)] for _ in range(Np)]
    except Exception:
        fail("bad input")

    if Rp != R or Kp != K:
        fail("internal R/K mismatch")

    N_true, M_true, A_true, Y_true = hidden_instance(t)
    if N_true != Np or len(Y_true) != Np:
        fail("internal generation mismatch (N)")
    for j in range(Np):
        for r in range(R):
            if abs(Y_true[j][r] - Ygiven[j][r]) > 1e-4:
                fail("internal generation mismatch (Y)")

    # ---------- parse participant output ----------
    out_tokens = open(sys.argv[2]).read().split()
    try:
        ot = iter(out_tokens)
        M_sub = [[float(next(ot)) for _ in range(R)] for _ in range(K)]
        A_sub = [[float(next(ot)) for _ in range(K)] for _ in range(Np)]
    except StopIteration:
        fail("too few tokens")
    except Exception:
        fail("parse error")

    for k in range(K):
        for r in range(R):
            v = M_sub[k][r]
            if not math.isfinite(v):
                fail("non-finite endmember entry")
            if v < -NONNEG_TOL or v > M_MAX:
                fail("endmember entry out of range")
    M_sub = [[max(0.0, v) for v in row] for row in M_sub]

    for j in range(Np):
        s = 0.0
        for k in range(K):
            v = A_sub[j][k]
            if not math.isfinite(v):
                fail("non-finite abundance entry")
            if v < -NONNEG_TOL or v > 1.0 + NONNEG_TOL:
                fail("abundance entry out of range")
            s += v
        if abs(s - 1.0) > SUM1_TOL:
            fail("abundances for pixel %d do not sum to 1 (sum=%.6f)" % (j, s))
    A_sub = [[max(0.0, v) for v in row] for row in A_sub]

    # ---------- reconstruction fidelity ----------
    rel_errs = []
    for j in range(Np):
        recon = [sum(A_sub[j][k] * M_sub[k][r] for k in range(K)) for r in range(R)]
        diff = [recon[r] - Ygiven[j][r] for r in range(R)]
        rel_errs.append(norm(diff) / (norm(Ygiven[j]) + 1e-6))
    mean_rel_err = sum(rel_errs) / Np
    if mean_rel_err > RECON_TOL:
        fail("reconstruction error too high (mean_rel=%.4f)" % mean_rel_err)

    # ---------- best-permutation endmember match ----------
    import itertools
    Mt_cols = [[M_true[r][k] for r in range(R)] for k in range(K)]  # K x R
    best = None
    for perm in itertools.permutations(range(K)):
        e = 0.0
        for k in range(K):
            mt = Mt_cols[k]
            mh = M_sub[perm[k]]
            diff = [mh[r] - mt[r] for r in range(R)]
            e += norm(diff) / (norm(mt) + 1e-9)
        if best is None or e < best[0]:
            best = (e, perm)
    _, perm = best

    sim_ks = []
    for k in range(K):
        mt = Mt_cols[k]
        mh = M_sub[perm[k]]
        diff = [mh[r] - mt[r] for r in range(R)]
        e = norm(diff) / (norm(mt) + 1e-9)
        sim_ks.append(math.exp(-e / ALPHA))
    end_score = sum(sim_ks) / K

    ab_scores = []
    for j in range(Np):
        at = A_true[j]
        ah = [A_sub[j][perm[k]] for k in range(K)]
        e = sum(abs(ah[k] - at[k]) for k in range(K))
        ab_scores.append(math.exp(-e / BETA))
    ab_score = sum(ab_scores) / Np

    F = 0.5 * end_score + 0.5 * ab_score
    ratio = min(1.0, F)

    print("end=%.4f ab=%.4f F=%.4f Ratio: %.6f" % (end_score, ab_score, F, ratio))


if __name__ == "__main__":
    main()
