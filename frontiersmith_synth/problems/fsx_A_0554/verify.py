#!/usr/bin/env python3
# verify.py <in> <out> <ans>   (ans ignored)  -- deterministic scorer, objective = MINIMIZE.
#
# Instance: full-rank integer lattice L in Z^n given by public basis B (rows).  The participant
# submits k lattice vectors (ambient integer coordinates).  Feasibility: each submitted vector
# must be NONZERO and lie in L (an integer combination of the rows of B), and the k vectors must
# be LINEARLY INDEPENDENT.  Objective F = sum of squared Euclidean norms (minimize).
#
# Normalization (minimization):  B_base = sum of the k smallest squared row-norms of B (a trivial
# feasible construction: k independent basis rows).  sc = min(1000, 100 * B_base / F);
# print "Ratio: sc/1000" so trivial ~ 0.1 and a much shorter harvest approaches 1.0.
import sys
from fractions import Fraction as Fr


def solve_int(B, v):
    """Return integer coeff vector x with x*B == v, or None. B: n x n int rows; v: len-n int."""
    n = len(B)
    # M x = v  where M[j][i] = B[i][j]  (M = B^T)
    M = [[Fr(B[i][j]) for i in range(n)] for j in range(n)]
    rhs = [Fr(v[j]) for j in range(n)]
    for c in range(n):
        piv = None
        for r in range(c, n):
            if M[r][c] != 0:
                piv = r
                break
        if piv is None:
            return None  # singular (should not happen for a real basis)
        M[c], M[piv] = M[piv], M[c]
        rhs[c], rhs[piv] = rhs[piv], rhs[c]
        pv = M[c][c]
        M[c] = [x / pv for x in M[c]]
        rhs[c] = rhs[c] / pv
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [M[r][j] - f * M[c][j] for j in range(n)]
                rhs[r] = rhs[r] - f * rhs[c]
    x = rhs
    if any(xi.denominator != 1 for xi in x):
        return None
    return [int(xi) for xi in x]


def rank_int(rows):
    M = [[Fr(x) for x in r] for r in rows]
    if not M:
        return 0
    ncol = len(M[0])
    r = 0
    for c in range(ncol):
        piv = None
        for i in range(r, len(M)):
            if M[i][c] != 0:
                piv = i
                break
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        pv = M[r][c]
        M[r] = [x / pv for x in M[r]]
        for i in range(len(M)):
            if i != r and M[i][c] != 0:
                f = M[i][c]
                M[i] = [M[i][j] - f * M[r][j] for j in range(ncol)]
        r += 1
        if r == len(M):
            break
    return r


def fail(reason):
    print("reason: %s" % reason)
    print("Ratio: 0.0")
    sys.exit(0)


def main():
    inf, outf = sys.argv[1], sys.argv[2]
    with open(inf) as f:
        toks = f.read().split()
    idx = 0
    n = int(toks[idx]); idx += 1
    p = int(toks[idx]); idx += 1
    k = int(toks[idx]); idx += 1
    B = []
    for i in range(n):
        row = [int(toks[idx + j]) for j in range(n)]
        idx += n
        B.append(row)

    # baseline: k smallest squared row norms of B
    row_norms = sorted(sum(x * x for x in row) for row in B)
    Bbase = sum(row_norms[:k])
    if Bbase <= 0:
        Bbase = 1

    # parse participant output: k lines, each n integers, all finite
    try:
        with open(outf) as f:
            otoks = f.read().split()
    except Exception:
        fail("cannot read output")
    if len(otoks) != n * k:
        fail("expected %d integer tokens (k=%d vectors of length n=%d), got %d"
             % (n * k, k, n, len(otoks)))
    vecs = []
    for t in otoks:
        # reject non-finite / non-integer tokens explicitly
        tl = t.lower()
        if "nan" in tl or "inf" in tl:
            fail("non-finite token")
        try:
            val = int(t)
        except ValueError:
            fail("non-integer token '%s'" % t[:16])
        vecs.append(val)
    V = [vecs[i * n:(i + 1) * n] for i in range(k)]

    # feasibility: nonzero, in lattice, independent
    for v in V:
        if all(x == 0 for x in v):
            fail("zero vector submitted")
        if solve_int(B, v) is None:
            fail("submitted vector is not in the lattice")
    if rank_int(V) != k:
        fail("submitted vectors are not linearly independent")

    F = sum(sum(x * x for x in v) for v in V)
    if F <= 0:
        fail("degenerate objective")

    sc = min(1000.0, 100.0 * Bbase / F)
    print("n=%d p=%d k=%d Bbase=%d F=%d" % (n, p, k, Bbase, F))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
