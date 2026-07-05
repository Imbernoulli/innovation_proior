# TIER: strong
# Best-of-three-axes slab rank factorization: for each of the three axes, slab
# the tensor and sum the exact rational ranks of the slabs; pick the axis with
# the smallest total and emit that decomposition.  R = min over axes of the
# slab-rank sum -- the classic slice-rank UPPER BOUND.  It is <= the fixed-axis
# greedy value and far below the per-entry trivial value.  It is NOT optimal:
# the planted rank is overcomplete and lies strictly below this bound, so ample
# headroom remains for a true CP-rank search (the uncovered ceiling).
import sys
from fractions import Fraction


def rref(M, m, n):
    A = [[Fraction(x) for x in row] for row in M]
    pivots = []
    r = 0
    for col in range(n):
        sel = None
        for rr in range(r, m):
            if A[rr][col] != 0:
                sel = rr; break
        if sel is None:
            continue
        A[r], A[sel] = A[sel], A[r]
        pv = A[r][col]
        A[r] = [x / pv for x in A[r]]
        for rr in range(m):
            if rr != r and A[rr][col] != 0:
                f = A[rr][col]
                A[rr] = [a - f * b for a, b in zip(A[rr], A[r])]
        pivots.append(col)
        r += 1
        if r == m:
            break
    return A, pivots


def factor(M, m, n):
    Rr, piv = rref(M, m, n)
    terms = []
    for t, pc in enumerate(piv):
        col = [Fraction(M[i][pc]) for i in range(m)]
        row = Rr[t][:]
        terms.append((col, row))
    return terms


def fr(x):
    x = Fraction(x)
    return str(x.numerator) if x.denominator == 1 else "%d/%d" % (x.numerator, x.denominator)


def decomp_axis(T, I, J, K, axis):
    stages = []
    if axis == 0:                       # slabs indexed by i, matrix over (j,k)
        for i in range(I):
            M = [[T[i][j][k] for k in range(K)] for j in range(J)]
            for col, row in factor(M, J, K):
                u = [Fraction(0)] * I; u[i] = Fraction(1)
                stages.append((u, col, row))
    elif axis == 1:                     # slabs indexed by j, matrix over (i,k)
        for j in range(J):
            M = [[T[i][j][k] for k in range(K)] for i in range(I)]
            for col, row in factor(M, I, K):
                v = [Fraction(0)] * J; v[j] = Fraction(1)
                stages.append((col, v, row))
    else:                               # slabs indexed by k, matrix over (i,j)
        for k in range(K):
            M = [[T[i][j][k] for j in range(J)] for i in range(I)]
            for col, row in factor(M, I, J):
                w = [Fraction(0)] * K; w[k] = Fraction(1)
                stages.append((col, row, w))
    return stages


def main():
    tok = sys.stdin.read().split()
    it = iter(tok)
    I = int(next(it)); J = int(next(it)); K = int(next(it))
    T = [[[int(next(it)) for _ in range(K)] for _ in range(J)] for _ in range(I)]

    best = None
    for axis in range(3):
        st = decomp_axis(T, I, J, K, axis)
        if best is None or len(st) < len(best):
            best = st

    outl = [str(len(best))]
    for (u, v, w) in best:
        outl.append(" ".join(fr(x) for x in list(u) + list(v) + list(w)))
    sys.stdout.write("\n".join(outl) + "\n")


if __name__ == "__main__":
    main()
