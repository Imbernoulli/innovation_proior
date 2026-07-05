# TIER: greedy
# Rank-factorize the tensor by slabbing along ONE fixed axis (axis 0, the
# "vessel" axis): each slab T[i] is a J x K matrix, rank-factorized exactly over
# the rationals into rank(T[i]) rank-1 crane stages.  R = sum_i rank(T[i]),
# which is <= number-of-nonzero-entries (beats trivial) but is data-dependently
# larger than the best-of-three-axes bound.
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
    # returns list of (colvec len m, rowvec len n) with M = sum col (x) row
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


def main():
    tok = sys.stdin.read().split()
    it = iter(tok)
    I = int(next(it)); J = int(next(it)); K = int(next(it))
    T = [[[int(next(it)) for _ in range(K)] for _ in range(J)] for _ in range(I)]

    stages = []
    for i in range(I):
        M = [[T[i][j][k] for k in range(K)] for j in range(J)]  # J x K
        for col, row in factor(M, J, K):
            u = [Fraction(0)] * I; u[i] = Fraction(1)
            v = col            # len J
            w = row            # len K
            stages.append((u, v, w))

    outl = [str(len(stages))]
    for (u, v, w) in stages:
        outl.append(" ".join(fr(x) for x in list(u) + list(v) + list(w)))
    sys.stdout.write("\n".join(outl) + "\n")


if __name__ == "__main__":
    main()
