# TIER: strong
# Slice-rank (column-space) factorization. Slice the tensor along each of the 3 modes; each
# slice is a matrix we EXACTLY factor as M = C * R (pivot columns C times rref rows R) into
# rank(M) rank-1 pieces. Total gadgets = sum of slice matrix-ranks; keep the cheapest of the
# three orientations. This is NOT a proven-optimal scheme: the planted tensor is over-complete
# so the true tensor rank stays unknown -- this only beats the fiber baselines.
import sys
from fractions import Fraction


def rref(A):
    A = [row[:] for row in A]
    rows = len(A)
    cols = len(A[0]) if rows else 0
    pivots = []
    r = 0
    for c in range(cols):
        piv = None
        for i in range(r, rows):
            if A[i][c] != 0:
                piv = i
                break
        if piv is None:
            continue
        A[r], A[piv] = A[piv], A[r]
        pv = A[r][c]
        A[r] = [x / pv for x in A[r]]
        for i in range(rows):
            if i != r and A[i][c] != 0:
                f = A[i][c]
                A[i] = [x - f * y for x, y in zip(A[i], A[r])]
        pivots.append(c)
        r += 1
        if r == rows:
            break
    return A, pivots, r


def factor(M):
    """Exact rank-1 factorization of integer matrix M -> list of (u, v) with M = sum u (x) v."""
    rows = len(M)
    Af = [[Fraction(x) for x in row] for row in M]
    red, pivots, r = rref(Af)
    if r == 0:
        return []
    C = [[Af[i][p] for p in pivots] for i in range(rows)]  # rows x r  (original pivot cols)
    pieces = []
    for t in range(r):
        u = [C[i][t] for i in range(rows)]
        v = red[t]
        pieces.append((u, v))
    return pieces


def main():
    tok = sys.stdin.read().split()
    it = iter(tok)
    I = int(next(it)); J = int(next(it)); K = int(next(it))
    T = [[[0] * K for _ in range(J)] for _ in range(I)]
    for k in range(K):
        for i in range(I):
            for j in range(J):
                T[i][j][k] = int(next(it))

    def e(n, x):
        return [Fraction(1) if p == x else Fraction(0) for p in range(n)]

    # mode-3: slice per k is I x J ; term = u(I) (x) v(J) (x) e_k
    def orient3():
        ts = []
        for k in range(K):
            M = [[T[i][j][k] for j in range(J)] for i in range(I)]
            for u, v in factor(M):
                ts.append((u, v, e(K, k)))
        return ts

    # mode-1: slice per i is J x K ; term = e_i (x) u(J) (x) v(K)
    def orient1():
        ts = []
        for i in range(I):
            M = [[T[i][j][k] for k in range(K)] for j in range(J)]
            for u, v in factor(M):
                ts.append((e(I, i), u, v))
        return ts

    # mode-2: slice per j is I x K ; term = u(I) (x) e_j (x) v(K)
    def orient2():
        ts = []
        for j in range(J):
            M = [[T[i][j][k] for k in range(K)] for i in range(I)]
            for u, v in factor(M):
                ts.append((u, e(J, j), v))
        return ts

    best = min([orient1(), orient2(), orient3()], key=len)

    out = [str(len(best))]
    for a, b, c in best:
        out.append(" ".join(str(x) for x in a))
        out.append(" ".join(str(x) for x in b))
        out.append(" ".join(str(x) for x in c))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
