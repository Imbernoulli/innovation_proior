# TIER: strong
# Slice-rank (column-space) factorization. Slice the tensor along each of the three modes;
# every slice is a matrix, exactly factored by fraction-free elimination into rank(slice)
# rank-1 pieces (column space x reduced rows). Total primitives = sum of slice matrix-ranks;
# keep the cheapest of the three orientations.
#
# This exploits the planted low rank WITHIN each slice, so it strictly beats the fiber
# decompositions. It is NOT proven optimal: the planted total rank is over-complete
# (rank > every mode dimension), so the true tensor rank stays unknown and this construction
# does not reach it -- it only closes part of the gap.
import sys
from fractions import Fraction


def rref(mat):
    """mat: list of rows of Fraction. Returns (reduced, pivot_cols, rank)."""
    A = [row[:] for row in mat]
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
    """Exact rank-1 factorization of matrix M -> [(u, v)] with M = sum u (x) v (Fractions)."""
    rows = len(M)
    Af = [[Fraction(x) for x in row] for row in M]
    red, pivots, r = rref(Af)
    if r == 0:
        return []
    col_space = [[Af[i][p] for p in pivots] for i in range(rows)]  # rows x r
    pieces = []
    for t in range(r):
        u = [col_space[i][t] for i in range(rows)]
        v = red[t]
        pieces.append((u, v))
    return pieces


def main():
    tok = sys.stdin.read().split()
    it = iter(tok)
    B = int(next(it)); L = int(next(it)); H = int(next(it))
    T = [[[0] * H for _ in range(L)] for _ in range(B)]
    for k in range(H):
        for i in range(B):
            for j in range(L):
                T[i][j][k] = int(next(it))

    def e(n, x):
        return [Fraction(1) if p == x else Fraction(0) for p in range(n)]

    # slice per harmonic band k : bus x line ; primitive = u(B) (x) v(L) (x) e_k(H)
    def orient_k():
        ts = []
        for k in range(H):
            M = [[T[i][j][k] for j in range(L)] for i in range(B)]
            for u, v in factor(M):
                ts.append((u, v, e(H, k)))
        return ts

    # slice per bus i : line x harmonic ; primitive = e_i(B) (x) u(L) (x) v(H)
    def orient_i():
        ts = []
        for i in range(B):
            M = [[T[i][j][k] for k in range(H)] for j in range(L)]
            for u, v in factor(M):
                ts.append((e(B, i), u, v))
        return ts

    # slice per line j : bus x harmonic ; primitive = u(B) (x) e_j(L) (x) v(H)
    def orient_j():
        ts = []
        for j in range(L):
            M = [[T[i][j][k] for k in range(H)] for i in range(B)]
            for u, v in factor(M):
                ts.append((u, e(L, j), v))
        return ts

    best = min([orient_i(), orient_j(), orient_k()], key=len)

    out = [str(len(best))]
    for a, b, c in best:
        out.append(" ".join(str(x) for x in a))
        out.append(" ".join(str(x) for x in b))
        out.append(" ".join(str(x) for x in c))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
