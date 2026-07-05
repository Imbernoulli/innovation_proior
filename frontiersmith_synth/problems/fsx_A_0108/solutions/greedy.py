# TIER: greedy
# Slice the tensor along the FIXED injection axis (mode 1).  Each i-slice H[i] is a
# b x c matrix; factor it exactly into rank(H[i]) rank-1 pieces (v (x) w) and prepend
# u = e_i.  Total modes = sum_i rank(H[i]).  Beats the per-fiber baseline because each
# slice has low rank, but never looks across slices.
import sys
from fractions import Fraction


def read_tensor():
    d = sys.stdin.read().split()
    a = int(d[0]); b = int(d[1]); c = int(d[2])
    it = iter(d[3:])
    H = [[[int(next(it)) for _ in range(c)] for _ in range(b)] for _ in range(a)]
    return a, b, c, H


def rank_factor(M):
    # exact rank factorization of matrix M (list of lists) -> list of (col, row)
    # with M == sum_t col_t (x) row_t and len == rank(M).
    m = len(M)
    n = len(M[0]) if m else 0
    A = [[Fraction(x) for x in row] for row in M]
    pivots = []
    r = 0
    for c in range(n):
        piv = None
        for i in range(r, m):
            if A[i][c] != 0:
                piv = i
                break
        if piv is None:
            continue
        A[r], A[piv] = A[piv], A[r]
        inv = Fraction(1) / A[r][c]
        A[r] = [x * inv for x in A[r]]
        for i in range(m):
            if i != r and A[i][c] != 0:
                f = A[i][c]
                A[i] = [A[i][t] - f * A[r][t] for t in range(n)]
        pivots.append(c)
        r += 1
        if r == m:
            break
    terms = []
    for t, pc in enumerate(pivots):
        col = [Fraction(M[i][pc]) for i in range(m)]
        row = A[t][:]
        terms.append((col, row))
    return terms


def fmt(x):
    return str(x)


def main():
    a, b, c, H = read_tensor()
    out_terms = []
    for i in range(a):
        for (col, row) in rank_factor(H[i]):  # col len b, row len c
            u = [Fraction(0)] * a
            u[i] = Fraction(1)
            out_terms.append((u, col, row))
    out = [str(len(out_terms))]
    for (u, v, w) in out_terms:
        out.append(" ".join(fmt(x) for x in list(u) + list(v) + list(w)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
