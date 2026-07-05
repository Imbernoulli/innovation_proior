# TIER: strong
"""Try slice-and-factor along ALL THREE modes and keep the cheapest.  Whichever axis
carries the planted low-rank structure gives the fewest rank-1 terms; the other two
axes are (generically) full rank.  This is only an UPPER bound on the tensor rank --
the true optimum (cross-slice recombination) is left open."""
import sys
from fractions import Fraction


def rref(M):
    M = [row[:] for row in M]
    rows = len(M); cols = len(M[0]) if rows else 0
    pivots = []; r = 0
    for col in range(cols):
        if r >= rows:
            break
        piv = None
        for i in range(r, rows):
            if M[i][col] != 0:
                piv = i; break
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        pv = M[r][col]
        M[r] = [x / pv for x in M[r]]
        for i in range(rows):
            if i != r and M[i][col] != 0:
                f = M[i][col]
                M[i] = [x - f * y for x, y in zip(M[i], M[r])]
        pivots.append(col); r += 1
    return M, pivots


def cr_terms(Mint):
    Mf = [[Fraction(x) for x in row] for row in Mint]
    rows = len(Mf)
    R, pivots = rref(Mf)
    terms = []
    for t, pcol in enumerate(pivots):
        col = [Mf[i][pcol] for i in range(rows)]
        row = R[t][:]
        terms.append((col, row))
    return terms


def unit(n, idx):
    v = [Fraction(0)] * n; v[idx] = Fraction(1); return v


def decompose(mode, a, b, c, T):
    terms = []
    if mode == 3:
        for k in range(c):
            S = [[T[i][j][k] for j in range(b)] for i in range(a)]
            for (col, row) in cr_terms(S):   # col len a (u), row len b (v)
                terms.append((col, row, unit(c, k)))
    elif mode == 1:
        for i in range(a):
            S = [[T[i][j][k] for k in range(c)] for j in range(b)]
            for (col, row) in cr_terms(S):   # col len b (v), row len c (w)
                terms.append((unit(a, i), col, row))
    else:  # mode == 2
        for j in range(b):
            S = [[T[i][j][k] for k in range(c)] for i in range(a)]
            for (col, row) in cr_terms(S):   # col len a (u), row len c (w)
                terms.append((col, unit(b, j), row))
    return terms


def main():
    tok = sys.stdin.read().split()
    it = iter(tok)
    a = int(next(it)); b = int(next(it)); c = int(next(it))
    T = [[[int(next(it)) for _ in range(c)] for _ in range(b)] for _ in range(a)]
    best = None
    for mode in (1, 2, 3):
        terms = decompose(mode, a, b, c, T)
        if best is None or len(terms) < len(best):
            best = terms
    out = [str(len(best))]
    for (u, v, w) in best:
        out.append(" ".join(str(x) for x in list(u) + list(v) + list(w)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
