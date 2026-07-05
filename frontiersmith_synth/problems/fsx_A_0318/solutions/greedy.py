# TIER: greedy
"""Slice-and-factor along a FIXED mode (mode 3 = frontal slices).  Each a x b frontal
slice is rank-factored exactly over the rationals (CR / rref decomposition), giving
sum_k rank(S_k) rank-1 terms.  Beats the fiber baseline only when the frontal slices
happen to be low rank (i.e. when the planted mode == 3)."""
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
    """return list of (col_vec, row_vec) exact rationals with sum col (x) row = M."""
    Mf = [[Fraction(x) for x in row] for row in Mint]
    rows = len(Mf);
    R, pivots = rref(Mf)
    terms = []
    for t, pcol in enumerate(pivots):
        col = [Mf[i][pcol] for i in range(rows)]
        row = R[t][:]
        terms.append((col, row))
    return terms


def decompose_mode3(a, b, c, T):
    terms = []
    for k in range(c):
        S = [[T[i][j][k] for j in range(b)] for i in range(a)]
        for (col, row) in cr_terms(S):
            w = [Fraction(0)] * c; w[k] = Fraction(1)
            terms.append((col, row, w))
    return terms


def emit(a, b, c, terms):
    out = [str(len(terms))]
    for (u, v, w) in terms:
        out.append(" ".join(str(x) for x in list(u) + list(v) + list(w)))
    sys.stdout.write("\n".join(out) + "\n")


def main():
    tok = sys.stdin.read().split()
    it = iter(tok)
    a = int(next(it)); b = int(next(it)); c = int(next(it))
    T = [[[int(next(it)) for _ in range(c)] for _ in range(b)] for _ in range(a)]
    emit(a, b, c, decompose_mode3(a, b, c, T))


if __name__ == "__main__":
    main()
