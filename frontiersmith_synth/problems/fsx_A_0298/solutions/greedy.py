# TIER: greedy
# Slice the tensor along the FIXED wavelength axis (axis 2, size c) and exactly
# rank-factor each a x b slice over the rationals.  Stage count = sum_k rank(S_k)
# = c * min(a,b) for a full-rank overcomplete tensor.  Because a <= b < c, this
# is strictly worse than the best axis -> beats trivial, loses to strong.
import sys
from fractions import Fraction


def rank_factor(M, nr, nc):
    """Exact column-rank factorisation M = sum_t col_t (x) row_t over Q.
    Returns list of (col vec len nr, row vec len nc)."""
    A = [[Fraction(x) for x in row] for row in M]
    pivots = []
    r = 0
    for col in range(nc):
        piv = None
        for rr in range(r, nr):
            if A[rr][col] != 0:
                piv = rr
                break
        if piv is None:
            continue
        A[r], A[piv] = A[piv], A[r]
        inv = Fraction(1) / A[r][col]
        A[r] = [x * inv for x in A[r]]
        for rr in range(nr):
            if rr != r and A[rr][col] != 0:
                f = A[rr][col]
                A[rr] = [x - f * y for x, y in zip(A[rr], A[r])]
        pivots.append(col)
        r += 1
        if r == nr:
            break
    terms = []
    for t in range(r):
        colv = [Fraction(M[i][pivots[t]]) for i in range(nr)]
        roww = A[t][:]                       # RREF nonzero row t (length nc)
        terms.append((colv, roww))
    return terms


def fmt(x):
    return str(x.numerator) if x.denominator == 1 else "%d/%d" % (x.numerator, x.denominator)


def main():
    tok = sys.stdin.read().split()
    it = iter(tok)
    a = int(next(it)); b = int(next(it)); c = int(next(it))
    T = [[[int(next(it)) for _ in range(c)] for _ in range(b)] for _ in range(a)]

    stages = []
    for k in range(c):
        S = [[T[i][j][k] for j in range(b)] for i in range(a)]   # a x b slice
        for (colv, roww) in rank_factor(S, a, b):
            u = colv                                            # length a
            v = roww                                            # length b
            w = [Fraction(0)] * c; w[k] = Fraction(1)
            stages.append((u, v, w))

    lines = [str(len(stages))]
    for (u, v, w) in stages:
        lines.append(" ".join(fmt(Fraction(x)) for x in list(u) + list(v) + list(w)))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
