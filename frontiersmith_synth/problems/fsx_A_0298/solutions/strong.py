# TIER: strong
# Best-of-three-axes slice-rank factorisation.  For each of the three axes, slice
# the tensor and exactly rank-factor every slice over Q; the total stage count for
# axis d is  (size of axis d) * (rank ceiling of the slices).  Emit the cheapest
# axis.  For a full-rank overcomplete tensor with a <= b < c this reaches a*b,
# strictly below greedy's c*a -- yet still an UPPER BOUND: the true CP rank (>= the
# planted overcomplete rank) is unknown, leaving headroom for a real search.
import sys
from fractions import Fraction


def rank_factor(M, nr, nc):
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
        roww = A[t][:]
        terms.append((colv, roww))
    return terms


def fmt(x):
    return str(x.numerator) if x.denominator == 1 else "%d/%d" % (x.numerator, x.denominator)


def decompose_axis(T, a, b, c, axis):
    """Return list of (u,v,w) stages using slices orthogonal to `axis`."""
    stages = []
    if axis == 0:                                   # slices over (j,k), size b x c
        for i in range(a):
            S = [[T[i][j][k] for k in range(c)] for j in range(b)]
            for (colv, roww) in rank_factor(S, b, c):
                u = [Fraction(0)] * a; u[i] = Fraction(1)
                stages.append((u, colv, roww))
    elif axis == 1:                                 # slices over (i,k), size a x c
        for j in range(b):
            S = [[T[i][j][k] for k in range(c)] for i in range(a)]
            for (colv, roww) in rank_factor(S, a, c):
                v = [Fraction(0)] * b; v[j] = Fraction(1)
                stages.append((colv, v, roww))
    else:                                           # slices over (i,j), size a x b
        for k in range(c):
            S = [[T[i][j][k] for j in range(b)] for i in range(a)]
            for (colv, roww) in rank_factor(S, a, b):
                w = [Fraction(0)] * c; w[k] = Fraction(1)
                stages.append((colv, roww, w))
    return stages


def main():
    tok = sys.stdin.read().split()
    it = iter(tok)
    a = int(next(it)); b = int(next(it)); c = int(next(it))
    T = [[[int(next(it)) for _ in range(c)] for _ in range(b)] for _ in range(a)]

    best = None
    for axis in (0, 1, 2):
        st = decompose_axis(T, a, b, c, axis)
        if best is None or len(st) < len(best):
            best = st

    lines = [str(len(best))]
    for (u, v, w) in best:
        lines.append(" ".join(fmt(Fraction(x)) for x in list(u) + list(v) + list(w)))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
