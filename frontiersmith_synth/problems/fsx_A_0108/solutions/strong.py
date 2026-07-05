# TIER: strong
# Multilinear-rank ("Tucker") compression, best of the three axes.
# For a chosen axis m: unfold the tensor with axis m as rows.  Exactly factor that
# unfolding into (coordinate vector along m) (x) (residual matrix over the other two
# axes); the number of pieces equals the *multilinear rank* g_m << n.  Then factor each
# residual matrix (which lives inside the compressed core, so it too has small rank).
# Total modes = sum over the g_m residual matrices of their ranks -- far below the plain
# per-slice bound because it shares the core basis across all slices.  We try all three
# axes and keep the cheapest.  Still only a heuristic: the core's true CP rank is smaller
# and unreachable by these polynomial linear-algebra steps.
import sys
from fractions import Fraction


def read_tensor():
    d = sys.stdin.read().split()
    a = int(d[0]); b = int(d[1]); c = int(d[2])
    it = iter(d[3:])
    H = [[[int(next(it)) for _ in range(c)] for _ in range(b)] for _ in range(a)]
    return a, b, c, H


def rank_factor(M):
    m = len(M)
    n = len(M[0]) if m else 0
    A = [[Fraction(x) for x in row] for row in M]
    pivots = []
    r = 0
    for col in range(n):
        piv = None
        for i in range(r, m):
            if A[i][col] != 0:
                piv = i
                break
        if piv is None:
            continue
        A[r], A[piv] = A[piv], A[r]
        inv = Fraction(1) / A[r][col]
        A[r] = [x * inv for x in A[r]]
        for i in range(m):
            if i != r and A[i][col] != 0:
                f = A[i][col]
                A[i] = [A[i][t] - f * A[r][t] for t in range(n)]
        pivots.append(col)
        r += 1
        if r == m:
            break
    terms = []
    for t, pc in enumerate(pivots):
        colv = [Fraction(M[i][pc]) for i in range(m)]
        row = A[t][:]
        terms.append((colv, row))
    return terms


def compress_axis(a, b, c, H, axis):
    # returns list of (u,v,w) rank-1 terms reconstructing H exactly
    terms = []
    if axis == 0:
        U = [[H[i][j][k] for j in range(b) for k in range(c)] for i in range(a)]
        for (coord, flat) in rank_factor(U):  # coord len a, flat len b*c
            N = [[flat[j * c + k] for k in range(c)] for j in range(b)]
            for (colv, row) in rank_factor(N):  # colv len b, row len c
                terms.append((list(coord), list(colv), list(row)))
    elif axis == 1:
        U = [[H[i][j][k] for i in range(a) for k in range(c)] for j in range(b)]
        for (coord, flat) in rank_factor(U):  # coord len b, flat len a*c
            N = [[flat[i * c + k] for k in range(c)] for i in range(a)]
            for (colu, row) in rank_factor(N):  # colu len a, row len c
                terms.append((list(colu), list(coord), list(row)))
    else:
        U = [[H[i][j][k] for i in range(a) for j in range(b)] for k in range(c)]
        for (coord, flat) in rank_factor(U):  # coord len c, flat len a*b
            N = [[flat[i * b + j] for j in range(b)] for i in range(a)]
            for (colu, rowv) in rank_factor(N):  # colu len a, rowv len b
                terms.append((list(colu), list(rowv), list(coord)))
    return terms


def main():
    a, b, c, H = read_tensor()
    best = None
    for axis in (0, 1, 2):
        cand = compress_axis(a, b, c, H, axis)
        if best is None or len(cand) < len(best):
            best = cand
    out = [str(len(best))]
    for (u, v, w) in best:
        out.append(" ".join(str(x) for x in u + v + w))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
