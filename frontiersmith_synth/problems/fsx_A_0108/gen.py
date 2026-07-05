import sys
import random
from fractions import Fraction

# gen.py <testId>  -- prints ONE instance of the geothermal thermal-coupling tensor.
#
# The tensor H (n x n x n, integer) is built with a PLANTED low-multilinear-rank
# (Tucker) structure of core size (g1 x g2 x g3) whose small dense core has an
# OVERCOMPLETE CP rank q > max(g1,g2,g3) -- so the true minimal number of separable
# "operating modes" is genuinely unknown (Jennrich / simultaneous-diagonalisation,
# which need rank <= dimension, cannot recover it).  The ambient dimensions n are
# large relative to the multilinear ranks, so cheap per-slice heuristics leave a
# lot of headroom.
#
# Difficulty ladder over testId:
#   n            = 6 + (t-1)%4              (6,7,8,9 cycling)  -- ambient size
#   g (perm)     = rotation of (3,4,5)      -- which mode is the thinnest
#   q            = 6                         -- planted (overcomplete) core CP rank
# The permutation makes the "best" compression axis differ across tests.

BASE_G = (3, 4, 5)
Q = 6  # planted overcomplete core rank


def matrank(M):
    # exact rank of integer/Fraction matrix via fraction-free-ish Gaussian elimination
    A = [[Fraction(x) for x in row] for row in M]
    m = len(A)
    n = len(A[0]) if m else 0
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
        pv = A[r][c]
        for i in range(m):
            if i != r and A[i][c] != 0:
                f = A[i][c] / pv
                A[i] = [A[i][t] - f * A[r][t] for t in range(n)]
        r += 1
        if r == m:
            break
    return r


def rand_full_rank(rng, rows, cols, lo, hi):
    # random integer rows x cols matrix with full COLUMN rank (== cols)
    assert rows >= cols
    for _ in range(2000):
        M = [[rng.randint(lo, hi) for _ in range(cols)] for _ in range(rows)]
        if matrank(M) == cols:
            return M
    raise RuntimeError("could not build full-column-rank matrix")


def main():
    t = int(sys.argv[1])
    rng = random.Random(1000 + 7919 * t)

    n = 6 + (t - 1) % 4
    shift = (t - 1) % 3
    g = tuple(BASE_G[(i + shift) % 3] for i in range(3))
    g1, g2, g3 = g

    # planted dense core C (g1 x g2 x g3) as a sum of Q random rank-1 terms
    C = [[[0] * g3 for _ in range(g2)] for _ in range(g1)]
    for _ in range(Q):
        x = [rng.randint(-2, 2) for _ in range(g1)]
        y = [rng.randint(-2, 2) for _ in range(g2)]
        z = [rng.randint(-2, 2) for _ in range(g3)]
        for p in range(g1):
            for qq in range(g2):
                for r in range(g3):
                    C[p][qq][r] += x[p] * y[qq] * z[r]

    # full-column-rank integer factor matrices
    A = rand_full_rank(rng, n, g1, -2, 2)  # n x g1
    B = rand_full_rank(rng, n, g2, -2, 2)  # n x g2
    D = rand_full_rank(rng, n, g3, -2, 2)  # n x g3

    # H[i][j][k] = sum_{p,q,r} C[p][q][r] A[i][p] B[j][q] D[k][r]
    H = [[[0] * n for _ in range(n)] for _ in range(n)]
    for i in range(n):
        Ai = A[i]
        for j in range(n):
            Bj = B[j]
            row = H[i][j]
            for k in range(n):
                Dk = D[k]
                s = 0
                for p in range(g1):
                    aip = Ai[p]
                    if aip == 0:
                        continue
                    for qq in range(g2):
                        cpq = aip * Bj[qq]
                        if cpq == 0:
                            continue
                        Cpq = C[p][qq]
                        for r in range(g3):
                            v = Cpq[r]
                            if v:
                                s += cpq * v * Dk[r]
                row[k] = s

    out = []
    out.append("%d %d %d" % (n, n, n))
    for i in range(n):
        for j in range(n):
            out.append(" ".join(str(x) for x in H[i][j]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
