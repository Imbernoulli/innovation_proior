# TIER: strong
# Best-of-three-axes slice-rank factorization.  For each of the three axes, slice
# the tensor into frontal matrices, exact-rank-factorize each slice, and sum the
# slice ranks; keep whichever axis yields the fewest total stages.  This beats the
# fixed-axis greedy (it never picks the worst axis) yet still sits ABOVE the true
# CP rank -- the minimal-rank ceiling stays open (rational ALS / CP search below
# the slice bound is left uncovered).
import sys
from fractions import Fraction


def read_tensor():
    data = sys.stdin.read().split()
    it = iter(data)
    a = int(next(it)); b = int(next(it)); c = int(next(it))
    T = [[[Fraction(0)] * c for _ in range(b)] for _ in range(a)]
    for i in range(a):
        for j in range(b):
            for k in range(c):
                T[i][j][k] = Fraction(int(next(it)))
    return a, b, c, T


def rref(M):
    A = [row[:] for row in M]
    rows = len(A); cols = len(A[0]) if rows else 0
    piv = []
    r = 0
    for cc in range(cols):
        pr = None
        for i in range(r, rows):
            if A[i][cc] != 0:
                pr = i; break
        if pr is None:
            continue
        A[r], A[pr] = A[pr], A[r]
        pv = A[r][cc]
        A[r] = [x / pv for x in A[r]]
        for i in range(rows):
            if i != r and A[i][cc] != 0:
                f = A[i][cc]
                A[i] = [x - f * y for x, y in zip(A[i], A[r])]
        piv.append(cc)
        r += 1
        if r == rows:
            break
    return A[:r], piv


def e(n, idx):
    v = [Fraction(0)] * n
    v[idx] = Fraction(1)
    return v


def slice_along(a, b, c, T, mode):
    stages = []
    if mode == 2:                       # slice by phase k -> M is (a x b)
        for k in range(c):
            M = [[T[i][j][k] for j in range(b)] for i in range(a)]
            red, piv = rref(M)
            for t, pc in enumerate(piv):
                u = [M[i][pc] for i in range(a)]
                v = red[t][:]
                stages.append((u, v, e(c, k)))
    elif mode == 0:                     # slice by approach i -> M is (b x c)
        for i in range(a):
            M = [[T[i][j][k] for k in range(c)] for j in range(b)]
            red, piv = rref(M)
            for t, pc in enumerate(piv):
                v = [M[j][pc] for j in range(b)]
                w = red[t][:]
                stages.append((e(a, i), v, w))
    else:                               # mode == 1: slice by movement j -> M is (a x c)
        for j in range(b):
            M = [[T[i][j][k] for k in range(c)] for i in range(a)]
            red, piv = rref(M)
            for t, pc in enumerate(piv):
                u = [M[i][pc] for i in range(a)]
                w = red[t][:]
                stages.append((u, e(b, j), w))
    return stages


def emit(stages):
    lines = [str(len(stages))]
    for u, v, w in stages:
        row = []
        for x in (u + v + w):
            row.append(str(x.numerator) if x.denominator == 1 else "%d/%d" % (x.numerator, x.denominator))
        lines.append(" ".join(row))
    sys.stdout.write("\n".join(lines) + "\n")


def main():
    a, b, c, T = read_tensor()
    best = None
    for mode in (0, 1, 2):
        s = slice_along(a, b, c, T, mode)
        if best is None or len(s) < len(best):
            best = s
    emit(best)


if __name__ == "__main__":
    main()
