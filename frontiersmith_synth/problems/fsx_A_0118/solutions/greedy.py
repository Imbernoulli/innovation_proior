# TIER: greedy
# Slice the tensor along the FIXED last (phase) axis and exact-rank-factorize each
# frontal slice.  R = sum_k rank(slice_k).  Because gen guarantees the last axis
# is the strictly-largest dimension, this fixed axis is the worst of the three and
# the total is larger than the best-axis strategy -- a real improvement over the
# per-entry baseline but not the best simple scheme.
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
    # M: list of rows of Fraction. Returns (reduced nonzero rows, pivot column idxs).
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


def slice_last_axis(a, b, c, T):
    # For each phase k: M (a x b) = T[:,:,k];  M = M[:,piv] * rref(M)[nz].
    stages = []
    for k in range(c):
        M = [[T[i][j][k] for j in range(b)] for i in range(a)]
        red, piv = rref(M)
        for t, pc in enumerate(piv):
            u = [M[i][pc] for i in range(a)]     # column pc of M (length a)
            v = red[t][:]                        # reduced row (length b)
            w = [Fraction(0)] * c; w[k] = Fraction(1)
            stages.append((u, v, w))
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
    emit(slice_last_axis(a, b, c, T))


if __name__ == "__main__":
    main()
