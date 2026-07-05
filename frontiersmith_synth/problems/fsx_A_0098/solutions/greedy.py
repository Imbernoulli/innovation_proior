# TIER: greedy
# Best-of-three-axes slice-rank factorization: slice the tensor along each mode,
# exact rank-1 peel every slice, keep the axis giving the fewest survey modes.
import sys
from fractions import Fraction as F


def read_tensor():
    toks = sys.stdin.read().split()
    it = iter(toks)
    I = int(next(it)); J = int(next(it)); K = int(next(it))
    T = [[[F(int(next(it))) for _ in range(K)] for _ in range(J)] for _ in range(I)]
    return I, J, K, T


def peel(M, nr, nc):
    """Rank-1 peel an nr x nc matrix -> list of (u[nr], v[nc]) with sum u.v^T = M."""
    A = [[M[i][j] for j in range(nc)] for i in range(nr)]
    terms = []
    while True:
        pi = pj = -1
        for i in range(nr):
            for j in range(nc):
                if A[i][j] != 0:
                    pi, pj = i, j
                    break
            if pi != -1:
                break
        if pi == -1:
            break
        piv = A[pi][pj]
        u = [A[i][pj] for i in range(nr)]
        v = [A[pi][j] / piv for j in range(nc)]
        terms.append((u, v))
        for i in range(nr):
            if u[i] == 0:
                continue
            for j in range(nc):
                if v[j] != 0:
                    A[i][j] -= u[i] * v[j]
    return terms


def slice_mode(T, I, J, K, mode):
    """CP terms (len-I, len-J, len-K vectors) from slicing along `mode`."""
    terms = []
    if mode == 2:            # fix k -> I x J
        for k in range(K):
            M = [[T[i][j][k] for j in range(J)] for i in range(I)]
            for (u, v) in peel(M, I, J):
                w = [F(0)] * K; w[k] = F(1)
                terms.append((u, v, w))
    elif mode == 1:          # fix j -> I x K
        for j in range(J):
            M = [[T[i][j][k] for k in range(K)] for i in range(I)]
            for (u, w) in peel(M, I, K):
                v = [F(0)] * J; v[j] = F(1)
                terms.append((u, v, w))
    else:                    # mode 0: fix i -> J x K
        for i in range(I):
            M = [[T[i][j][k] for k in range(K)] for j in range(J)]
            for (v, w) in peel(M, J, K):
                u = [F(0)] * I; u[i] = F(1)
                terms.append((u, v, w))
    return terms


def main():
    I, J, K, T = read_tensor()
    best = None
    for mode in (0, 1, 2):
        terms = slice_mode(T, I, J, K, mode)
        if best is None or len(terms) < len(best):
            best = terms
    out = [str(len(best))]
    for (u, v, w) in best:
        out.append(" ".join(str(x) for x in (u + v + w)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
