# TIER: strong
"""Exact Tucker (HOSVD-style) compression followed by a core slab.

Compute the exact multilinear ranks (r1,r2,r3) of T via rational Gaussian
elimination, project onto a rational basis of each mode's fiber space to get a
small r1 x r2 x r3 core G, slab G along its LARGEST mode, and map every rank-one
core term back through the basis matrices.  Result: an EXACT CP decomposition
of rank = product of the two smallest multilinear ranks -- an upper bound on the
(open) CP rank, strictly better than any raw slab whenever T is genuinely
low-multilinear-rank.
"""
import sys
from fractions import Fraction as F


def pivot_cols(rows):
    M = [row[:] for row in rows]
    nr = len(M)
    nc = len(M[0]) if nr else 0
    piv = []
    r = 0
    for c in range(nc):
        sel = None
        for i in range(r, nr):
            if M[i][c] != 0:
                sel = i
                break
        if sel is None:
            continue
        M[r], M[sel] = M[sel], M[r]
        pv = M[r][c]
        M[r] = [x / pv for x in M[r]]
        for i in range(nr):
            if i != r and M[i][c] != 0:
                f = M[i][c]
                M[i] = [x - f * y for x, y in zip(M[i], M[r])]
        piv.append(c)
        r += 1
        if r == nr:
            break
    return piv


def matinv(A):
    n = len(A)
    M = [[F(A[i][j]) for j in range(n)] + [F(1) if i == j else F(0) for j in range(n)]
         for i in range(n)]
    for c in range(n):
        sel = None
        for i in range(c, n):
            if M[i][c] != 0:
                sel = i
                break
        M[c], M[sel] = M[sel], M[c]
        pv = M[c][c]
        M[c] = [x / pv for x in M[c]]
        for i in range(n):
            if i != c and M[i][c] != 0:
                f = M[i][c]
                M[i] = [x - f * y for x, y in zip(M[i], M[c])]
    return [row[n:] for row in M]


def matmul(A, B):
    n = len(A)
    m = len(B[0])
    k = len(B)
    out = [[F(0)] * m for _ in range(n)]
    for i in range(n):
        Ai = A[i]
        for t in range(k):
            a = Ai[t]
            if a == 0:
                continue
            Bt = B[t]
            oi = out[i]
            for j in range(m):
                oi[j] += a * Bt[j]
    return out


def transpose(A):
    return [list(col) for col in zip(*A)]


def pinv(U):
    # left inverse of full-column-rank U (dim x r) -> (r x dim)
    Ut = transpose(U)
    return matmul(matinv(matmul(Ut, U)), Ut)


def mode1(P, T, a, b, c):
    r = len(P)
    out = [[[F(0)] * c for _ in range(b)] for _ in range(r)]
    for p in range(r):
        for i in range(a):
            pv = P[p][i]
            if pv == 0:
                continue
            for j in range(b):
                for k in range(c):
                    out[p][j][k] += pv * T[i][j][k]
    return out


def mode2(P, T, d0, b, c):
    r = len(P)
    out = [[[F(0)] * c for _ in range(r)] for _ in range(d0)]
    for p in range(d0):
        for q in range(r):
            for j in range(b):
                pv = P[q][j]
                if pv == 0:
                    continue
                for k in range(c):
                    out[p][q][k] += pv * T[p][j][k]
    return out


def mode3(P, T, d0, d1, c):
    r = len(P)
    out = [[[F(0)] * r for _ in range(d1)] for _ in range(d0)]
    for p in range(d0):
        for q in range(d1):
            for s in range(r):
                pv = P[s]
                acc = F(0)
                for k in range(c):
                    if pv[k] != 0:
                        acc += pv[k] * T[p][q][k]
                out[p][q][s] = acc
    return out


def col(U, idx):
    return [U[i][idx] for i in range(len(U))]


def mv(U, vec):
    # U (dim x r) times vec (len r) -> len dim
    dim = len(U)
    r = len(vec)
    out = [F(0)] * dim
    for i in range(dim):
        Ui = U[i]
        s = F(0)
        for t in range(r):
            if vec[t] != 0:
                s += Ui[t] * vec[t]
        out[i] = s
    return out


def fmt(x):
    return str(x)


def main():
    toks = sys.stdin.read().split()
    a, b, c = int(toks[0]), int(toks[1]), int(toks[2])
    body = list(map(int, toks[3:3 + a * b * c]))
    T = [[[F(0)] * c for _ in range(b)] for _ in range(a)]
    idx = 0
    for i in range(a):
        for j in range(b):
            for k in range(c):
                T[i][j][k] = F(body[idx]); idx += 1

    # unfoldings (rows = mode dim, cols = the other two)
    M1 = [[T[i][j][k] for j in range(b) for k in range(c)] for i in range(a)]
    M2 = [[T[i][j][k] for i in range(a) for k in range(c)] for j in range(b)]
    M3 = [[T[i][j][k] for i in range(a) for j in range(b)] for k in range(c)]

    p1 = pivot_cols(M1)
    p2 = pivot_cols(M2)
    p3 = pivot_cols(M3)
    r1, r2, r3 = len(p1), len(p2), len(p3)

    # basis matrices (dim x r) = selected independent columns
    U1 = [[M1[i][cc] for cc in p1] for i in range(a)]
    U2 = [[M2[j][cc] for cc in p2] for j in range(b)]
    U3 = [[M3[k][cc] for cc in p3] for k in range(c)]

    P1 = pinv(U1)  # r1 x a
    P2 = pinv(U2)  # r2 x b
    P3 = pinv(U3)  # r3 x c

    # core G = T x1 P1 x2 P2 x3 P3   -> shape r1 x r2 x r3
    Ta = mode1(P1, T, a, b, c)          # r1 x b x c
    Tb = mode2(P2, Ta, r1, b, c)        # r1 x r2 x c
    G = mode3(P3, Tb, r1, r2, c)        # r1 x r2 x r3

    dims = (r1, r2, r3)
    slab = dims.index(max(dims))        # slab the largest core mode

    terms = []
    if slab == 2:                       # fiber over s (mode3)
        for p in range(r1):
            for q in range(r2):
                fib = [G[p][q][s] for s in range(r3)]
                u = col(U1, p)
                v = col(U2, q)
                w = mv(U3, fib)
                terms.append(u + v + w)
    elif slab == 0:                     # fiber over p (mode1)
        for q in range(r2):
            for s in range(r3):
                fib = [G[p][q][s] for p in range(r1)]
                u = mv(U1, fib)
                v = col(U2, q)
                w = col(U3, s)
                terms.append(u + v + w)
    else:                               # slab == 1, fiber over q (mode2)
        for p in range(r1):
            for s in range(r3):
                fib = [G[p][q][s] for q in range(r2)]
                u = col(U1, p)
                v = mv(U2, fib)
                w = col(U3, s)
                terms.append(u + v + w)

    out = [str(len(terms))]
    for row in terms:
        out.append(" ".join(fmt(x) for x in row))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
