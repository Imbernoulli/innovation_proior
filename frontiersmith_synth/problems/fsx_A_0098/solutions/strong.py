# TIER: strong
# Tucker-compress the tensor to its multilinear rank on every axis (exact rational
# HOSVD-style basis extraction), then best-of-three-axes slice the tiny core and map
# each core mode back through the factor matrices.  Exploits low multilinear rank, so
# it needs strictly fewer survey modes than plain axis slicing when the tensor is
# compressible -- but it does NOT reach the (unknown) true CP optimum.
import sys
from fractions import Fraction as F


def read_tensor():
    toks = sys.stdin.read().split()
    it = iter(toks)
    I = int(next(it)); J = int(next(it)); K = int(next(it))
    T = [[[F(int(next(it))) for _ in range(K)] for _ in range(J)] for _ in range(I)]
    return I, J, K, T


# ---------------- exact linear algebra ----------------
def mat_rank(rows):
    M = [[F(x) for x in r] for r in rows]
    if not M:
        return 0
    nc = len(M[0]); r = 0
    for c in range(nc):
        piv = None
        for i in range(r, len(M)):
            if M[i][c] != 0:
                piv = i; break
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        pv = M[r][c]; M[r] = [x / pv for x in M[r]]
        for i in range(len(M)):
            if i != r and M[i][c] != 0:
                f = M[i][c]; M[i] = [a - f * b for a, b in zip(M[i], M[r])]
        r += 1
        if r == len(M):
            break
    return r


def col_basis(cols):
    """cols: list of column vectors. Return indices of an independent spanning set."""
    chosen = []
    cur = 0
    for idx, col in enumerate(cols):
        cand = chosen + [col]
        # rank of matrix whose rows are the chosen columns (transpose is fine for rank)
        if mat_rank(cand) > cur:
            chosen.append(col)
            cur += 1
    # recompute indices
    picks = []
    chosen2 = []
    cur = 0
    for idx, col in enumerate(cols):
        if mat_rank(chosen2 + [col]) > cur:
            chosen2.append(col); picks.append(idx); cur += 1
    return picks


def mat_inv(A):
    n = len(A)
    M = [[F(A[i][j]) for j in range(n)] + [F(1) if j == i else F(0) for j in range(n)]
         for i in range(n)]
    for c in range(n):
        piv = None
        for i in range(c, n):
            if M[i][c] != 0:
                piv = i; break
        M[c], M[piv] = M[piv], M[c]
        pv = M[c][c]; M[c] = [x / pv for x in M[c]]
        for i in range(n):
            if i != c and M[i][c] != 0:
                f = M[i][c]; M[i] = [a - f * b for a, b in zip(M[i], M[c])]
    return [row[n:] for row in M]


def solve(A, targets):
    """A: d x r (full column rank).  targets: list of length-d vectors.
    Return coords: list (same length) of length-r vectors x with A x = target."""
    d = len(A); r = len(A[0])
    # normal equations: (A^T A) x = A^T t
    G = [[sum(A[i][p] * A[i][q] for i in range(d)) for q in range(r)] for p in range(r)]
    Ginv = mat_inv(G)
    out = []
    for t in targets:
        atb = [sum(A[i][p] * t[i] for i in range(d)) for p in range(r)]
        x = [sum(Ginv[p][q] * atb[q] for q in range(r)) for p in range(r)]
        out.append(x)
    return out


def reduce_mode(T, dims, mode):
    d0, d1, d2 = dims
    if mode == 0:
        fibers = [[T[i][j][k] for i in range(d0)] for j in range(d1) for k in range(d2)]
    elif mode == 1:
        fibers = [[T[i][j][k] for j in range(d1)] for i in range(d0) for k in range(d2)]
    else:
        fibers = [[T[i][j][k] for k in range(d2)] for i in range(d0) for j in range(d1)]
    picks = col_basis(fibers)
    Fac = [[fibers[c][row] for c in picks] for row in range(len(fibers[0]))]  # d_m x r
    coords = solve(Fac, fibers)  # one r-vector per fiber
    r = len(picks)
    # rebuild tensor with mode dim -> r
    if mode == 0:
        nd = (r, d1, d2)
        Tn = [[[F(0)] * d2 for _ in range(d1)] for _ in range(r)]
        idx = 0
        for j in range(d1):
            for k in range(d2):
                x = coords[idx]; idx += 1
                for p in range(r):
                    Tn[p][j][k] = x[p]
    elif mode == 1:
        nd = (d0, r, d2)
        Tn = [[[F(0)] * d2 for _ in range(r)] for _ in range(d0)]
        idx = 0
        for i in range(d0):
            for k in range(d2):
                x = coords[idx]; idx += 1
                for p in range(r):
                    Tn[i][p][k] = x[p]
    else:
        nd = (d0, d1, r)
        Tn = [[[F(0)] * r for _ in range(d1)] for _ in range(d0)]
        idx = 0
        for i in range(d0):
            for j in range(d1):
                x = coords[idx]; idx += 1
                for p in range(r):
                    Tn[i][j][p] = x[p]
    return Fac, Tn, nd


# ---------------- slicing on the core ----------------
def peel(M, nr, nc):
    A = [[M[i][j] for j in range(nc)] for i in range(nr)]
    terms = []
    while True:
        pi = pj = -1
        for i in range(nr):
            for j in range(nc):
                if A[i][j] != 0:
                    pi, pj = i, j; break
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


def best_slice(core, a, b, c):
    best = None
    for mode in (0, 1, 2):
        terms = []
        if mode == 2:
            for k in range(c):
                M = [[core[i][j][k] for j in range(b)] for i in range(a)]
                for (u, v) in peel(M, a, b):
                    w = [F(0)] * c; w[k] = F(1); terms.append((u, v, w))
        elif mode == 1:
            for j in range(b):
                M = [[core[i][j][k] for k in range(c)] for i in range(a)]
                for (u, w) in peel(M, a, c):
                    v = [F(0)] * b; v[j] = F(1); terms.append((u, v, w))
        else:
            for i in range(a):
                M = [[core[i][j][k] for k in range(c)] for j in range(b)]
                for (v, w) in peel(M, b, c):
                    u = [F(0)] * a; u[i] = F(1); terms.append((u, v, w))
        if best is None or len(terms) < len(best):
            best = terms
    return best


def matvec(M, x):
    return [sum(M[i][p] * x[p] for p in range(len(x))) for i in range(len(M))]


def main():
    I, J, K, T = read_tensor()
    dims = (I, J, K)
    A, T1, d1 = reduce_mode(T, dims, 0)
    B, T2, d2 = reduce_mode(T1, d1, 1)
    C, core, d3 = reduce_mode(T2, d2, 2)
    a, b, c = d3
    core_terms = best_slice(core, a, b, c)
    out = [str(len(core_terms))]
    for (u, v, w) in core_terms:
        av = matvec(A, u)   # length I
        bv = matvec(B, v)   # length J
        cv = matvec(C, w)   # length K
        out.append(" ".join(str(x) for x in (av + bv + cv)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
