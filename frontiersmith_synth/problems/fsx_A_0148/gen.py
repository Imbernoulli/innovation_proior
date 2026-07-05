#!/usr/bin/env python3
"""gen.py <testId>  -> prints ONE instance of the 'Festival Vibe Tensor' problem.

Instance = a small integer 3D tensor T of shape a x b x c, built as a Tucker
product  T = G x1 A x2 B x3 C  with PLANTED multilinear ranks (r1,r2,r3), each
strictly small relative to the dimensions.  The CP-rank (minimum number of
rank-one 'acts' needed to reproduce T exactly) is therefore genuinely OPEN:
it lies between max(r1,r2,r3) and the product of the two smallest multilinear
ranks, and no polynomial method recovers it in general.

STDOUT (token stream, whitespace separated):
  line 1:  a b c
  then     a*b*c integers, index order i (0..a-1) outer, j (0..b-1), k (0..c-1)
           laid out as a blocks of b lines, each line holding c integers.
"""
import sys
from fractions import Fraction as F

# testId -> (a, b, c, r1, r2, r3)   with c < a, r_n < dim_n (mostly), and
# product-of-two-smallest-ranks  <  min pairwise product of dims.
CFG = {
    1:  (5, 4, 3, 3, 3, 2),
    2:  (5, 4, 2, 3, 3, 2),
    3:  (4, 4, 3, 3, 2, 2),
    4:  (5, 3, 3, 3, 2, 2),
    5:  (5, 5, 4, 4, 3, 3),
    6:  (5, 4, 3, 4, 3, 2),
    7:  (4, 3, 2, 3, 2, 2),
    8:  (5, 4, 4, 3, 3, 3),
    9:  (5, 5, 3, 4, 4, 2),
    10: (5, 4, 3, 3, 2, 2),
}


def mat_rank(rows):
    """Exact rank of a matrix given as a list of rows (lists of numbers)."""
    M = [[F(x) for x in row] for row in rows]
    nr = len(M)
    nc = len(M[0]) if nr else 0
    r = 0
    for c in range(nc):
        piv = None
        for i in range(r, nr):
            if M[i][c] != 0:
                piv = i
                break
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        pv = M[r][c]
        M[r] = [x / pv for x in M[r]]
        for i in range(nr):
            if i != r and M[i][c] != 0:
                f = M[i][c]
                M[i] = [a - f * b for a, b in zip(M[i], M[r])]
        r += 1
        if r == nr:
            break
    return r


def rand_matrix(rng, nrows, ncols, lo, hi):
    return [[rng.randint(lo, hi) for _ in range(ncols)] for _ in range(nrows)]


def full_col_rank_matrix(rng, nrows, ncols):
    # nrows x ncols integer matrix with column rank == ncols
    for _ in range(2000):
        M = rand_matrix(rng, nrows, ncols, -2, 2)
        if mat_rank(M) == ncols:
            return M
    raise RuntimeError("could not build full-rank factor")


def full_multilinear_core(rng, r1, r2, r3):
    for _ in range(4000):
        G = [[[rng.randint(-3, 3) for _ in range(r3)] for _ in range(r2)]
             for _ in range(r1)]
        u1 = [[G[p][q][s] for q in range(r2) for s in range(r3)] for p in range(r1)]
        u2 = [[G[p][q][s] for p in range(r1) for s in range(r3)] for q in range(r2)]
        u3 = [[G[p][q][s] for p in range(r1) for q in range(r2)] for s in range(r3)]
        if mat_rank(u1) == r1 and mat_rank(u2) == r2 and mat_rank(u3) == r3:
            return G
    raise RuntimeError("could not build full multilinear-rank core")


def main():
    if len(sys.argv) < 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    tid = int(sys.argv[1])
    key = tid if tid in CFG else ((tid - 1) % 10) + 1
    a, b, c, r1, r2, r3 = CFG[key]

    import random
    rng = random.Random(100003 + 1009 * tid)

    A = full_col_rank_matrix(rng, a, r1)   # a x r1
    B = full_col_rank_matrix(rng, b, r2)   # b x r2
    C = full_col_rank_matrix(rng, c, r3)   # c x r3
    G = full_multilinear_core(rng, r1, r2, r3)

    # T = G x1 A x2 B x3 C  (all integer)
    T = [[[0 for _ in range(c)] for _ in range(b)] for _ in range(a)]
    for i in range(a):
        for j in range(b):
            for k in range(c):
                acc = 0
                for p in range(r1):
                    aip = A[i][p]
                    if aip == 0:
                        continue
                    for q in range(r2):
                        bjq = B[j][q]
                        if bjq == 0:
                            continue
                        for s in range(r3):
                            g = G[p][q][s]
                            if g == 0:
                                continue
                            acc += aip * bjq * C[k][s] * g
                T[i][j][k] = acc

    out = ["%d %d %d" % (a, b, c)]
    for i in range(a):
        for j in range(b):
            out.append(" ".join(str(T[i][j][k]) for k in range(c)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
