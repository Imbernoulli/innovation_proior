#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE instance (a 3-D "greenhouse coupling tensor") to stdout.

GREENHOUSE ZONES skin (family: tensor-decomposition-rank, format D, AlphaEvolve-inspired):
An automated greenhouse recomputes, every control tick, a 3-way coupling table
    T[i][j][k] = the growth response produced jointly by
                 light-zone i, nutrient-band j and humidity-tier k.
The controller evaluates the whole table as a sum of rank-1 "actuator gadgets"
(a_r . x)(b_r . y)(c_r . z); each gadget costs ONE scalar multiply on the hot path.

We ship a coupling tensor built as an ASYMMETRIC low-multilinear-rank (Tucker) tensor:
    T = G x1 A x2 B x3 C,   A:IxP, B:JxQ, C:KxS,   core G:PxQxS,   P,Q,S < the dims.
Because the core is asymmetric (P != Q != S) every one of the three mode-slicings is
rank-deficient by a DIFFERENT amount, so which orientation is cheapest varies per test.
The planted (multi-linear) rank exceeds no single formula: the true tensor rank is
over-complete and genuinely unknown -- polynomial diagonalization (rank <= dimension)
does not recover it, so the optimum stays open.

Deterministic: everything seeded by testId only. Integer entries.  Scale: LARGE.

STDOUT format:
    I J K
    then K frontal slices; slice k is I lines of J integers  (value = T[i][j][k]).
"""
import sys
import random
from fractions import Fraction

# (I, J, K, P, Q, S): dims of the tensor and of the asymmetric Tucker core.
#   P,Q,S < min of the participating dims -> every mode-slice is rank-deficient.
#   Mode-3 slice rank <= min(P,Q); mode-1 <= min(Q,S); mode-2 <= min(P,S)  (generically =).
#   The planted structure is over-complete (true rank unknown); large-scale ladder.
PARAMS = {
    1:  (8,  7, 6, 3, 3, 4),
    2:  (8,  8, 6, 3, 4, 3),
    3:  (9,  8, 6, 4, 3, 3),
    4:  (9,  8, 7, 3, 4, 4),
    5:  (9,  9, 7, 4, 3, 4),
    6:  (10, 8, 7, 3, 5, 4),
    7:  (10, 9, 7, 4, 4, 3),
    8:  (10, 9, 8, 4, 5, 3),
    9:  (11, 9, 8, 4, 4, 5),
    10: (11, 10, 8, 5, 4, 4),
}


def col_rank(M):
    """Exact column rank of an integer matrix (rows x cols) via rational elimination."""
    rows = len(M)
    cols = len(M[0]) if rows else 0
    A = [[Fraction(x) for x in row] for row in M]
    r = 0
    for c in range(cols):
        piv = None
        for i in range(r, rows):
            if A[i][c] != 0:
                piv = i
                break
        if piv is None:
            continue
        A[r], A[piv] = A[piv], A[r]
        pv = A[r][c]
        A[r] = [x / pv for x in A[r]]
        for i in range(rows):
            if i != r and A[i][c] != 0:
                f = A[i][c]
                A[i] = [x - f * y for x, y in zip(A[i], A[r])]
        r += 1
        if r == rows:
            break
    return r


def full_col_rank_matrix(rng, rows, cols, lo=-2, hi=2):
    """Random integer matrix (rows x cols) with full column rank == cols."""
    for _ in range(2000):
        M = [[rng.randint(lo, hi) for _ in range(cols)] for _ in range(rows)]
        if col_rank(M) == cols:
            return M
    raise RuntimeError("could not build a full-column-rank factor")


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if t not in PARAMS:
        t = ((t - 1) % len(PARAMS)) + 1
    I, J, K, P, Q, S = PARAMS[t]
    rng = random.Random(716123 + 9973 * t)

    # Full-column-rank factor matrices -> multilinear ranks are exactly (P, Q, S).
    A = full_col_rank_matrix(rng, I, P)
    B = full_col_rank_matrix(rng, J, Q)
    C = full_col_rank_matrix(rng, K, S)

    # Asymmetric integer core G[x][y][z], guaranteed not all-zero.
    while True:
        G = [[[rng.randint(-2, 2) for _ in range(S)] for _ in range(Q)] for _ in range(P)]
        if any(G[x][y][z] != 0 for x in range(P) for y in range(Q) for z in range(S)):
            break

    # T[i][j][k] = sum_{x,y,z} A[i][x] B[j][y] C[k][z] G[x][y][z]
    # Compute in stages to keep it O(I J K (P+Q+S)).
    # H[i][y][z] = sum_x A[i][x] G[x][y][z]
    H = [[[0] * S for _ in range(Q)] for _ in range(I)]
    for i in range(I):
        for x in range(P):
            aix = A[i][x]
            if aix == 0:
                continue
            for y in range(Q):
                for z in range(S):
                    H[i][y][z] += aix * G[x][y][z]
    # W[i][j][z] = sum_y B[j][y] H[i][y][z]
    W = [[[0] * S for _ in range(J)] for _ in range(I)]
    for i in range(I):
        for j in range(J):
            for y in range(Q):
                bjy = B[j][y]
                if bjy == 0:
                    continue
                Hiy = H[i][y]
                Wij = W[i][j]
                for z in range(S):
                    Wij[z] += bjy * Hiy[z]
    # T[i][j][k] = sum_z C[k][z] W[i][j][z]
    T = [[[0] * K for _ in range(J)] for _ in range(I)]
    for i in range(I):
        for j in range(J):
            Wij = W[i][j]
            for k in range(K):
                Ck = C[k]
                s = 0
                for z in range(S):
                    ckz = Ck[z]
                    if ckz:
                        s += ckz * Wij[z]
                T[i][j][k] = s

    out = ["%d %d %d" % (I, J, K)]
    for k in range(K):
        for i in range(I):
            out.append(" ".join(str(T[i][j][k]) for j in range(J)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
