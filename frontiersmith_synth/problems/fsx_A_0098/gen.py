#!/usr/bin/env python3
# gen.py <testId>  -> prints ONE coral-reef survey-response tensor instance to stdout.
# The tensor is a PLANTED Tucker tensor with a chosen multilinear rank (r1,r2,r3),
# r_n < dim_n, embedded in a larger index space.  Its true CP rank is unknown and
# well above max(dim) (no polynomial recovery), so the optimum stays open.
import sys
from fractions import Fraction as F

# ------------------------------------------------------------------ ladder
# (I, J, K, r1, r2, r3)  -- r_n < dim_n so the tensor is Tucker-compressible.
LADDER = {
    1:  (3, 3, 3, 2, 2, 2),
    2:  (4, 4, 4, 3, 3, 3),
    3:  (4, 4, 4, 3, 3, 3),
    4:  (5, 5, 5, 4, 4, 4),
    5:  (5, 5, 5, 4, 4, 4),
    6:  (4, 5, 5, 3, 4, 4),
    7:  (5, 4, 5, 4, 3, 4),
    8:  (5, 5, 4, 4, 4, 3),
    9:  (5, 5, 5, 4, 4, 4),
    10: (4, 4, 4, 3, 3, 3),
}

# ---------------------------------------------------------------- lin alg (exact)
def mat_rank(rows):
    """Exact rank of a matrix (list of lists of Fraction/int)."""
    M = [[F(x) for x in r] for r in rows]
    if not M:
        return 0
    nc = len(M[0])
    r = 0
    for c in range(nc):
        piv = None
        for i in range(r, len(M)):
            if M[i][c] != 0:
                piv = i
                break
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        pv = M[r][c]
        M[r] = [x / pv for x in M[r]]
        for i in range(len(M)):
            if i != r and M[i][c] != 0:
                f = M[i][c]
                M[i] = [a - f * b for a, b in zip(M[i], M[r])]
        r += 1
        if r == len(M):
            break
    return r


def full_col_rank(mat, want):
    return mat_rank(mat) == want


def build_instance(tid):
    import random
    I, J, K, r1, r2, r3 = LADDER[tid]
    rng = random.Random(90000 + 137 * tid)

    def rand_factor(rows, cols):
        # ±1 entries, full column rank (reseeded until satisfied)
        for _ in range(2000):
            M = [[rng.choice((-1, 1)) for _ in range(cols)] for _ in range(rows)]
            if full_col_rank(M, cols):
                return M
        raise RuntimeError("factor rank")

    A = rand_factor(I, r1)
    B = rand_factor(J, r2)
    C = rand_factor(K, r3)

    def core_ok(G):
        # every mode-n unfolding full rank AND every slice (all 3 modes) full rank,
        # so plain-slice and Tucker-slice rank counts are pinned to the targets.
        # mode-3 slices: r1 x r2, rank min(r1,r2)
        for k in range(r3):
            if mat_rank([[G[p][q][k] for q in range(r2)] for p in range(r1)]) != min(r1, r2):
                return False
        # mode-2 slices: r1 x r3, rank min(r1,r3)
        for q in range(r2):
            if mat_rank([[G[p][q][k] for k in range(r3)] for p in range(r1)]) != min(r1, r3):
                return False
        # mode-1 slices: r2 x r3, rank min(r2,r3)
        for p in range(r1):
            if mat_rank([[G[p][q][k] for k in range(r3)] for q in range(r2)]) != min(r2, r3):
                return False
        return True

    G = None
    for _ in range(4000):
        cand = [[[rng.choice((-2, -1, 1, 2)) for _ in range(r3)]
                 for _ in range(r2)] for _ in range(r1)]
        if core_ok(cand):
            G = cand
            break
    if G is None:
        raise RuntimeError("core rank")

    # T[i][j][k] = sum_{p,q,s} A[i][p] B[j][q] C[k][s] G[p][q][s]
    T = [[[0] * K for _ in range(J)] for _ in range(I)]
    for i in range(I):
        for j in range(J):
            for k in range(K):
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
                            if g:
                                acc += aip * bjq * C[k][s] * g
                T[i][j][k] = acc
    return I, J, K, T


def main():
    tid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if tid not in LADDER:
        tid = ((tid - 1) % len(LADDER)) + 1
    I, J, K, T = build_instance(tid)
    out = ["%d %d %d" % (I, J, K)]
    for i in range(I):
        for j in range(J):
            out.append(" ".join(str(T[i][j][k]) for k in range(K)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
