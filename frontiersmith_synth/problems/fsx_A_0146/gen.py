#!/usr/bin/env python3
"""gen.py <testId> -> prints ONE 'archaeology dig grid' instance to stdout.

Instance = an N x N grid. Some cells are already excavated: their artifact
orientation is fixed to +1 or -1. The rest are unexcavated (marked 0) and must
be filled with +/-1 by the solver.  N is odd for every testId, so no Hadamard
matrix exists and the true maximum determinant is unknown / unreachable.

Output format:
    line 1 : N
    next N lines : N integers each, in {-1, 0, +1}   (0 = free/unexcavated)

Determinism: everything is seeded from testId only. The generator also verifies
(via exact Bareiss elimination) that the checker's minimal 'triangular' baseline
completion is non-singular for the chosen fixed cells; if not, it deterministically
reseeds, so the scoring baseline is always well defined.
"""
import sys, random


def det_bareiss(M):
    """Exact integer determinant via Bareiss fraction-free elimination."""
    n = len(M)
    A = [row[:] for row in M]
    sign = 1
    prev = 1
    for k in range(n - 1):
        if A[k][k] == 0:
            piv = -1
            for i in range(k + 1, n):
                if A[i][k] != 0:
                    piv = i
                    break
            if piv == -1:
                return 0
            A[k], A[piv] = A[piv], A[k]
            sign = -sign
        akk = A[k][k]
        for i in range(k + 1, n):
            aik = A[i][k]
            Ai = A[i]
            Ak = A[k]
            for j in range(k + 1, n):
                Ai[j] = (Ai[j] * akk - aik * Ak[j]) // prev
        prev = akk
    return sign * A[n - 1][n - 1]


def baseline_matrix(fixed, N):
    """Minimal 'triangular' sign completion: +1 on/below diagonal, -1 above,
    then overwrite the excavated (fixed) cells. MUST match verify.py exactly."""
    M = [[1 if j <= i else -1 for j in range(N)] for i in range(N)]
    for (i, j), v in fixed.items():
        M[i][j] = v
    return M


def build_instance(testId):
    N = 7 + 2 * testId          # 9, 11, 13, ... , 27  (all odd)
    frac = 0.60                 # fraction of cells pre-excavated (fixed)
    k = max(1, round(frac * N * N))
    base_seed = 8191 * testId + 17
    attempt = 0
    while True:
        rng = random.Random(base_seed + attempt)
        cells = rng.sample(range(N * N), k)
        fixed = {}
        for c in cells:
            fixed[(c // N, c % N)] = rng.choice((-1, 1))
        M = baseline_matrix(fixed, N)
        if det_bareiss(M) != 0:      # baseline must be non-singular
            return N, fixed
        attempt += 1


def main():
    testId = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    N, fixed = build_instance(testId)
    out = [str(N)]
    for i in range(N):
        row = []
        for j in range(N):
            row.append(str(fixed.get((i, j), 0)))
        out.append(" ".join(row))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
