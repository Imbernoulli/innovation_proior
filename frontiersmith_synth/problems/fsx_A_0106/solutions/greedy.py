# TIER: greedy
# Seeded multi-restart random search: draw many random +/-1 matrices, rank them
# by a fast float log-determinant, and emit the best one. Random +/-1 matrices
# have determinants vastly larger than the arrow baseline (2^(N-1)), so this
# beats the trivial tier -- but it does no local refinement, so it stays below
# the hill-climbed / structured tiers.
import sys
import random
from math import log

def slogdet(A):
    """Return log|det| via float LU with partial pivoting (-inf if singular)."""
    n = len(A)
    A = [row[:] for row in A]
    ld = 0.0
    for k in range(n):
        p = k
        best = abs(A[k][k])
        for i in range(k + 1, n):
            v = abs(A[i][k])
            if v > best:
                best = v
                p = i
        if best < 1e-9:
            return float("-inf")
        if p != k:
            A[k], A[p] = A[p], A[k]
        piv = A[k][k]
        ld += log(abs(piv))
        Ak = A[k]
        for i in range(k + 1, n):
            Ai = A[i]
            f = Ai[k] / piv
            if f != 0.0:
                for j in range(k + 1, n):
                    Ai[j] -= f * Ak[j]
    return ld

def main():
    n = int(sys.stdin.read().split()[0])
    rng = random.Random(1000 + n)

    # keep total float-LU work bounded across sizes
    restarts = max(6, min(40, 900000 // (n * n * n // 4 + 1)))

    best = None
    best_ld = float("-inf")
    for _ in range(restarts):
        M = [[1 if rng.random() < 0.5 else -1 for _ in range(n)] for _ in range(n)]
        ld = slogdet([[float(x) for x in row] for row in M])
        if ld > best_ld:
            best_ld = ld
            best = M

    out = "\n".join(" ".join(map(str, row)) for row in best)
    sys.stdout.write(out + "\n")

if __name__ == "__main__":
    main()
