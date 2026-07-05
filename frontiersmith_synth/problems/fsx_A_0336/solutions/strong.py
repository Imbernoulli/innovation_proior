# TIER: strong
"""Hill-climbing local search. Start from a seeded random +/-1 schedule (row 0 pinned to
the beacon r), then repeatedly flip single interior entries whenever the flip increases
|det| (evaluated exactly with Bareiss). Deterministic: fixed RNG seed and a fixed
evaluation budget (so the result is bit-for-bit reproducible and time-bounded)."""
import sys
import random

EVAL_BUDGET = 2500
PASSES = 5


def bareiss_det(M):
    n = len(M)
    A = [row[:] for row in M]
    sign = 1
    prev = 1
    for k in range(n - 1):
        if A[k][k] == 0:
            sw = None
            for i in range(k + 1, n):
                if A[i][k] != 0:
                    sw = i
                    break
            if sw is None:
                return 0
            A[k], A[sw] = A[sw], A[k]
            sign = -sign
        akk = A[k][k]
        for i in range(k + 1, n):
            Ai = A[i]
            aik = Ai[k]
            Ak = A[k]
            for j in range(k + 1, n):
                Ai[j] = (Ai[j] * akk - aik * Ak[j]) // prev
        prev = akk
    return sign * A[n - 1][n - 1]


def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    r = [int(x) for x in data[1:1 + N]]

    rnd = random.Random(N)  # deterministic
    M = [[rnd.choice([-1, 1]) for _ in range(N)] for _ in range(N)]
    for j in range(N):
        M[0][j] = r[j]

    best = abs(bareiss_det(M))
    evals = 1
    for _ in range(PASSES):
        improved = False
        for i in range(1, N):          # never touch the beacon row
            for j in range(N):
                if evals >= EVAL_BUDGET:
                    improved = False
                    break
                M[i][j] = -M[i][j]
                d = abs(bareiss_det(M))
                evals += 1
                if d > best:
                    best = d
                    improved = True
                else:
                    M[i][j] = -M[i][j]
            if evals >= EVAL_BUDGET:
                break
        if not improved:
            break

    out = [" ".join(str(x) for x in M[i]) for i in range(N)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
