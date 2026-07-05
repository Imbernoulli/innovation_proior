# TIER: greedy
# Seeded multi-restart random sampling of feasible 0/1 grids; keep the best
# by a fast float log|det|.  Beats the sparse block baseline, weaker than the
# hill-climbing 'strong' solution.
import sys, random, math


def logabsdet(M, n):
    """Float log|det| via partial-pivot Gaussian elimination; -inf if singular."""
    A = [row[:] for row in M]
    total = 0.0
    for k in range(n):
        piv = k
        best = abs(A[k][k])
        for i in range(k + 1, n):
            if abs(A[i][k]) > best:
                best = abs(A[i][k]); piv = i
        if best < 1e-12:
            return -float("inf")
        if piv != k:
            A[k], A[piv] = A[piv], A[k]
        akk = A[k][k]
        total += math.log(abs(akk))
        for i in range(k + 1, n):
            f = A[i][k] / akk
            if f != 0.0:
                row_i = A[i]; row_k = A[k]
                for j in range(k, n):
                    row_i[j] -= f * row_k[j]
    return total


def main():
    data = sys.stdin.read().split()
    idx = 0
    N = int(data[idx]); idx += 1
    K = int(data[idx]); idx += 1
    fixed = {}
    for _ in range(K):
        r = int(data[idx]); c = int(data[idx + 1]); v = int(data[idx + 2]); idx += 3
        fixed[(r, c)] = v

    rng = random.Random(4242 + N)
    free = [(i, j) for i in range(N) for j in range(N) if (i, j) not in fixed]

    best = None
    best_ld = -float("inf")
    RESTARTS = 24
    for _ in range(RESTARTS):
        M = [[0] * N for _ in range(N)]
        for (r, c), v in fixed.items():
            M[r][c] = v
        for (i, j) in free:
            M[i][j] = rng.randint(0, 1)
        ld = logabsdet(M, N)
        if ld > best_ld:
            best_ld = ld
            best = [row[:] for row in M]

    if best is None:  # all singular fallback: identity on free diagonal
        best = [[0] * N for _ in range(N)]
        for (r, c), v in fixed.items():
            best[r][c] = v
        for i in range(N):
            if (i, i) not in fixed:
                best[i][i] = 1

    out = []
    for i in range(N):
        out.append(" ".join(str(x) for x in best[i]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
