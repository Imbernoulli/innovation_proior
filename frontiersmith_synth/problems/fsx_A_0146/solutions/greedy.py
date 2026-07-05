# TIER: greedy
"""Greedy completion: fill the free cells with a fixed pseudo-random sign
pattern, run TWO sweeps of single-cell sign flips that increase |det| (evaluated
with a fast floating-point determinant), and fall back to the triangular
baseline if that ever does worse. Beats the trivial baseline but stops well
short of the multi-restart 'strong' search -- one starting basin, no restarts."""
import sys, math


def logabsdet(A0):
    n = len(A0)
    A = [row[:] for row in A0]
    s = 0.0
    for k in range(n):
        p = k
        mx = abs(A[k][k])
        for i in range(k + 1, n):
            if abs(A[i][k]) > mx:
                mx = abs(A[i][k]); p = i
        if mx < 1e-12:
            return -1e18
        if p != k:
            A[k], A[p] = A[p], A[k]
        piv = A[k][k]
        s += math.log(abs(piv))
        for i in range(k + 1, n):
            f = A[i][k] / piv
            if f != 0.0:
                Ai = A[i]; Ak = A[k]
                for j in range(k + 1, n):
                    Ai[j] -= f * Ak[j]
    return s


def main():
    toks = sys.stdin.read().split()
    N = int(toks[0])
    vals = list(map(int, toks[1:1 + N * N]))
    G = [vals[i * N:(i + 1) * N] for i in range(N)]
    free = [(i, j) for i in range(N) for j in range(N) if G[i][j] == 0]

    # baseline fallback (triangular sign + fixed)
    B = [[1 if j <= i else -1 for j in range(N)] for i in range(N)]
    for i in range(N):
        for j in range(N):
            if G[i][j] != 0:
                B[i][j] = G[i][j]

    # deterministic pseudo-random init
    M = [[G[i][j] for j in range(N)] for i in range(N)]
    st = 0x9e3779b9 ^ (N * 2654435761)
    for (i, j) in free:
        st = (st * 1103515245 + 12345) & 0x7fffffff
        M[i][j] = 1 if (st >> 8) & 1 else -1

    cur = logabsdet(M)
    for _ in range(2):
        for (i, j) in free:
            M[i][j] = -M[i][j]
            cand = logabsdet(M)
            if cand > cur + 1e-9:
                cur = cand
            else:
                M[i][j] = -M[i][j]

    if logabsdet(B) > cur:      # never do worse than the baseline
        M = B

    out = [" ".join(str(M[i][j]) for j in range(N)) for i in range(N)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
