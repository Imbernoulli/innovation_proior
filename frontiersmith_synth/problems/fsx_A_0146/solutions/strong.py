# TIER: strong
"""Strong completion: multi-restart coordinate descent on |det|. Each restart
fills the free cells randomly (deterministic seeds), then repeatedly sweeps all
free cells flipping any sign that raises |det|, using a Sherman-Morrison rank-1
update of the inverse so each flip is evaluated in O(1) and applied in O(N^2).
Runs to convergence over several passes/restarts and keeps the best matrix. The
true optimum is unknown for these odd orders, so it never saturates the score."""
import sys, math


def inv_and_logdet(M):
    n = len(M)
    A = [[float(M[i][j]) for j in range(n)] + [1.0 if j == i else 0.0 for j in range(n)]
         for i in range(n)]
    logdet = 0.0
    for k in range(n):
        p = k
        mx = abs(A[k][k])
        for i in range(k + 1, n):
            if abs(A[i][k]) > mx:
                mx = abs(A[i][k]); p = i
        if mx < 1e-9:
            return None, -1e18
        if p != k:
            A[k], A[p] = A[p], A[k]
        piv = A[k][k]
        logdet += math.log(abs(piv))
        inv = 1.0 / piv
        row = A[k]
        for j in range(2 * n):
            row[j] *= inv
        for i in range(n):
            if i != k:
                f = A[i][k]
                if f != 0.0:
                    Ai = A[i]
                    for j in range(2 * n):
                        Ai[j] -= f * row[j]
    Minv = [A[i][n:] for i in range(n)]
    return Minv, logdet


def descend(M, free, N, max_passes):
    """In-place coordinate descent raising |det|; returns final logdet."""
    Minv, logdet = inv_and_logdet(M)
    if Minv is None:
        return -1e18
    for _ in range(max_passes):
        improved = False
        for (i, j) in free:
            delta = -2.0 * M[i][j]          # flip: new = -old
            factor = 1.0 + delta * Minv[j][i]
            if abs(factor) > 1.0 + 1e-7:
                # accept flip
                M[i][j] = -M[i][j]
                logdet += math.log(abs(factor))
                coef = delta / factor
                col_i = [Minv[r][i] for r in range(N)]
                row_j = Minv[j]
                for r in range(N):
                    c = coef * col_i[r]
                    Mr = Minv[r]
                    for cc in range(N):
                        Mr[cc] -= c * row_j[cc]
                improved = True
        # refresh inverse to kill float drift
        Minv, logdet = inv_and_logdet(M)
        if Minv is None:
            return -1e18
        if not improved:
            break
    return logdet


def main():
    toks = sys.stdin.read().split()
    N = int(toks[0])
    vals = list(map(int, toks[1:1 + N * N]))
    G = [vals[i * N:(i + 1) * N] for i in range(N)]
    free = [(i, j) for i in range(N) for j in range(N) if G[i][j] == 0]

    best_M = None
    best_ld = -1e18
    RESTARTS = 6
    for r in range(RESTARTS):
        M = [[G[i][j] for j in range(N)] for i in range(N)]
        st = (0x2545F491 + 2246822519 * (r + 1) + 3266489917 * N) & 0xffffffff
        for (i, j) in free:
            st = (st * 1103515245 + 12345) & 0x7fffffff
            M[i][j] = 1 if (st >> 9) & 1 else -1
        ld = descend(M, free, N, max_passes=12)
        if ld > best_ld:
            best_ld = ld
            best_M = [row[:] for row in M]

    out = [" ".join(str(best_M[i][j]) for j in range(N)) for i in range(N)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
