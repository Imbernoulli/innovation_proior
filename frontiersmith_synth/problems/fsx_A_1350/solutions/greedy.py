# TIER: greedy
"""The obvious recipe: at each step, find the single eigenvalue with the
worst error, find the single edge most sensitive to THAT eigenvalue alone
(the standard first-order eigenvalue-perturbation sensitivity), and nudge
just that edge toward fixing just that eigenvalue. Repeat.

This treats each target eigenvalue as an independent control problem. It
ignores that a rank-1 stiffness bump on one edge shifts EVERY eigenvalue at
once (Weyl / interlacing), so when several targets are clustered close
together the "worst eigenvalue" flips between the near-degenerate indices
from iteration to iteration and the edge weight oscillates instead of
converging.
"""
import sys, math


def jacobi_eigh(A, n, max_sweeps=100, tol=1e-13):
    A = [row[:] for row in A]
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(max_sweeps):
        off = sum(A[i][j] * A[i][j] for i in range(n) for j in range(n) if i != j)
        if off < tol:
            break
        for p in range(n):
            for q in range(p + 1, n):
                apq = A[p][q]
                if abs(apq) < 1e-300:
                    continue
                theta = (A[q][q] - A[p][p]) / (2.0 * apq)
                t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                app, aqq = A[p][p], A[q][q]
                A[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq
                A[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq
                A[p][q] = 0.0
                A[q][p] = 0.0
                for i in range(n):
                    if i != p and i != q:
                        aip, aiq = A[i][p], A[i][q]
                        A[i][p] = c * aip - s * aiq
                        A[p][i] = A[i][p]
                        A[i][q] = s * aip + c * aiq
                        A[q][i] = A[i][q]
                for i in range(n):
                    vip, viq = V[i][p], V[i][q]
                    V[i][p] = c * vip - s * viq
                    V[i][q] = s * vip + c * viq
    eigvals = [A[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda i: eigvals[i])
    sv = [eigvals[i] for i in order]
    sV = [[V[r][i] for i in order] for r in range(n)]
    return sv, sV


def build_laplacian(n, edges, w):
    L = [[0.0] * n for _ in range(n)]
    for (u, v), we in zip(edges, w):
        L[u][u] += we
        L[v][v] += we
        L[u][v] -= we
        L[v][u] -= we
    return L


def main():
    toks = sys.stdin.read().split()
    p = 0
    n = int(toks[p]); p += 1
    m = int(toks[p]); p += 1
    edges = []
    bounds = []
    for _ in range(m):
        u = int(toks[p]) - 1; p += 1
        v = int(toks[p]) - 1; p += 1
        lo = float(toks[p]); p += 1
        hi = float(toks[p]); p += 1
        edges.append((u, v))
        bounds.append((lo, hi))
    target = [float(toks[p + i]) for i in range(n)]

    w = [(lo + hi) / 2.0 for (lo, hi) in bounds]

    ITERS = 200
    for it in range(ITERS):
        L = build_laplacian(n, edges, w)
        lam, V = jacobi_eigh(L, n)

        # worst single eigenvalue (index 1..n-1; index 0 is the structural 0)
        worst_i, worst_res = 1, -1.0
        for i in range(1, n):
            r = abs(lam[i] - target[i])
            if r > worst_res:
                worst_res = r
                worst_i = i
        residual = target[worst_i] - lam[worst_i]
        if abs(residual) < 1e-9:
            break

        # edge most sensitive to THIS eigenvalue only (ignores coupling to
        # all the other eigenvalues -- the trap)
        best_e, best_sens = 0, -1.0
        for e, (u, v) in enumerate(edges):
            bv = V[u][worst_i] - V[v][worst_i]
            sens = bv * bv
            if sens > best_sens:
                best_sens = sens
                best_e = e

        # single-variable Newton step for THIS eigenvalue alone:
        # d(lambda_worst_i)/d(w_best_e) ~= best_sens, so move w_best_e by
        # residual / best_sens. This converges nicely when eigenvalues are
        # well separated (the local single-variable model is accurate), but
        # when targets are clustered the same edge is highly sensitive to
        # SEVERAL nearby eigenvalues at once (their eigenvectors overlap),
        # the single-variable model ignores that shared sensitivity, and the
        # "worst eigenvalue" flips between the cluster members each round --
        # so the edge weight overshoots back and forth instead of settling.
        lo, hi = bounds[best_e]
        span = hi - lo
        sens = max(best_sens, 1e-6)
        step = residual / sens
        trust = 0.5 * span
        step = max(-trust, min(trust, step))
        w[best_e] = min(hi, max(lo, w[best_e] + step))

    print(" ".join("%.6f" % x for x in w))


if __name__ == "__main__":
    main()
