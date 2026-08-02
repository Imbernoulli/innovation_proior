# TIER: strong
"""Interlacing-aware joint placement.

Rank-1 fact: L(w) = sum_e w_e * b_e b_e^T with each b_e b_e^T positive
semidefinite (b_e is the +-1 incidence vector of edge e). Raising any single
w_e is therefore a positive rank-1 bump, so by Weyl's monotonicity /
eigenvalue interlacing theorem it moves EVERY eigenvalue, not just the one an
edge "belongs to" -- and it moves them in a constrained relative order.
Chasing one eigenvalue at a time (the greedy recipe) fights this coupling and
oscillates whenever targets are clustered, because nearby eigenvalues share
almost the same sensitive directions.

The insight used here: solve for the weight UPDATE that respects the full
coupling at once. First-order eigenvalue perturbation gives an exact local
Jacobian d(lambda_i)/d(w_e) = (b_e . v_i)^2 (v_i = current eigenvector). We
build that Jacobian for ALL target eigenvalues simultaneously and solve the
regularized joint least-squares system

    (J^T J + mu I) dw = J^T r

(a damped Gauss-Newton / Levenberg-Marquardt step) instead of moving one edge
for one eigenvalue at a time. This lets weight increases and decreases on
different edges cancel out on the eigenvalues that are already correct while
still pushing the ones that are off -- the correct ORDER-respecting move that
plain per-eigenvalue gradient descent cannot represent.
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


def solve_linear(Amat, bvec, k):
    """Gaussian elimination with partial pivoting on a k x k system."""
    A = [row[:] + [bvec[i]] for i, row in enumerate(Amat)]
    for col in range(k):
        piv = max(range(col, k), key=lambda r: abs(A[r][col]))
        if abs(A[piv][col]) < 1e-14:
            continue
        A[col], A[piv] = A[piv], A[col]
        pv = A[col][col]
        for j in range(col, k + 1):
            A[col][j] /= pv
        for r in range(k):
            if r != col:
                f = A[r][col]
                if f != 0.0:
                    for j in range(col, k + 1):
                        A[r][j] -= f * A[col][j]
    return [A[i][k] for i in range(k)]


def residual_norm(n, edges, w, target):
    L = build_laplacian(n, edges, w)
    lam, _ = jacobi_eigh(L, n)
    lam = sorted(lam)
    return sum((lam[i] - target[i]) ** 2 for i in range(1, n))


def optimize(w0, n, m, edges, bounds, target, iters=80):
    """Damped Gauss-Newton / Levenberg-Marquardt on the joint eigenvalue
    Jacobian, with adaptive damping (mu grows on a failed step, shrinks on a
    successful one) so it keeps making progress even where the Jacobian is
    ill-conditioned (near-degenerate / clustered target eigenvalues)."""
    w = list(w0)
    k = n - 1
    mu_scale = None
    mu_mult = 1.0
    for _ in range(iters):
        L = build_laplacian(n, edges, w)
        lam, V = jacobi_eigh(L, n)
        r = [target[i] - lam[i] for i in range(1, n)]
        base_norm = sum(x * x for x in r)
        if base_norm < 1e-16:
            break

        J = [[0.0] * m for _ in range(k)]
        for e, (u, v) in enumerate(edges):
            for i in range(k):
                bv = V[u][i + 1] - V[v][i + 1]
                J[i][e] = bv * bv

        JTJ = [[sum(J[t][a] * J[t][b] for t in range(k)) for b in range(m)] for a in range(m)]
        JTr = [sum(J[t][a] * r[t] for t in range(k)) for a in range(m)]
        if mu_scale is None:
            mu_scale = max(1e-9, sum(JTJ[a][a] for a in range(m)) / max(1, m))

        improved = False
        for _try in range(8):
            mu = mu_mult * mu_scale
            M = [row[:] for row in JTJ]
            for a in range(m):
                M[a][a] += mu
            dw = solve_linear(M, JTr, m)
            trial = [min(bounds[e][1], max(bounds[e][0], w[e] + dw[e])) for e in range(m)]
            nrm = residual_norm(n, edges, trial, target)
            if nrm < base_norm:
                w = trial
                mu_mult = max(1e-6, mu_mult * 0.5)
                improved = True
                break
            mu_mult *= 3.0
        if not improved:
            break
    L = build_laplacian(n, edges, w)
    lam, _ = jacobi_eigh(L, n)
    lam = sorted(lam)
    final_norm = sum((lam[i] - target[i]) ** 2 for i in range(1, n))
    return w, final_norm


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

    # deterministic multi-start: the joint Gauss-Newton solve can land in
    # different basins depending on where it starts, so try a few structured
    # starting points (no randomness) and keep whichever converges best.
    starts = [
        [(lo + hi) / 2.0 for (lo, hi) in bounds],
        [lo + 0.25 * (hi - lo) for (lo, hi) in bounds],
        [lo + 0.75 * (hi - lo) for (lo, hi) in bounds],
    ]

    best_w, best_norm = None, float("inf")
    for w0 in starts:
        w, nrm = optimize(w0, n, m, edges, bounds, target)
        if nrm < best_norm:
            best_norm = nrm
            best_w = w

    print(" ".join("%.6f" % x for x in best_w))


if __name__ == "__main__":
    main()
