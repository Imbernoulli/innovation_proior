#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- checker for vibrating-graph-spectrum.
Deterministic: no randomness, no wall-time. Prints 'Ratio: <float in [0,1]>'.
"""
import sys, math


def jacobi_eigh(A, n, max_sweeps=100, tol=1e-13):
    A = [row[:] for row in A]
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
    eigvals = [A[i][i] for i in range(n)]
    return sorted(eigvals)


def build_laplacian(n, edges, w):
    L = [[0.0] * n for _ in range(n)]
    for (u, v), we in zip(edges, w):
        L[u][u] += we
        L[v][v] += we
        L[u][v] -= we
        L[v][u] -= we
    return L


def spectral_error(n, edges, w, target):
    L = build_laplacian(n, edges, w)
    lam = jacobi_eigh(L, n)
    lam = sorted(max(0.0, x) for x in lam)
    # skip index 0: every weighted Laplacian has a structural 0 eigenvalue
    # (all-ones vector), so it carries no placement information.
    m2 = sum((lam[i] - target[i]) ** 2 for i in range(1, n))
    return math.sqrt(m2 / (n - 1))


def quality(err, scale):
    eps = 0.018 * scale
    return 1.0 / (err + eps)


def fail(msg):
    print("INFEASIBLE: %s" % msg)
    print("Ratio: 0.0")
    sys.exit(0)


def main():
    inf, outf = sys.argv[1], sys.argv[2]

    with open(inf) as f:
        toks = f.read().split()
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

    try:
        with open(outf) as f:
            out_toks = f.read().split()
    except FileNotFoundError:
        fail("no output produced")

    if len(out_toks) != m:
        fail("expected %d weights, got %d" % (m, len(out_toks)))

    w = []
    for tok in out_toks:
        try:
            v = float(tok)
        except ValueError:
            fail("non-numeric token %r" % tok)
        if not math.isfinite(v):
            fail("non-finite weight %r" % tok)
        w.append(v)

    EPS = 1e-6
    for i, (val, (lo, hi)) in enumerate(zip(w, bounds)):
        if val < lo - EPS or val > hi + EPS:
            fail("edge %d weight %.6f out of bounds [%.6f, %.6f]" % (i, val, lo, hi))

    scale = max(1e-6, target[-1])
    err = spectral_error(n, edges, w, target)
    F = quality(err, scale)

    # internal baseline: the trivial feasible construction (midpoint weights),
    # scored the same way the participant's artifact is scored.
    w_base = [(lo + hi) / 2.0 for (lo, hi) in bounds]
    err_base = spectral_error(n, edges, w_base, target)
    B = quality(err_base, scale)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%.8f B=%.8f err=%.8f err_base=%.8f" % (F, B, err, err_base))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
