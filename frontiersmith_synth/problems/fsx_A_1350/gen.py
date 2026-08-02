#!/usr/bin/env python3
"""gen.py <testId> -- vibrating-graph-spectrum instance generator.
Prints: a connected weighted-Laplacian topology (edges + weight bounds) and a
target spectrum. Deterministic: all randomness seeded from testId only.
"""
import sys, random, math


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
    return sv


def build_laplacian(n, edges, w):
    L = [[0.0] * n for _ in range(n)]
    for (u, v), we in zip(edges, w):
        L[u][u] += we
        L[v][v] += we
        L[u][v] -= we
        L[v][u] -= we
    return L


def random_connected_graph(rng, n, extra):
    """Random recursive-tree spanning structure + a few extra chords."""
    edges = []
    order = list(range(n))
    rng.shuffle(order)
    for i in range(1, n):
        j = order[rng.randrange(i)]
        edges.append((min(order[i], j), max(order[i], j)))
    have = set(edges)
    tries = 0
    while extra > 0 and tries < 50:
        tries += 1
        a, b = rng.randrange(n), rng.randrange(n)
        if a == b:
            continue
        e = (min(a, b), max(a, b))
        if e in have:
            continue
        have.add(e)
        edges.append(e)
        extra -= 1
    return edges


def main():
    testId = int(sys.argv[1])
    rng = random.Random(0xF57 * testId + 13)

    # scale ladder: small graphs throughout ("scale": small). Sizes are
    # deliberately NOT monotone in whether a case is a trap case, so the
    # trap/non-trap contrast below isolates the clustering effect rather
    # than just "bigger instances are harder".
    sizes = [4, 5, 6, 7, 8, 5, 6, 7, 8, 9]
    extras = [0, 0, 1, 1, 1, 0, 1, 2, 1, 2]
    TRAP_IDS = {2, 3, 4, 6, 7, 8, 10}  # 7 of the 10, spanning small->large
    n = sizes[(testId - 1) % len(sizes)]
    extra = extras[(testId - 1) % len(extras)]
    edges = random_connected_graph(rng, n, extra)
    m = len(edges)

    bounds = []
    for _ in edges:
        lo = rng.randint(1, 3)
        hi = lo + rng.randint(5, 9)
        bounds.append((lo, hi))

    # seed weights -> a feasible base spectrum
    w_seed = [rng.uniform(lo, hi) for (lo, hi) in bounds]
    L = build_laplacian(n, edges, w_seed)
    lam = jacobi_eigh(L, n)
    lam = [max(0.0, x) for x in lam]
    lam[0] = 0.0

    # trap cases: TRAP_IDS (7 of the 10, spanning small->large n) get
    # clustered targets -- pull 2-4 of the interior eigenvalues together
    # into a tight cluster so the per-eigenvalue-independent approach has
    # to fight nearly-degenerate directions.
    is_trap = testId in TRAP_IDS
    if is_trap:
        k = rng.randint(2, min(4, n - 1))
        idxs = sorted(rng.sample(range(1, n), k))
        center = sum(lam[i] for i in idxs) / k
        span = max(1e-6, 0.015 * max(1.0, center))
        for j, i in enumerate(idxs):
            jitter = (j - (k - 1) / 2.0) * span / max(1, k - 1) if k > 1 else 0.0
            lam[i] = max(0.0, center + jitter)
        lam[0] = 0.0
        lam = sorted(lam)
        lam[0] = 0.0

    target = [round(x, 6) for x in lam]
    target[0] = 0.0

    out = []
    out.append(f"{n} {m}")
    for (u, v), (lo, hi) in zip(edges, bounds):
        out.append(f"{u + 1} {v + 1} {lo} {hi}")
    out.append(" ".join(f"{x:.6f}" for x in target))
    print("\n".join(out))


if __name__ == "__main__":
    main()
